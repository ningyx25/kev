"""QueueBatchEngine: continuous batching, the state-prefix cache, and OOM resilience.

Ported from kev/serve.py's Server model thread. One daemon thread drains a queue into batches (up to
settings.max_batch) and runs each batch through the backend's answer_batch_with_prefix / answer_batch /
answer; the batch's model time is what every response reports as latency_ms (not the queue wait). The
prefix cache is per engine (per backend): its entries hold that model's KV / recurrent states.

A pass that runs out of device memory while states are cached drops the cache and runs once more: the
cache only saves time, and kept resident it failed every later batch of that size (#75).
"""
from __future__ import annotations

import asyncio
import atexit
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import Future

from ..device import empty_cache, out_of_memory, sync
from .prefix_cache import PrefixCache


class QueueBatchEngine:
    """The model thread, the queue it drains, and the prefix cache the batches share."""

    def __init__(self, backend, settings):
        self.backend = backend
        self.device = getattr(backend, "device", "cpu")
        self.max_batch = settings.max_batch
        min_tokens = settings.prefix_min_tokens
        if min_tokens is None: min_tokens = int(getattr(backend, "prefix_min_tokens", 0) or 0)
        self.prefix_cache = PrefixCache(settings.prefix_cache, min_tokens, settings.prefix_max_tokens)
        self.lock = threading.Lock()
        self.batches = 0
        self.batched_requests = 0
        self.queue: queue.Queue = queue.Queue()
        self.stopping = threading.Event()
        # the model thread gives up the GIL at every CUDA sync and waits to get it back while the event
        # loop parses and answers requests; at Python's default 5 ms switch interval those waits
        # stretched a batch's model time ~2x. Process-wide, so close() puts it back.
        self.switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)
        self.thread = threading.Thread(target=self._work, name="kev-serve-model", daemon=True)
        self.thread.start()
        atexit.register(self.close)   # a daemon thread killed inside a CUDA call at interpreter exit aborts the process

    # ── request side (the event loop) ────────────────────────────────────────────────────────

    async def run(self, d: dict) -> dict:
        """Queue one request; the model thread batches it and answers. -> the response body."""
        if self.stopping.is_set(): raise RuntimeError("the server is stopping")
        done: Future = Future()
        self.queue.put((d, done))
        return await asyncio.wrap_future(done)

    def wait_idle(self):
        """Block until every queued request is answered (benchmarks, warm-up)."""
        self.queue.join()

    def close(self):
        """Stop the model thread after its current batch; requests still queued fail, and so do later ones."""
        if self.stopping.is_set(): return
        self.stopping.set(); self.thread.join()
        while not self.queue.empty():
            self.queue.get_nowait()[1].set_exception(RuntimeError("the server stopped")); self.queue.task_done()
        sys.setswitchinterval(self.switch_interval)

    def stats(self) -> dict:
        return {"batches": {"count": self.batches, "requests": self.batched_requests, "queued": self.queue.qsize()},
                "prefix_cache": self.prefix_cache.stats()}

    # ── model thread ─────────────────────────────────────────────────────────────────────────

    def _work(self):
        while not self.stopping.is_set():
            try: batch = [self.queue.get(timeout=0.05)]
            except queue.Empty: continue
            while len(batch) < self.max_batch:
                try: batch.append(self.queue.get_nowait())
                except queue.Empty: break
            try:
                with self.lock: results = self._run([d for d, _ in batch])
            except Exception as e:   # every request of the batch gets the error; the thread lives on
                traceback.clear_frames(e.__traceback__)   # the batch's futures keep the exception, and its frames held the failed pass's tensors; the traceback keeps its lines
                results = [e] * len(batch)
            for (_, done), result in zip(batch, results):
                (done.set_exception if isinstance(result, Exception) else done.set_result)(result)
                self.queue.task_done()

    def _batch_call(self, reqs, cached, keep):
        """One batch through the backend. -> (responses, prefixes) — prefixes all None when the backend
        does not cache."""
        b = self.backend
        if hasattr(b, "answer_batch_with_prefix"):
            return b.answer_batch_with_prefix(reqs, cached, keep)
        if hasattr(b, "answer_batch"):
            return b.answer_batch(reqs), [None] * len(reqs)
        return [b.answer(d) for d in reqs], [None] * len(reqs)

    def _prefix_items(self, reqs):
        """(key, n_tokens) per request; (None, 0) for a backend or request that cannot be cached."""
        hook = getattr(self.backend, "prefix_key", None)
        if hook is None: return [(None, 0)] * len(reqs)
        return [hook(d) or (None, 0) for d in reqs]

    def _run(self, reqs):
        """One batch, with the prefix cache and the one-shot OOM retry. -> one response per request,
        each carrying latency_ms (the batch's model time)."""
        sync(self.device); t = time.time()
        items = self._prefix_items(reqs)
        for retry in (False, True):
            keys, cached, keep = self.prefix_cache.plan(items)
            self.prefix_cache.make_room(keys, cached, keep)
            try:
                resps, prefixes = self._batch_call(reqs, cached, keep); break
            except Exception as e:   # states held: the cache's, and this batch's hits make_room already dropped from it
                held = self.prefix_cache.entries or any(c is not None for c in cached)
                if retry or not held or not out_of_memory(e): raise
            cached = None; self.prefix_cache.clear(); self.prefix_cache.oom_retries += 1   # after the except: its traceback holds the failed pass's tensors
            empty_cache(self.device)
        sync(self.device); dt = round((time.time() - t) * 1000, 1)
        self.prefix_cache.store(keys, cached, prefixes)
        self.batches += 1; self.batched_requests += len(reqs)
        # latency_ms is the batch's model time (the old Server's semantics), not the request's queue wait
        return [{**resp, "latency_ms": dt} for resp in resps]
