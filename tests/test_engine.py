"""PrefixCache + QueueBatchEngine (P1-1/P1-2/P1-3): the ported LRU semantics, the batch path, and the
one-shot OOM retry. Pure Python, CPU-only — these are the baselines the old monorepo pinned."""
import threading
import weakref

import pytest
import torch

from kev_serve.config import Settings
from kev_serve.engine import QueueBatchEngine, SyncLockEngine
from kev_serve.engine.prefix_cache import PrefixCache


def enc(state, n=3):
    """A cacheable item stands for one request: the old encs carried ids/seg; the generalised plan() takes
    (key, n_tokens) straight."""
    return (tuple(state), n) if state else (None, 0)


def test_prefix_cache_keeps_what_survives_the_batch():
    """A batch keeps only its last `size` distinct cacheable states; hits are reinserted as most recent,
    short states and size 0 are never cached."""
    c = PrefixCache(size=2, min_tokens=3)
    batch = [enc("abc"), enc("abd"), enc("abe"), enc("abd"), enc("ab", 2)]
    keys, cached, keep = c.plan(batch)
    assert cached == [None] * 5 and keep == [False, True, True, True, False] and keys[4] is None
    c.store(keys, cached, [None, "p2", "p3", "p2", None])
    assert list(c.entries.values()) == ["p3", "p2"] and (c.hits, c.misses) == (0, 4)
    keys, cached, keep = c.plan([enc("abe"), enc("abf")])
    assert cached == ["p3", None] and keep == [True, True]
    c.store(keys, cached, ["p3", "p4"])
    assert list(c.entries.values()) == ["p3", "p4"] and c.hits == 1
    assert PrefixCache(size=0, min_tokens=0).plan([enc("abc")])[2] == [False]


def test_prefix_cache_bounds_the_state_tokens_it_holds():
    """A key is the state's token-id tuple; its length is the token count the budget is drawn against."""
    c = PrefixCache(size=4, min_tokens=1, max_tokens=10)
    assert c.plan([(tuple(range(12)), 12)])[0] == [None]      # one state longer than the whole budget
    four = [tuple(range(i * 100, i * 100 + 4)) for i in range(3)]        # three distinct 4-token states: 12 > 10
    _, _, keep = c.plan([(k, 4) for k in four])
    assert keep == [False, True, True]                        # keep from most recent while they fit (4, then 8)
    assert PrefixCache(size=4, min_tokens=1, max_tokens=10).plan([(four[0], 4), (four[1], 4)])[2] == [True, True]
    c.store([(k if kp else None) for k, kp in zip(four, keep)], [None] * 3, ["p" if kp else None for kp in keep])
    assert sum(len(k) for k in c.entries) <= 10


def test_prefix_cache_makes_room_before_the_batch():
    """make_room drops exactly what store() evicts: with and without it the cache ends identical."""
    import random
    for size, bound in ((2, 12), (3, 20), (1, 6)):
        rng = random.Random(size)
        a = PrefixCache(size, 1, bound); b = PrefixCache(size, 1, bound)
        for _ in range(300):
            batch = [enc("".join(rng.choice("abcd") for _ in range(rng.randint(1, 5)))) for _ in range(rng.randint(1, 4))]
            ka, ca, ke = a.plan(batch)
            a.make_room(ka, ca, ke)
            _, _, _ = b.plan(batch)                           # b never calls make_room
            pa = ["p" for k, c_, p in zip(ka, ca, ke) if k is not None and (c_ is not None or p)] or [None] * len(batch)
            a.store(ka, ca, pa); b.store(ka, ca, pa)
        assert list(a.entries) == list(b.entries) and (a.hits, a.misses) == (b.hits, b.misses)


class _Pass: pass   # weakref supports instances of user classes, not bare object()


class CacheModel:
    """A fake batch-capable backend for the engine: records calls, can fail on cache hits (an OOM stand-in)."""
    prefix_min_tokens = 1

    def __init__(self, fail=None):
        self.fail, self.calls, self.passes = fail, 0, []
        self.supported_primitives = frozenset({"noul"})
        self.kind = "fake"; self.device = "cpu"; self.requested = "x"; self.release_date = "2026-01-01"

    def prefix_key(self, d):
        return (tuple(d["state"]), len(d["state"]))

    def answer_batch_with_prefix(self, reqs, cached, keep):
        self.calls += 1
        holder = _Pass(); self.passes.append(weakref.ref(holder))
        if self.fail == "always" or self.fail == "cached" and any(c is not None for c in cached):
            raise torch.OutOfMemoryError("CUDA out of memory")
        if self.fail == "other":
            raise ValueError("not memory")
        return ([{"answers": {}} for _ in reqs], [ ("prefix", self.calls) if k else None for k in keep ])

    def answer(self, d): return {"answers": {}}
    def model_card(self): return {}
    def close(self): pass


def _engine(backend):
    return QueueBatchEngine(backend, Settings(prefix_min_tokens=1))


def _run(engine, state):
    import asyncio
    return asyncio.run(engine.run({"model": "m", "state": state, "questions": {"n": {"type": "noul", "instructions": "?"}}}))


def test_out_of_memory_drops_the_prefix_cache_and_retries_once():
    b = CacheModel()
    e = _engine(b)
    try:
        _run(e, "abc")
        assert len(e.prefix_cache.entries) == 1
        b.fail, b.calls = "cached", 0
        _run(e, "abc")                                   # the hit fails, the retry runs it as a miss
        assert b.calls == 2 and e.prefix_cache.oom_retries == 1
        assert list(e.prefix_cache.entries.values()) == [("prefix", 2)]     # only the retry's prefix
        b.fail, b.calls = "always", 0
        with pytest.raises(torch.OutOfMemoryError): _run(e, "abc")
        assert b.calls == 2 and e.prefix_cache.entries == {} and e.prefix_cache.oom_retries == 2
        e.wait_idle()
        assert all(ref() is None for ref in b.passes), "a failed batch's pass outlives it"
        b.calls = 0
        with pytest.raises(torch.OutOfMemoryError): _run(e, "zzz")   # nothing cached: nothing to drop
        assert b.calls == 1 and e.prefix_cache.oom_retries == 2
        b.fail = None; _run(e, "abc"); b.fail, b.calls = "other", 0
        with pytest.raises(ValueError): _run(e, "abc")
        assert b.calls == 1 and len(e.prefix_cache.entries) == 1      # a non-OOM error is never retried
    finally:
        e.close()


def test_batch_drains_the_queue_into_one_call():
    """Concurrent run()s become one model pass (the old Server's MAX_BATCH behaviour)."""
    class CountingBackend(CacheModel):
        barrier = threading.Barrier(3, timeout=2)
        def answer_batch_with_prefix(self, reqs, cached, keep):
            self.calls += len(reqs)
            return [{"answers": {}, "n": len(reqs)} for _ in reqs], [None] * len(reqs)
    b = CountingBackend()
    e = _engine(b)
    import asyncio
    async def three():
        return await asyncio.gather(*[e.run({"model": "m", "state": f"s{i}", "questions": {"n": {"type": "noul", "instructions": "?"}}}) for i in range(3)])
    out = asyncio.run(three())
    e.close()
    assert all(r["n"] == 3 for r in out) or b.calls == 3        # batched together, or at worst 3 singles
    assert "latency_ms" in out[0]


def test_prefix_min_tokens_and_stats():
    b = CacheModel()
    e = _engine(b)
    _run(e, "abc")
    s = e.stats()
    assert s["prefix_cache"]["hits"] == 0 and s["prefix_cache"]["misses"] == 1
    assert s["batches"]["requests"] == 1
    e.close()


def test_out_of_memory_predicate():
    from kev_serve.device import out_of_memory
    assert out_of_memory(RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"))
    assert not out_of_memory(RuntimeError("shape mismatch"))
