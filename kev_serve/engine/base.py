"""Execution engines: how a backend's forward passes are scheduled.

An engine is bound to one backend instance (its `.engine`) and runs that backend's requests off the
event loop: `run(d) -> response body`. SyncLockEngine serialises blocking calls through one lock (a
backend with a single blocking forward pass and no batching). QueueBatchEngine (engine/batch.py) is
the continuous-batching one: a model thread drains a queue into batches and runs them through the
backend's answer_batch(_with_prefix), with the state-prefix cache and OOM resilience.
"""
import asyncio
import threading
from typing import Protocol


class ExecutionEngine(Protocol):
    async def run(self, d: dict) -> dict:
        """Answer one request dict (the shape backends.base.Backend.answer takes)."""
        ...

    def close(self) -> None: ...

    def stats(self) -> dict | None:
        """Batching/caching counters for the /v1/models card; None when the engine keeps none."""
        ...


class SyncLockEngine:
    """Serialise blocking calls through one lock and run them on the event loop's default executor.
    Right for backends with a single blocking forward pass per request and no batching (GUI-Actor)."""

    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()

    def _run(self, fn):
        with self.lock:
            return fn()

    async def run(self, d: dict) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, lambda: self.backend.answer(d))

    def close(self) -> None:
        pass

    def stats(self) -> dict | None:
        return None
