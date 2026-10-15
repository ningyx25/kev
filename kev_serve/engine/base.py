"""Execution engines: how a backend's forward passes are scheduled.

An engine turns a blocking call into an awaitable for the HTTP layer. ClefBackend uses SyncLockEngine
(a lock plus the event loop's executor). The slot for the future kev pointer-head backend is QueueBatchEngine
(one model thread draining a queue into batches, with a state-prefix cache) — same interface, no HTTP changes
when it arrives."""
import asyncio
import threading
from typing import Any, Awaitable, Callable, Protocol


class ExecutionEngine(Protocol):
    def submit(self, fn: Callable[[], Any]) -> Awaitable[Any]: ...
    def close(self) -> None: ...


class SyncLockEngine:
    """Serialise blocking calls through one lock and run them on the event loop's default executor.
    Right for backends with a single blocking forward pass per request (clef, GUI-Actor)."""

    def __init__(self):
        self.lock = threading.Lock()

    def _run(self, fn):
        with self.lock:
            return fn()

    async def submit(self, fn: Callable[[], Any]) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, fn)

    def close(self) -> None:
        pass
