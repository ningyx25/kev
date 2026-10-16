from .base import ExecutionEngine, SyncLockEngine
from .batch import QueueBatchEngine
from .prefix_cache import PrefixCache


def make_engine(backend, settings) -> ExecutionEngine:
    """The engine a backend runs on: QueueBatchEngine for one that declares a cache hook
    (answer_batch / answer_batch_with_prefix — both clef and gui_actor do), SyncLockEngine otherwise.
    A backend with neither caches nothing and needs no model thread. settings=None (library callers,
    tests) always takes the lock."""
    if settings is not None and (hasattr(backend, "answer_batch_with_prefix") or hasattr(backend, "answer_batch")):
        return QueueBatchEngine(backend, settings)
    return SyncLockEngine(backend)


__all__ = ["ExecutionEngine", "PrefixCache", "QueueBatchEngine", "SyncLockEngine", "make_engine"]
