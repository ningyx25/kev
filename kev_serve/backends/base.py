"""Backend contract and the checkpoint-sniffing registry.

Adding a backend = a package under kev_serve/backends/ that calls register_backend(kind, sniff, factory)
at import time (and is imported by backends/__init__.py, or ships an entry point in the group
"kev_serve.backends" for third-party discovery). The HTTP layer and ModelRouter never change.

Box is reserved: a backend that can answer box questions implements answer_box() and includes "box" in
supported_primitives (planned: GUI-Actor's per-patch distribution aggregated into a connected region,
normalised to the four edges). Until one registers, the HTTP layer answers box questions with 501.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

from ..api.primitives import BoxAnswer


class Backend(Protocol):
    kind: str                                  # family name in the /v1/models card
    supported_primitives: frozenset[str]       # subset of api.QUESTION_TYPES this backend answers
    release_date: str

    def answer(self, req_dict: dict[str, Any]) -> dict[str, Any]: ...
    def answer_box(self, instructions: str, images: list) -> BoxAnswer:
        """Answer one box question against `images` (PIL). Default: not implemented — a backend that
        puts "box" in supported_primitives must override this."""
        raise NotImplementedError(f"the {self.kind} backend does not implement answer_box")

    def model_card(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


# kind -> (sniff(path) -> bool, factory(path, device, dtype) -> Backend)
_REGISTRY: dict[str, tuple[Callable[[str], bool], Callable[..., "Backend"]]] = {}


def register_backend(kind: str, sniff: Callable[[str], bool], factory: Callable[..., "Backend"]) -> None:
    if kind in _REGISTRY: raise ValueError(f"backend kind {kind!r} is already registered")
    _REGISTRY[kind] = (sniff, factory)


def load_backend(path: str, device: str, dtype) -> "Backend":
    """Sniff the checkpoint directory with each registered backend, load the first match."""
    for kind, (sniff, factory) in _REGISTRY.items():
        if sniff(path):
            return factory(path, device, dtype)
    known = " / ".join(sorted(_REGISTRY)) or "none registered"
    raise ValueError(f"{path!r} matches no registered backend ({known})")


def discover_entry_points() -> None:
    """Import every third-party module that registers a backend via the kev_serve.backends entry-point group."""
    from importlib.metadata import entry_points
    for ep in entry_points(group="kev_serve.backends"):
        ep.load()
