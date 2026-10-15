"""Backend abstraction: which model family answers a /v1/systemone request.

A backend declares the primitives it answers (SUPPORTED_PRIMITIVES); the HTTP layer (serve.py) intercepts
the rest (box -> 501 until a backend registers for it). Adding the kev pointer head or GUI-Actor later means
implementing Backend and registering it in load_backend() — the routes do not change."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from .clef_model import is_clef, load_clef_checkpoint


class Backend(Protocol):
    kind: str                                  # family name in the /v1/models card
    supported_primitives: frozenset[str]       # subset of api.QUESTION_TYPES this backend answers
    release_date: str

    def answer(self, req_dict: dict[str, Any]) -> dict[str, Any]: ...
    def model_card(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


SUPPORTED_PRIMITIVES = {"clef": frozenset({"noul", "choice", "score"})}


@dataclass
class ClefBackend:
    """clef-flash JointSchemaHead on Qwen3.5 VL (kev_serve.clef_model). Requests are serialised through a lock:
    one blocking forward pass per request."""
    model: object
    processor: object
    requested: str                             # checkpoint path as given on the command line
    device: str
    release_date: str
    kind: str = "clef"
    supported_primitives: frozenset[str] = field(default_factory=lambda: SUPPORTED_PRIMITIVES["clef"])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def answer(self, req_dict: dict[str, Any]) -> dict[str, Any]:
        from .clef_model import systemone as clef_systemone
        with self.lock:
            return clef_systemone(self.model, self.processor, req_dict)

    def model_card(self) -> dict[str, Any]:
        return {"description": f"Clef joint-schema head on Qwen3.5, serving {self.requested}",
                "supported_primitives": sorted(self.supported_primitives)}

    def close(self) -> None:
        pass   # no background thread


def file_release_date(path: str) -> str:
    """ISO date from the newest file mtime in `path` (local checkpoints without a Hub commit date)."""
    import datetime
    from pathlib import Path
    MTIME_FLOOR = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    files = sorted((p.stat().st_mtime for p in Path(path).iterdir() if p.is_file()), reverse=True)
    stamp = next((t for t in files if t >= MTIME_FLOOR), None)
    return "unknown" if stamp is None else datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).date().isoformat()


def load_backend(path: str, device: str, dtype) -> Backend:
    """Sniff the checkpoint directory and load the matching backend. Unknown shapes raise ValueError."""
    if is_clef(path):
        model, processor = load_clef_checkpoint(path, device, dtype)
        return ClefBackend(model, processor, path, device, file_release_date(path))
    raise ValueError(f"{path!r} is not a checkpoint this service knows how to load "
                     f"(clef: a directory with joint_head_config.json + joint_head.safetensors)")
