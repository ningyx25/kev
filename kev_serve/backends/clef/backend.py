"""ClefBackend: a thin serving wrapper around the vendored ClefDecisionModel.

Images arrive decoded (PIL) in req_dict["images"]; the HTTP layer does base64 / assets handling."""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from ..base import Backend
from ...engine import SyncLockEngine

SUPPORTED_PRIMITIVES = frozenset({"noul", "choice", "score"})


def file_release_date(path: str) -> str:
    """ISO date from the newest file mtime in `path` (local checkpoints without a Hub commit date)."""
    MTIME_FLOOR = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    files = sorted((p.stat().st_mtime for p in Path(path).iterdir() if p.is_file()), reverse=True)
    stamp = next((t for t in files if t >= MTIME_FLOOR), None)
    return "unknown" if stamp is None else datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).date().isoformat()


class ClefBackend:
    """Answers /v1/systemone requests through clef's systemone(); every pass runs on the engine
    (SyncLockEngine: one blocking forward at a time, off the event loop)."""
    kind = "clef"
    supported_primitives = SUPPORTED_PRIMITIVES

    def __init__(self, model, processor, requested: str, device: str, engine: SyncLockEngine | None = None):
        self.model, self.processor = model, processor
        self.requested, self.device = requested, device
        self.release_date = file_release_date(requested)
        self.engine = engine or SyncLockEngine()

    def answer(self, req_dict: dict[str, Any]) -> dict[str, Any]:
        from .clef_model import systemone as clef_systemone
        return clef_systemone(self.model, self.processor, req_dict)

    def model_card(self) -> dict[str, Any]:
        return {"description": f"Clef joint-schema head on Qwen3.5, serving {self.requested}",
                "supported_primitives": sorted(self.supported_primitives)}

    def close(self) -> None:
        self.engine.close()
