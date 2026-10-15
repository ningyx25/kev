"""GuiActorBackend: serving wrapper around the vendored GUI-Actor pointer head.

Images arrive decoded (PIL) in req_dict["images"]; the HTTP layer does base64 / assets handling.
Requests run on the engine (SyncLockEngine): every grounding pass is blocking, batch=1."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...api.primitives import GroundAnswer
from ...engine import SyncLockEngine

SUPPORTED_PRIMITIVES = frozenset({"ground"})
# Alternative region centres a ground answer carries; KEV_GUI_ACTOR_TOPK=0 keeps only the point and confidence.
TOPK = int(os.environ.get("KEV_GUI_ACTOR_TOPK", "5"))


def file_release_date(path: str) -> str:
    """ISO date from the newest file mtime in `path` (local checkpoints without a Hub commit date)."""
    import datetime
    MTIME_FLOOR = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    files = sorted((p.stat().st_mtime for p in Path(path).iterdir() if p.is_file()), reverse=True)
    stamp = next((t for t in files if t >= MTIME_FLOOR), None)
    return "unknown" if stamp is None else datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).date().isoformat()


class GuiActorBackend:
    kind = "gui_actor"
    supported_primitives = SUPPORTED_PRIMITIVES

    def __init__(self, model, processor, requested: str, device: str, model_type: str,
                 topk: int = TOPK, temperature: float | None = None, engine=None):
        self.model, self.processor = model, processor
        self.requested, self.device = requested, device
        self.model_type = model_type            # transformers model_type (qwen2_5_vl / qwen3_vl / qwen3_5 / ...)
        self.topk = topk
        self.temperature_override = temperature # KEV_TEMPERATURE; None = the checkpoint's fitted pointer_temperature
        self.release_date = file_release_date(requested)
        self.engine = engine or SyncLockEngine(self)

    def answer(self, req_dict: dict[str, Any]) -> dict[str, Any]:
        """One grounding pass per distinct instruction; each answer validated against the
        GroundAnswer contract (0-999 point, confidence in [0, 1]) before it is returned."""
        from .gui_actor_model import systemone as gui_actor_systemone
        try:
            resp = gui_actor_systemone(self.model, self.processor, req_dict, self.model_type, topk=self.topk,
                                       temperature=self.temperature_override)
        except ValueError as e:   # no image, or a question type this backend does not answer
            from fastapi import HTTPException
            raise HTTPException(422, str(e))
        resp["answers"] = {qid: GroundAnswer(**a).model_dump() for qid, a in resp["answers"].items()}
        return resp

    def answer_box(self, instructions: str, images: list):
        """Box is not implemented by this backend (its per-patch distribution gives a point /
        region, not calibrated edges); a future head may override this."""
        raise NotImplementedError("gui_actor answers ground, not box")

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", None)

    @property
    def temperature(self) -> float:
        """The calibration the confidence is produced at: KEV_TEMPERATURE, else the fitted pointer_temperature."""
        from .gui_actor_model import pointer_temperature
        return pointer_temperature(self.model, self.temperature_override)

    def model_card(self) -> dict[str, Any]:
        from .gui_actor_model import get_merged_token_pixels
        return {"description": f"GUI-Actor pointer head on {self.model_type}, serving {self.requested}",
                "supported_primitives": sorted(self.supported_primitives),
                "model_type": self.model_type, "patch_pixels": get_merged_token_pixels(self.model_type),
                "temperature": self.temperature, "topk": self.topk}

    def close(self) -> None:
        self.engine.close()
