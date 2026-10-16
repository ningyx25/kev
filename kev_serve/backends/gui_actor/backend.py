"""GuiActorBackend: serving wrapper around the vendored GUI-Actor pointer head.

Images arrive decoded (PIL) in req_dict["images"]; the HTTP layer does base64 / assets handling.

Each grounding pass is blocking and batch=1 (collating gains nothing), so the backend declares no
plain answer_batch; it declares answer_batch_with_prefix instead, which is what carries the cache —
that also routes it to the QueueBatchEngine. The cached unit is the image prefix (system + user
header + screenshot, dominating the prefill — only the instruction varies between requests): it runs
once and every later request over the same (image, system text) reads the cache. The pointer head's image side (the embed-layer hidden state at the
image tokens) is cached with the KV. The key is (image bytes, system text): both must match, since
either can move the prefix. The continuation's M-RoPE 3D position ids are the tail of the joined
sequence's ids (get_rope_index over the full prompt, sliced per stage) — letting a stale rope_deltas
stand in does not reproduce a one-pass forward (probed: cached vs one-pass max|dp| ~1e-3 bf16, same
argmax, GUI-Actor-3B-Qwen2.5-VL and the Qwen3.5-2B checkpoint, 2026-10)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...api.primitives import GroundAnswer
from ...engine import SyncLockEngine

SUPPORTED_PRIMITIVES = frozenset({"ground"})
# Alternative region centres a ground answer carries; KEV_GUI_ACTOR_TOPK=0 keeps only the point and confidence.
TOPK = int(os.environ.get("KEV_GUI_ACTOR_TOPK", "5"))


def _first_instruction(d: dict) -> str:
    """The first question's instruction: image_prefix only needs it to locate the split point (the
    prefix tokens are the same for every instruction over one image)."""
    from ...api import render
    qs = list((d.get("questions") or {}).values())
    return render(qs[0].get("instructions")) if qs else ""


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

    def prefix_key(self, d: dict):
        """(prefix token ids, n_tokens) for the image prefix this request would reuse, or None when it
        cannot cache. The ids are exactly the tokens before the continuation (system + user header +
        image), so they encode both the screenshot and the system text — the two things a shared
        prefix must agree on. Encoding is the processor's, not the model's: no forward pass."""
        images = d.get("images") or []
        if not images: return None
        from .gui_actor_model import GROUNDING_SYSTEM_MESSAGE, build_prompt, render, split_at_image
        system_text = render(d.get("state")).strip() or GROUNDING_SYSTEM_MESSAGE
        instr = _first_instruction(d)
        try:
            ids = self.processor(text=[build_prompt(self.processor, images[-1], system_text, instr)],
                                 images=[images[-1]], return_tensors="pt")["input_ids"]
            p = split_at_image(self.processor, ids)
        except ValueError:
            return None     # no vision marker / unprocessable image: this request just goes uncached
        if p <= 0 or p > ids.shape[1]:
            return None
        return tuple(ids[0, :p].tolist()), p

    def answer_batch_with_prefix(self, req_dicts, cached, keep):
        """The engine's batch call. Each request grounds against images[-1]; those sharing an image
        reuse one cache entry. The first request over an image builds it (image_prefix), the rest
        continue from it. A grounding pass is batch=1, so the requests still run one at a time — the
        win is that each pays only for its instruction, not the image."""
        from .gui_actor_model import GROUNDING_SYSTEM_MESSAGE, image_prefix, render
        from .gui_actor_model import systemone as gui_actor_systemone
        resps, prefixes = [None] * len(req_dicts), [None] * len(req_dicts)
        built: dict = {}                     # within this batch: one prefix serves every request that shares a key
        for i, d in enumerate(req_dicts):
            pfx = cached[i]
            if pfx is None and keep[i]:
                item = self.prefix_key(d)
                if item is not None:
                    key = item[0]
                    if key in built:
                        pfx = built[key]
                    else:
                        system_text = render(d.get("state")).strip() or GROUNDING_SYSTEM_MESSAGE
                        pfx = image_prefix(self.model, self.processor, d["images"][-1], system_text, _first_instruction(d))
                        built[key] = pfx
                    prefixes[i] = pfx
            elif pfx is not None:
                prefixes[i] = pfx            # a hit is reinserted as most recent
            resps[i] = gui_actor_systemone(self.model, self.processor, d, self.model_type, topk=self.topk,
                                           temperature=self.temperature_override, prefix=pfx)
            resps[i]["answers"] = {qid: GroundAnswer(**a).model_dump() for qid, a in resps[i]["answers"].items()}
        return resps, prefixes

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
