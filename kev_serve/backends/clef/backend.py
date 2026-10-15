"""ClefBackend: a thin serving wrapper around the vendored ClefDecisionModel.

Images arrive decoded (PIL) in req_dict["images"]; the HTTP layer does base64 / assets handling.
On the QueueBatchEngine the backend answers whole batches, and text-only requests reuse a cached
state prefix (ClefDecisionModel.state_pass / forward_continuation): the second request over the same
state runs only its schema. Temperature is a scalar on the head's logits (docs: KEV_TEMPERATURE)."""
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
    """Answers /v1/systemone requests through clef's model; single requests go one blocking forward
    pass at a time on the engine, batches through answer_batch(_with_prefix)."""
    kind = "clef"
    supported_primitives = SUPPORTED_PRIMITIVES
    prefix_min_tokens = 384     # a shorter state costs less to re-run than to keep and copy

    def __init__(self, model, processor, requested: str, device: str, temperature: float | None = None,
                 engine=None):
        self.model, self.processor = model, processor
        self.requested, self.device = requested, device
        self.release_date = file_release_date(requested)
        self.temperature = temperature if temperature is not None else 1.0   # clef ships no fitted scalar; KEV_TEMPERATURE overrides
        self.engine = engine or SyncLockEngine(self)

    # ── optional backend hooks ───────────────────────────────────────────────────────────────

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def _encode(self, d: dict):
        from .clef_model import encode_record
        r = {"state": d.get("state"), "questions": d.get("questions") or {}}
        if d.get("images"): r["images"] = d["images"]
        return encode_record(self.processor.tokenizer, r, max_length=16384,
                             processor=self.processor if d.get("images") else None)

    def prefix_key(self, d: dict):
        """(tuple of prefix token ids, prefix length) for a text-only request; None when the request
        carries media (its mrope positions do not slice) or cannot be encoded (answer() will raise)."""
        if d.get("images"): return None
        try:
            enc = self._encode(d)
        except ValueError: return None
        if enc.prefix_len < 2: return None
        return tuple(enc.input_ids[:enc.prefix_len]), enc.prefix_len

    # ── answering ────────────────────────────────────────────────────────────────────────────

    def answer(self, req_dict: dict[str, Any]) -> dict[str, Any]:
        from .clef_model import systemone as clef_systemone
        return clef_systemone(self.model, self.processor, req_dict, temperature=self.temperature)

    def answer_batch(self, req_dicts: list[dict]) -> list[dict]:
        """One collated forward pass per batch: encode every request, run model.probs_batch, shape answers.
        (The prefix path subsumes this when the engine caches; this is the fallback for batch-capable,
        cache-less runs.)"""
        from .clef_model import _systemone_answer
        encs = [self._encode(d) for d in req_dicts]
        probs = self.model.probs_batch(encs, temperature=self.temperature)
        return [self._body(d, e, p) for d, e, p in zip(req_dicts, encs, probs)]

    def answer_batch_with_prefix(self, req_dicts, cached, keep):
        """The engine's batch call. cached[i] is the request's prefix (cache, hidden) or None; keep[i]
        says a new prefix may be stored. -> (responses, prefixes). A request without a prefix_key (media,
        uncacheable) runs the plain one-pass batch path."""
        from .clef_model import _systemone_answer
        encs = [self._encode(d) for d in req_dicts]
        resps, prefixes = [None] * len(encs), [None] * len(encs)
        plain = []
        for i, (d, enc) in enumerate(zip(req_dicts, encs)):
            if enc.prefix_len == 0 or d.get("images"):
                plain.append(i); continue
            if cached[i] is not None:
                logits = self.model.forward_continuation(enc, cached[i])
                resps[i] = self._body(d, enc, [self.model._softmax(z, self.temperature) for z in logits])
                prefixes[i] = cached[i]                                  # hits are reinserted as most recent
            elif keep[i]:
                pfx = self.model.state_pass(enc)
                logits = self.model.forward_continuation(enc, pfx)
                resps[i] = self._body(d, enc, [self.model._softmax(z, self.temperature) for z in logits])
                prefixes[i] = pfx                                        # one state pass + one schema pass
            else:
                plain.append(i)                                          # not worth caching: one full pass
        if plain:
            ps = self.model.probs_batch([encs[i] for i in plain], temperature=self.temperature)
            for i, p in zip(plain, ps):
                resps[i] = self._body(req_dicts[i], encs[i], p)
        return resps, prefixes

    def _body(self, d: dict, enc, probs: list[list[float]]) -> dict:
        from .clef_model import _systemone_answer
        questions = d.get("questions") or {}
        answers = {q.question_id: _systemone_answer(questions[q.question_id], dict(zip(q.option_ids, p)))
                   for q, p in zip(enc.questions, probs)}
        return {"model": d.get("model", "clef"), "answers": answers,
                "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}

    def model_card(self) -> dict[str, Any]:
        return {"description": f"Clef joint-schema head on Qwen3.5, serving {self.requested}",
                "supported_primitives": sorted(self.supported_primitives),
                "temperature": self.temperature}

    def close(self) -> None:
        self.engine.close()
