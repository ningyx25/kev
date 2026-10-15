"""State-length admission: refuse (422) or truncate an over-length state before it reaches a backend.

Ported from kev/model.py's admit()/ContextOverflow and kev/serve.py's truncation marks, reworked as a
service-layer check driven by two optional backend hooks (the getattr convention router.py already uses):

    state_text(d) -> str    the exact text the backend will encode (default: api.render(d["state"]))
    tokenizer               a transformers tokenizer (absent = the backend opts out of admission)

A state over the limit is a 422 whose detail names the length and the fix; with KEV_TRUNCATE_STATES=1
the first `limit` tokens are read instead and every response says whether that happened
(`truncated`, usage.state_tokens / state_tokens_used).
"""
from __future__ import annotations

from fastapi import HTTPException

from .api import render
from .config import Settings

DEFAULT_MAX_STATE = 65536     # the old serving default when the backbone does not name its window
BRANCH_SLACK = 8192           # question tokens a request may add on top of the state (kev's SERVE_MAX_BRANCH - SERVE_MAX_STATE)


class ContextOverflow(ValueError):
    """A record does not encode within its context. state_tokens / max_state are set when the state is
    what overflows, which is exactly the case the truncating mode can fix."""

    def __init__(self, message, state_tokens=None, max_state=None):
        super().__init__(message)
        self.state_tokens, self.max_state = state_tokens, max_state


def state_text(backend, d: dict) -> str:
    """The state as the backend will encode it: the backend's own renderer when it has one."""
    hook = getattr(backend, "state_text", None)
    return hook(d) if hook is not None else render(d.get("state"))


def max_state_tokens(backend, settings: Settings) -> int:
    """KEV_MAX_STATE_TOKENS, else the backbone's max_position_embeddings, else DEFAULT_MAX_STATE."""
    if settings.max_state_tokens: return int(settings.max_state_tokens)
    cfg = getattr(getattr(backend, "model", None), "config", None)
    for c in (cfg, getattr(cfg, "text_config", None)):
        n = getattr(c, "max_position_embeddings", None)
        if isinstance(n, int) and n > 0: return n
    return DEFAULT_MAX_STATE


def _count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def admit(backend, d: dict, settings: Settings) -> tuple[dict, dict]:
    """-> (the backend dict to run, marks). marks is {} unless truncate_states is on, in which case it
    carries {"truncated", "state_tokens", "state_tokens_used"} for the response body. Raises
    HTTPException(422) for a state over the limit (unless truncating) or an over-long question."""
    tok = getattr(backend, "tokenizer", None)
    if tok is None: return d, {}                      # backend opts out (no tokenizer to count with)
    limit = max_state_tokens(backend, settings)
    text = state_text(backend, d)
    ids = tok(text, add_special_tokens=False).input_ids
    n = len(ids)
    if n > limit and not settings.truncate_states:
        raise HTTPException(422, f"state is {n:,} tokens, over the {limit:,}-token limit (the <state> token included): "
                                 "shorten the document or split it across requests"
                                 f"; or start the server with KEV_TRUNCATE_STATES=1 to read only its first {limit:,} tokens "
                                 "(responses then say truncated: true)")
    used = n
    if n > limit:   # truncating: read the first `limit` tokens, and say so on the response
        d = {**d, "state": tok.decode(ids[:limit])}
        used = limit
    q_tokens = sum(_count(tok, render(q.get("instructions"))) + _count(tok, render(q.get("criteria") or []))
                   for q in (d.get("questions") or {}).values())
    if q_tokens > BRANCH_SLACK:
        raise HTTPException(422, f"branch too long: this request's questions need {q_tokens:,} tokens, over the "
                                 f"{BRANCH_SLACK:,}-token question budget (the state is counted separately): "
                                 "shorten the instructions or criteria")
    marks = {"state_tokens": n, "state_tokens_used": used, "truncated": used < n} if settings.truncate_states else {}
    return d, marks


def truncation_marks(body: dict, part: dict) -> dict:
    """A demo endpoint built from several /v1/systemone answers on one state carries a truncating server's
    marks from any one of them: each read the same tokens (kev/serve.py truncation_marks)."""
    if "truncated" in part:
        body["truncated"] = part["truncated"]
        body.setdefault("usage", {}).update({k: part["usage"][k] for k in ("state_tokens", "state_tokens_used")})
    return body
