"""App assembly: create_app(settings, router) wires the middleware stack and the routes.
Route functions only do parse -> gate -> admit -> router -> engine; everything else lives in a module."""
from __future__ import annotations

import io
import random
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .admission import admit, truncation_marks
from .api import SystemOneRequest, with_date_facts
from .config import Settings
from .middlewares import build_stack
from .request_body import image_bytes, read_body
from .router import ModelRouter


def _decode_image(data: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(data))


def _to_backend_dict(req: SystemOneRequest, parts: dict, settings: Settings) -> dict:
    """Request -> the dict a text backend's answer() takes: images (assets or legacy list) decoded to PIL,
    questions as raw dicts. Box/ground questions never reach this (the route 501s unsupported types first)."""
    d = req.model_copy(update={"state": with_date_facts(req.state)} if settings.date_facts else {}).model_dump(exclude={"assets", "images"})
    raw = image_bytes(req.assets, parts) if req.assets else []
    if req.images:
        if req.assets: raise HTTPException(422, "send either `images` or `assets`, not both")
        import base64
        raw = [base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s) for s in req.images]
    if raw:
        d["images"] = [_decode_image(b) for b in raw]
    d["questions"] = {qid: q.model_dump() for qid, q in req.questions.items()}
    return d


def _json_safe(v):
    """Error details must be JSON-encodable. A failed multipart request puts raw image bytes in the
    error's `input`, and FastAPI's encoder decodes bytes as UTF-8 -> a 500 instead of the 422."""
    if isinstance(v, (bytes, bytearray)): return f"<{len(v)} bytes>"
    if isinstance(v, dict): return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_json_safe(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)): return v
    return str(v)


class PermuteSystemOne(BaseModel):
    """The /v1/systemone/permute body. Module-level: FastAPI resolves route annotations by name, and a
    class defined inside create_app is invisible to it (the model would arrive as a query param)."""
    request: SystemOneRequest
    question: str
    n_perm: int = Field(default=6, ge=1, le=64)   # each order is a forward pass; 0 divided by nothing, unbounded counts ran forever (#30)
    seed: int = 0


def create_app(settings: Settings, router: ModelRouter) -> FastAPI:
    app = FastAPI(title="kev-serve")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["x-typesafe-request-id", "server-timing"])

    for mw in build_stack(settings):
        app.middleware("http")(mw)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse({"detail": _json_safe(exc.errors())}, 422)

    async def run_one(req: SystemOneRequest, parts: dict) -> dict:
        """The whole pipeline for one parsed request: resolve -> primitive gate -> dict -> admit -> engine."""
        backend = router.resolve(req.model)
        unsupported = [qid for qid, q in req.questions.items() if q.type not in backend.supported_primitives]
        if unsupported:
            raise HTTPException(501, f"question(s) {', '.join(sorted(unsupported))}: the {backend.kind} backend does not answer "
                                     f"{sorted({req.questions[q].type for q in unsupported})} questions yet")
        t = time.time()
        d = _to_backend_dict(req, parts, settings)
        d, marks = admit(backend, d, settings)
        resp = await backend.engine.run(d)
        resp.setdefault("latency_ms", round((time.time() - t) * 1000, 1))   # the batch engine reports its batch time
        if marks:
            resp["truncated"] = marks["truncated"]
            resp.setdefault("usage", {}).update(state_tokens=marks["state_tokens"], state_tokens_used=marks["state_tokens_used"])
        return resp

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        req, parts = await read_body(request)
        return await run_one(req, parts)

    @app.post("/v1/systemone/permute")
    async def systemone_permute(r: PermuteSystemOne):
        """Re-run one Choice question under n_perm option orders. Returns per-order probabilities keyed by option name."""
        backend = router.resolve(r.request.model)
        if "choice" not in backend.supported_primitives:
            raise HTTPException(422, f"option-order permutation needs a choice-answering backend; {backend.kind} answers "
                                     f"{sorted(backend.supported_primitives)}")
        q = r.request.questions.get(r.question)
        if q is None or q.type != "choice": raise HTTPException(422, "question must be an existing choice question")
        rng = random.Random(r.seed); keys = list(q.criteria); runs = []
        for i in range(r.n_perm):
            order = list(keys)
            if i > 0: rng.shuffle(order)
            one = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
            resp = await run_one(one, {}); a = resp["answers"][r.question]
            runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": resp["latency_ms"]})
        spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
        return truncation_marks({"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}, resp)

    @app.post("/v1/systemone/separate")
    async def systemone_separate(request: Request):
        """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
        req, parts = await read_body(request)
        router.resolve(req.model)   # unknown model: the same 404 as /v1/systemone
        parts_ = [await run_one(req.model_copy(update={"questions": {qid: q}}), parts) for qid, q in req.questions.items()]
        answers = {qid: a for p in parts_ for qid, a in p["answers"].items()}
        body = {"model": req.model, "answers": answers,
                "usage": {"input_tokens": sum(p["usage"]["input_tokens"] for p in parts_), "output_tokens": 0},
                "latency_ms": round(sum(p["latency_ms"] for p in parts_), 1)}
        return truncation_marks(body, parts_[0]) if parts_ else body

    @app.get("/v1/models")
    def models():
        return {"models": router.cards(settings)}

    return app
