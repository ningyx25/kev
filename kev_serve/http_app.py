"""App assembly: create_app(settings, router) wires the middleware stack and the two routes.
Route functions only do parse -> gate -> router -> engine; everything else lives in a module."""
from __future__ import annotations

import io
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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


def create_app(settings: Settings, router: ModelRouter) -> FastAPI:
    app = FastAPI(title="kev-serve")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["x-typesafe-request-id", "server-timing"])

    for mw in build_stack(settings):
        app.middleware("http")(mw)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse({"detail": _json_safe(exc.errors())}, 422)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        req, parts = await read_body(request)
        backend = router.resolve(req.model)
        unsupported = [qid for qid, q in req.questions.items() if q.type not in backend.supported_primitives]
        if unsupported:
            raise HTTPException(501, f"question(s) {', '.join(sorted(unsupported))}: the {backend.kind} backend does not answer "
                                     f"{sorted({req.questions[q].type for q in unsupported})} questions yet")
        t = time.time()
        d = _to_backend_dict(req, parts, settings)
        resp = await backend.engine.submit(lambda: backend.answer(d))
        resp["latency_ms"] = round((time.time() - t) * 1000, 1)
        return resp

    @app.get("/v1/models")
    def models():
        return {"models": router.cards(settings)}

    return app
