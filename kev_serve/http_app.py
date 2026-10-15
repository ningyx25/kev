"""App assembly: create_app(settings, router) wires the middleware stack and the two routes.
Route functions only do parse -> gate -> router -> engine; everything else lives in a module."""
from __future__ import annotations

import base64
import io
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .api import SystemOneRequest, with_date_facts
from .config import Settings
from .middlewares import build_stack
from .router import ModelRouter


def _decode_base64_image(src: str):
    from PIL import Image
    if src.startswith("data:"):
        src = src.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(src)))


def _to_backend_dict(req: SystemOneRequest, settings: Settings) -> dict:
    """Request -> the dict a text backend's answer() takes: images (assets or legacy list) decoded to PIL,
    questions as raw dicts. Box questions never reach this (the route 501s unsupported types first)."""
    d = req.model_copy(update={"state": with_date_facts(req.state)} if settings.date_facts else {}).model_dump(exclude={"assets", "images"})
    images = [a for a in req.images] if req.images else ([a.data_base64 for a in (req.assets or {}).values()])
    if images:
        d["images"] = [_decode_base64_image(a if isinstance(a, str) else a["data_base64"]) for a in images]
    d["questions"] = {qid: q.model_dump() for qid, q in req.questions.items()}
    return d


def create_app(settings: Settings, router: ModelRouter) -> FastAPI:
    app = FastAPI(title="kev-serve")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["x-typesafe-request-id", "server-timing"])

    for mw in build_stack(settings):
        app.middleware("http")(mw)

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest):
        backend = router.resolve(req.model)
        unsupported = [qid for qid, q in req.questions.items() if q.type not in backend.supported_primitives]
        if unsupported:
            raise HTTPException(501, f"question(s) {', '.join(sorted(unsupported))}: the {backend.kind} backend does not answer "
                                     f"{sorted({req.questions[q].type for q in unsupported})} questions yet")
        t = time.time()
        d = _to_backend_dict(req, settings)
        resp = await backend.engine.submit(lambda: backend.answer(d))
        resp["latency_ms"] = round((time.time() - t) * 1000, 1)
        return resp

    @app.get("/v1/models")
    def models():
        return {"models": router.cards(settings)}

    return app
