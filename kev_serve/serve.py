"""FastAPI service: one clef checkpoint, Jev-like (TypeSafe System One) API.

Run: python -m kev_serve --run <clef_checkpoint> --host 127.0.0.1 --port 8008

TypeSafe-compatible: POST /v1/systemone (Noul / Choice / Score; Box returns 501 until a Box backend
is registered), GET /v1/models, the `x-typesafe-request-id` response header, and bearer auth when
KEV_API_KEY is set (unset = open server). Images arrive either the Jev way — a top-level `assets`
map of base64 images referenced by key from state / instructions, enabled per request with the
`TypeSafe-Danger-Zone-Images: true` header — or the legacy kev way (base64 `images` list), which
also requires the header. KEV_DATE_FACTS=1 opts into date preprocessing (api.with_date_facts).
"""
import argparse, asyncio, base64, hmac, io, os, time, uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import IMAGES_HEADER, SystemOneRequest, with_date_facts
from .backends import Backend, load_backend
from .device import DEVICES, DEVICE_HELP, select

DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"
API_KEY = os.environ.get("KEV_API_KEY")              # unset = open server; set = require Authorization: Bearer <key>
MODEL_NAMES = ("kev-latest", "jev-latest")           # jev-latest is the TypeSafe SDK default, so an unconfigured client works


def prepare(req: SystemOneRequest):
    """Opt-in preprocessing applied to every request before the model sees it."""
    return req.model_copy(update={"state": with_date_facts(req.state)}) if DATE_FACTS else req


def _decode_base64_image(src: str):
    from PIL import Image
    if src.startswith("data:"):
        src = src.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(src)))


def to_backend_dict(req: SystemOneRequest) -> dict:
    """Request -> the dict clef's systemone() takes: legacy images plus the assets referenced by the request,
    decoded to PIL Images in assets-map order. Only text primitives reach this (box is intercepted earlier)."""
    d = prepare(req).model_dump(exclude={"assets"})
    d.pop("assets", None)
    images = list(req.images or [])
    if req.assets:
        if req.images and len(req.images) != len(req.assets):
            raise HTTPException(422, "send either `images` or `assets`, not both")
        images = list(req.assets.values()) if not req.images else images
    if images:
        d["images"] = [_decode_base64_image(a.data_base64 if hasattr(a, "data_base64") else a) for a in images]
    else:
        d.pop("images", None)
    # clef's encoder takes the raw question dicts (type / instructions / criteria); box questions were filtered out
    d["questions"] = {qid: q.model_dump() for qid, q in req.questions.items()}
    return d


def check_images_enabled(req: SystemOneRequest, request: Request):
    """TypeSafe gates images behind the danger-zone header; mirror that (docs.typesafe.ai/images)."""
    has_images = bool(req.assets or req.images) or any(q.type == "box" for q in req.questions.values())
    if has_images and request.headers.get(IMAGES_HEADER) != "true":
        raise HTTPException(422, f"requests with images or box questions need the header {IMAGES_HEADER}: true")


app = FastAPI(title="kev-serve")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["x-typesafe-request-id", "server-timing"])


@app.middleware("http")
async def typesafe(request, call_next):
    """Bearer auth (when API_KEY is set) and the request id every TypeSafe client reads off the response."""
    started = time.perf_counter()
    if API_KEY and request.url.path.startswith("/v1") and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}"):
        resp = JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"}, 401, {"www-authenticate": "Bearer"})
    else:
        resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    resp.headers["server-timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
    return resp


def server() -> Backend:
    return app.state.server


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest, request: Request):
    """Jev-like endpoint: typed questions in, typed answers out, one prefill pass per request."""
    s = server()
    unsupported = [qid for qid, q in req.questions.items() if q.type not in s.supported_primitives]
    if unsupported:
        # today only box lands here: the schema is served, no backend is registered for it yet
        raise HTTPException(501, f"question(s) {', '.join(sorted(unsupported))}: the {s.kind} backend does not answer "
                                 f"{sorted({req.questions[q].type for q in unsupported})} questions yet; "
                                 f"send noul / choice / score, or wait for a box backend")
    check_images_enabled(req, request)
    t = time.time()
    d = to_backend_dict(req)
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, s.answer, d)
    resp["latency_ms"] = round((time.time() - t) * 1000, 1)
    return resp


@app.get("/v1/models")
def models():
    """One TypeSafe model card (name, description, release_date) per accepted model name, plus serving details
    a client may ignore: kind, supported_primitives."""
    s = server()
    card = {"release_date": s.release_date, "kind": s.kind, "run": getattr(s, "requested", None), "device": getattr(s, "device", None), **s.model_card()}
    return {"models": [{"name": name, **card} for name in MODEL_NAMES]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="clef checkpoint directory (joint_head_config.json + joint_head.safetensors)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind; 0.0.0.0 to serve beyond this machine")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", choices=("auto", *DEVICES), default="auto", help=DEVICE_HELP)
    a = ap.parse_args()
    import torch
    dev = select(a.device, dtype=torch.bfloat16)
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    print(f"loading {a.run} on {dev} ({dtype})")
    app.state.server = load_backend(a.run, dev, dtype)
    card = app.state.server.model_card()
    print(f"serving {a.run} on {dev} {a.host}:{a.port}; primitives {card.get('supported_primitives')}, box -> 501")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
