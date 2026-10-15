"""One address, several models: route /v1/systemone by the request's `model` field.

kev.serve serves one checkpoint per process and echoes the request's `model` rather than routing on
it. This gateway sits in front of several kev.serve instances (each on its own port and GPU) and picks
the backend from the request body:

    # clef on GPU0:8011, GUI-Actor on GPU1:8010
    uv run python scripts/kev_gateway.py \
        --route clef=http://127.0.0.1:8011 --route gui-actor=http://127.0.0.1:8010 \
        --default clef --port 8020

    curl -s localhost:8020/v1/systemone -d '{"model": "gui-actor", "images": ["..."], "questions": {...}}'

GET /v1/models merges the backends' cards under the route names (each carrying its `backend`), so a
client can discover what the gateway serves; a request naming an unknown model is a 404 that lists
them. With --default set (a route name or a URL) anything else -- e.g. the TypeSafe SDK's default
"jev-latest" -- lands there. Every other path is forwarded verbatim under /v1/: the gateway does not
touch request or response bodies, and a backend's own 4xx/5xx (a 422 refusal, say) passes through
unchanged, x-typesafe-request-id included.

The gateway holds no model: it is CPU-only, starts in a second, and each backend can be restarted
underneath it. Backends that are down are reported in /v1/models rather than failing the whole card.
httpx comes from the dev dependency group, which `uv run` installs by default.
"""
import argparse, asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

app = FastAPI(title="kev-gateway")
ROUTES: dict[str, str] = {}
DEFAULT: str | None = None
TIMEOUT = 600.0


def backend_for(model) -> str:
    """The backend URL a request's `model` names: an exact route name, else --default."""
    if model in ROUTES:
        return ROUTES[model]
    if DEFAULT:
        return DEFAULT
    raise HTTPException(404, f"unknown model {model!r}; this gateway serves {sorted(ROUTES)} "
                             f"(start it with --default to send everything else to one backend)")


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT)   # httpx returns 4xx/5xx as responses; only transport errors raise


@app.get("/v1/models")
async def models():
    """The backends' cards under the route names; a backend that is down is reported, not fatal."""
    async with client() as c:
        async def card(name, url):
            try:
                r = await c.get(f"{url}/v1/models")
                body = r.json()["models"][0] if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
            except Exception as e:   # unreachable, a bad body, a timeout: the route is still listed
                body = {"error": str(e)}
            return {"name": name, **{k: v for k, v in body.items() if k != "name"}, "backend": url}
        return {"models": await asyncio.gather(*(card(name, url) for name, url in ROUTES.items()))}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    """Forward under /v1/ to the backend the body's `model` names (the default when absent)."""
    body = await request.body()
    model = None
    if body:
        try:
            model = json.loads(body).get("model")
        except ValueError:
            pass                                    # not JSON: the default backend answers (or refuses)
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    if "authorization" in request.headers:
        headers["authorization"] = request.headers["authorization"]
    try:
        async with client() as c:
            r = await c.request(request.method, backend_for(model) + "/" + path, content=body or None, headers=headers)
    except httpx.RequestError as e:                 # connection refused, a timeout: the backend's problem, not the client's
        raise HTTPException(502, f"backend unreachable: {e}")
    out = Response(content=r.content, status_code=r.status_code,
                   media_type=r.headers.get("content-type", "application/json"))
    if request_id := r.headers.get("x-typesafe-request-id"):
        out.headers["x-typesafe-request-id"] = request_id
    return out


def main():
    global DEFAULT, TIMEOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", action="append", default=[], metavar="NAME=URL",
                    help="a backend the request's `model` can name; repeat per model")
    ap.add_argument("--default", default=None, help="a route name or a URL for requests naming anything else")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds a backend round trip may take")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    a = ap.parse_args()
    for route in a.route:
        name, _, url = route.partition("=")
        if not name or not url:
            ap.error(f"--route wants NAME=URL, got {route!r}")
        ROUTES[name] = url.rstrip("/")
    if not ROUTES:
        ap.error("give at least one --route NAME=URL")
    DEFAULT = ROUTES.get(a.default, a.default.rstrip("/") if a.default else None)
    TIMEOUT = a.timeout

    import uvicorn
    print(f"gateway on {a.host}:{a.port} -> " + ", ".join(f"{name}={url}" for name, url in ROUTES.items())
          + (f"; everything else -> {DEFAULT}" if DEFAULT else ""))
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
