"""The images gate: TypeSafe serves images behind the danger-zone header (docs.typesafe.ai/images).
A request body with assets / images / box questions and no `TypeSafe-Danger-Zone-Images: true` -> 422.
Returns the response rather than raising: an exception here is outside FastAPI's exception handlers
(it never reaches the route), so it would surface as a 500."""
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..api.primitives import IMAGES_HEADER


async def images_gate(request: Request, call_next):
    if (request.method == "POST" and request.url.path.endswith("/systemone")
            and request.headers.get(IMAGES_HEADER) != "true"):
        body = await request.json()
        has_images = bool(body.get("assets") or body.get("images")
                          or any(q.get("type") in ("box", "ground") for q in (body.get("questions") or {}).values()))
        if has_images:
            return JSONResponse({"detail": f"requests with images or box questions need the header {IMAGES_HEADER}: true"}, 422)
    return await call_next(request)
