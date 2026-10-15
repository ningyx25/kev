"""Reading a /v1/systemone body.

Two wire formats, both accepted:

- a plain JSON body (curl, the SDK without images, `assets` carrying `data_base64`);
- `multipart/form-data` — what the TypeSafe SDK sends when the request has images: a `request` part
  holding the JSON body, followed by one part per image. The body's `assets` entries name their part
  (`{"type": "image", "part": "asset-0"}`) instead of carrying the bytes inline.

The route and the images gate both need the body, so parsing lives here; Starlette caches the parsed
form/body on the request, so the second reader does not re-read the socket."""
from __future__ import annotations

import json

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from .api import SystemOneRequest

MULTIPART = "multipart/form-data"
JSON_PART = "request"


def _validated(body) -> SystemOneRequest:
    """Parse the JSON body into a request, or a 422 carrying the same error shape FastAPI uses."""
    from fastapi.exceptions import RequestValidationError
    try:
        return SystemOneRequest.model_validate(body)
    except ValidationError as e:
        raise RequestValidationError(e.errors()) from None


async def read_body(request: Request) -> tuple[SystemOneRequest, dict[str, bytes]]:
    """-> (the request, {part name: raw image bytes}). The parts map is empty for a JSON body."""
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith(MULTIPART):
        try:
            return _validated(await request.json()), {}
        except json.JSONDecodeError as e:
            raise HTTPException(422, f"the request body is not valid JSON: {e}") from None

    form = await request.form()
    raw = form.get(JSON_PART)
    if raw is None:
        raise HTTPException(422, f"a multipart /v1/systemone body needs a {JSON_PART!r} part holding the JSON request")
    if not isinstance(raw, str):   # sent as a file part: read it
        raw = (await raw.read()).decode()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"the {JSON_PART!r} part is not valid JSON: {e}") from None
    parts = {name: await value.read() for name, value in form.multi_items()
             if name != JSON_PART and hasattr(value, "read")}
    return _validated(body), parts


async def peek(request: Request) -> tuple[dict, bool]:
    """-> (the body's JSON as a dict, whether the request carries file parts), without validating any of
    it. The images gate only needs to know whether a request carries images; a body that cannot be read
    is left to the route, which reports it properly. The body is cached, so the route still reads it."""
    try:
        ctype = request.headers.get("content-type", "")
        if ctype.startswith(MULTIPART):
            form = await request.form()
            part = form.get(JSON_PART)
            if part is None:
                return {}, True
            body = json.loads(part if isinstance(part, str) else (await part.read()).decode())
            return body, any(hasattr(v, "read") for k, v in form.multi_items() if k != JSON_PART)
        return json.loads(await request.body()), False
    except Exception:
        return {}, False


def image_bytes(assets: dict | None, parts: dict[str, bytes]) -> list[bytes]:
    """The raw bytes of every asset, in assets-map order. An asset names a part (multipart) or carries
    data_base64 (JSON); a part no asset claims is refused, like TypeSafe does."""
    import base64

    out, claimed = [], set()
    for ref, asset in (assets or {}).items():
        if asset.part is not None:
            if asset.part not in parts:
                raise HTTPException(422, f"asset {ref!r} names part {asset.part!r}, which the request does not carry")
            claimed.add(asset.part)
            out.append(parts[asset.part])
        else:
            src = asset.data_base64.split(",", 1)[1] if asset.data_base64.startswith("data:") else asset.data_base64
            try:
                out.append(base64.b64decode(src, validate=True))
            except Exception:
                raise HTTPException(422, f"asset {ref!r}: data_base64 is not valid base64") from None
    unclaimed = sorted(set(parts) - claimed)
    if unclaimed:
        raise HTTPException(422, f"part(s) {', '.join(unclaimed)} are not claimed by any asset")
    return out
