"""Bearer auth (KEV_API_KEY): unset = open server, set = require Authorization: Bearer <key>."""
import hmac

from fastapi.responses import JSONResponse
from starlette.requests import Request


async def auth(request: Request, call_next, api_key: str):
    if request.url.path.startswith("/v1") and not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {api_key}"):
        return JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"},
                            401, {"www-authenticate": "Bearer"})
    return await call_next(request)
