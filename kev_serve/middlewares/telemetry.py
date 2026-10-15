"""Telemetry: the request id every TypeSafe client reads off the response, and server-timing
(time inside this process, for telling it from the network)."""
import time
import uuid

from starlette.requests import Request


async def telemetry(request: Request, call_next):
    started = time.perf_counter()
    resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    resp.headers["server-timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
    return resp
