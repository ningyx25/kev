"""Middleware protocol and stack assembly.

A middleware is any async callable (request, call_next) -> response, mounted in the order
build_stack returns them. Adding one = a module here + a line in build_stack (or an entry in
Settings.middleware_order later if ordering needs to be runtime-tunable).
"""
from __future__ import annotations

from typing import Awaitable, Callable, Sequence

from starlette.requests import Request
from starlette.responses import Response

from ..config import Settings

Middleware = Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]


def build_stack(settings: Settings) -> Sequence[Middleware]:
    """The middlewares every request passes through, outermost first: telemetry wraps everything (so it
    times auth rejections too), then auth, then the images gate."""
    from .telemetry import telemetry
    from .auth import auth
    from .images_gate import images_gate

    stack: list[Middleware] = [telemetry]
    if settings.api_key:
        from functools import partial
        stack.append(partial(auth, api_key=settings.api_key))
    stack.append(images_gate)
    return stack
