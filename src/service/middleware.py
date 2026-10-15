from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .logging import request_id_var
from .metrics import DURATION, IN_FLIGHT, REQUESTS
from .telemetry import request_span

logger = logging.getLogger("vle.http")


class BodyTooLarge(Exception):
    pass


class MaxBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    response = JSONResponse(status_code=413, content={"detail": "request body too large"})
                    await response(scope, receive, send)
                    return
            except ValueError:
                response = JSONResponse(status_code=400, content={"detail": "invalid content-length"})
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except BodyTooLarge:
            response = JSONResponse(status_code=413, content={"detail": "request body too large"})
            await response(scope, receive, send)


async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", "").strip()
    if not request_id or len(request_id) > 128 or any(ord(ch) < 32 for ch in request_id):
        request_id = str(uuid.uuid4())
    token = request_id_var.set(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    IN_FLIGHT.inc()
    response = None
    status_code = 500
    try:
        with request_span(
            "http.request",
            {"http.request.method": request.method, "url.path": request.url.path, "request.id": request_id},
        ) as span:
            response = await call_next(request)
            status_code = response.status_code
            span.set_attribute("http.response.status_code", status_code)
        return response
    finally:
        elapsed = time.perf_counter() - started
        IN_FLIGHT.dec()
        route = request.scope.get("route")
        route_name = getattr(route, "path", "unmatched")
        REQUESTS.labels(request.method, route_name, str(status_code)).inc()
        DURATION.labels(request.method, route_name).observe(elapsed)
        logger.info(
            "request_completed",
            extra={"method": request.method, "path": route_name, "status_code": status_code, "duration_ms": round(elapsed * 1000, 3)},
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Cache-Control"] = "no-store"
        request_id_var.reset(token)
