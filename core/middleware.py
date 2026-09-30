"""Pure-ASGI request logging middleware (no BaseHTTPMiddleware).

Logs method, path (NO query string, NO headers, NO body), status code and
duration, and sets an X-Request-ID response header. Never logs credentials.
"""

import logging
import time
import uuid

logger = logging.getLogger("app.request")

_QUIET_PATHS = {"/health", "/health/live", "/health/ready"}  # probes: keep logs readable


class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        state = {"status": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "unhandled error",
                extra={"request_id": request_id, "method": scope.get("method"), "path": scope.get("path")},
            )
            raise
        finally:
            path = scope.get("path", "")
            level = logging.DEBUG if path in _QUIET_PATHS and state["status"] < 400 else logging.INFO
            logger.log(
                level,
                "%s %s -> %s",
                scope.get("method"),
                path,
                state["status"],
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": path,
                    "status": state["status"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
