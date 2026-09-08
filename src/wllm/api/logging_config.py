"""Structured (JSON) logging: one JSON object per log record, including one
per HTTP request (method, path, status, latency_ms, client) -- easy to grep
or ship to a log aggregator, unlike uvicorn's default plain-text access log.
"""
from __future__ import annotations

import json
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("wllm")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    """Routes the root logger AND uvicorn's own loggers through one JSON
    handler, rather than leaving uvicorn's default plain-text handlers in
    place -- so access/error logs from uvicorn itself stay structured too,
    not just this project's own log calls.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = [handler]
        uv_logger.propagate = False


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Deliberately the outermost middleware (added first, see server.py) so
    it logs every request, including ones auth/rate-limit reject -- an
    operator debugging "why am I getting 401s" needs those in the log too.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "request",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                    "client": request.client.host if request.client else None,
                }
            },
        )
        return response
