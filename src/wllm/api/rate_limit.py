"""Fixed-window per-client rate limiting (opt-in via `rate_limit_per_minute`
on create_app / --rate-limit-rpm on the CLI). In-memory only -- matches this
project's single-process, single-machine scope (a multi-worker or
multi-machine deployment would need a shared store, e.g. Redis, instead;
out of scope here, same as everywhere else this project targets one process
on one GPU). Keyed by API key when auth is enabled (fair across clients
behind the same NAT/proxy), falling back to client IP otherwise.

Known limitation: a client key's entry in `_hits` is never removed once
created (only its deque is trimmed to empty) -- an unbounded number of
distinct clients over a long server lifetime is a slow memory leak. Not
addressed here; a real deployment would want an idle-eviction sweep.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int):
        super().__init__(app)
        self.limit = requests_per_minute
        self.window_seconds = 60.0
        self._hits: dict[str, deque] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()

        if len(hits) >= self.limit:
            retry_after = max(0.0, self.window_seconds - (now - hits[0]))
            return JSONResponse(
                {"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        hits.append(now)
        return await call_next(request)
