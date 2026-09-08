"""API-key authentication (opt-in via `api_keys` on create_app / --api-key on
the CLI): every /v1/* request must present a configured key via
`Authorization: Bearer <key>` -- OpenAI's own convention, so existing OpenAI
client libraries work unchanged (just point base_url + api_key at this
server). Entirely disabled (middleware not even installed) when no keys are
configured -- the default for local/dev use, matching every other opt-in
feature in this codebase (prefix caching, CPU swap, etc.). /health and
/metrics are deliberately never gated -- a load balancer or monitoring agent
needs to reach those without a key.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, valid_keys: frozenset[str]):
        super().__init__(app)
        self.valid_keys = valid_keys

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth.removeprefix("Bearer ") not in self.valid_keys:
            return JSONResponse(
                {"error": {"message": "invalid or missing API key", "type": "invalid_request_error"}},
                status_code=401,
            )
        return await call_next(request)
