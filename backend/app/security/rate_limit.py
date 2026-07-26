"""Per-caller rate limiting (CLAUDE.md 4.7).

Keyed by authenticated user where possible, falling back to client IP for unauthenticated
requests. Keying on the user matters because the expensive endpoints are the authenticated
ones, and a single user behind a corporate NAT should not be throttled by their colleagues.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

#: Uploads are heavy: each one costs disk, a worker slot, and a model pass.
UPLOAD_RATE_LIMIT = "20/minute"
CHAT_RATE_LIMIT = "60/minute"


def rate_limit_key(request: Request) -> str:
    """Identify the caller for limiting purposes.

    The Authorization header is used verbatim rather than decoded: this runs on every
    request, including ones that will fail verification, so it must not do crypto. Two
    requests with the same token share a bucket, which is exactly the intent.
    """
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return f"token:{hash(authorization[7:])}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=rate_limit_key, default_limits=[])


def register_rate_limiting(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)


__all__ = [
    "CHAT_RATE_LIMIT",
    "UPLOAD_RATE_LIMIT",
    "limiter",
    "rate_limit_key",
    "register_rate_limiting",
]
