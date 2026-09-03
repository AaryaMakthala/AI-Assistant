"""Workspace-switching support.

When a user belongs to multiple workspaces, they need a way to select which
one is active for a given request.  Rather than re-issuing JWTs (which are
managed by Supabase), the frontend sends an optional ``X-Workspace-ID``
header.  This module provides:

1. A context variable (``_requested_workspace_id_ctx``) that stores the
   workspace override for the current request.
2. A middleware that reads the ``X-Workspace-ID`` header and sets the
   context variable.
3. A helper (``get_requested_workspace_id``) for ``auth.py`` to read it.

If the header is absent, the workspace is resolved from the JWT claim (the
existing default).  If the header is present but the user is not a member of
the requested workspace, ``get_principal`` returns 403.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

#: Context variable holding the workspace override for the current request.
_requested_workspace_id_ctx: ContextVar[uuid.UUID | None] = ContextVar(
    "_requested_workspace_id", default=None
)


async def workspace_switch_middleware(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Read an optional ``X-Workspace-ID`` header and store it in a context variable.

    This runs before route handlers, so ``get_principal`` can pick up the
    override without any route-level changes.
    """
    x_workspace_id = request.headers.get("x-workspace-id")

    token = None
    if x_workspace_id is not None:
        try:
            requested_ws = uuid.UUID(x_workspace_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workspace ID format.",
            )
        token = _requested_workspace_id_ctx.set(requested_ws)

    try:
        response = await call_next(request)
        return response
    finally:
        if token is not None:
            _requested_workspace_id_ctx.reset(token)


def get_requested_workspace_id() -> uuid.UUID | None:
    """Read the workspace override from the context variable, if set by the middleware."""
    return _requested_workspace_id_ctx.get()
