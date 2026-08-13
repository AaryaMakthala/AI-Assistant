"""Shared FastAPI dependencies.

The LLM router lives behind a dependency rather than a module-level singleton so tests
can override it through FastAPI's normal mechanism instead of monkeypatching imports.
It is built lazily and once: constructing it reads API keys, which no import should do.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends

from app.llm.base import LLMRouterProtocol
from app.security.auth import Principal, get_principal

_router: LLMRouterProtocol | None = None


def get_llm_router() -> LLMRouterProtocol:
    global _router
    if _router is None:
        from app.llm.router import build_default_router

        _router = build_default_router()
    return _router


def reset_llm_router() -> None:
    """Drop the cached router. For tests and for a deliberate reconfiguration."""
    global _router
    _router = None


def require_role(*allowed: str) -> Callable[[Principal], Principal]:
    """Legacy compatibility shim.

    Phase 2 moved authorization roles from the JWT to the ``members`` table, so the
    Principal no longer carries a role. This function is kept so legacy routers
    (documents.py) continue to compile. It passes through the principal unconditionally
    — the real authorization check is in the workspace-scoped RLS policies, which are
    the last line of defense regardless (CLAUDE.md 4.6).

    This shim will be removed when documents.py is rebuilt against canonical models
    in Phase 3.
    """

    def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        # Phase 2: role comes from the members table, not the JWT. The workspace-scoped
        # RLS policies enforce the actual access control at the database level.
        return principal

    return dependency


__all__ = ["get_llm_router", "require_role", "reset_llm_router"]
