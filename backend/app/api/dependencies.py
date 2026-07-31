"""Shared FastAPI dependencies.

The LLM router lives behind a dependency rather than a module-level singleton so tests
can override it through FastAPI's normal mechanism instead of monkeypatching imports.
It is built lazily and once: constructing it reads API keys, which no import should do.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, status

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
    """Dependency factory admitting only the listed roles.

    This is the API-layer half of CLAUDE.md 4.6 — the fast rejection. It is not the only
    check: the data a gated route returns is still fetched through `tenant_session`, so
    RLS constrains the rows independently. A role check that were somehow skipped would
    widen *which endpoint* a caller reaches, never *whose data* it returns.

    The role comes from the verified token's `org_role` claim, so it cannot be set by the
    client. Roles are compared exactly; there is no hierarchy, because an implicit
    ordering ("owner outranks admin") is the kind of assumption that silently grants
    access when a new role is added later. Callers list every role they mean.
    """
    permitted = frozenset(allowed)

    def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if principal.role not in permitted:
            # Deliberately not 404: the caller is authenticated and the route exists.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your role does not permit this action.",
            )
        return principal

    return dependency


__all__ = ["get_llm_router", "require_role", "reset_llm_router"]
