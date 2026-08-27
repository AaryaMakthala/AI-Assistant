"""Shared FastAPI dependencies.

The canonical LLM provider is behind a dependency so tests can override it
through FastAPI's normal mechanism instead of monkeypatching imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends

from app.llm.base import LLMProvider
from app.security.auth import Principal, get_principal


def get_generic_llm() -> LLMProvider:
    """The canonical Section 13 LLM provider (CLAUDE.md sections 2 and 13).

    Built per request rather than cached: constructing it only reads settings, and
    not caching means a reconfiguration takes effect without a process restart.

    When multiple provider API keys are configured, returns a
    :class:`~app.llm.fallback.FallbackChainProvider` that implements sequential
    failover (primary → fallback → secondary fallback).  When only one key is
    present, returns a plain :class:`~app.llm.generic.GenericProvider`.
    """
    from app.config import get_settings

    settings = get_settings()
    chain_count = sum(
        1
        for key in (
            settings.gemini_api_key,
            settings.groq_api_key,
            settings.openrouter_api_key,
        )
        if key is not None
    )

    if chain_count > 1:
        from app.llm.fallback import FallbackChainProvider

        return FallbackChainProvider()

    from app.llm.generic import GenericProvider

    return GenericProvider()


def require_role(*allowed: str) -> Callable[[Principal], Principal]:
    """Legacy compatibility shim.

    Phase 2 moved authorization roles from the JWT to the ``members`` table, so the
    Principal no longer carries a role. This function is kept so legacy routers
    continue to compile. It passes through the principal unconditionally — the real
    authorization check is in the workspace-scoped RLS policies, which are
    the last line of defense regardless (CLAUDE.md 4.6).
    """

    def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        return principal

    return dependency


__all__ = ["get_generic_llm", "require_role"]
