"""Shared FastAPI dependencies.

The LLM router lives behind a dependency rather than a module-level singleton so tests
can override it through FastAPI's normal mechanism instead of monkeypatching imports.
It is built lazily and once: constructing it reads API keys, which no import should do.
"""

from __future__ import annotations

from app.llm.base import LLMRouterProtocol

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


__all__ = ["get_llm_router", "reset_llm_router"]
