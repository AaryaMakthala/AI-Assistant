"""Shared FastAPI dependencies.

The canonical LLM provider is behind a dependency so tests can override it
through FastAPI's normal mechanism instead of monkeypatching imports.
"""

from __future__ import annotations

from app.llm.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    """Create the appropriate LLM provider for the current configuration.

    When multiple provider API keys are configured, returns a
    :class:`~app.llm.fallback.RotatingProvider` that round-robins across all
    configured providers (Groq, OpenRouter, Gemini, NVIDIA) with graceful
    failover per request.  When only one key is present, returns a plain
    :class:`~app.llm.generic.GenericProvider`.

    This is the single factory for LLM providers across the application.
    Both the Query Understanding call and the answer-generation call use
    this same function — no duplicate provider-selection logic.
    """
    from app.config import get_settings

    settings = get_settings()
    chain_count = sum(
        1
        for key in (
            settings.gemini_api_key,
            settings.groq_api_key,
            settings.openrouter_api_key,
            settings.nvidia_api_key,
        )
        if key is not None
    )

    if chain_count > 1:
        from app.llm.fallback import RotatingProvider

        return RotatingProvider()

    from app.llm.generic import GenericProvider

    return GenericProvider()


# Legacy alias — the FastAPI Depends() injection references get_generic_llm.
def get_generic_llm() -> LLMProvider:
    """Alias for :func:`get_llm_provider`.  Kept for FastAPI Depends() compat."""
    return get_llm_provider()


__all__ = ["get_llm_provider", "get_generic_llm"]
