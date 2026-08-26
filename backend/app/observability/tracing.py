"""Agent tracing — disabled.

CLAUDE.md Section 2 explicitly excludes paid observability platforms, including
LangSmith. This module exists as a no-op so ``main.py`` can call
``configure_tracing()`` without an ImportError or a conditional — the call is
harmless and the import stays stable if tracing is ever re-evaluated.
"""

from __future__ import annotations

import os

from loguru import logger

from app.config import Settings, get_settings


def configure_tracing(settings: Settings | None = None) -> bool:
    """No-op: LangSmith tracing is excluded from the canonical architecture.

    Called at startup in the API process. Idempotent.
    """
    settings = settings or get_settings()
    # Clear any stale tracing environment variables.
    for name in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
    ):
        os.environ.pop(name, None)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logger.info("Agent tracing is disabled (LangSmith excluded from architecture).")
    return False


def tracing_is_enabled() -> bool:
    """Whether this process is currently shipping traces (always False)."""
    return False


__all__ = ["configure_tracing", "tracing_is_enabled"]
