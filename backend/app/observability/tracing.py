"""LangSmith agent tracing (Phase 11).

CLAUDE.md is explicit about the constraint: agent traces are for dev and staging, and
production user data must not reach a third-party trace tool without an explicit review of
the data-handling policy. A trace of this system's supervisor contains the user's question
and the document passages retrieved for it — tenant content, in other words, and the same
content RLS exists to keep inside one organization.

So the rule is enforced in code rather than in a deploy checklist:

- Tracing is off unless `LANGSMITH_TRACING_ENABLED` is explicitly true and an API key is set.
- In production it stays off regardless, unless `LANGSMITH_ALLOW_PRODUCTION` is also true —
  which is the recorded outcome of that review, not a convenience flag.
- Even then, run inputs and outputs are suppressed in production, so an approved production
  trace records structure and timings without the content that made it sensitive.

LangSmith is driven entirely by environment variables read by `langsmith` and LangChain at
call time, so this module's job is to set them — and, just as importantly, to *unset* them
when tracing is disabled. A stale `LANGSMITH_TRACING=true` inherited from a developer's
shell would otherwise silently start shipping production traces.
"""

from __future__ import annotations

import os

from loguru import logger

from app.config import Settings, get_settings

#: The variables this module owns. Cleared as a set when tracing is off, so a partially
#: configured environment cannot leave tracing half-enabled.
_TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_HIDE_INPUTS",
    "LANGSMITH_HIDE_OUTPUTS",
)


def _disable() -> None:
    for name in _TRACING_VARS:
        os.environ.pop(name, None)
    # Both spellings written explicitly: LangChain reads the legacy name, and an unset
    # variable and one set to "false" are not the same to every code path in the chain.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def configure_tracing(settings: Settings | None = None) -> bool:
    """Enable or disable LangSmith for this process. Returns whether tracing is on.

    Called at startup in both the API process and the Celery worker. Idempotent.
    """
    settings = settings or get_settings()

    if not settings.langsmith_enabled:
        _disable()
        if settings.langsmith_tracing_enabled and settings.is_production:
            # Worth a warning rather than silence: someone asked for tracing and did not
            # get it, and the reason is a policy decision they may not know about.
            logger.warning(
                "LangSmith tracing was requested but is disabled in production. Set "
                "LANGSMITH_ALLOW_PRODUCTION=true only after reviewing the data-handling "
                "policy — traces carry user questions and retrieved document text."
            )
        else:
            logger.info("LangSmith tracing is disabled; agent traces stay local.")
        return False

    key = settings.langsmith_api_key.get_secret_value()  # type: ignore[union-attr]
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    # "true" hides the field; the variable's absence means "send it".
    record_io = settings.langsmith_records_io
    os.environ["LANGSMITH_HIDE_INPUTS"] = str(not record_io).lower()
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = str(not record_io).lower()

    logger.info(
        "LangSmith tracing enabled for project {project} (inputs/outputs {io})",
        project=settings.langsmith_project,
        io="recorded" if record_io else "suppressed",
    )
    return True


def tracing_is_enabled() -> bool:
    """Whether this process is currently shipping traces."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"


__all__ = ["configure_tracing", "tracing_is_enabled"]
