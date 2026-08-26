"""Observability wiring (Phase 11): errors, agent traces, correlated logs.

Three concerns, one package:

- :mod:`app.observability.sentry` — error reporting, with secret redaction as a hard
  guarantee rather than a default that someone might change.
- :mod:`app.observability.tracing` — LangSmith agent traces, off unless deliberately
  enabled and never in production without a recorded review.
- :mod:`app.observability.context` — the request-scoped facts every signal is tagged with,
  so a log line, an error and a trace for the same request can be joined by `request_id`.

Nothing here is required for the application to run. Every entry point is safe to call
with observability unconfigured, and each one reports what it did at startup so a
misconfiguration is visible in the logs rather than as silence in a dashboard.
"""

from app.observability.context import (
    bind_principal,
    bind_request,
    clear_observability_context,
    observability_tags,
)
from app.observability.sentry import (
    capture_exception,
    configure_sentry,
    sentry_is_active,
)
from app.observability.tracing import configure_tracing, tracing_is_enabled
from app.observability.usage import record_token_usage

__all__ = [
    "bind_principal",
    "bind_request",
    "capture_exception",
    "clear_observability_context",
    "configure_sentry",
    "configure_tracing",
    "observability_tags",
    "record_token_usage",
    "sentry_is_active",
    "tracing_is_enabled",
]
