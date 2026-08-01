"""Token accounting and the cost/quota alert (Phase 11).

The Risk Register names "cost/quota overrun" with the mitigation "log token usage per
request from Phase 4; alert threshold in Phase 11". Phase 4 delivered the first half — every
generation records a :class:`TokenUsage` — and this module is the second: one place that
decides when a turn's usage is worth a warning, so the rule is not scattered across the
handlers that happen to know a usage figure.

Deliberately not a hard limit. A turn that has already been generated has already been paid
for, so refusing to deliver it would waste the spend rather than save it. The threshold is a
signal — a prompt that grew unexpectedly, a retrieval that stopped trimming, a client
looping — and it is routed to Sentry so it can carry an alert rule rather than sitting in a
log nobody reads.
"""

from __future__ import annotations

from loguru import logger

from app.config import get_settings
from app.llm.base import Completion
from app.observability.context import observability_tags

#: Sentry message level for a threshold breach. A warning rather than an error: nothing has
#: failed, and paging on it would train people to ignore the channel.
_ALERT_LEVEL = "warning"


def record_token_usage(completion: Completion, *, route: str = "chat") -> bool:
    """Log one generation's token usage and alert if it crosses the threshold.

    Returns whether an alert was raised, so a caller can assert on it without reaching into
    the logging or Sentry internals.
    """
    settings = get_settings()
    usage = completion.usage
    tags = observability_tags()

    # Structured rather than interpolated, so usage can be aggregated from logs directly.
    logger.bind(
        token_usage=True,
        route=route,
        provider=completion.provider,
        model=completion.model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        **tags,
    ).info(
        "Token usage for {route}: {total} tokens ({prompt} prompt, {completion} completion) "
        "from {provider}",
        route=route,
        total=usage.total_tokens,
        prompt=usage.prompt_tokens,
        completion=usage.completion_tokens,
        provider=completion.provider or "unknown",
    )

    threshold = settings.token_usage_alert_threshold
    if not threshold or usage.total_tokens < threshold:
        return False

    logger.warning(
        "Token usage for {route} reached {total}, above the alert threshold of {threshold}",
        route=route,
        total=usage.total_tokens,
        threshold=threshold,
    )
    _report_threshold_breach(
        route=route,
        total_tokens=usage.total_tokens,
        threshold=threshold,
        provider=completion.provider,
        model=completion.model,
    )
    return True


def _report_threshold_breach(
    *,
    route: str,
    total_tokens: int,
    threshold: int,
    provider: str,
    model: str,
) -> None:
    """Send the breach to Sentry as a message. Never raises."""
    from app.observability.sentry import sentry_is_active

    if not sentry_is_active():
        return
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.set_tag("route", route)
            scope.set_tag("alert", "token_usage_threshold")
            if provider:
                scope.set_tag("llm_provider", provider)
            # Counts only — no prompt, no answer, no question.
            scope.set_context(
                "token_usage",
                {
                    "total_tokens": total_tokens,
                    "threshold": threshold,
                    "model": model,
                },
            )
            sentry_sdk.capture_message(
                f"Token usage for {route} exceeded the configured threshold.",
                level=_ALERT_LEVEL,
            )
    except Exception:
        # Cost telemetry is not worth failing a delivered answer over.
        return


__all__ = ["record_token_usage"]
