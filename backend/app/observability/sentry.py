"""Sentry error reporting (Phase 11).

The phase's acceptance criterion is precise: *a deliberately triggered error shows up in
Sentry with useful context and no leaked secrets*. Both halves are enforced here in code,
because both are easy to get wrong in the direction that is invisible until it matters.

**Useful context** comes from three sources. Request-scoped tags (`request_id`, `org_id`,
`user_id`) let an error be joined to the log lines that led to it. Loguru breadcrumbs carry
the last few log lines from the same request, which is usually the whole story. And the
Starlette/FastAPI integration supplies the route, method and status.

**No leaked secrets** is the harder half, and rests on four layers, in order:

1. ``include_local_variables=False``. This is the big one. Sentry's default is to snapshot
   every stack frame's locals, so an exception raised anywhere below ``get_settings()``
   would ship the whole ``Settings`` object — and while pydantic's ``SecretStr`` hides its
   value in ``repr``, a raw ``os.environ`` string or a database URL with an inline password
   has no such protection. This is the same reasoning that already sets Loguru's
   ``diagnose=False`` in :mod:`app.logging_config`.
2. ``max_request_body_size="never"``. Request bodies here are chat messages and uploaded
   files — tenant content by definition, and never needed to debug a crash.
3. A recursive :class:`EventScrubber` over an extended denylist. The SDK's default list is
   shallow and stops at the top level of each dict; nested structures are exactly where a
   config dump would hide.
4. :func:`_before_send`, the final gate, which walks the serialized event and redacts any
   string that matches a configured secret's actual value. Layers 1–3 are structural and
   catch a secret by the *name of the field holding it*; this one catches a secret that
   arrived somewhere unexpected — interpolated into an exception message, most likely — by
   its value.

Nothing in this module raises. Error reporting that can itself take down a request is worse
than no error reporting, so every entry point degrades to a no-op.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import SecretStr

from app.config import Settings, get_settings

#: Replaces any secret value found in an outgoing event.
REDACTED = "[redacted]"

#: Added to Sentry's own denylist. These are the field names this project would plausibly
#: put a secret or tenant content into, matched case-insensitively against every dict key in
#: the event when the scrubber runs recursively.
_EXTRA_DENYLIST = [
    "access_token",
    "anon_key",
    "api_key",
    "bearer",
    "chunk",
    "content",
    "database_url",
    "dsn",
    "excerpt",
    "gemini_api_key",
    "github_token",
    "groq_api_key",
    "jwt",
    "jwt_secret",
    "message",
    "passage",
    "question",
    "refresh_token",
    "secret_key",
    "sentry_dsn",
    "service_role_key",
    "supabase_anon_key",
    "supabase_service_role_key",
]

#: A secret shorter than this is not searched for by value. Short strings produce false
#: positives that would redact unrelated text and make events unreadable; the structural
#: layers above already cover a short secret held in a named field.
_MIN_SECRET_LENGTH = 8

#: How deep :func:`_redact` descends. Sentry events are shallow in practice; the bound
#: exists so a self-referential structure cannot spin here.
_MAX_REDACTION_DEPTH = 12

_secret_values: tuple[str, ...] = ()


def _collect_secret_values(settings: Settings) -> tuple[str, ...]:
    """Every configured secret, as the literal strings to search outgoing events for.

    Read from the live settings rather than a hardcoded list of field names, so a secret
    added to :class:`Settings` in a later phase is covered without anyone remembering to
    update this module. `SecretStr` fields are unwrapped deliberately — the whole point is
    to hold the plaintext so it can be recognised if it turns up somewhere it should not.
    """
    values: set[str] = set()

    for name in type(settings).model_fields:
        raw = getattr(settings, name, None)
        text = raw.get_secret_value() if isinstance(raw, SecretStr) else None
        # A DSN carries an inline password, so the whole URL is treated as a secret.
        if text is None and name.endswith(("_url", "_dsn")) and raw is not None:
            text = str(raw)
        if text and len(text) >= _MIN_SECRET_LENGTH:
            values.add(text)

    # Longest first, so a secret that contains another (a URL containing its own password)
    # is redacted as a whole rather than leaving a recognisable remainder.
    return tuple(sorted(values, key=len, reverse=True))


def _redact_text(text: str) -> str:
    for secret in _secret_values:
        if secret in text:
            text = text.replace(secret, REDACTED)
    return text


def _redact(value: Any, depth: int = 0) -> Any:
    """Recursively replace every configured secret found in `value`.

    Containers are rebuilt rather than mutated in place: a Sentry event holds tuples and
    frozen structures the SDK may reuse, and mutating one would be a side effect on an
    object this module does not own.
    """
    if depth > _MAX_REDACTION_DEPTH:
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {key: _redact(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, depth + 1) for item in value)
    return value


def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any] | None:
    """Last gate before an event leaves the process.

    Runs after the scrubber, so it sees the event exactly as it would be transmitted.
    Failures are swallowed and the event dropped: a redaction step that errors must not
    fall through to sending an unredacted event.
    """
    try:
        return _redact(event)  # type: ignore[no-any-return]
    except Exception:
        # No logger call here — this runs inside the SDK, and logging an error from within
        # the error pipeline is how a reporting loop starts.
        return None


def configure_sentry(settings: Settings | None = None, *, component: str = "api") -> bool:
    """Initialise Sentry for this process. Returns whether it was enabled.

    `component` identifies this process in the Sentry UI.

    Called once per process, at startup. Safe to call again — the SDK replaces the client
    rather than stacking clients — but the integrations it installs are global, which is why
    the API and the worker each configure their own process rather than sharing one.
    """
    global _secret_values

    settings = settings or get_settings()

    if not settings.sentry_enabled:
        logger.info("Sentry is not configured (SENTRY_DSN is empty); errors stay local.")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.loguru import LoggingLevels, LoguruIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.scrubber import DEFAULT_DENYLIST, EventScrubber
    except ImportError:
        logger.warning("sentry-sdk is not installed; error reporting is disabled.")
        return False

    # Populated before init: _before_send may run for an event raised during startup.
    _secret_values = _collect_secret_values(settings)

    event_level = getattr(LoggingLevels, settings.sentry_event_level).value

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        release=settings.sentry_release,
        # Errors are always captured; this governs performance transactions only.
        traces_sample_rate=settings.sentry_traces_sample_rate,
        # Layer 1 — see the module docstring. The single most important option here.
        include_local_variables=False,
        # Layer 2. Chat messages and uploaded files are the request bodies in this system.
        max_request_body_size="never",
        # No IP addresses, no cookies, no usernames. Identity is attached deliberately in
        # app/observability/context.py as opaque ids instead.
        send_default_pii=False,
        # Layer 3.
        event_scrubber=EventScrubber(
            denylist=DEFAULT_DENYLIST + _EXTRA_DENYLIST,
            recursive=True,
        ),
        # Layer 4.
        before_send=_before_send,
        integrations=[
            StarletteIntegration(),
            FastApiIntegration(),
            LoguruIntegration(
                # Below `event_level`, log lines attach as breadcrumbs — the trail of what
                # the request did before it failed, which is most of an error's usefulness.
                level=LoggingLevels.INFO.value,
                event_level=event_level,
            ),
        ],
    )

    sentry_sdk.set_tag("component", component)

    logger.info(
        "Sentry initialised for {component} in {environment} (traces at {rate:.0%})",
        component=component,
        environment=settings.environment,
        rate=settings.sentry_traces_sample_rate,
    )
    return True


def sentry_is_active() -> bool:
    """Whether a Sentry client is installed and would transmit an event.

    `Client.is_active()` rather than `is_initialized()`: the latter stays true after an
    `init` with an empty DSN, which is the shape of a *deliberate shutdown*. Asking whether
    the client would actually send is the question every caller here means.
    """
    try:
        import sentry_sdk
    except ImportError:
        return False
    try:
        return bool(sentry_sdk.get_client().is_active())
    except Exception:
        return False


def set_tag(key: str, value: str) -> None:
    """Tag the current scope. No-op when Sentry is not active."""
    if not sentry_is_active():
        return
    import sentry_sdk

    sentry_sdk.set_tag(key, value)


def set_user(user: dict[str, str]) -> None:
    """Attach the caller to the current scope. No-op when Sentry is not active."""
    if not sentry_is_active():
        return
    import sentry_sdk

    sentry_sdk.set_user(user)


def capture_exception(exc: BaseException, **tags: str) -> str | None:
    """Report an exception the application handled itself, returning Sentry's event id.

    Needed because the paths that matter most here are the ones that *do not* propagate:
    a sub-agent failure is swallowed into an `AgentOutcome`, a failed audit write is logged
    and moved past. Those are invisible to any middleware, and they are exactly the failures
    worth knowing about.

    Returns None when Sentry is inactive, so a caller can log the id when there is one
    without branching on configuration.
    """
    if not sentry_is_active():
        return None
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            for key, value in tags.items():
                scope.set_tag(key, value)
            return sentry_sdk.capture_exception(exc)
    except Exception:
        # Reporting an error must never be the thing that breaks the request.
        return None


__all__ = [
    "REDACTED",
    "capture_exception",
    "configure_sentry",
    "sentry_is_active",
    "set_tag",
    "set_user",
]
