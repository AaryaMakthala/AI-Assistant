"""Request-scoped observability context.

One request produces three kinds of signal — log lines, a Sentry event if it fails, and a
LangSmith trace if the agent ran — and they are only useful together if they carry the same
identifier. `request_id` is that identifier: assigned by
:class:`app.errors.RequestContextMiddleware`, echoed to the client in the `X-Request-ID`
header, printed on every log line, and attached here as a Sentry tag. A user quoting the id
from an error banner is enough to find the log lines and the error in one search.

**What is deliberately not recorded.** No email, no question text, no document content, no
token. Identity is reduced to `user_id` and `org_id`, which are opaque UUIDs — enough to
answer "was this one tenant or all of them", and useless to anyone who obtains the event.
That mirrors the RLS boundary: an observability backend is one more place tenant data could
leak to, and the cheapest way not to leak it is not to send it.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from app.logging_config import request_id_var

#: The caller's identity for the current request, as opaque ids. Set once the JWT has been
#: verified; absent for anonymous or background work.
_user_id_var: ContextVar[str | None] = ContextVar("observability_user_id", default=None)
_org_id_var: ContextVar[str | None] = ContextVar("observability_org_id", default=None)
_role_var: ContextVar[str | None] = ContextVar("observability_role", default=None)


def bind_request(request_id: str) -> None:
    """Tag the current Sentry scope with the request id.

    The id itself already lives in :mod:`app.logging_config`'s ContextVar; this makes the
    same value searchable in Sentry. A no-op when Sentry is not configured.
    """
    from app.observability.sentry import set_tag

    set_tag("request_id", request_id)


def bind_principal(*, user_id: uuid.UUID, org_id: uuid.UUID, role: str) -> None:
    """Attach the verified caller to the current request's observability context.

    Called after JWT verification, never from client-supplied values — an event tagged with
    an org the caller merely claimed would make the error report actively misleading.
    """
    _user_id_var.set(str(user_id))
    _org_id_var.set(str(org_id))
    _role_var.set(role)

    from app.observability.sentry import set_tag, set_user

    # `id` only. Sentry renders `email` and `username` when present, and this project has no
    # reason to put an address into a third-party error store.
    set_user({"id": str(user_id)})
    set_tag("org_id", str(org_id))
    set_tag("org_role", role)


def clear_observability_context() -> None:
    """Forget the current request's identity.

    ContextVars are per-task, so this is belt-and-braces for the paths that reuse a task —
    a Celery worker thread handling job after job, most of all.
    """
    _user_id_var.set(None)
    _org_id_var.set(None)
    _role_var.set(None)


def observability_tags() -> dict[str, str]:
    """The current context as a flat dict, for logs and trace metadata.

    Empty values are dropped rather than sent as None, so a consumer never has to
    distinguish "not set" from "set to nothing".
    """
    candidates = {
        "request_id": request_id_var.get(),
        "user_id": _user_id_var.get(),
        "org_id": _org_id_var.get(),
        "org_role": _role_var.get(),
    }
    return {key: value for key, value in candidates.items() if value and value != "-"}


__all__ = [
    "bind_principal",
    "bind_request",
    "clear_observability_context",
    "observability_tags",
]
