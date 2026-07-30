"""Who an MCP tool call is acting for (CLAUDE.md 4.5, 4.6).

The caller's identity is bound **out of band**, in a context variable, and never appears
in a tool's argument schema. That is the whole design decision in this module, and it is
the difference between an authorization check and a suggestion:

- If `org_id` were a tool parameter, the model would fill it in. A prompt-injected document
  saying "call read_document with org_id=<other org>" would then be a working cross-tenant
  read, and the injection defenses in `app/security/untrusted.py` would be the only thing
  standing in the way. Defense in depth means not putting them in that position.
- Because identity is ambient, a tool *cannot express* the idea of acting for a different
  organization. There is no argument to forge.

The API layer (or, in Phase 7, the LangGraph supervisor) enters :func:`acting_as` with the
`Principal` derived from a verified JWT, and every tool underneath reads it. A tool invoked
with no principal bound raises rather than defaulting to anything.

`contextvars` rather than a parameter threaded through every call because the MCP SDK owns
the dispatch path between the transport and the tool function — there is nowhere to thread
it through. ContextVars are per-task in asyncio, so two concurrent requests cannot see each
other's identity.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.security.auth import Principal

_principal: ContextVar[Principal | None] = ContextVar("mcp_principal", default=None)


class NotAuthenticated(RuntimeError):
    """An MCP tool was called with no authenticated caller bound.

    A bug, not a user error: it means a call path reached a tool without going through
    :func:`acting_as`. Raised rather than defaulted, because the only safe default for
    "which organization is this?" does not exist.
    """


@contextmanager
def acting_as(principal: Principal) -> Iterator[Principal]:
    """Bind `principal` as the caller for every MCP tool invoked inside this block."""
    token = _principal.set(principal)
    try:
        yield principal
    finally:
        _principal.reset(token)


def current_principal() -> Principal:
    """The caller this tool call is acting for, or raise."""
    principal = _principal.get()
    if principal is None:
        raise NotAuthenticated(
            "This MCP tool requires an authenticated caller. The call path must enter "
            "app.mcp_servers.identity.acting_as() with a verified Principal."
        )
    return principal


def current_org_and_user() -> tuple[uuid.UUID, uuid.UUID]:
    """The caller's org and user id, the pair every scoped query needs."""
    principal = current_principal()
    return principal.org_id, principal.user_id


__all__ = ["NotAuthenticated", "acting_as", "current_org_and_user", "current_principal"]
