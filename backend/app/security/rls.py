"""Row-level-security plumbing.

RLS is the last line of defense (CLAUDE.md 4.6): API-layer checks can be forgotten, but
a policy cannot. Every tenant-facing session must run through :func:`tenant_session` so
Postgres knows who is asking; a session without claims sees nothing at all.

Two things have to be true for that to hold, and only one of them is about claims. Managed
providers hand out an administrative login role — Supabase's `postgres` has BYPASSRLS —
and such a role ignores every policy, FORCE included. So each tenant transaction also
drops into `app_tenant`, which is NOBYPASSRLS. Setting claims without switching roles
would produce a session that looks correctly scoped and is not scoped at all.
"""

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session_factory

# Matches Supabase's own claim setting, so the policies defined in the migration behave
# identically whether a query arrives through this backend or Supabase's REST layer.
_CLAIMS_SETTING = "request.jwt.claims"

# set_config(..., is_local => true) scopes the claims to the current transaction, so a
# pooled connection cannot leak one tenant's identity into the next request.
_SET_CLAIMS = text(f"SELECT set_config('{_CLAIMS_SETTING}', :claims, true)")

# LOCAL for the same reason: the role reverts when the transaction ends, so the connection
# returns to the pool as it was found. The role name is a constant, never interpolated.
_SET_TENANT_ROLE = text("SET LOCAL ROLE app_tenant")


async def use_tenant_role(session: AsyncSession) -> None:
    """Drop to the NOBYPASSRLS role for the remainder of this transaction."""
    await session.execute(_SET_TENANT_ROLE)


async def set_tenant_claims(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    user_id: uuid.UUID | None = None,
    role: str | None = None,
) -> None:
    """Bind the caller's identity to the session's current transaction.

    `user_id` may be omitted for background work that acts on behalf of an organization
    rather than a person (document ingestion). Org-scoped policies still apply; the
    user-scoped ones (chat) then match nothing, which is the correct outcome — a worker
    has no business reading anyone's chat history.
    """
    claims: dict[str, str] = {"org_id": str(org_id)}
    if user_id is not None:
        claims["sub"] = str(user_id)
    if role is not None:
        claims["role"] = role
    await session.execute(_SET_CLAIMS, {"claims": json.dumps(claims)})


@asynccontextmanager
async def tenant_session(
    *, org_id: uuid.UUID, user_id: uuid.UUID | None = None, role: str | None = None
) -> AsyncIterator[AsyncSession]:
    """A session whose queries are already constrained to one user and org."""
    async with get_session_factory()() as session:
        await session.begin()
        # Order matters: claims are set while still privileged, because app_tenant is
        # granted with INHERIT FALSE and need not hold rights on the config function.
        await set_tenant_claims(session, org_id=org_id, user_id=user_id, role=role)
        await use_tenant_role(session)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


__all__ = ["set_tenant_claims", "tenant_session", "use_tenant_role"]
