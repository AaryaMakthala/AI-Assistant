"""Row-level-security plumbing.

RLS is the last line of defense (CLAUDE.md 4.6): API-layer checks can be forgotten, but
a policy cannot. Every tenant-facing session must run through :func:`tenant_session` so
Postgres knows who is asking; a session without claims sees nothing at all.
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


async def set_tenant_claims(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    org_id: uuid.UUID,
    role: str | None = None,
) -> None:
    """Bind the caller's identity to the session's current transaction."""
    claims: dict[str, str] = {"sub": str(user_id), "org_id": str(org_id)}
    if role is not None:
        claims["role"] = role
    await session.execute(_SET_CLAIMS, {"claims": json.dumps(claims)})


@asynccontextmanager
async def tenant_session(
    *, user_id: uuid.UUID, org_id: uuid.UUID, role: str | None = None
) -> AsyncIterator[AsyncSession]:
    """A session whose queries are already constrained to one user and org."""
    async with get_session_factory()() as session:
        await session.begin()
        await set_tenant_claims(session, user_id=user_id, org_id=org_id, role=role)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
