"""Identity and organization membership.

Two endpoints, both answering questions the frontend cannot answer for itself.

`/me` exists because the UI must not read its own role out of the JWT. A browser can
decode a token without verifying it, so anything derived that way is a claim the client
made about itself — fine for cosmetics, useless as a security boundary, and impossible to
tell apart from the real thing once it is in a React state variable. Asking the server
keeps one answer to "who am I", produced by the same verification path that guards every
other route.

`/org/members` is the first role-gated route in the system (CLAUDE.md 4.6). It is gated
twice over: `require_role` rejects a non-admin before any query runs, and the query itself
goes through `tenant_session`, so the `users_read_own_org` policy from migration 0001
constrains the rows regardless of what the API layer decided.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.dependencies import require_role
from app.db.models import Organization, User
from app.security.auth import CurrentPrincipal, Principal
from app.security.rls import tenant_session

router = APIRouter(tags=["identity"])

#: Roles permitted to see the member list. Spelled out rather than "not member", so adding
#: a role later is a decision instead of an accident.
ADMIN_ROLES = ("owner", "admin")


class MeResponse(BaseModel):
    """The caller as the server understands them, derived from verified claims only."""

    user_id: uuid.UUID
    org_id: uuid.UUID
    role: str
    email: str | None = None
    org_name: str | None = None


class MemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str | None = None
    role: str
    created_at: datetime


class MemberListResponse(BaseModel):
    members: list[MemberResponse]


@router.get("/me", response_model=MeResponse, summary="The authenticated caller")
async def read_me(principal: CurrentPrincipal) -> MeResponse:
    # The org name is a convenience for the UI header and the only part of this response
    # that needs the database; identity itself comes entirely from the verified token.
    org_name: str | None = None
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        org_name = (
            await session.execute(
                select(Organization.name).where(Organization.id == principal.org_id)
            )
        ).scalar_one_or_none()

    return MeResponse(
        user_id=principal.user_id,
        org_id=principal.org_id,
        role=principal.role,
        email=principal.email,
        org_name=org_name,
    )


@router.get(
    "/org/members",
    response_model=MemberListResponse,
    summary="Members of the caller's organization (admins only)",
)
async def list_members(
    principal: Annotated[Principal, Depends(require_role(*ADMIN_ROLES))],
) -> MemberListResponse:
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        # No org_id filter in the query: RLS applies it. Adding one here would hide the
        # fact that the policy is what makes this safe, and would still be redundant.
        rows = (
            (await session.execute(select(User).order_by(User.created_at.asc()).limit(500)))
            .scalars()
            .all()
        )

    return MemberListResponse(members=[MemberResponse.model_validate(row) for row in rows])


__all__ = ["ADMIN_ROLES", "router"]
