"""Identity endpoint (Phase 2).

``/me`` exists because the UI must not read its own role out of the JWT. A browser can
decode a token without verifying it, so anything derived that way is a claim the client
made about itself — fine for cosmetics, useless as a security boundary, and impossible to
tell apart from the real thing once it is in a React state variable. Asking the server
keeps one answer to "who am I", produced by the same verification path that guards every
other route.

The workspace role is looked up from the canonical ``members`` table — never from the JWT
(CLAUDE.md section 4: the database is the source of truth for authorization).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select

from app.db.models import Member, Workspace
from app.security.auth import CurrentPrincipal
from app.security.rls import tenant_session

router = APIRouter(tags=["identity"])


class MeResponse(BaseModel):
    """The caller as the server understands them, derived from verified claims only."""

    user_id: uuid.UUID
    workspace_id: uuid.UUID
    workspace_name: str | None = None
    role: str | None = None
    email: str | None = None


@router.get("/me", response_model=MeResponse, summary="The authenticated caller")
async def read_me(principal: CurrentPrincipal) -> MeResponse:
    """Return the caller's identity and their role in their default workspace.

    The workspace name and role require the database; identity itself comes entirely from
    the verified token.
    """
    workspace_name: str | None = None
    role: str | None = None

    async with tenant_session(
        workspace_id=principal.workspace_id, user_id=principal.user_id
    ) as session:
        ws = (
            await session.execute(
                select(Workspace.name).where(Workspace.id == principal.workspace_id)
            )
        ).scalar_one_or_none()
        workspace_name = ws

        member = (
            await session.execute(
                select(Member.role).where(
                    Member.workspace_id == principal.workspace_id,
                    Member.user_id == principal.user_id,
                    Member.status == "ACTIVE",
                )
            )
        ).scalar_one_or_none()
        role = member

    return MeResponse(
        user_id=principal.user_id,
        workspace_id=principal.workspace_id,
        workspace_name=workspace_name,
        role=role,
        email=principal.email,
    )


__all__ = ["router"]
