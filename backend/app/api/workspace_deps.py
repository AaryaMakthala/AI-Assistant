"""Workspace-specific FastAPI dependencies (Phase 2).

Workspace authorization is database-driven: the ``members`` table is the source of truth
for whether a user belongs to a workspace and what role they hold. The JWT only provides
the user's identity (``sub``) and default workspace (``workspace_id``); the *role* is
never extracted from the token.

Two authorization levels exist (CLAUDE.md section 4):
* **OWNER**: full workspace control — approve documents, manage members, invite users.
* **MEMBER**: read, upload (pending), chat — the default for invited users.

Workspace isolation: a user requesting ``/workspaces/{workspace_id}/...`` must have an
``ACTIVE`` membership row in that workspace. If not, the request is rejected with 404
(to prevent workspace ID enumeration).
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy import select

from app.db.models import Member
from app.security.auth import Principal, get_principal
from app.security.rls import tenant_session


@dataclass(frozen=True)
class WorkspaceContext:
    """The authenticated caller's context within a specific workspace."""

    workspace_id: uuid.UUID
    workspace_role: str
    principal: Principal


async def get_workspace_member(workspace_id: uuid.UUID, principal: Principal) -> Member:
    """Fetch the caller's ACTIVE membership record for a workspace, or 404 if not found.

    Uses a tenant session scoped to the *requested* workspace — not the principal's
    default workspace — so RLS enforces isolation. A missing record returns 404
    rather than 403 to prevent workspace ID enumeration.
    """
    async with tenant_session(
        workspace_id=workspace_id,
        user_id=principal.user_id,
    ) as session:
        stmt = select(Member).where(
            Member.workspace_id == workspace_id,
            Member.user_id == principal.user_id,
            Member.status == "ACTIVE",
        )
        result = await session.execute(stmt)
        member = result.scalar_one_or_none()

        if member is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found.",
            )
        return member


async def assert_workspace_role(
    workspace_id: uuid.UUID, principal: Principal, *allowed: str
) -> str:
    """Check a workspace role outside the dependency system, returning the caller's role.

    `require_workspace_role` covers endpoints whose path carries the workspace id. This
    covers the ones where it arrives in the body instead — document upload, say — which
    FastAPI cannot resolve into a path-parameter dependency.
    """
    member = await get_workspace_member(workspace_id, principal)
    if allowed and member.role not in frozenset(allowed):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your workspace role does not permit this action.",
        )
    return member.role


def require_workspace_role(*allowed: str) -> Callable[..., Awaitable[WorkspaceContext]]:
    """Dependency factory admitting only the listed workspace roles.

    The workspace_id is automatically extracted from the path parameters by FastAPI.
    Role is looked up from the canonical ``members`` table — never from the JWT.
    """
    permitted = frozenset(allowed)

    async def dependency(
        workspace_id: uuid.UUID,
        principal: Annotated[Principal, Depends(get_principal)],
    ) -> WorkspaceContext:
        member = await get_workspace_member(workspace_id, principal)

        if member.role not in permitted:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your workspace role does not permit this action.",
            )

        return WorkspaceContext(
            workspace_id=workspace_id,
            workspace_role=member.role,
            principal=principal,
        )

    return dependency


#: OWNER only — full workspace control (approve, invite, manage).
WorkspaceOwner = Annotated[WorkspaceContext, Depends(require_workspace_role("OWNER"))]
#: Any ACTIVE membership — read, upload, chat.
WorkspaceMemberAny = Annotated[WorkspaceContext, Depends(require_workspace_role("OWNER", "MEMBER"))]

__all__ = [
    "WorkspaceContext",
    "WorkspaceMemberAny",
    "WorkspaceOwner",
    "assert_workspace_role",
    "get_workspace_member",
    "require_workspace_role",
]
