"""Workspace-specific FastAPI dependencies for Phase 12."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy import select

from app.db.legacy_models import WorkspaceMember
from app.security.auth import Principal, get_principal
from app.security.rls import tenant_session


@dataclass(frozen=True)
class WorkspaceContext:
    """The authenticated caller's context within a specific workspace."""

    workspace_id: uuid.UUID
    workspace_role: str
    principal: Principal
    org_id: uuid.UUID


async def get_workspace_member(workspace_id: uuid.UUID, principal: Principal) -> WorkspaceMember:
    """Fetch the caller's membership record for a workspace, or 404 if not found.

    Uses a tenant session to ensure data isolation. A missing record returns 404
    rather than 403 to prevent workspace ID enumeration.
    """
    async with tenant_session(
        org_id=principal.org_id,
        user_id=principal.user_id,
        role=principal.role,
    ) as session:
        stmt = select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == principal.user_id,
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

    Like `require_role`, but checks the caller's role within a specific workspace
    rather than their org-level role. The workspace_id is automatically extracted
    from the path parameters by FastAPI.
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
            org_id=principal.org_id,
        )

    return dependency


WorkspaceOwner = Annotated[WorkspaceContext, Depends(require_workspace_role("owner"))]
WorkspaceAdmin = Annotated[WorkspaceContext, Depends(require_workspace_role("owner", "admin"))]
WorkspaceEditor = Annotated[
    WorkspaceContext, Depends(require_workspace_role("owner", "admin", "editor"))
]
WorkspaceMemberAny = Annotated[
    WorkspaceContext, Depends(require_workspace_role("owner", "admin", "editor", "viewer"))
]

__all__ = [
    "WorkspaceAdmin",
    "WorkspaceContext",
    "WorkspaceEditor",
    "WorkspaceMemberAny",
    "WorkspaceOwner",
    "assert_workspace_role",
    "get_workspace_member",
    "require_workspace_role",
]
