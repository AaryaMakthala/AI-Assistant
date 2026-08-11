"""Workspace CRUD, membership, and usage statistics (Phase 12).

A workspace is a sub-org grouping that owns documents and conversations. Every member has
a workspace-level role that governs what they can do inside it, independently of their
org-level role.

Isolation is enforced at two layers:
* The API layer checks workspace membership through ``workspace_deps``.
* RLS constrains the underlying queries to the caller's ``org_id``, so even a bypassed
  membership check cannot reach another organization's workspaces.

Slug generation is deterministic from the name — lowercase, hyphens for spaces, stripped
to alphanumeric+hyphen. A collision within the same org appends a short random suffix.
"""

from __future__ import annotations

import re
import secrets
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.workspace_deps import (
    WorkspaceAdmin,
    WorkspaceMemberAny,
    WorkspaceOwner,
)
from app.db.legacy_models import (
    ChatMessage,
    ChatSession,
    Document,
    User,
    Workspace,
    WorkspaceMember,
)
from app.security.auth import CurrentPrincipal
from app.security.rls import tenant_session

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WorkspaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    slug: str | None = Field(default=None, min_length=1, max_length=100)


class WorkspaceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    slug: str | None = Field(default=None, min_length=1, max_length=100)


class WorkspaceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID
    name: str
    slug: str
    owner_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspaceResponse]


class MemberAdd(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: uuid.UUID
    role: str = Field(default="viewer", pattern="^(owner|admin|editor|viewer)$")


class MemberUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern="^(owner|admin|editor|viewer)$")


class WorkspaceMemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    role: str
    created_at: datetime
    # Joined from the user table for convenience.
    email: str | None = None
    full_name: str | None = None


class WorkspaceMemberListResponse(BaseModel):
    members: list[WorkspaceMemberResponse]


class WorkspaceStatsResponse(BaseModel):
    document_count: int
    conversation_count: int
    member_count: int
    total_storage_bytes: int
    total_messages: int


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9-]")


def _slugify(name: str) -> str:
    """Deterministic slug from a name: lowercase, hyphens, no special chars."""
    slug = name.lower().strip().replace(" ", "-")
    slug = _SLUG_RE.sub("", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:100] or "workspace"


async def _unique_slug(
    session: AsyncSession,
    org_id: uuid.UUID,
    base_slug: str,
    *,
    exclude_id: uuid.UUID | None = None,
) -> str:
    """Return ``base_slug`` if it is available within the org, else append a suffix.

    ``exclude_id`` skips the workspace being updated. Without it, renaming a workspace
    while keeping its slug would collide with itself and gain a pointless random suffix.
    """
    stmt = select(Workspace.id).where(Workspace.org_id == org_id, Workspace.slug == base_slug)
    if exclude_id is not None:
        stmt = stmt.where(Workspace.id != exclude_id)
    exists = (await session.execute(stmt)).scalar_one_or_none()
    if exists is None:
        return base_slug
    return f"{base_slug[:90]}-{secrets.token_hex(4)}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a workspace",
)
async def create_workspace(
    principal: CurrentPrincipal, payload: WorkspaceCreate
) -> WorkspaceResponse:
    """Create a new workspace in the caller's organization.

    The creator becomes the workspace owner automatically.
    """
    slug_base = _slugify(payload.slug or payload.name)

    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        slug = await _unique_slug(session, principal.org_id, slug_base)

        ws_row = (
            await session.execute(
                insert(Workspace)
                .values(
                    org_id=principal.org_id,
                    name=payload.name.strip(),
                    slug=slug,
                    owner_id=principal.user_id,
                )
                .returning(Workspace)
            )
        ).one()

        workspace = WorkspaceResponse.model_validate(ws_row)

        # Creator is the owner of the workspace.
        await session.execute(
            insert(WorkspaceMember).values(
                workspace_id=workspace.id,
                user_id=principal.user_id,
                org_id=principal.org_id,
                role="owner",
            )
        )

    logger.info(
        "Workspace {ws} created by user {user} in org {org}",
        ws=workspace.id,
        user=principal.user_id,
        org=principal.org_id,
    )
    return workspace


@router.get(
    "",
    response_model=WorkspaceListResponse,
    summary="List workspaces the caller belongs to",
)
async def list_workspaces(principal: CurrentPrincipal) -> WorkspaceListResponse:
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        # Only workspaces the caller is a member of.
        rows = (
            await session.execute(
                select(Workspace)
                .join(
                    WorkspaceMember,
                    (WorkspaceMember.workspace_id == Workspace.id)
                    & (WorkspaceMember.user_id == principal.user_id),
                )
                .where(Workspace.org_id == principal.org_id)
                .order_by(Workspace.created_at.asc())
            )
        ).scalars()
        workspaces = [WorkspaceResponse.model_validate(row) for row in rows]
    return WorkspaceListResponse(workspaces=workspaces)


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Workspace detail",
)
async def get_workspace(ctx: WorkspaceMemberAny) -> WorkspaceResponse:
    async with tenant_session(
        org_id=ctx.principal.org_id,
        user_id=ctx.principal.user_id,
        role=ctx.principal.role,
    ) as session:
        row = (
            await session.execute(select(Workspace).where(Workspace.id == ctx.workspace_id))
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found.",
            )
        return WorkspaceResponse.model_validate(row)


@router.patch(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Update a workspace",
)
async def update_workspace(ctx: WorkspaceAdmin, payload: WorkspaceUpdate) -> WorkspaceResponse:
    updates: dict[str, Any] = {}
    if payload.name is not None:
        updates["name"] = payload.name.strip()
    if payload.slug is not None:
        updates["slug"] = _slugify(payload.slug)

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields to update.",
        )

    async with tenant_session(
        org_id=ctx.principal.org_id,
        user_id=ctx.principal.user_id,
        role=ctx.principal.role,
    ) as session:
        if "slug" in updates:
            updates["slug"] = await _unique_slug(
                session, ctx.principal.org_id, updates["slug"], exclude_id=ctx.workspace_id
            )
        updates["updated_at"] = func.now()
        await session.execute(
            update(Workspace).where(Workspace.id == ctx.workspace_id).values(**updates)
        )
        row = (
            await session.execute(select(Workspace).where(Workspace.id == ctx.workspace_id))
        ).scalar_one()
        return WorkspaceResponse.model_validate(row)


@router.delete(
    "/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a workspace",
)
async def delete_workspace(ctx: WorkspaceOwner) -> None:
    """Remove a workspace and, by cascade, its memberships.

    Documents and chat sessions that belonged to this workspace have their
    ``workspace_id`` set to NULL (FK ON DELETE SET NULL), so they are not lost — they
    become unassigned.
    """
    async with tenant_session(
        org_id=ctx.principal.org_id,
        user_id=ctx.principal.user_id,
        role=ctx.principal.role,
    ) as session:
        deleted = (
            await session.execute(delete(Workspace).where(Workspace.id == ctx.workspace_id))
        ).rowcount
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found.")
    logger.info(
        "Workspace {ws} deleted by user {user}",
        ws=ctx.workspace_id,
        user=ctx.principal.user_id,
    )


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


@router.post(
    "/{workspace_id}/members",
    response_model=WorkspaceMemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a member to a workspace",
)
async def add_member(ctx: WorkspaceAdmin, payload: MemberAdd) -> WorkspaceMemberResponse:
    """Add a user to this workspace with the given role.

    The target user must be in the same organization. Adding a user who is already a
    member returns 409.
    """
    async with tenant_session(
        org_id=ctx.principal.org_id,
        user_id=ctx.principal.user_id,
        role=ctx.principal.role,
    ) as session:
        # Verify the target user belongs to the same org.
        target = (
            await session.execute(
                select(User.id, User.email, User.full_name).where(User.id == payload.user_id)
            )
        ).first()
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in your organization.",
            )

        # Check for existing membership.
        existing = (
            await session.execute(
                select(WorkspaceMember.id).where(
                    WorkspaceMember.workspace_id == ctx.workspace_id,
                    WorkspaceMember.user_id == payload.user_id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User is already a member of this workspace.",
            )

        row = (
            await session.execute(
                insert(WorkspaceMember)
                .values(
                    workspace_id=ctx.workspace_id,
                    user_id=payload.user_id,
                    org_id=ctx.principal.org_id,
                    role=payload.role,
                )
                .returning(WorkspaceMember)
            )
        ).one()
        member = WorkspaceMemberResponse(
            id=row.id,
            workspace_id=row.workspace_id,
            user_id=row.user_id,
            role=row.role,
            created_at=row.created_at,
            email=target.email,
            full_name=target.full_name,
        )

    logger.info(
        "User {target} added to workspace {ws} as {role} by {actor}",
        target=payload.user_id,
        ws=ctx.workspace_id,
        role=payload.role,
        actor=ctx.principal.user_id,
    )
    return member


@router.get(
    "/{workspace_id}/members",
    response_model=WorkspaceMemberListResponse,
    summary="List workspace members",
)
async def list_members(ctx: WorkspaceMemberAny) -> WorkspaceMemberListResponse:
    async with tenant_session(
        org_id=ctx.principal.org_id,
        user_id=ctx.principal.user_id,
        role=ctx.principal.role,
    ) as session:
        rows = (
            await session.execute(
                select(
                    WorkspaceMember.id,
                    WorkspaceMember.workspace_id,
                    WorkspaceMember.user_id,
                    WorkspaceMember.role,
                    WorkspaceMember.created_at,
                    User.email,
                    User.full_name,
                )
                .join(User, User.id == WorkspaceMember.user_id)
                .where(WorkspaceMember.workspace_id == ctx.workspace_id)
                .order_by(WorkspaceMember.created_at.asc())
            )
        ).all()
        members = [
            WorkspaceMemberResponse(
                id=row.id,
                workspace_id=row.workspace_id,
                user_id=row.user_id,
                role=row.role,
                created_at=row.created_at,
                email=row.email,
                full_name=row.full_name,
            )
            for row in rows
        ]
    return WorkspaceMemberListResponse(members=members)


@router.patch(
    "/{workspace_id}/members/{member_id}",
    response_model=WorkspaceMemberResponse,
    summary="Change a member's role",
)
async def update_member(
    ctx: WorkspaceAdmin,
    member_id: uuid.UUID,
    payload: MemberUpdate,
) -> WorkspaceMemberResponse:
    """Change a workspace member's role.

    A member cannot change their own role. The workspace owner cannot be demoted by an
    admin — only by themselves (effectively transferring ownership).
    """
    async with tenant_session(
        org_id=ctx.principal.org_id,
        user_id=ctx.principal.user_id,
        role=ctx.principal.role,
    ) as session:
        row = (
            await session.execute(
                select(WorkspaceMember).where(
                    WorkspaceMember.id == member_id,
                    WorkspaceMember.workspace_id == ctx.workspace_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found.",
            )

        # Prevent self-role-change.
        if row.user_id == ctx.principal.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You cannot change your own role.",
            )

        # Prevent an admin from demoting the owner.
        if row.role == "owner" and ctx.workspace_role != "owner":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the workspace owner can change the owner's role.",
            )

        await session.execute(
            update(WorkspaceMember).where(WorkspaceMember.id == member_id).values(role=payload.role)
        )

        updated = (
            await session.execute(
                select(
                    WorkspaceMember.id,
                    WorkspaceMember.workspace_id,
                    WorkspaceMember.user_id,
                    WorkspaceMember.role,
                    WorkspaceMember.created_at,
                    User.email,
                    User.full_name,
                )
                .join(User, User.id == WorkspaceMember.user_id)
                .where(WorkspaceMember.id == member_id)
            )
        ).one()
        return WorkspaceMemberResponse(
            id=updated.id,
            workspace_id=updated.workspace_id,
            user_id=updated.user_id,
            role=updated.role,
            created_at=updated.created_at,
            email=updated.email,
            full_name=updated.full_name,
        )


@router.delete(
    "/{workspace_id}/members/{member_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a member from a workspace",
)
async def remove_member(ctx: WorkspaceAdmin, member_id: uuid.UUID) -> None:
    """Remove a member from this workspace.

    The workspace owner cannot be removed — delete the workspace instead.
    """
    async with tenant_session(
        org_id=ctx.principal.org_id,
        user_id=ctx.principal.user_id,
        role=ctx.principal.role,
    ) as session:
        row = (
            await session.execute(
                select(WorkspaceMember.role, WorkspaceMember.user_id).where(
                    WorkspaceMember.id == member_id,
                    WorkspaceMember.workspace_id == ctx.workspace_id,
                )
            )
        ).first()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")

        if row.role == "owner":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="The workspace owner cannot be removed.",
            )

        await session.execute(delete(WorkspaceMember).where(WorkspaceMember.id == member_id))

    logger.info(
        "Member {member} removed from workspace {ws} by {actor}",
        member=member_id,
        ws=ctx.workspace_id,
        actor=ctx.principal.user_id,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get(
    "/{workspace_id}/stats",
    response_model=WorkspaceStatsResponse,
    summary="Usage statistics for a workspace",
)
async def workspace_stats(ctx: WorkspaceMemberAny) -> WorkspaceStatsResponse:
    """Dashboard data: counts and totals for this workspace."""
    async with tenant_session(
        org_id=ctx.principal.org_id,
        user_id=ctx.principal.user_id,
        role=ctx.principal.role,
    ) as session:
        doc_count = (
            await session.execute(
                select(func.count())
                .select_from(Document)
                .where(Document.workspace_id == ctx.workspace_id)
            )
        ).scalar_one()

        storage = (
            await session.execute(
                select(func.coalesce(func.sum(Document.size_bytes), 0)).where(
                    Document.workspace_id == ctx.workspace_id
                )
            )
        ).scalar_one()

        conv_count = (
            await session.execute(
                select(func.count())
                .select_from(ChatSession)
                .where(ChatSession.workspace_id == ctx.workspace_id)
            )
        ).scalar_one()

        member_count = (
            await session.execute(
                select(func.count())
                .select_from(WorkspaceMember)
                .where(WorkspaceMember.workspace_id == ctx.workspace_id)
            )
        ).scalar_one()

        msg_count = (
            await session.execute(
                select(func.count())
                .select_from(ChatMessage)
                .join(ChatSession, ChatSession.id == ChatMessage.session_id)
                .where(ChatSession.workspace_id == ctx.workspace_id)
            )
        ).scalar_one()

    return WorkspaceStatsResponse(
        document_count=doc_count,
        conversation_count=conv_count,
        member_count=member_count,
        total_storage_bytes=storage,
        total_messages=msg_count,
    )


__all__ = ["router"]
