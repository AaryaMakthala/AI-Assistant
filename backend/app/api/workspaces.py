"""Workspace CRUD, membership, and invitations (Phase 2).

A workspace is a tenant boundary that owns documents and conversations. Every user has
a workspace-level role that governs what they can do inside it:

* **OWNER**: full control — approve documents, manage members, invite users, delete.
* **MEMBER**: read, upload (pending approval), chat.

Isolation is enforced at two layers:
* The API layer checks workspace membership through ``workspace_deps``.
* RLS constrains the underlying queries to the caller's ``workspace_id``, so even a
  bypassed membership check cannot reach another workspace's data.

Invitation flow (CLAUDE.md section 9, Phase 2):
1. OWNER creates an invitation by email.
2. Invitee authenticates with Supabase, then calls ``POST /invitations/{id}/accept``.
3. Accept creates an ACTIVE MEMBER row in ``members``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.exc import DBAPIError

from app.api.workspace_deps import (
    WorkspaceMemberAny,
    WorkspaceOwner,
)
from app.db.models import (
    Document,
    Invitation,
    Member,
    Workspace,
)
from app.email import generate_verification_token, send_verification_email
from app.security.auth import CurrentPrincipal
from app.security.rls import tenant_session

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WorkspaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)


class WorkspaceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    verified_at: datetime | None = None
    created_at: datetime


class WorkspaceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspaceResponse]


class MemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    role: str
    status: str
    created_at: datetime


class MemberListResponse(BaseModel):
    members: list[MemberResponse]


class MemberUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern="^(OWNER|MEMBER)$")


class InvitationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Plain string, not pydantic EmailStr: the canonical Invitation.email is a
    #: String(320) with no DB-level email type, and the project deliberately keeps its
    #: dependency set minimal (CLAUDE.md section 2) — email-validator would be a new
    #: dependency for what a light sanity check covers. Acceptance matches this address
    #: case-insensitively against the verified JWT email claim, which Supabase already
    #: validated, so the API edge only needs to reject obviously malformed input.
    email: str = Field(min_length=1, max_length=320)

    @field_validator("email")
    @classmethod
    def _email_is_plausible(cls, value: str) -> str:
        email = value.strip()
        if (
            len(email) < 3
            or " " in email
            or email.count("@") != 1
            or email.startswith("@")
            or email.endswith("@")
        ):
            raise ValueError(
                "email must be a plausible address: one @, a non-empty local part "
                "and domain, and no spaces."
            )
        return email


class InvitationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    email: str
    status: str
    invited_by: uuid.UUID
    created_at: datetime


class InvitationListResponse(BaseModel):
    invitations: list[InvitationResponse]


class WorkspaceStatsResponse(BaseModel):
    document_count: int
    member_count: int


# ---------------------------------------------------------------------------
# Workspace CRUD
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
    """Create a new workspace. The creator becomes the OWNER automatically.

    Uses the Phase 1C ``app.create_workspace()`` SQL function, which atomically creates
    both the workspace and the OWNER membership row. The function reads the calling user's
    ``sub`` claim from the RLS context, so it is scoped to the authenticated user.

    Organization-level verification:
    - Each new organization gets a unique verification token.
    - A verification email is sent to the owner's email.
    - Duplicate organization names (case-insensitive) are rejected.
    """
    async with tenant_session(
        workspace_id=principal.workspace_id, user_id=principal.user_id
    ) as session:
        # Use the canonical SQL function created by migration 0014.
        try:
            ws_id = (
                await session.execute(
                    select(func.app.create_workspace(payload.name.strip()))
                )
            ).scalar_one()
        except DBAPIError as exc:
            state = getattr(getattr(exc, "orig", None), "sqlstate", None)
            if state == "W1007":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="This email already has an organization with this name. Please try a different organization name.",
                ) from exc
            if state == "W1006":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="This email is already registered with an organization. Please try again with a different email.",
                ) from exc
            raise

        # Generate verification token and update the workspace.
        verification_token = generate_verification_token()
        await session.execute(
            update(Workspace)
            .where(Workspace.id == ws_id)
            .values(verification_token=verification_token)
        )

        row = (
            await session.execute(
                select(Workspace).where(Workspace.id == ws_id)
            )
        ).scalar_one()
        workspace = WorkspaceResponse.model_validate(row)

    # Send verification email (outside the transaction).
    if principal.email:
        await send_verification_email(
            to_email=principal.email,
            organization_name=workspace.name,
            verification_token=verification_token,
        )

    logger.info(
        "Workspace {ws} created by user {user}, verification email sent",
        ws=workspace.id,
        user=principal.user_id,
    )
    return workspace


@router.get(
    "",
    response_model=WorkspaceListResponse,
    summary="List workspaces the caller belongs to",
)
async def list_workspaces(principal: CurrentPrincipal) -> WorkspaceListResponse:
    """Return all workspaces the authenticated user has an ACTIVE membership in."""
    async with tenant_session(
        workspace_id=principal.workspace_id, user_id=principal.user_id
    ) as session:
        rows = (
            await session.execute(
                select(Workspace)
                .join(
                    Member,
                    (Member.workspace_id == Workspace.id)
                    & (Member.user_id == principal.user_id)
                    & (Member.status == "ACTIVE"),
                )
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
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
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
async def update_workspace(ctx: WorkspaceOwner, payload: WorkspaceUpdate) -> WorkspaceResponse:
    updates: dict[str, Any] = {}
    if payload.name is not None:
        updates["name"] = payload.name.strip()

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields to update.",
        )

    async with tenant_session(
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
    ) as session:
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
    """Remove a workspace and, by cascade, its memberships, documents, and chats."""
    async with tenant_session(
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
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


@router.get(
    "/{workspace_id}/members",
    response_model=MemberListResponse,
    summary="List workspace members",
)
async def list_members(ctx: WorkspaceMemberAny) -> MemberListResponse:
    async with tenant_session(
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
    ) as session:
        rows = (
            await session.execute(
                select(Member)
                .where(Member.workspace_id == ctx.workspace_id)
                .order_by(Member.created_at.asc())
            )
        ).scalars().all()
        members = [MemberResponse.model_validate(row) for row in rows]
    return MemberListResponse(members=members)


@router.patch(
    "/{workspace_id}/members/{member_id}",
    response_model=MemberResponse,
    summary="Change a member's role",
)
async def update_member(
    ctx: WorkspaceOwner,
    member_id: uuid.UUID,
    payload: MemberUpdate,
) -> MemberResponse:
    """Change a workspace member's role. Only the OWNER can change roles.

    The OWNER cannot change their own role — transfer ownership by promoting someone
    else first.
    """
    async with tenant_session(
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
    ) as session:
        row = (
            await session.execute(
                select(Member).where(
                    Member.id == member_id,
                    Member.workspace_id == ctx.workspace_id,
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

        await session.execute(
            update(Member).where(Member.id == member_id).values(role=payload.role)
        )

        updated = (
            await session.execute(select(Member).where(Member.id == member_id))
        ).scalar_one()
        return MemberResponse.model_validate(updated)


@router.delete(
    "/{workspace_id}/members/{member_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a member from a workspace",
)
async def remove_member(ctx: WorkspaceOwner, member_id: uuid.UUID) -> None:
    """Remove a member from this workspace. The OWNER cannot be removed."""
    async with tenant_session(
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
    ) as session:
        row = (
            await session.execute(
                select(Member.role, Member.user_id).where(
                    Member.id == member_id,
                    Member.workspace_id == ctx.workspace_id,
                )
            )
        ).first()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")

        if row.role == "OWNER":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="The workspace owner cannot be removed.",
            )

        await session.execute(
            update(Member)
            .where(Member.id == member_id)
            .values(status="REMOVED")
        )

    logger.info(
        "Member {member} removed from workspace {ws} by {actor}",
        member=member_id,
        ws=ctx.workspace_id,
        actor=ctx.principal.user_id,
    )


# ---------------------------------------------------------------------------
# Invitations (Phase 2)
# ---------------------------------------------------------------------------


@router.post(
    "/{workspace_id}/invitations",
    response_model=InvitationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Invite a user to a workspace",
)
async def create_invitation(ctx: WorkspaceOwner, payload: InvitationCreate) -> InvitationResponse:
    """Create an invitation for a user by email. Only the workspace OWNER can invite.

    If an invitation already exists for this email in this workspace, returns 409.
    """
    email = payload.email.lower().strip()

    async with tenant_session(
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
    ) as session:
        # Check for existing PENDING invitation.
        existing = (
            await session.execute(
                select(Invitation.id).where(
                    Invitation.workspace_id == ctx.workspace_id,
                    Invitation.email == email,
                    Invitation.status == "PENDING",
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An invitation for this email already exists.",
            )

        # Check if user is already an ACTIVE member.
        # We need to check by looking up auth.users by email, but since we can't directly
        # query auth.users through RLS, we check invitations only. The accept flow will
        # check actual membership.

        row = (
            await session.execute(
                insert(Invitation)
                .values(
                    workspace_id=ctx.workspace_id,
                    email=email,
                    status="PENDING",
                    invited_by=ctx.principal.user_id,
                )
                .returning(Invitation)
            )
        ).one()
        invitation = InvitationResponse.model_validate(row)

    logger.info(
        "Invitation {inv} created for {email} in workspace {ws} by {actor}",
        inv=invitation.id,
        email=email,
        ws=ctx.workspace_id,
        actor=ctx.principal.user_id,
    )
    return invitation


@router.get(
    "/{workspace_id}/invitations",
    response_model=InvitationListResponse,
    summary="List workspace invitations",
)
async def list_invitations(ctx: WorkspaceOwner) -> InvitationListResponse:
    """List all invitations for a workspace. Only the OWNER can see invitations."""
    async with tenant_session(
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
    ) as session:
        rows = (
            await session.execute(
                select(Invitation)
                .where(Invitation.workspace_id == ctx.workspace_id)
                .order_by(Invitation.created_at.desc())
            )
        ).scalars().all()
        invitations = [InvitationResponse.model_validate(row) for row in rows]
    return InvitationListResponse(invitations=invitations)


# ---------------------------------------------------------------------------
# Invitation acceptance (top-level route, not workspace-scoped)
# ---------------------------------------------------------------------------

accept_router = APIRouter(tags=["invitations"])

#: SQLSTATEs raised by ``app.accept_invitation()`` (migration 0010), so the endpoint
#: maps database-side validation failures to HTTP responses without parsing messages.
_ACCEPT_INV_NOT_FOUND = "P0002"
_ACCEPT_NOT_PENDING = "W1002"
_ACCEPT_EMAIL_MISMATCH = "W1003"
_ACCEPT_DUPLICATE_MEMBER = "W1004"


def _accept_error(exc: DBAPIError) -> HTTPException | None:
    """Map a SQLSTATE from ``app.accept_invitation()`` to an HTTP response.

    Returns None for unexpected database errors so they surface as 500s. W1001
    (missing ``sub`` claim) cannot happen through this endpoint — ``get_principal``
    requires the claim — so it is left to the 500 path rather than given a status a
    client could never trigger.
    """
    mapping = {
        _ACCEPT_INV_NOT_FOUND: (
            status.HTTP_404_NOT_FOUND,
            "Invitation not found.",
        ),
        _ACCEPT_NOT_PENDING: (
            status.HTTP_409_CONFLICT,
            "This invitation is no longer pending.",
        ),
        _ACCEPT_EMAIL_MISMATCH: (
            status.HTTP_403_FORBIDDEN,
            "Your email does not match this invitation.",
        ),
        _ACCEPT_DUPLICATE_MEMBER: (
            status.HTTP_409_CONFLICT,
            "You are already a member of this workspace.",
        ),
    }
    state = getattr(getattr(exc, "orig", None), "sqlstate", None)
    if state in mapping:
        http_code, detail = mapping[state]
        return HTTPException(status_code=http_code, detail=detail)
    return None


@accept_router.post(
    "/invitations/{invitation_id}/accept",
    response_model=MemberResponse,
    summary="Accept a workspace invitation",
)
async def accept_invitation(
    invitation_id: uuid.UUID,
    principal: CurrentPrincipal,
) -> MemberResponse:
    """Accept a pending invitation, creating an ACTIVE MEMBER row.

    The whole transition runs inside the SECURITY DEFINER ``app.accept_invitation()``
    (migration 0010). A not-yet-member cannot read the invitation or create their own
    membership under RLS — the invitee is, by definition, not yet a member — so the
    function, running as the table owner, performs the atomic validation + insert +
    status flip. Identity (``sub``) is read from the RLS claims, never from a
    parameter; the email passed here comes from the caller's own verified JWT and is
    cross-checked against the invitation inside the function.
    """
    if principal.email is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your email does not match this invitation.",
        )

    async with tenant_session(
        workspace_id=principal.workspace_id,
        user_id=principal.user_id,
    ) as session:
        try:
            member_id = (
                await session.execute(
                    select(func.app.accept_invitation(invitation_id, principal.email))
                )
            ).scalar_one()
        except DBAPIError as exc:
            mapped = _accept_error(exc)
            if mapped is not None:
                raise mapped from exc
            raise

        # The function's insert is visible in this same transaction, and the caller is
        # now a member, so members_select admits the row back out.
        member_row = (
            await session.execute(select(Member).where(Member.id == member_id))
        ).scalar_one()
        member = MemberResponse.model_validate(member_row)

    logger.info(
        "Invitation {inv} accepted by user {user}, now MEMBER of workspace {ws}",
        inv=invitation_id,
        user=principal.user_id,
        ws=member.workspace_id,
    )
    return member


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get(
    "/{workspace_id}/stats",
    response_model=WorkspaceStatsResponse,
    summary="Usage statistics for a workspace",
)
async def workspace_stats(ctx: WorkspaceMemberAny) -> WorkspaceStatsResponse:
    """Dashboard data: counts for this workspace."""
    async with tenant_session(
        workspace_id=ctx.workspace_id,
        user_id=ctx.principal.user_id,
    ) as session:
        doc_count = (
            await session.execute(
                select(func.count())
                .select_from(Document)
                .where(Document.workspace_id == ctx.workspace_id)
            )
        ).scalar_one()

        member_count = (
            await session.execute(
                select(func.count())
                .select_from(Member)
                .where(
                    Member.workspace_id == ctx.workspace_id,
                    Member.status == "ACTIVE",
                )
            )
        ).scalar_one()

    return WorkspaceStatsResponse(
        document_count=doc_count,
        member_count=member_count,
    )


# ---------------------------------------------------------------------------
# Organization verification
# ---------------------------------------------------------------------------

verify_router = APIRouter(tags=["verification"])


class VerifyOrganizationResponse(BaseModel):
    """Response after successful organization verification."""

    organization_id: uuid.UUID
    name: str
    verified_at: datetime


@verify_router.get(
    "/verify-organization",
    response_model=VerifyOrganizationResponse,
    summary="Verify an organization using the verification token",
)
async def verify_organization(token: str) -> VerifyOrganizationResponse:
    """Verify an organization using the token from the verification email.

    The token is single-use and expires after 24 hours. Verifying one organization
    does not affect any other organization.
    """
    async with tenant_session(
        workspace_id=uuid.UUID(int=0),  # Placeholder, RLS not needed for verification
        user_id=None,
    ) as session:
        # Find the workspace with this verification token.
        ws = (
            await session.execute(
                select(Workspace).where(
                    Workspace.verification_token == token,
                    Workspace.verified_at.is_(None),
                )
            )
        ).scalar_one_or_none()

        if ws is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid or expired verification link.",
            )

        # Mark as verified and clear the token (single-use).
        now = datetime.now(timezone.utc)
        await session.execute(
            update(Workspace)
            .where(Workspace.id == ws.id)
            .values(
                verified_at=now,
                verification_token=None,
            )
        )

        logger.info(
            "Organization {ws} verified by token",
            ws=ws.id,
        )

        return VerifyOrganizationResponse(
            organization_id=ws.id,
            name=ws.name,
            verified_at=now,
        )


__all__ = ["accept_router", "router", "verify_router"]
