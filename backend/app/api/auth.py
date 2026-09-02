"""Identity and auth-precheck endpoints.

``/me`` exists because the UI must not read its own role out of the JWT. A browser can
decode a token without verifying it, so anything derived that way is a claim the client
made about itself — fine for cosmetics, useless as a security boundary, and impossible to
tell apart from the real thing once it is in a React state variable. Asking the server
keeps one answer to "who am I", produced by the same verification path that guards every
other route.

The workspace role is looked up from the canonical ``members`` table — never from the JWT
(CLAUDE.md section 4: the database is the source of truth for authorization).

``/auth/check-email`` lets the login form distinguish "email not registered" from
"wrong password" without exposing Supabase error details. It returns only a boolean —
no user IDs, no timestamps, no other account metadata.
"""

from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Member, Workspace
from app.security.auth import CurrentPrincipal
from app.security.rls import tenant_session

router = APIRouter(tags=["identity"])


# ---------------------------------------------------------------------------
# Email existence check (pre-login)
# ---------------------------------------------------------------------------


class CheckEmailRequest(BaseModel):
    """Body for the check-email endpoint."""

    email: str = Field(min_length=1, max_length=320)


class CheckEmailResponse(BaseModel):
    """Whether an account with this email exists in Supabase Auth."""

    exists: bool


@router.post(
    "/auth/check-email",
    response_model=CheckEmailResponse,
    summary="Check whether an email is registered",
)
async def check_email(body: CheckEmailRequest) -> CheckEmailResponse:
    """Return whether an account with the given email exists in Supabase Auth.

    Used by the login form to show a specific message: "No account found" vs.
    "Incorrect password." Returns only a boolean — no user IDs, timestamps,
    or other account metadata are exposed.

    Rate-limited to prevent email enumeration (see slowapi config).
    """
    settings = get_settings()
    supabase_url = str(settings.supabase_url).rstrip("/")
    service_key = settings.supabase_service_role_key.get_secret_value()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{supabase_url}/auth/v1/admin/users",
                headers={
                    "apikey": service_key,
                    "Authorization": f"Bearer {service_key}",
                },
                params={"email": body.email},
            )
            # Supabase admin API returns 200 with a users array.
            if response.status_code == 200:
                data = response.json()
                users = data.get("users", [])
                return CheckEmailResponse(exists=len(users) > 0)

            # Non-200 from Supabase: treat as "unknown" — fail open so the
            # normal signInWithPassword flow handles its own errors.
            logger.warning(
                "Supabase admin users lookup returned {status}: {body}",
                status=response.status_code,
                body=response.text[:200],
            )
            return CheckEmailResponse(exists=False)

    except httpx.HTTPError:
        # Network failure reaching Supabase: fail open.
        logger.exception("Failed to reach Supabase admin API for email check")
        return CheckEmailResponse(exists=False)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


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
