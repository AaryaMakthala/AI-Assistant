"""Demo entry endpoint — drops a visitor into a pre-seeded demo workspace as a member.

The demo flow creates an ephemeral guest user in Supabase Auth, links them to the
demo workspace as a MEMBER, and returns credentials the frontend uses to sign in.
No manual account creation is required.

Guest cleanup is handled by a scheduled task (see ``app.demo.cleanup``).
"""

from __future__ import annotations

import secrets
import uuid

import httpx
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import delete, select

from app.config import get_settings
from app.db.models import Member, Workspace
from app.db.session import get_session_factory
from app.demo.seed import get_seeded_workspace_id

router = APIRouter(tags=["demo"])


class DemoEnterResponse(BaseModel):
    """Credentials and redirect target for the demo entry flow."""

    user_id: uuid.UUID
    email: str
    password: str
    workspace_id: uuid.UUID
    redirect_url: str


@router.post(
    "/demo/enter",
    response_model=DemoEnterResponse,
    summary="Enter the demo as a guest member",
)
async def demo_enter() -> DemoEnterResponse:
    """Create an ephemeral guest user and return auth credentials for the demo workspace.

    1. Look up the demo workspace (must be seeded beforehand).
    2. Create a guest user in Supabase Auth via the Admin API.
       The ``on_auth_user_created`` trigger fires and provisions a
       temporary workspace + OWNER membership for the guest.
    3. In one transaction: delete the trigger-created workspace and
       membership, then insert the correct MEMBER membership linking the
       guest to the demo workspace.
    4. Return credentials the frontend uses to sign in.

    The guest user's email is ``guest_<uuid>@demo.local`` — no real email
    is required.  The password is random per session and never stored.
    """
    settings = get_settings()

    if not settings.demo_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demo mode is not enabled.",
        )

    supabase_url = str(settings.supabase_url).rstrip("/")
    service_key = settings.supabase_service_role_key.get_secret_value()

    # 1. Look up the demo workspace.
    #    Priority: (a) explicit demo_workspace_id from settings, (b) the ID stored
    #    by seed_demo_workspace() at startup, (c) name-based fallback.
    async with get_session_factory()() as session:
        ws = None

        # (a) Explicit DEMO_WORKSPACE_ID in settings — must exist.
        if settings.demo_workspace_id:
            try:
                target_id = uuid.UUID(settings.demo_workspace_id)
            except ValueError:
                target_id = None
            if target_id is not None:
                ws = (
                    await session.execute(
                        select(Workspace).where(Workspace.id == target_id)
                    )
                ).scalar_one_or_none()

        # (b) ID stored by seed_demo_workspace() at startup.
        if ws is None:
            stored_id = get_seeded_workspace_id()
            if stored_id is not None:
                ws = (
                    await session.execute(
                        select(Workspace).where(Workspace.id == stored_id)
                    )
                ).scalar_one_or_none()

        # (c) Name-based fallback (legacy / when no ID was used).
        if ws is None:
            ws = (
                await session.execute(
                    select(Workspace).where(Workspace.name == settings.demo_workspace_name)
                )
            ).scalar_one_or_none()

    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demo workspace is not seeded. Please try again later.",
        )

    # 2. Create the guest user in Supabase Auth.
    #    members.user_id has a FK to auth.users.id, so the user must exist
    #    before we can insert a membership row.  The on_auth_user_created
    #    trigger fires here and provisions a temporary "Demo User" workspace
    #    + OWNER membership — cleaned up in step 3.
    guest_uuid = uuid.uuid4()
    guest_email = f"guest_{guest_uuid}@demo.local"
    guest_password = secrets.token_urlsafe(24)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{supabase_url}/auth/v1/admin/users",
                headers={
                    "apikey": service_key,
                    "Authorization": f"Bearer {service_key}",
                },
                json={
                    "email": guest_email,
                    "password": guest_password,
                    "email_confirm": True,
                    "user_metadata": {
                        "is_guest": True,
                        "full_name": "Demo User",
                    },
                },
            )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Supabase returned {response.status_code}: {response.text[:300]}"
                )

            user_data = response.json()
            created_user_id = uuid.UUID(user_data["id"])
    except Exception as exc:
        logger.error(
            "Supabase Admin API failed to create demo guest: {error}",
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not create demo session. Please try again later.",
        )

    # 3. Clean up the trigger-created workspace and membership, then insert
    #    the correct membership linking the guest to the demo workspace.
    #    All four operations run in a single transaction so the guest never
    #    ends up in a partial state (e.g. no membership at all if the app
    #    crashes mid-cleanup).
    #
    #    Deletion order matters: the FK from members → workspaces requires
    #    deleting the membership row before the workspace row.
    try:
        async with get_session_factory()() as session:
            async with session.begin():
                # Find the trigger-created membership row (OWNER in the
                # temporary workspace, any workspace except the target).
                trigger_membership = (
                    await session.execute(
                        select(Member).where(
                            Member.user_id == created_user_id,
                            Member.workspace_id != ws.id,
                        )
                    )
                ).scalar_one_or_none()

                if trigger_membership is not None:
                    trigger_ws_id = trigger_membership.workspace_id

                    # Delete the trigger-created membership first (FK order).
                    await session.execute(
                        delete(Member).where(Member.id == trigger_membership.id)
                    )

                    # Delete the now-orphaned trigger-created workspace.
                    await session.execute(
                        delete(Workspace).where(Workspace.id == trigger_ws_id)
                    )

                # Insert the correct membership linking guest → demo workspace.
                session.add(
                    Member(
                        workspace_id=ws.id,
                        user_id=created_user_id,
                        role="MEMBER",
                        status="ACTIVE",
                    )
                )
    except Exception as exc:
        logger.error(
            "Failed to set up demo membership for {user}: {error}",
            user=created_user_id,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not create demo session. Please try again later.",
        )

    logger.info(
        "Demo guest {email} created, linked to workspace {ws}",
        email=guest_email,
        ws=ws.id,
    )

    return DemoEnterResponse(
        user_id=created_user_id,
        email=guest_email,
        password=guest_password,
        workspace_id=ws.id,
        redirect_url="/",
    )
