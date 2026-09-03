"""Demo guest cleanup — removes expired guest users and their memberships.

Guest users are created by the demo entry flow and flagged with
``user_metadata.is_guest = True`` in Supabase Auth.  This module provides
a cleanup function that deletes guests older than a configurable TTL
(default 24 hours).

Called on a schedule from the app lifespan (see ``app.main``).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from loguru import logger
from sqlalchemy import delete, select

from app.config import get_settings
from app.db.models import Member, Workspace
from app.db.session import get_session_factory


async def cleanup_demo_guests() -> int:
    """Delete expired demo guest users and their member rows.

    A guest is expired when:
    1. ``user_metadata.is_guest == True`` in Supabase Auth, AND
    2. The user's Supabase ``created_at`` is older than ``demo_guest_ttl_hours``.

    Returns the number of guests deleted.
    """
    settings = get_settings()

    if not settings.demo_enabled:
        return 0

    supabase_url = str(settings.supabase_url).rstrip("/")
    service_key = settings.supabase_service_role_key.get_secret_value()

    # Find the demo workspace.
    demo_ws_name = settings.demo_workspace_name
    async with get_session_factory()() as session:
        ws = (
            await session.execute(
                select(Workspace.id).where(Workspace.name == demo_ws_name)
            )
        ).scalar_one_or_none()

    if ws is None:
        return 0

    # List users from Supabase Auth, looking for guest accounts.
    # Supabase admin API paginates; we scan up to 1000 users per run.
    expired_user_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=15.0) as client:
        page = 0
        per_page = 100
        max_pages = 10  # Safety limit: 1000 users max

        while page < max_pages:
            response = await client.get(
                f"{supabase_url}/auth/v1/admin/users",
                headers={
                    "apikey": service_key,
                    "Authorization": f"Bearer {service_key}",
                },
                params={"page": page, "per_page": per_page},
            )

            if response.status_code >= 400:
                logger.warning(
                    "Supabase admin users list failed: {status}",
                    status=response.status_code,
                )
                break

            users = response.json().get("users", [])
            if not users:
                break

            for user in users:
                meta = user.get("user_metadata", {})
                if not meta.get("is_guest"):
                    continue

                created_at_str = user.get("created_at", "")
                if not created_at_str:
                    continue

                try:
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                except ValueError:
                    continue

                now = datetime.now(timezone.utc)  # noqa: UP017 — Python 3.10 compat
                age_hours = (now - created_at).total_seconds() / 3600

                if age_hours >= settings.demo_guest_ttl_hours:
                    expired_user_ids.add(user["id"])

            page += 1

    if not expired_user_ids:
        return 0

    deleted_count = 0

    # Delete member rows first (can't delete Supabase users via DB).
    async with get_session_factory()() as session:
        async with session.begin():
            result = await session.execute(
                delete(Member).where(
                    Member.workspace_id == ws,
                    Member.user_id.in_(
                        list(expired_user_ids)
                    ),
                )
            )
            deleted_count = result.rowcount

    # Delete the Supabase Auth users (cannot cascade from DB).
    async with httpx.AsyncClient(timeout=15.0) as client:
        for user_id in expired_user_ids:
            try:
                response = await client.delete(
                    f"{supabase_url}/auth/v1/admin/users/{user_id}",
                    headers={
                        "apikey": service_key,
                        "Authorization": f"Bearer {service_key}",
                    },
                    params={"gotrue": "true"},
                )
                if response.status_code >= 400:
                    logger.warning(
                        "Failed to delete guest user {uid}: {status}",
                        uid=user_id,
                        status=response.status_code,
                    )
            except httpx.HTTPError as exc:
                logger.warning(
                    "Network error deleting guest user {uid}: {error}",
                    uid=user_id,
                    error=str(exc)[:200],
                )

    logger.info(
        "Demo cleanup: removed {count} expired guest(s) from workspace {ws}",
        count=deleted_count,
        ws=ws,
    )

    return deleted_count


__all__ = ["cleanup_demo_guests"]
