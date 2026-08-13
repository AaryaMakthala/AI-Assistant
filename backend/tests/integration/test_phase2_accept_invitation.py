"""Phase 2 invitation acceptance integration tests (migration 0010).

These tests apply the *real* Alembic migrations (0001..0010) to a disposable
Postgres, seed a workspace with an OWNER and PENDING invitations, then exercise
``app.accept_invitation()`` — the SECURITY DEFINER function the
``POST /invitations/{id}/accept`` endpoint calls — through the same ``app_tenant`` +
claims path the API uses. Assertions cover the four contract cases:

* successful acceptance: ACTIVE MEMBER row + invitation ACCEPTED, atomically
* wrong email: refused, nothing written
* already-accepted invitation: refused
* duplicate membership: refused

Plus one RLS boundary check: the invitee cannot read the invitation before accepting,
which is exactly why the definer function (rather than a policy relaxation) exists.

**Safety.** The database is chosen via PHASE1C_TEST_DATABASE_URL, read from the
*environment only* — never from the repo .env, which may point at a real deployment.
The URL must be localhost and its database name must contain "test", or the module
skips. The tests reset that database (DROP SCHEMA public/auth CASCADE) and rebuild it
from scratch; point it at anything but a disposable local database and it will not
run.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from alembic import command

_TEST_DATABASE_URL = "PHASE1C_TEST_DATABASE_URL"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND = _REPO_ROOT / "backend"
_ALEMBIC_INI = _BACKEND / "alembic.ini"
_AUTH_BOOTSTRAP = _BACKEND / "scripts" / "dev_auth_schema.sql"

#: A complete, valid environment for app.config.Settings — the conftest strips .env,
#: so these must be provided explicitly whenever anything calls get_settings()
#: (alembic/env.py does, to resolve the database URL).
_REQUIRED_ENV = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_ANON_KEY": "test-anon-key",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "LLM_PROVIDER": "test-provider",
    "LLM_MODEL": "test-model",
    "LLM_API_KEY": "test-llm-api-key",
    "JWT_SECRET": "test-jwt-secret-that-is-long-enough-to-pass-validation",
}

#: SQLSTATEs raised by app.accept_invitation() (migration 0010).
_WRONG_EMAIL = "W1003"
_ALREADY_ACCEPTED = "W1002"
_DUPLICATE_MEMBER = "W1004"


def _test_database_url() -> str | None:
    url = os.environ.get(_TEST_DATABASE_URL)
    if not url:
        return None
    if "localhost" not in url and "127.0.0.1" not in url:
        pytest.skip(f"{_TEST_DATABASE_URL} must point at a local disposable database")
    if "test" not in url.rsplit("/", 1)[-1]:
        pytest.skip("the database name in PHASE1C_TEST_DATABASE_URL must contain 'test'")
    return url


def _split_sql(script: str) -> list[str]:
    """Split a SQL script on top-level ';', respecting $$ dollar-quoted bodies."""
    statements: list[str] = []
    current: list[str] = []
    in_dollar = False
    i = 0
    n = len(script)
    while i < n:
        ch = script[i]
        if not in_dollar and script.startswith("$$", i):
            in_dollar = True
            current.append("$$")
            i += 2
            continue
        if in_dollar and script.startswith("$$", i):
            in_dollar = False
            current.append("$$")
            i += 2
            continue
        if not in_dollar and ch == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    statement = "".join(current).strip()
    if statement:
        statements.append(statement)
    return statements


def _upgrade(rev: str) -> None:
    config = Config(str(_ALEMBIC_INI))
    command.upgrade(config, rev)


def _reset_and_bootstrap(url: str) -> None:
    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
                await conn.execute(text("CREATE SCHEMA public"))
                await conn.execute(text("DROP SCHEMA IF EXISTS auth CASCADE"))
            for statement in _split_sql(_AUTH_BOOTSTRAP.read_text(encoding="utf-8")):
                async with engine.begin() as conn:
                    await conn.execute(text(statement))
        finally:
            await engine.dispose()

    asyncio.run(_run())


@pytest.fixture(scope="module")
def accepted_db() -> dict:
    """Fresh 0001..0010 on a disposable database, with a seeded target workspace.

    Signups run through the real provisioning trigger (0008), so each auth.users row
    gains its own default workspace + OWNER membership, exactly like production; the
    test's target workspace and its OWNER membership are then added explicitly.
    """
    url = _test_database_url()
    if url is None:
        pytest.skip(f"{_TEST_DATABASE_URL} is not set — no disposable database")

    import app.config as config_module

    os.environ["DATABASE_URL"] = url
    for key, value in _REQUIRED_ENV.items():
        os.environ[key] = value
    config_module.get_settings.cache_clear()

    _reset_and_bootstrap(url)
    _upgrade("head")

    async def _seed() -> dict:
        owner = uuid.uuid4()
        invitee = uuid.uuid4()
        owner_email = "owner@example.test"
        invitee_email = "invitee@example.test"
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                # Trigger provisions each with a default workspace + OWNER membership.
                await conn.execute(
                    text("INSERT INTO auth.users (id, email) VALUES (:id, :email)"),
                    [
                        {"id": owner, "email": owner_email},
                        {"id": invitee, "email": invitee_email},
                    ],
                )
                ws = uuid.uuid4()
                await conn.execute(
                    text(
                        "INSERT INTO workspaces (id, name, owner_id) "
                        "VALUES (:id, 'Target', :owner)"
                    ),
                    {"id": ws, "owner": owner},
                )
                await conn.execute(
                    text(
                        "INSERT INTO members (workspace_id, user_id, role, status) "
                        "VALUES (:ws, :owner, 'OWNER', 'ACTIVE')"
                    ),
                    {"ws": ws, "owner": owner},
                )
        finally:
            await engine.dispose()
        return {
            "url": url,
            "ws": ws,
            "owner": owner,
            "owner_email": owner_email,
            "invitee": invitee,
            "invitee_email": invitee_email,
        }

    return asyncio.run(_seed())


async def _engine(url: str) -> AsyncEngine:
    return create_async_engine(url, connect_args={"statement_cache_size": 0})


async def _tenant_conn(
    url: str,
    *,
    workspace_id: uuid.UUID | None,
    user_id: uuid.UUID | None,
    svc: str | None = None,
) -> tuple[AsyncConnection, AsyncEngine]:
    """A connection acting as app_tenant with the given claims, in a transaction."""
    engine = await _engine(url)
    conn = await engine.connect()
    await conn.begin()
    claims: dict[str, str] = {}
    if workspace_id is not None:
        claims["workspace_id"] = str(workspace_id)
    if user_id is not None:
        claims["sub"] = str(user_id)
    if svc is not None:
        claims["svc"] = svc
    await conn.execute(
        text("SELECT set_config('request.jwt.claims', :claims, true)"),
        {"claims": json.dumps(claims)},
    )
    await conn.execute(text("SET LOCAL ROLE app_tenant"))
    return conn, engine


async def _close(conn: AsyncConnection, engine: AsyncEngine) -> None:
    await conn.rollback()
    await conn.close()
    await engine.dispose()


async def _insert_invitation(
    url: str,
    *,
    ws: uuid.UUID,
    email: str,
    invited_by: uuid.UUID,
    status: str = "PENDING",
) -> uuid.UUID:
    """Insert an invitation as the superuser (bypassing RLS), returning its id."""
    engine = await _engine(url)
    try:
        async with engine.begin() as conn:
            return (
                await conn.execute(
                    text(
                        "INSERT INTO invitations (workspace_id, email, status, invited_by) "
                        "VALUES (:ws, :email, :status, :by) RETURNING id"
                    ),
                    {"ws": ws, "email": email, "status": status, "by": invited_by},
                )
            ).scalar_one()
    finally:
        await engine.dispose()


async def _invitation_status(url: str, invitation_id: uuid.UUID) -> str:
    engine = await _engine(url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT status FROM invitations WHERE id = :id"),
                    {"id": invitation_id},
                )
            ).scalar_one()
    finally:
        await engine.dispose()


async def _member_row(url: str, member_id: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID, str, str]:
    engine = await _engine(url)
    try:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT workspace_id, user_id, role, status FROM members "
                        "WHERE id = :id"
                    ),
                    {"id": member_id},
                )
            ).one()
    finally:
        await engine.dispose()
    return row.workspace_id, row.user_id, row.role, row.status


async def _accept(
    conn: AsyncConnection, invitation_id: uuid.UUID, email: str
) -> uuid.UUID:
    """Call app.accept_invitation() as the caller the claims describe."""
    return (
        await conn.execute(
            text("SELECT app.accept_invitation(:inv, :email)"),
            {"inv": invitation_id, "email": email},
        )
    ).scalar_one()


def _sqlstate(exc: BaseException) -> str | None:
    return getattr(getattr(exc, "orig", None), "sqlstate", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_acceptance_creates_active_member(accepted_db: dict) -> None:
    db = accepted_db
    invitation_id = await _insert_invitation(
        db["url"], ws=db["ws"], email=db["invitee_email"], invited_by=db["owner"]
    )

    conn, engine = await _tenant_conn(
        db["url"], workspace_id=db["ws"], user_id=db["invitee"]
    )
    try:
        member_id = await _accept(conn, invitation_id, db["invitee_email"])

        # The membership is created as MEMBER/ACTIVE and readable back under RLS now
        # that the caller is a member (same transaction).
        visible = (
            await conn.execute(text("SELECT id FROM members WHERE id = :id"), {"id": member_id})
        ).scalar_one()
        assert visible == member_id
    finally:
        await _close(conn, engine)

    ws, user, role, status = await _member_row(db["url"], member_id)
    assert (ws, user, role, status) == (db["ws"], db["invitee"], "MEMBER", "ACTIVE")
    assert await _invitation_status(db["url"], invitation_id) == "ACCEPTED"


@pytest.mark.asyncio
async def test_wrong_email_is_rejected(accepted_db: dict) -> None:
    db = accepted_db
    invitation_id = await _insert_invitation(
        db["url"], ws=db["ws"], email=db["invitee_email"], invited_by=db["owner"]
    )

    conn, engine = await _tenant_conn(
        db["url"], workspace_id=db["ws"], user_id=db["invitee"]
    )
    try:
        with pytest.raises(DBAPIError) as excinfo:
            await _accept(conn, invitation_id, "someone.else@example.test")
        assert _sqlstate(excinfo.value) == _WRONG_EMAIL
    finally:
        await _close(conn, engine)

    # Nothing was written: the invitation is untouched and no membership exists.
    assert await _invitation_status(db["url"], invitation_id) == "PENDING"
    engine = await _engine(db["url"])
    try:
        async with engine.connect() as conn:
            count = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM members "
                        "WHERE workspace_id = :ws AND user_id = :user"
                    ),
                    {"ws": db["ws"], "user": db["invitee"]},
                )
            ).scalar_one()
    finally:
        await engine.dispose()
    assert count == 0


@pytest.mark.asyncio
async def test_already_accepted_invitation_is_rejected(accepted_db: dict) -> None:
    db = accepted_db
    invitation_id = await _insert_invitation(
        db["url"],
        ws=db["ws"],
        email=db["invitee_email"],
        invited_by=db["owner"],
        status="ACCEPTED",
    )

    conn, engine = await _tenant_conn(
        db["url"], workspace_id=db["ws"], user_id=db["invitee"]
    )
    try:
        with pytest.raises(DBAPIError) as excinfo:
            await _accept(conn, invitation_id, db["invitee_email"])
        assert _sqlstate(excinfo.value) == _ALREADY_ACCEPTED
    finally:
        await _close(conn, engine)


@pytest.mark.asyncio
async def test_duplicate_membership_is_rejected(accepted_db: dict) -> None:
    db = accepted_db
    # The invitee is already an ACTIVE member of the target workspace.
    engine = await _engine(db["url"])
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO members (workspace_id, user_id, role, status) "
                    "VALUES (:ws, :user, 'MEMBER', 'ACTIVE')"
                ),
                {"ws": db["ws"], "user": db["invitee"]},
            )
    finally:
        await engine.dispose()
    invitation_id = await _insert_invitation(
        db["url"], ws=db["ws"], email=db["invitee_email"], invited_by=db["owner"]
    )

    conn, engine = await _tenant_conn(
        db["url"], workspace_id=db["ws"], user_id=db["invitee"]
    )
    try:
        with pytest.raises(DBAPIError) as excinfo:
            await _accept(conn, invitation_id, db["invitee_email"])
        assert _sqlstate(excinfo.value) == _DUPLICATE_MEMBER
    finally:
        await _close(conn, engine)

    # The invitation was not consumed.
    assert await _invitation_status(db["url"], invitation_id) == "PENDING"


@pytest.mark.asyncio
async def test_invitee_cannot_read_invitation_under_rls(accepted_db: dict) -> None:
    """Before accepting, the invitee cannot see the invitation — RLS is unweakened.

    This is the property that justifies the SECURITY DEFINER function: the invitee is
    not a member of the target workspace, so the member-scoped policy admits nothing,
    and the function (running as the table owner) is the only path into the
    pre-membership transition.
    """
    db = accepted_db
    invitation_id = await _insert_invitation(
        db["url"], ws=db["ws"], email=db["invitee_email"], invited_by=db["owner"]
    )

    conn, engine = await _tenant_conn(
        db["url"], workspace_id=db["ws"], user_id=db["invitee"]
    )
    try:
        visible = (
            await conn.execute(
                text("SELECT count(*) FROM invitations WHERE id = :id"),
                {"id": invitation_id},
            )
        ).scalar_one()
    finally:
        await _close(conn, engine)

    assert visible == 0
