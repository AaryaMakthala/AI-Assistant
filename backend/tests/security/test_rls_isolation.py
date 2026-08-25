"""Proves RLS blocks cross-workspace reads at the database, not just in application code.

These tests run as `app_tenant` — the NOLOGIN, NOBYPASSRLS role the migration creates —
so they exercise the policies the way a real request does. A superuser connection ignores
RLS entirely, which is why the app must never connect as one.

Requires a live Postgres with migrations already applied (through 0008+), addressed by
``TEST_DATABASE_URL``; skipped when that is unset or unreachable. It deliberately does not
read ``DATABASE_URL`` from settings: the test suite stubs that value out, so these tests
would silently skip against a placeholder host and report a pass they never earned.
"""

import os
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.security.rls import set_tenant_claims

_TEST_DATABASE_URL = "TEST_DATABASE_URL"
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Canonical workspace-scoped tables that must have RLS enabled and forced.
WORKSPACE_SCOPED_TABLES = (
    "workspaces",
    "members",
    "documents",
    "document_chunks",
    "chat_sessions",
    "chat_messages",
    "invitations",
)


def _test_database_url() -> str | None:
    """The environment wins; the repo-root .env is a convenience fallback."""
    from_env = os.environ.get(_TEST_DATABASE_URL)
    if from_env:
        return from_env
    return dotenv_values(_REPO_ROOT / ".env").get(_TEST_DATABASE_URL) or None


async def _ensure_auth_users(conn: AsyncConnection, user_ids: list[uuid.UUID]) -> None:
    """Insert rows into auth.users so FK references succeed.

    The canonical schema FKs to auth.users; the old org-centric schema had its own
    local ``users`` table with no FK to auth.users. We insert directly as superuser
    (bypassing any RLS on auth schema).
    """
    for uid in user_ids:
        # Use ON CONFLICT to be idempotent — the same user may appear in multiple
        # test runs within the same database.
        await conn.execute(
            text(
                "INSERT INTO auth.users (id, email, raw_app_meta_data, raw_user_meta_data, "
                "created_at, updated_at) "
                "VALUES (:id, :email, '{}'::jsonb, '{}'::jsonb, now(), now()) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": uid, "email": f"{uid.hex[:12]}@rls-test.example"},
        )


async def _seeded(conn: AsyncConnection) -> dict[str, uuid.UUID]:
    """Insert two workspaces with members and documents, bypassing RLS as table owner."""
    ids = {
        "ws_a": uuid.uuid4(),
        "ws_b": uuid.uuid4(),
        "user_a": uuid.uuid4(),
        "user_b": uuid.uuid4(),
        "user_c": uuid.uuid4(),  # extra user for cross-user chat privacy test
        "doc_a": uuid.uuid4(),
        "doc_b": uuid.uuid4(),
    }

    # Ensure auth.users rows exist for FK references.
    await _ensure_auth_users(conn, [ids["user_a"], ids["user_b"], ids["user_c"]])

    # Force RLS off for seeding (the superuser would bypass it anyway, but be explicit).
    for table in WORKSPACE_SCOPED_TABLES:
        await conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))

    try:
        for side in ("a", "b"):
            ws_id = ids[f"ws_{side}"]
            user_id = ids[f"user_{side}"]
            doc_id = ids[f"doc_{side}"]

            # Workspace — owner_id references auth.users.
            await conn.execute(
                text(
                    "INSERT INTO workspaces (id, name, owner_id) "
                    "VALUES (:id, :name, :owner)"
                ),
                {"id": ws_id, "name": f"Workspace {side.upper()}", "owner": user_id},
            )

            # Member — OWNER membership for the workspace creator.
            await conn.execute(
                text(
                    "INSERT INTO members (id, workspace_id, user_id, role, status) "
                    "VALUES (:id, :ws, :user, 'OWNER', 'ACTIVE')"
                ),
                {"id": uuid.uuid4(), "ws": ws_id, "user": user_id},
            )

            # Document — READY status, uploaded_by references auth.users.
            await conn.execute(
                text(
                    "INSERT INTO documents "
                    "(id, workspace_id, uploaded_by, filename, mime_type, file_size, "
                    "checksum, file_data, status) "
                    "VALUES (:id, :ws, :u, :f, 'text/plain', 10, :ck, E'\\\\x00', 'READY')"
                ),
                {
                    "id": doc_id,
                    "ws": ws_id,
                    "u": user_id,
                    "f": f"secret-{side}.txt",
                    "ck": f"sha256-{side}-{uuid.uuid4().hex[:8]}",
                },
            )
    finally:
        # Drain any queued trigger events before re-enabling FORCE.
        await conn.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
        for table in WORKSPACE_SCOPED_TABLES:
            await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))

    return ids


@pytest.fixture
async def tenant_conn():
    """A connection seeded with two workspaces, acting as `app_tenant`, rolled back after."""
    url = _test_database_url()
    if not url:
        pytest.skip(f"{_TEST_DATABASE_URL} is not set — no database to prove RLS against")
    # statement_cache_size=0 keeps this working through a transaction-mode pooler
    # (e.g. Supabase's 6543 port), which cannot hold asyncpg's prepared statements.
    engine = create_async_engine(url, poolclass=None, connect_args={"statement_cache_size": 0})
    try:
        conn = await engine.connect()
    except (OSError, SQLAlchemyError) as exc:
        await engine.dispose()
        pytest.skip(f"no reachable database at DATABASE_URL: {exc}")
    try:
        # Everything happens in one transaction that is never committed, so the test
        # leaves no rows behind even if it fails partway.
        await conn.begin()
        ids = await _seeded(conn)
        await conn.execute(text("SET LOCAL ROLE app_tenant"))
        yield conn, ids
        await conn.rollback()
    finally:
        await conn.close()
        await engine.dispose()


async def _visible_doc_ids(conn: AsyncConnection) -> set[uuid.UUID]:
    result = await conn.execute(text("SELECT id FROM documents"))
    return {row[0] for row in result}


async def test_tenant_role_does_not_bypass_rls(tenant_conn) -> None:
    conn, _ = tenant_conn
    row = (
        await conn.execute(
            text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
        )
    ).one()
    assert row.rolsuper is False, "app role must not be a superuser — superusers ignore RLS"
    assert row.rolbypassrls is False, "app role must not have BYPASSRLS"


async def test_user_sees_only_own_workspace_documents(tenant_conn) -> None:
    conn, ids = tenant_conn
    await set_tenant_claims(conn, workspace_id=ids["ws_a"], user_id=ids["user_a"])
    visible = await _visible_doc_ids(conn)
    assert ids["doc_a"] in visible
    assert ids["doc_b"] not in visible


async def test_targeting_another_workspaces_document_by_id_returns_nothing(tenant_conn) -> None:
    """Knowing the exact primary key must not be enough to read another workspace's row."""
    conn, ids = tenant_conn
    await set_tenant_claims(conn, workspace_id=ids["ws_a"], user_id=ids["user_a"])
    result = await conn.execute(
        text("SELECT id FROM documents WHERE id = :id"), {"id": ids["doc_b"]}
    )
    assert result.first() is None


async def test_session_without_claims_sees_nothing(tenant_conn) -> None:
    """A request that forgets to set claims must fail closed, not open."""
    conn, _ = tenant_conn
    assert await _visible_doc_ids(conn) == set()


async def test_cannot_insert_document_into_another_workspace(tenant_conn) -> None:
    conn, ids = tenant_conn
    await set_tenant_claims(conn, workspace_id=ids["ws_a"], user_id=ids["user_a"])
    with pytest.raises(DBAPIError):
        await conn.execute(
            text(
                "INSERT INTO documents "
                "(workspace_id, uploaded_by, filename, mime_type, file_size, "
                "checksum, file_data, status) "
                "VALUES (:ws, :u, 'planted.txt', 'text/plain', 1, "
                "'sha256-planted', E'\\\\x00', 'PENDING')"
            ),
            {"ws": ids["ws_b"], "u": ids["user_a"]},
        )
    await conn.rollback()


async def test_cannot_move_own_document_to_another_workspace(tenant_conn) -> None:
    """WITH CHECK must stop an update that would hand a row to a different tenant."""
    conn, ids = tenant_conn
    await set_tenant_claims(conn, workspace_id=ids["ws_a"], user_id=ids["user_a"])
    with pytest.raises(DBAPIError):
        await conn.execute(
            text("UPDATE documents SET workspace_id = :ws_b WHERE id = :id"),
            {"ws_b": ids["ws_b"], "id": ids["doc_a"]},
        )
    await conn.rollback()


async def test_chat_sessions_are_private_to_their_author(tenant_conn) -> None:
    conn, ids = tenant_conn

    await set_tenant_claims(conn, workspace_id=ids["ws_a"], user_id=ids["user_a"])
    session_id = (
        await conn.execute(
            text(
                "INSERT INTO chat_sessions (workspace_id, user_id) "
                "VALUES (:ws, :user) RETURNING id"
            ),
            {"ws": ids["ws_a"], "user": ids["user_a"]},
        )
    ).scalar_one()

    # Same workspace, different user: workspace-level access is not enough for chat history.
    await set_tenant_claims(conn, workspace_id=ids["ws_a"], user_id=ids["user_c"])
    result = await conn.execute(
        text("SELECT id FROM chat_sessions WHERE id = :id"), {"id": session_id}
    )
    assert result.first() is None


async def test_rls_enabled_and_forced_on_every_workspace_scoped_table(tenant_conn) -> None:
    conn, _ = tenant_conn
    result = await conn.execute(
        text(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class"
            " WHERE relname = ANY(:names)"
        ),
        {"names": list(WORKSPACE_SCOPED_TABLES)},
    )
    rows = {r.relname: (r.relrowsecurity, r.relforcerowsecurity) for r in result}
    assert set(rows) == set(WORKSPACE_SCOPED_TABLES)
    for table, (enabled, forced) in rows.items():
        assert enabled, f"{table} has RLS disabled"
        assert forced, f"{table} does not FORCE RLS — its owner would bypass every policy"
