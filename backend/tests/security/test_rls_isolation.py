"""Proves RLS blocks cross-org reads at the database, not just in application code.

These tests run as `app_tenant` — the NOLOGIN, NOBYPASSRLS role the migration creates —
so they exercise the policies the way a real request does. A superuser connection ignores
RLS entirely, which is why the app must never connect as one.

Requires a live Postgres with migrations already applied, addressed by `TEST_DATABASE_URL`;
skipped when that is unset or unreachable. It deliberately does not read `DATABASE_URL`
from settings: the test suite stubs that value out, so these tests would silently skip
against a placeholder host and report a pass they never earned.
"""

import os
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.db.models import ORG_SCOPED_TABLES
from app.security.rls import set_tenant_claims

_TEST_DATABASE_URL = "TEST_DATABASE_URL"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _test_database_url() -> str | None:
    """The environment wins; the repo-root .env is a convenience fallback."""
    from_env = os.environ.get(_TEST_DATABASE_URL)
    if from_env:
        return from_env
    return dotenv_values(_REPO_ROOT / ".env").get(_TEST_DATABASE_URL) or None


async def _seeded(conn: AsyncConnection) -> dict[str, uuid.UUID]:
    """Insert two orgs with a user and a document each, bypassing RLS as table owner."""
    ids = {
        "org_a": uuid.uuid4(),
        "org_b": uuid.uuid4(),
        "user_a": uuid.uuid4(),
        "user_b": uuid.uuid4(),
        "doc_a": uuid.uuid4(),
        "doc_b": uuid.uuid4(),
    }
    for table in ORG_SCOPED_TABLES:
        await conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))
    try:
        for side in ("a", "b"):
            await conn.execute(
                text("INSERT INTO organizations (id, name, slug) VALUES (:id, :n, :s)"),
                {
                    "id": ids[f"org_{side}"],
                    "n": f"Org {side.upper()}",
                    "s": f"rls-{side}-{uuid.uuid4().hex[:8]}",
                },
            )
            await conn.execute(
                text("INSERT INTO users (id, org_id, email) VALUES (:id, :org, :email)"),
                {
                    "id": ids[f"user_{side}"],
                    "org": ids[f"org_{side}"],
                    "email": f"{side}-{uuid.uuid4().hex[:8]}@example.test",
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO documents (id, org_id, uploaded_by, filename, mime_type,"
                    " size_bytes) VALUES (:id, :org, :u, :f, 'text/plain', 10)"
                ),
                {
                    "id": ids[f"doc_{side}"],
                    "org": ids[f"org_{side}"],
                    "u": ids[f"user_{side}"],
                    "f": f"secret-{side}.txt",
                },
            )
    finally:
        # Since migration 0007 `fk_documents_uploaded_by` is DEFERRABLE INITIALLY DEFERRED,
        # so the document inserts above leave trigger events queued until commit. Postgres
        # refuses to ALTER a table with pending trigger events, which would make the
        # re-enable below fail and take every test in this module with it. Forcing the
        # constraint to check now drains that queue; the inserts are valid, so nothing here
        # can fail — this only moves the check earlier.
        await conn.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
        for table in ORG_SCOPED_TABLES:
            await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
    return ids


@pytest.fixture
async def tenant_conn():
    """A connection seeded with two orgs, acting as `app_tenant`, rolled back after."""
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


async def test_user_sees_only_own_org_documents(tenant_conn) -> None:
    conn, ids = tenant_conn
    await set_tenant_claims(conn, user_id=ids["user_a"], org_id=ids["org_a"])  # type: ignore[arg-type]
    visible = await _visible_doc_ids(conn)
    assert ids["doc_a"] in visible
    assert ids["doc_b"] not in visible


async def test_targeting_another_orgs_document_by_id_returns_nothing(tenant_conn) -> None:
    """Knowing the exact primary key must not be enough to read another org's row."""
    conn, ids = tenant_conn
    await set_tenant_claims(conn, user_id=ids["user_a"], org_id=ids["org_a"])  # type: ignore[arg-type]
    result = await conn.execute(
        text("SELECT id FROM documents WHERE id = :id"), {"id": ids["doc_b"]}
    )
    assert result.first() is None


async def test_session_without_claims_sees_nothing(tenant_conn) -> None:
    """A request that forgets to set claims must fail closed, not open."""
    conn, _ = tenant_conn
    assert await _visible_doc_ids(conn) == set()


async def test_cannot_insert_document_into_another_org(tenant_conn) -> None:
    conn, ids = tenant_conn
    await set_tenant_claims(conn, user_id=ids["user_a"], org_id=ids["org_a"])  # type: ignore[arg-type]
    with pytest.raises(DBAPIError):
        await conn.execute(
            text(
                "INSERT INTO documents (org_id, filename, mime_type, size_bytes)"
                " VALUES (:org, 'planted.txt', 'text/plain', 1)"
            ),
            {"org": ids["org_b"]},
        )
    await conn.rollback()


async def test_cannot_move_own_document_to_another_org(tenant_conn) -> None:
    """WITH CHECK must stop an update that would hand a row to a different tenant."""
    conn, ids = tenant_conn
    await set_tenant_claims(conn, user_id=ids["user_a"], org_id=ids["org_a"])  # type: ignore[arg-type]
    with pytest.raises(DBAPIError):
        await conn.execute(
            text("UPDATE documents SET org_id = :org_b WHERE id = :id"),
            {"org_b": ids["org_b"], "id": ids["doc_a"]},
        )
    await conn.rollback()


async def test_chat_sessions_are_private_to_their_author(tenant_conn) -> None:
    conn, ids = tenant_conn
    other_user = uuid.uuid4()
    await set_tenant_claims(conn, user_id=ids["user_a"], org_id=ids["org_a"])  # type: ignore[arg-type]
    session_id = (
        await conn.execute(
            text(
                "INSERT INTO chat_sessions (org_id, user_id, title)"
                " VALUES (:org, :user, 'mine') RETURNING id"
            ),
            {"org": ids["org_a"], "user": ids["user_a"]},
        )
    ).scalar_one()

    # Same org, different user: org-level access is not enough for chat history.
    await set_tenant_claims(conn, user_id=other_user, org_id=ids["org_a"])  # type: ignore[arg-type]
    result = await conn.execute(
        text("SELECT id FROM chat_sessions WHERE id = :id"), {"id": session_id}
    )
    assert result.first() is None


async def test_rls_enabled_and_forced_on_every_org_scoped_table(tenant_conn) -> None:
    conn, _ = tenant_conn
    result = await conn.execute(
        text(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class"
            " WHERE relname = ANY(:names)"
        ),
        {"names": list(ORG_SCOPED_TABLES)},
    )
    rows = {r.relname: (r.relrowsecurity, r.relforcerowsecurity) for r in result}
    assert set(rows) == set(ORG_SCOPED_TABLES)
    for table, (enabled, forced) in rows.items():
        assert enabled, f"{table} has RLS disabled"
        assert forced, f"{table} does not FORCE RLS — its owner would bypass every policy"
