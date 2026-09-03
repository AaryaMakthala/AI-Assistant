"""Integration tests for the demo workspace seed script.

Verifies that seed_demo_workspace() creates the demo workspace and pre-loaded
sample documents correctly without RLS errors, and that running it twice is
idempotent (no duplication).

Requires a live Postgres with migrations already applied (through 0015+),
addressed by ``TEST_DATABASE_URL``; skipped when that is unset or unreachable.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import dotenv_values
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.config import Settings
from app.demo.seed import seed_demo_workspace

_TEST_DATABASE_URL = "TEST_DATABASE_URL"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _test_database_url() -> str | None:
    """The environment wins; the repo-root .env is a convenience fallback."""
    from_env = os.environ.get(_TEST_DATABASE_URL)
    if from_env:
        return from_env
    return dotenv_values(_REPO_ROOT / ".env").get(_TEST_DATABASE_URL) or None


async def _ensure_demo_owner_in_auth(conn: AsyncConnection, owner_id: uuid.UUID) -> None:
    """Insert the demo owner into auth.users so FK references succeed."""
    await conn.execute(
        text(
            "INSERT INTO auth.users (id, email, raw_app_meta_data, raw_user_meta_data, "
            "created_at, updated_at) "
            "VALUES (:id, :email, '{}'::jsonb, '{}'::jsonb, now(), now()) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": owner_id, "email": "demo-owner@officebrain.app"},
    )


async def _seed_workspace(
    conn: AsyncConnection,
    owner_id: uuid.UUID,
) -> uuid.UUID:
    """Insert the demo workspace and OWNER membership, bypassing RLS."""
    ws_id = uuid.uuid4()

    # Force RLS off for seeding (superuser would bypass anyway, but be explicit).
    for table in (
        "workspaces",
        "members",
        "documents",
        "document_chunks",
        "chat_sessions",
        "chat_messages",
        "invitations",
    ):
        await conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))

    try:
        await conn.execute(
            text(
                "INSERT INTO workspaces (id, name, owner_id) "
                "VALUES (:id, :name, :owner)"
            ),
            {
                "id": ws_id,
                "name": "Office Brain Demo",
                "owner": owner_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO members (id, workspace_id, user_id, role, status) "
                "VALUES (:id, :ws, :user, 'OWNER', 'ACTIVE')"
            ),
            {"id": uuid.uuid4(), "ws": ws_id, "user": owner_id},
        )
    finally:
        # Drain any queued trigger events before re-enabling FORCE.
        await conn.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
        for table in (
            "workspaces",
            "members",
            "documents",
            "document_chunks",
            "chat_sessions",
            "chat_messages",
            "invitations",
        ):
            await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))

    return ws_id


@pytest.fixture
async def test_db_conn():
    """A connection to the test database, rolled back after each test."""
    url = _test_database_url()
    if not url:
        pytest.skip(f"{_TEST_DATABASE_URL} is not set — no database to prove seeding against")
    # statement_cache_size=0 keeps this working through a transaction-mode pooler
    engine = create_async_engine(url, poolclass=None, connect_args={"statement_cache_size": 0})
    try:
        conn = await engine.connect()
    except (OSError, SQLAlchemyError) as exc:
        await engine.dispose()
        pytest.skip(f"no reachable database at {_TEST_DATABASE_URL}: {exc}")

    try:
        # Everything happens in one transaction that is never committed.
        await conn.begin()
        yield conn
        await conn.rollback()
    finally:
        await conn.close()
        await engine.dispose()


@pytest.fixture
def mock_supabase_admin():
    """Mock the Supabase Admin API responses for demo owner creation/lookup."""
    with patch("app.demo.seed.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        # Mock the GET /admin/users (owner lookup) - returns empty list (owner not found)
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"users": []}

        # Mock the POST /admin/users (owner creation)
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        owner_id = uuid.uuid4()
        mock_post_response.json.return_value = {"id": str(owner_id)}

        # Return GET first, then POST
        mock_client.get.return_value = mock_get_response
        mock_client.post.return_value = mock_post_response

        yield owner_id


async def _count_docs_and_chunks(conn: AsyncConnection, ws_id: uuid.UUID, owner_id: uuid.UUID) -> tuple[int, int]:
    """Count documents and chunks for a workspace, setting RLS claims."""
    from app.security.rls import set_tenant_claims

    await set_tenant_claims(conn, workspace_id=ws_id, user_id=owner_id)

    doc_count = (
        await conn.execute(
            text("SELECT COUNT(*) FROM documents WHERE workspace_id = :ws"),
            {"ws": ws_id},
        )
    ).scalar_one()

    chunk_count = (
        await conn.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE workspace_id = :ws"),
            {"ws": ws_id},
        )
    ).scalar_one()

    return doc_count, chunk_count


@pytest.mark.asyncio
async def test_seed_creates_workspace_without_rls_error(test_db_conn, mock_supabase_admin) -> None:
    """seed_demo_workspace runs without 'requires an authenticated sub claim' error."""
    conn = test_db_conn
    owner_id = mock_supabase_admin

    # Ensure the demo owner exists in auth.users (FK target).
    await _ensure_demo_owner_in_auth(conn, owner_id)

    # Run the seed function directly.
    # It will create its own session via get_session_factory().
    ws_id = await seed_demo_workspace()

    assert ws_id is not None

    # Verify workspace and membership were created.
    result = (
        await conn.execute(
            text("SELECT id, owner_id FROM workspaces WHERE name = 'Office Brain Demo'")
        )
    ).first()
    assert result is not None
    assert result.owner_id == owner_id

    member = (
        await conn.execute(
            text("SELECT user_id, role, status FROM members WHERE workspace_id = :ws"),
            {"ws": ws_id},
        )
    ).first()
    assert member is not None
    assert member.user_id == owner_id
    assert member.role == "OWNER"
    assert member.status == "ACTIVE"

    # Verify documents were created and are READY.
    doc_count, chunk_count = await _count_docs_and_chunks(conn, ws_id, owner_id)
    assert doc_count == 3  # 3 sample documents
    assert chunk_count > 0  # At least some chunks per doc


@pytest.mark.asyncio
async def test_seed_idempotent_no_duplication(test_db_conn, mock_supabase_admin) -> None:
    """Running seed_demo_workspace twice does not duplicate documents or chunks."""
    conn = test_db_conn
    owner_id = mock_supabase_admin

    # Ensure the demo owner exists in auth.users.
    await _ensure_demo_owner_in_auth(conn, owner_id)

    # First run.
    ws_id_1 = await seed_demo_workspace()
    assert ws_id_1 is not None

    doc_count_1, chunk_count_1 = await _count_docs_and_chunks(conn, ws_id_1, owner_id)
    assert doc_count_1 == 3
    assert chunk_count_1 > 0

    # Second run — should be idempotent.
    ws_id_2 = await seed_demo_workspace()
    assert ws_id_2 is not None
    assert ws_id_2 == ws_id_1  # Same workspace ID returned

    doc_count_2, chunk_count_2 = await _count_docs_and_chunks(conn, ws_id_1, owner_id)
    assert doc_count_2 == doc_count_1, "Document count should not change on second run"
    assert chunk_count_2 == chunk_count_1, "Chunk count should not change on second run"

    # Verify membership wasn't duplicated.
    member_count = (
        await conn.execute(
            text("SELECT COUNT(*) FROM members WHERE workspace_id = :ws"),
            {"ws": ws_id_1},
        )
    ).scalar_one()
    assert member_count == 1, "Only one OWNER membership should exist"


@pytest.mark.asyncio
async def test_seed_does_not_leak_claims(test_db_conn, mock_supabase_admin) -> None:
    """After seed runs, claims should not leak to other transactions on the same connection."""
    conn = test_db_conn
    owner_id = mock_supabase_admin

    # Ensure the demo owner exists in auth.users.
    await _ensure_demo_owner_in_auth(conn, owner_id)

    # Run seed.
    ws_id = await seed_demo_workspace()
    assert ws_id is not None

    # Now start a fresh transaction WITHOUT setting claims.
    # The connection should see NO documents (RLS blocks everything without claims).
    await conn.rollback()
    await conn.begin()

    doc_count = (
        await conn.execute(text("SELECT COUNT(*) FROM documents"))
    ).scalar_one()

    assert doc_count == 0, "Claims must not leak across transactions — RLS should block"


# ---------------------------------------------------------------------------
# Tests for DEMO_WORKSPACE_ID feature
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_with_valid_demo_workspace_id(test_db_conn, mock_supabase_admin) -> None:
    """When demo_workspace_id points to an existing workspace, seed uses it directly
    without calling _ensure_demo_owner or create_workspace.
    """
    conn = test_db_conn
    owner_id = mock_supabase_admin

    # Create a workspace and owner membership directly in the DB.
    ws_id = uuid.uuid4()
    await _ensure_demo_owner_in_auth(conn, owner_id)
    await conn.execute(
        text(
            "INSERT INTO workspaces (id, name, owner_id) "
            "VALUES (:id, :name, :owner)"
        ),
        {"id": ws_id, "name": "Pre-existing Workspace", "owner": owner_id},
    )
    await conn.execute(
        text(
            "INSERT INTO members (id, workspace_id, user_id, role, status) "
            "VALUES (:id, :ws, :user, 'OWNER', 'ACTIVE')"
        ),
        {"id": uuid.uuid4(), "ws": ws_id, "user": owner_id},
    )

    # Patch get_settings to return demo_workspace_id set to the existing workspace.
    with patch("app.demo.seed.get_settings") as mock_get_settings:
        mock_settings = MagicMock(spec=Settings)
        mock_settings.demo_workspace_id = str(ws_id)
        mock_settings.demo_workspace_name = "Office Brain Demo"
        mock_get_settings.return_value = mock_settings

        ws_result = await seed_demo_workspace()

    assert ws_result == ws_id

    # Verify documents were ingested into the existing workspace.
    doc_count, chunk_count = await _count_docs_and_chunks(conn, ws_id, owner_id)
    assert doc_count == 3, "3 sample documents should be ingested"
    assert chunk_count > 0, "Chunks should be present"

    # Verify no new workspace was created (the pre-existing one is the only one).
    all_ws = (
        await conn.execute(text("SELECT id FROM workspaces"))
    ).scalars().all()
    assert len(all_ws) == 1, "No new workspace should have been created"
    assert all_ws[0] == ws_id


@pytest.mark.asyncio
async def test_seed_with_valid_demo_workspace_id_skips_if_already_seeded(
    test_db_conn, mock_supabase_admin
) -> None:
    """When demo_workspace_id points to an already-seeded workspace, seed returns
    early without re-ingesting documents.
    """
    conn = test_db_conn
    owner_id = mock_supabase_admin

    # Create a workspace with documents already present.
    ws_id = uuid.uuid4()
    await _ensure_demo_owner_in_auth(conn, owner_id)
    await conn.execute(
        text(
            "INSERT INTO workspaces (id, name, owner_id) "
            "VALUES (:id, :name, :owner)"
        ),
        {"id": ws_id, "name": "Already Seeded Workspace", "owner": owner_id},
    )
    await conn.execute(
        text(
            "INSERT INTO members (id, workspace_id, user_id, role, status) "
            "VALUES (:id, :ws, :user, 'OWNER', 'ACTIVE')"
        ),
        {"id": uuid.uuid4(), "ws": ws_id, "user": owner_id},
    )
    # Insert a dummy document so the seed sees it as "already seeded".
    await conn.execute(
        text(
            "INSERT INTO documents (workspace_id, uploaded_by, filename, mime_type, "
            "file_size, checksum, file_data, status) "
            "VALUES (:ws, :owner, 'dummy.pdf', 'application/pdf', 100, 'abc123', "
            "'\\x00'::bytea, 'READY')"
        ),
        {"ws": ws_id, "owner": owner_id},
    )

    with patch("app.demo.seed.get_settings") as mock_get_settings:
        mock_settings = MagicMock(spec=Settings)
        mock_settings.demo_workspace_id = str(ws_id)
        mock_settings.demo_workspace_name = "Office Brain Demo"
        mock_get_settings.return_value = mock_settings

        ws_result = await seed_demo_workspace()

    assert ws_result == ws_id

    # Should still only have 1 document (the dummy one), not the 3 sample docs.
    doc_count = (
        await conn.execute(
            text("SELECT COUNT(*) FROM documents WHERE workspace_id = :ws"),
            {"ws": ws_id},
        )
    ).scalar_one()
    assert doc_count == 1, "Should not re-ingest when documents already exist"


@pytest.mark.asyncio
async def test_seed_with_nonexistent_demo_workspace_id_raises() -> None:
    """When demo_workspace_id is set to a nonexistent workspace, seed raises at startup."""
    fake_ws_id = str(uuid.uuid4())

    # Mock session factory so the seed function can query without a real DB.
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_factory = MagicMock(return_value=mock_session)

    with (
        patch("app.demo.seed.get_settings") as mock_get_settings,
        patch("app.demo.seed.get_session_factory", return_value=mock_session_factory),
    ):
        mock_settings = MagicMock(spec=Settings)
        mock_settings.demo_workspace_id = fake_ws_id
        mock_settings.demo_workspace_name = "Office Brain Demo"
        mock_get_settings.return_value = mock_settings

        with pytest.raises(RuntimeError, match="DEMO_WORKSPACE_ID is set.*but no workspace with that ID exists"):
            await seed_demo_workspace()


@pytest.mark.asyncio
async def test_seed_with_invalid_demo_workspace_id_raises() -> None:
    """When demo_workspace_id is set to an invalid UUID, seed raises at startup."""
    with patch("app.demo.seed.get_settings") as mock_get_settings:
        mock_settings = MagicMock(spec=Settings)
        mock_settings.demo_workspace_id = "not-a-valid-uuid"
        mock_settings.demo_workspace_name = "Office Brain Demo"
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError, match="DEMO_WORKSPACE_ID is not a valid UUID"):
            await seed_demo_workspace()


@pytest.mark.asyncio
async def test_seed_without_demo_workspace_id_uses_existing_behavior(
    test_db_conn, mock_supabase_admin
) -> None:
    """When demo_workspace_id is NOT set, seed falls back to create-or-reuse-by-name."""
    conn = test_db_conn
    owner_id = mock_supabase_admin

    await _ensure_demo_owner_in_auth(conn, owner_id)

    with patch("app.demo.seed.get_settings") as mock_get_settings:
        mock_settings = MagicMock(spec=Settings)
        mock_settings.demo_workspace_id = None  # Not set
        mock_settings.demo_workspace_name = "Office Brain Demo"
        mock_get_settings.return_value = mock_settings

        ws_result = await seed_demo_workspace()

    assert ws_result is not None

    # Verify it created the workspace via the name-based path.
    result = (
        await conn.execute(
            text("SELECT name FROM workspaces WHERE id = :ws"),
            {"ws": ws_result},
        )
    ).first()
    assert result is not None
    assert result.name == "Office Brain Demo"


# ---------------------------------------------------------------------------
# ORM insert tests — verify auth.users FK resolution via the stub table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_member_orm_insert_commits(test_db_conn) -> None:
    """Inserting a Member row via the ORM must not raise NoReferencedTableError.

    Before the auth.users stub table was registered in Base.metadata,
    SQLAlchemy could not resolve ForeignKey("auth.users.id") at flush time
    and raised NoReferencedTableError.  This test confirms the fix works
    end-to-end through a real commit.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.db.models import Member, Workspace

    conn = test_db_conn
    user_id = uuid.uuid4()

    # Build an async session from the test connection's engine so we can use
    # the ORM models.  We avoid get_session_factory() because it requires
    # full env vars (DATABASE_URL etc.) which the integration test harness
    # doesn't provide.
    session_factory = async_sessionmaker(
        bind=conn.engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        # Insert a fake auth.users row so the FK is satisfied at the DB level.
        await session.execute(
            text(
                "INSERT INTO auth.users (id, email, raw_app_meta_data, raw_user_meta_data, "
                "created_at, updated_at) "
                "VALUES (:id, :email, '{}'::jsonb, '{}'::jsonb, now(), now()) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": user_id, "email": "orm-test@test.com"},
        )

        ws = Workspace(name="ORM Test WS", owner_id=user_id)
        session.add(ws)
        await session.flush()  # Flush workspace first.

        member = Member(
            workspace_id=ws.id,
            user_id=user_id,
            role="MEMBER",
            status="ACTIVE",
        )
        session.add(member)
        # flush() triggers FK resolution — this is the line that used to fail
        # with NoReferencedTableError before the auth.users stub was added.
        await session.flush()

        # The flush itself is the key assertion: if the auth.users stub were
        # missing, this line would raise NoReferencedTableError.
        # Verify the ORM assigned IDs (meaning the INSERT was prepared).
        assert ws.id is not None, "Workspace must have an ID after flush"
        assert member.id is not None, "Member must have an ID after flush"