"""Phase 10 acceptance: deleting a document removes every resource it owns.

Deletion is the requirement most likely to be *nearly* right — a row disappears, the UI
looks correct, and the vectors stay behind and keep surfacing in retrieval. So these tests
assert on what is left in the database and on disk afterwards, not on the response code.

The other half is isolation: one org must not be able to delete, or even confirm the
existence of, another org's document. That is enforced by RLS rather than by application
code, which is exactly why it is worth testing through a real database.

Requires `TEST_DATABASE_URL` with migrations applied. Skips cleanly when absent — a skip
means unproven, not passed.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.db.models import ORG_SCOPED_TABLES

pytestmark = pytest.mark.usefixtures("valid_env")

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: A 384-dimension vector, matching the pinned embedding model.
_VECTOR = "[" + ",".join(["0.1"] * 384) + "]"


def _database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL") or (
        dotenv_values(_REPO_ROOT / ".env").get("TEST_DATABASE_URL") or None
    )


@pytest.fixture
def delete_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    url = _database_url()
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set — cannot prove deletion end to end")

    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


def _engine(url: str):
    return create_async_engine(url, connect_args={"statement_cache_size": 0})


async def _seed_document(url: str, *, org_slug: str) -> dict[str, uuid.UUID]:
    """Create an org, user, document, chunk and failure row, bypassing RLS as owner."""
    ids = {
        "org": uuid.uuid4(),
        "user": uuid.uuid4(),
        "doc": uuid.uuid4(),
        "storage_key": uuid.uuid4(),
    }
    engine = _engine(url)
    async with engine.begin() as conn:
        for table in ORG_SCOPED_TABLES:
            await conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))
        try:
            await conn.execute(
                text("INSERT INTO organizations (id, name, slug) VALUES (:id, 'Del', :slug)"),
                {"id": ids["org"], "slug": org_slug},
            )
            await conn.execute(
                text("INSERT INTO users (id, org_id, email) VALUES (:id, :org, :email)"),
                {
                    "id": ids["user"],
                    "org": ids["org"],
                    "email": f"del-{uuid.uuid4().hex[:8]}@example.test",
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO documents (id, org_id, uploaded_by, filename, storage_key,"
                    " mime_type, size_bytes, status, chunk_count) VALUES (:id, :org, :user,"
                    " 'policy.pdf', :key, 'application/pdf', 1024, 'ready', 1)"
                ),
                {
                    "id": ids["doc"],
                    "org": ids["org"],
                    "user": ids["user"],
                    "key": ids["storage_key"],
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO document_chunks (org_id, document_id, chunk_index, content,"
                    " embedding) VALUES (:org, :doc, 0, 'Refunds are issued.',"
                    f" '{_VECTOR}'::vector)"
                ),
                {"org": ids["org"], "doc": ids["doc"]},
            )
            await conn.execute(
                text(
                    "INSERT INTO ingestion_failures (org_id, document_id, stage, reason,"
                    " attempts) VALUES (:org, :doc, 'embedding', 'a chunk failed', 1)"
                ),
                {"org": ids["org"], "doc": ids["doc"]},
            )
        finally:
            for table in ORG_SCOPED_TABLES:
                await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
    await engine.dispose()
    return ids


async def _cleanup(url: str, *org_ids: uuid.UUID) -> None:
    engine = _engine(url)
    async with engine.begin() as conn:
        for table in ORG_SCOPED_TABLES:
            await conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))
        try:
            for org_id in org_ids:
                await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})
        finally:
            for table in ORG_SCOPED_TABLES:
                await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
    await engine.dispose()


async def _counts(url: str, ids: dict[str, uuid.UUID]) -> dict[str, int]:
    """Count what survives, as the table owner so nothing is hidden by RLS."""
    engine = _engine(url)
    async with engine.begin() as conn:
        for table in ORG_SCOPED_TABLES:
            await conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))
        try:
            result = {}
            for name, table, column in (
                ("documents", "documents", "id"),
                ("chunks", "document_chunks", "document_id"),
                ("failures", "ingestion_failures", "document_id"),
            ):
                count = (
                    await conn.execute(
                        text(f"SELECT count(*) FROM {table} WHERE {column} = :doc"),  # noqa: S608
                        {"doc": ids["doc"]},
                    )
                ).scalar_one()
                result[name] = count
        finally:
            for table in ORG_SCOPED_TABLES:
                await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
    await engine.dispose()
    return result


async def _delete_as(org_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID) -> int:
    """Run the endpoint's delete through a tenant session, returning rows removed."""
    from sqlalchemy import delete as sql_delete

    from app.db.models import Document, DocumentChunk
    from app.db.session import dispose_engine
    from app.security.rls import tenant_session

    try:
        async with tenant_session(org_id=org_id, user_id=user_id, role="owner") as session:
            await session.execute(
                sql_delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            return (
                await session.execute(sql_delete(Document).where(Document.id == document_id))
            ).rowcount
    finally:
        # `tenant_session` uses the module-level engine, which binds to whichever loop
        # first touched it. Each asyncio.run here closes its loop on the way out, so the
        # engine has to go with it or the next call finds dead connections. Production
        # keeps one loop for the process's life and never hits this.
        await dispose_engine()


def test_delete_removes_the_row_its_chunks_and_its_failures(delete_env: str) -> None:
    try:
        ids = asyncio.run(_seed_document(delete_env, org_slug=f"del-{uuid.uuid4().hex[:8]}"))
    except (OSError, SQLAlchemyError) as exc:
        pytest.skip(f"test database unreachable: {exc}")

    try:
        before = asyncio.run(_counts(delete_env, ids))
        assert before == {"documents": 1, "chunks": 1, "failures": 1}

        deleted = asyncio.run(_delete_as(ids["org"], ids["user"], ids["doc"]))
        assert deleted == 1

        after = asyncio.run(_counts(delete_env, ids))
        # Chunks and failures go by cascade — the vectors must not outlive the document.
        assert after == {"documents": 0, "chunks": 0, "failures": 0}
    finally:
        asyncio.run(_cleanup(delete_env, ids["org"]))


def test_delete_removes_the_stored_file(delete_env: str) -> None:
    """The bytes on disk are the fourth resource, and the one a cascade cannot reach."""
    from app.api.documents import _unlink_stored_file
    from app.security.uploads import storage_path_for

    storage_key = uuid.uuid4()
    path = storage_path_for(storage_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 stored bytes")
    assert path.exists()

    _unlink_stored_file(storage_key, uuid.uuid4())

    assert not path.exists()


def test_unlinking_an_already_missing_file_is_not_an_error(delete_env: str) -> None:
    """A retried delete must not fail on the file it already removed."""
    from app.api.documents import _unlink_stored_file

    _unlink_stored_file(uuid.uuid4(), uuid.uuid4())


def test_one_org_cannot_delete_another_orgs_document(delete_env: str) -> None:
    """The isolation guarantee: RLS makes a cross-org delete affect zero rows."""
    try:
        owner = asyncio.run(_seed_document(delete_env, org_slug=f"own-{uuid.uuid4().hex[:8]}"))
        attacker = asyncio.run(_seed_document(delete_env, org_slug=f"atk-{uuid.uuid4().hex[:8]}"))
    except (OSError, SQLAlchemyError) as exc:
        pytest.skip(f"test database unreachable: {exc}")

    try:
        # The attacker knows the victim's document id and asks for it directly.
        deleted = asyncio.run(_delete_as(attacker["org"], attacker["user"], owner["doc"]))
        assert deleted == 0

        survived = asyncio.run(_counts(delete_env, owner))
        assert survived == {"documents": 1, "chunks": 1, "failures": 1}
    finally:
        asyncio.run(_cleanup(delete_env, owner["org"], attacker["org"]))


def test_one_org_cannot_see_another_orgs_chunks(delete_env: str) -> None:
    """Isolation of the vectors themselves, not just of the parent document."""
    from sqlalchemy import func, select

    from app.db.models import DocumentChunk
    from app.security.rls import tenant_session

    try:
        owner = asyncio.run(_seed_document(delete_env, org_slug=f"own-{uuid.uuid4().hex[:8]}"))
        attacker = asyncio.run(_seed_document(delete_env, org_slug=f"atk-{uuid.uuid4().hex[:8]}"))
    except (OSError, SQLAlchemyError) as exc:
        pytest.skip(f"test database unreachable: {exc}")

    async def visible_chunks(org_id: uuid.UUID, user_id: uuid.UUID) -> int:
        from app.db.session import dispose_engine

        try:
            async with tenant_session(org_id=org_id, user_id=user_id, role="owner") as session:
                return (
                    await session.execute(
                        select(func.count())
                        .select_from(DocumentChunk)
                        .where(DocumentChunk.document_id == owner["doc"])
                    )
                ).scalar_one()
        finally:
            # See _delete_as: the shared engine cannot outlive this asyncio.run's loop.
            await dispose_engine()

    try:
        assert asyncio.run(visible_chunks(owner["org"], owner["user"])) == 1
        assert asyncio.run(visible_chunks(attacker["org"], attacker["user"])) == 0
    finally:
        asyncio.run(_cleanup(delete_env, owner["org"], attacker["org"]))
