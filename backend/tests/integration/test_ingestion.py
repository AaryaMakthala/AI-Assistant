"""Phase 3 acceptance: a sample PDF becomes queryable chunks with correct metadata.

Runs the ingestion task's body directly rather than through Celery — the broker adds no
coverage here, and the task is written so its logic is independent of how it was queued.

Requires `TEST_DATABASE_URL` (a Postgres with migrations applied) and the embedding model,
which is downloaded on first use. Skips cleanly when either is unavailable; a skip means
this criterion is unproven, not that it passed.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.db.legacy_models import ORG_SCOPED_TABLES

pytestmark = pytest.mark.usefixtures("valid_env")

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL") or (
        dotenv_values(_REPO_ROOT / ".env").get("TEST_DATABASE_URL") or None
    )


@pytest.fixture
def ingestion_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the app at the test database and a scratch upload directory."""
    url = _database_url()
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set — cannot prove ingestion end to end")

    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


def _sample_pdf(path: Path) -> Path:
    import pymupdf

    document = pymupdf.open()
    for body in ("Refunds are issued within 30 days.", "Escalations go to the duty manager."):
        page = document.new_page()
        page.insert_text((72, 72), body)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()
    return path


async def _seed(url: str, storage_key: uuid.UUID) -> dict[str, uuid.UUID]:
    """Create an org, a user and a pending document row, bypassing RLS as table owner."""
    ids = {"org": uuid.uuid4(), "user": uuid.uuid4(), "doc": uuid.uuid4()}
    engine = create_async_engine(url, connect_args={"statement_cache_size": 0})
    async with engine.begin() as conn:
        for table in ORG_SCOPED_TABLES:
            await conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))
        try:
            await conn.execute(
                text("INSERT INTO organizations (id, name, slug) VALUES (:id, 'Ingest', :slug)"),
                {"id": ids["org"], "slug": f"ingest-{uuid.uuid4().hex[:8]}"},
            )
            await conn.execute(
                text("INSERT INTO users (id, org_id, email) VALUES (:id, :org, :email)"),
                {
                    "id": ids["user"],
                    "org": ids["org"],
                    "email": f"ingest-{uuid.uuid4().hex[:8]}@example.test",
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO documents (id, org_id, uploaded_by, filename, storage_key,"
                    " mime_type, size_bytes, status) VALUES (:id, :org, :user, 'policy.pdf',"
                    " :key, 'application/pdf', 1024, 'pending')"
                ),
                {"id": ids["doc"], "org": ids["org"], "user": ids["user"], "key": storage_key},
            )
        finally:
            for table in ORG_SCOPED_TABLES:
                await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
    await engine.dispose()
    return ids


async def _cleanup(url: str, org_id: uuid.UUID) -> None:
    engine = create_async_engine(url, connect_args={"statement_cache_size": 0})
    async with engine.begin() as conn:
        for table in ORG_SCOPED_TABLES:
            await conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))
        try:
            # Cascades to users, documents and chunks.
            await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})
        finally:
            for table in ORG_SCOPED_TABLES:
                await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
    await engine.dispose()


def test_pdf_upload_produces_queryable_chunks(ingestion_env: str, tmp_path: Path) -> None:
    from app.rag.embeddings import get_model
    from app.security.uploads import storage_path_for
    from app.workers.ingestion import _ingest

    try:
        get_model()
    except (OSError, SQLAlchemyError, RuntimeError) as exc:
        pytest.skip(f"embedding model unavailable: {exc}")

    storage_key = uuid.uuid4()
    _sample_pdf(storage_path_for(storage_key))

    try:
        ids = asyncio.run(_seed(ingestion_env, storage_key))
    except (OSError, SQLAlchemyError) as exc:
        pytest.skip(f"test database unreachable: {exc}")

    try:
        result = asyncio.run(_ingest(ids["doc"], ids["org"]))
        assert result["status"] == "ready"
        assert result["chunks"] > 0

        rows, document = asyncio.run(_read_back(ingestion_env, ids))

        assert document.status == "ready"
        assert document.page_count == 2
        # Phase 10 progress columns: recorded once, and consistent with what was stored.
        assert document.chunk_count == len(rows)
        assert document.word_count > 0
        assert document.processing_started_at is not None
        assert document.processing_completed_at is not None
        assert document.processing_completed_at >= document.processing_started_at
        assert rows, "no chunks were stored"
        # Every chunk carries the owning org — this is what RLS filters on.
        assert {row.org_id for row in rows} == {ids["org"]}
        # ...and the citation metadata the UI renders as a source chip.
        assert {row.page for row in rows} == {1, 2}
        assert all(row.chunk_metadata["source"] == "policy.pdf" for row in rows)
        assert all(row.chunk_metadata["locator"].startswith("page ") for row in rows)
        # Phase 10 requires each vector to carry its own provenance, so a record is
        # interpretable without joining back to the document it came from.
        for row in rows:
            assert row.chunk_metadata["org_id"] == str(ids["org"])
            assert row.chunk_metadata["document_id"] == str(ids["doc"])
            assert row.chunk_metadata["page"] == row.page
            assert isinstance(row.chunk_metadata["chunk_index"], int)
        assert any("Refunds are issued" in row.content for row in rows)
        assert all(len(row.embedding) == get_settings().embedding_dim for row in rows)
    finally:
        asyncio.run(_cleanup(ingestion_env, ids["org"]))


async def _read_back(url: str, ids: dict[str, uuid.UUID]):
    """Read the stored chunks back through RLS, as the application would.

    The embedding is selected through the ORM column rather than as raw SQL text. pgvector
    registers a SQLAlchemy type that decodes `vector` into a list of floats; a `text()`
    query bypasses it and hands back the literal string `'[0.1,0.2,...]'`, so `len()` would
    count characters instead of dimensions and the assertion would fail against perfectly
    valid data.
    """
    from app.db.legacy_models import DocumentChunk
    from app.security.rls import set_tenant_claims

    engine = create_async_engine(url, connect_args={"statement_cache_size": 0})
    async with engine.connect() as conn:
        await conn.begin()
        await set_tenant_claims(conn, org_id=ids["org"])  # type: ignore[arg-type]
        chunks = (
            await conn.execute(
                select(
                    DocumentChunk.org_id,
                    DocumentChunk.page,
                    DocumentChunk.content,
                    DocumentChunk.embedding,
                    DocumentChunk.chunk_metadata,
                )
                .where(DocumentChunk.document_id == ids["doc"])
                .order_by(DocumentChunk.chunk_index)
            )
        ).all()
        document = (
            await conn.execute(
                text(
                    "SELECT status, page_count, chunk_count, word_count,"
                    " processing_started_at, processing_completed_at"
                    " FROM documents WHERE id = :id"
                ),
                {"id": ids["doc"]},
            )
        ).one()
    await engine.dispose()
    return chunks, document


def test_ingestion_marks_an_unreadable_file_as_failed(ingestion_env: str) -> None:
    """A document that cannot be parsed must reach a terminal status, not hang in limbo."""
    from app.security.uploads import storage_path_for
    from app.workers.ingestion import _ingest

    storage_key = uuid.uuid4()
    path = storage_path_for(storage_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 truncated and corrupt")

    try:
        ids = asyncio.run(_seed(ingestion_env, storage_key))
    except (OSError, SQLAlchemyError) as exc:
        pytest.skip(f"test database unreachable: {exc}")

    try:
        result = asyncio.run(_ingest(ids["doc"], ids["org"]))
        assert result["status"] == "failed"
        assert result["reason"]
    finally:
        asyncio.run(_cleanup(ingestion_env, ids["org"]))
