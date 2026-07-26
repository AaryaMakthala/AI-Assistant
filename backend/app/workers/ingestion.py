"""Document ingestion: extract → chunk → embed → store.

The task owns the document's status for its whole life. Every exit path writes a terminal
status, because a document stuck in `processing` forever is indistinguishable in the UI
from one that is merely slow, and that ambiguity is what makes ingestion failures
invisible.

Ingestion is idempotent: it deletes the document's existing chunks before inserting, so a
task redelivered after a worker crash produces one clean set rather than duplicates.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.db.models import Document, DocumentChunk
from app.rag.chunking import chunk_pages
from app.rag.embeddings import embed_passages
from app.rag.extraction import ExtractionError, extract_pages
from app.security.rls import set_tenant_claims
from app.security.uploads import ALLOWED_TYPES, storage_path_for
from app.workers.celery_app import celery_app

#: mime type -> extension, so the extractor is chosen from validated stored metadata
#: rather than from a filename we no longer trust or even keep on disk.
_EXTENSION_BY_MIME = {allowed.mime_type: allowed.extension for allowed in ALLOWED_TYPES.values()}


async def _open_session() -> tuple[AsyncEngine, AsyncSession]:
    """An engine plus a session bound to it, both owned by the caller.

    A fresh engine per task: the module-level engine in app.db.session would be bound to
    a loop that `asyncio.run` closes on the way out, and every later task in the same
    prefork worker would fail on a dead connection. statement_cache_size=0 keeps this
    working through a transaction-mode pooler, which cannot hold prepared statements.
    """
    engine = create_async_engine(
        str(get_settings().database_url),
        pool_pre_ping=True,
        connect_args={"statement_cache_size": 0},
    )
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return engine, factory()


async def _set_status(
    session: AsyncSession,
    *,
    document_id: uuid.UUID,
    org_id: uuid.UUID,
    status: str,
    error_message: str | None = None,
    page_count: int | None = None,
) -> None:
    values: dict[str, Any] = {"status": status, "error_message": error_message}
    if page_count is not None:
        values["page_count"] = page_count
    await session.execute(
        update(Document).where(Document.id == document_id, Document.org_id == org_id).values(values)
    )


async def _ingest(document_id: uuid.UUID, org_id: uuid.UUID) -> dict[str, Any]:
    engine, session = await _open_session()
    try:
        async with session:
            await session.begin()
            # Claims first: every statement below is then filtered by RLS, so a wrong
            # org_id here yields zero rows rather than another tenant's document.
            await set_tenant_claims(session, org_id=org_id)

            row = (
                await session.execute(
                    select(Document.storage_key, Document.mime_type, Document.filename).where(
                        Document.id == document_id, Document.org_id == org_id
                    )
                )
            ).first()
            if row is None:
                logger.warning(
                    "Ingestion skipped: document {doc} not visible to org {org}",
                    doc=document_id,
                    org=org_id,
                )
                return {"status": "missing", "document_id": str(document_id)}

            storage_key, mime_type, filename = row
            await _set_status(session, document_id=document_id, org_id=org_id, status="processing")
            await session.commit()

            extension = _EXTENSION_BY_MIME.get(mime_type)
            path = storage_path_for(storage_key)

            await session.begin()
            await set_tenant_claims(session, org_id=org_id)
            try:
                if extension is None:
                    raise ExtractionError(f"Unsupported stored media type '{mime_type}'.")
                pages = extract_pages(Path(path), extension=extension)
                chunks = chunk_pages(pages)
                if not chunks:
                    raise ExtractionError("Document produced no indexable text.")
                vectors = embed_passages([chunk.content for chunk in chunks])
            except ExtractionError as exc:
                logger.info("Ingestion rejected {file}: {reason}", file=filename, reason=str(exc))
                await _set_status(
                    session,
                    document_id=document_id,
                    org_id=org_id,
                    status="failed",
                    error_message=str(exc),
                )
                await session.commit()
                return {"status": "failed", "document_id": str(document_id), "reason": str(exc)}

            # Idempotency: a redelivered task replaces its chunks instead of duplicating.
            await session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == document_id, DocumentChunk.org_id == org_id
                )
            )
            await session.execute(
                insert(DocumentChunk),
                [
                    {
                        "org_id": org_id,
                        "document_id": document_id,
                        "chunk_index": chunk.index,
                        "content": chunk.content,
                        "page": chunk.page,
                        "token_count": len(chunk.content.split()),
                        "embedding": vector,
                        "chunk_metadata": {"source": filename, "locator": chunk.label},
                    }
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            page_count = len({page.page for page in pages if page.page is not None}) or None
            await _set_status(
                session,
                document_id=document_id,
                org_id=org_id,
                status="ready",
                page_count=page_count,
            )
            await session.commit()

            logger.info(
                "Ingested {file}: {n} chunks for org {org}",
                file=filename,
                n=len(chunks),
                org=org_id,
            )
            return {
                "status": "ready",
                "document_id": str(document_id),
                "chunks": len(chunks),
            }
    finally:
        await engine.dispose()


async def _mark_failed(document_id: uuid.UUID, org_id: uuid.UUID, message: str) -> None:
    engine, session = await _open_session()
    try:
        async with session:
            await session.begin()
            await set_tenant_claims(session, org_id=org_id)
            await _set_status(
                session,
                document_id=document_id,
                org_id=org_id,
                status="failed",
                error_message=message,
            )
            await session.commit()
    finally:
        await engine.dispose()


@celery_app.task(bind=True, name="app.workers.ingestion.ingest_document", max_retries=2)
def ingest_document(self: Any, document_id: str, org_id: str) -> dict[str, Any]:
    """Ingest one uploaded document. Identifiers are passed as strings for JSON transport."""
    doc_uuid = uuid.UUID(document_id)
    org_uuid = uuid.UUID(org_id)
    try:
        return asyncio.run(_ingest(doc_uuid, org_uuid))
    except SoftTimeLimitExceeded:
        # The hard limit is two minutes out, which is enough to record the outcome.
        asyncio.run(_mark_failed(doc_uuid, org_uuid, "Processing timed out."))
        raise
    except Exception as exc:
        logger.opt(exception=exc).error("Ingestion crashed for document {doc}", doc=document_id)
        if self.request.retries >= self.max_retries:
            # Final attempt: leave a terminal status so the UI stops showing a spinner.
            asyncio.run(
                _mark_failed(doc_uuid, org_uuid, "Processing failed after repeated attempts.")
            )
            raise
        raise self.retry(exc=exc, countdown=30 * (2**self.request.retries)) from exc


__all__ = ["ingest_document"]
