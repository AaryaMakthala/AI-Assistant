"""Document ingestion: extract → chunk → embed → store.

The task owns the document's status for its whole life. Every exit path writes a terminal
status, because a document stuck in `processing` forever is indistinguishable in the UI
from one that is merely slow, and that ambiguity is what makes ingestion failures
invisible.

Ingestion is idempotent: it deletes the document's existing chunks before inserting, so a
task redelivered after a worker crash produces one clean set rather than duplicates.

Failure is graded rather than binary (Phase 10). A document that cannot be read at all
fails outright, but one where a few chunks fail to embed is stored without them and marked
`ready` with a reduced `chunk_count` — losing a whole manual because one page tripped the
model is the pipeline choosing nothing over almost everything. When retries are exhausted
the reason is also written to `ingestion_failures`, which is the operator's copy of an
event the user only sees one line of.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import UTC, datetime
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
from app.db.models import Document, DocumentChunk, IngestionFailure
from app.observability import capture_exception
from app.observability.sentry import set_tag
from app.rag.chunking import chunk_pages
from app.rag.embeddings import embed_passages_resilient
from app.rag.extraction import ExtractionError, count_words, extract_pages
from app.security.rls import set_tenant_claims, use_tenant_role
from app.security.uploads import ALLOWED_TYPES, storage_path_for
from app.workers.celery_app import celery_app

#: mime type -> extension, so the extractor is chosen from validated stored metadata
#: rather than from a filename we no longer trust or even keep on disk.
_EXTENSION_BY_MIME = {allowed.mime_type: allowed.extension for allowed in ALLOWED_TYPES.values()}

#: Longest a failure reason is stored. An exception string is attacker-influenced text
#: (it can quote document content), so it is bounded before it reaches the database.
_MAX_REASON_CHARS = 2000

#: Identifies this worker to the document policies added in migration 0007. Those policies
#: scope reads and writes by uploader, and ingestion has no user to be — without this claim
#: a personal document would be invisible to the very task meant to process it, so its
#: status writes would affect zero rows and it would sit `pending` while the reaper
#: re-queued it forever. Not forgeable from a token: see app/security/rls.py.
_INGESTION_SVC = "ingestion"


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
    **fields: Any,
) -> None:
    """Write a status and whatever progress columns came with it.

    `error_message` is always written, including as None on success, so a document that
    succeeds on retry does not keep the previous attempt's error next to a `ready` status.
    """
    values: dict[str, Any] = {"status": status, "error_message": error_message, **fields}
    await session.execute(
        update(Document).where(Document.id == document_id, Document.org_id == org_id).values(values)
    )


async def _record_failure(
    session: AsyncSession,
    *,
    document_id: uuid.UUID,
    org_id: uuid.UUID,
    stage: str,
    reason: str,
    attempts: int,
) -> None:
    """Append a dead-letter row. Never raises: a failed log must not mask the failure.

    Wrapped in a savepoint rather than a bare try/except. In Postgres a failed statement
    aborts the entire transaction, so swallowing the exception without one would leave the
    session poisoned and the *status update* — the part the user actually sees — would be
    lost at commit. The savepoint confines the damage to this insert.
    """
    try:
        async with session.begin_nested():
            await session.execute(
                insert(IngestionFailure).values(
                    id=uuid.uuid4(),
                    org_id=org_id,
                    document_id=document_id,
                    stage=stage,
                    reason=reason[:_MAX_REASON_CHARS],
                    attempts=attempts,
                )
            )
    except Exception as exc:
        logger.opt(exception=exc).error(
            "Could not write dead-letter row for document {doc}", doc=document_id
        )


async def _ingest(document_id: uuid.UUID, org_id: uuid.UUID, *, attempt: int = 0) -> dict[str, Any]:
    engine, session = await _open_session()
    try:
        async with session:
            await session.begin()
            # Claims first: every statement below is then filtered by RLS, so a wrong
            # org_id here yields zero rows rather than another tenant's document.
            await set_tenant_claims(session, org_id=org_id, svc=_INGESTION_SVC)
            await use_tenant_role(session)

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
            await _set_status(
                session,
                document_id=document_id,
                org_id=org_id,
                status="processing",
                processing_started_at=datetime.now(UTC),
                processing_completed_at=None,
                retry_count=attempt,
            )
            await session.commit()

            extension = _EXTENSION_BY_MIME.get(mime_type)
            path = storage_path_for(storage_key)

            await session.begin()
            await set_tenant_claims(session, org_id=org_id, svc=_INGESTION_SVC)
            await use_tenant_role(session)
            stage = "extraction"
            try:
                if extension is None:
                    raise ExtractionError(f"Unsupported stored media type '{mime_type}'.")
                pages = extract_pages(Path(path), extension=extension)
                word_count = sum(count_words(page.text) for page in pages)

                stage = "chunking"
                chunks = chunk_pages(pages)
                if not chunks:
                    raise ExtractionError("Document produced no indexable text.")
            except ExtractionError as exc:
                # A document that cannot be read is a permanent failure, not a transient
                # one: retrying a corrupt PDF produces the same corrupt PDF. It is marked
                # terminal here rather than raised, so the task does not retry it.
                logger.info("Ingestion rejected {file}: {reason}", file=filename, reason=str(exc))
                await _set_status(
                    session,
                    document_id=document_id,
                    org_id=org_id,
                    status="failed",
                    error_message=str(exc),
                    processing_completed_at=datetime.now(UTC),
                )
                await _record_failure(
                    session,
                    document_id=document_id,
                    org_id=org_id,
                    stage=stage,
                    reason=str(exc),
                    attempts=attempt + 1,
                )
                await session.commit()
                return {"status": "failed", "document_id": str(document_id), "reason": str(exc)}

            settings = get_settings()
            embedded = embed_passages_resilient(
                [chunk.content for chunk in chunks],
                batch_size=settings.embedding_batch_size,
                max_attempts=settings.embedding_max_attempts,
            )
            if embedded.failed:
                logger.warning(
                    "Storing {file} without {n} of {total} chunks: they could not be embedded",
                    file=filename,
                    n=embedded.failed,
                    total=len(chunks),
                )
                await _record_failure(
                    session,
                    document_id=document_id,
                    org_id=org_id,
                    stage="embedding",
                    reason=(
                        f"{embedded.failed} of {len(chunks)} chunks could not be embedded and "
                        f"were not indexed (chunk indices: {embedded.failed_indices[:20]})."
                    ),
                    attempts=attempt + 1,
                )

            stored = [
                (chunk, vector)
                for chunk, vector in zip(chunks, embedded.vectors, strict=True)
                if vector is not None
            ]
            if not stored:
                # Nothing embedded at all. This is a real failure rather than a partial
                # success, and it is worth retrying: it usually means the model could not
                # load, which is transient in a way a corrupt file is not.
                raise RuntimeError("No chunk in this document could be embedded.")

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
                        "chunk_metadata": {
                            "source": filename,
                            "locator": chunk.label,
                            "page": chunk.page,
                            "chunk_index": chunk.index,
                            "org_id": str(org_id),
                            "document_id": str(document_id),
                        },
                    }
                    for chunk, vector in stored
                ],
            )
            page_count = len({page.page for page in pages if page.page is not None}) or None
            await _set_status(
                session,
                document_id=document_id,
                org_id=org_id,
                status="ready",
                page_count=page_count,
                chunk_count=len(stored),
                word_count=word_count,
                processing_completed_at=datetime.now(UTC),
            )
            await session.commit()

            logger.info(
                "Ingested {file}: {n} chunks for org {org}",
                file=filename,
                n=len(stored),
                org=org_id,
            )
            return {
                "status": "ready",
                "document_id": str(document_id),
                "chunks": len(stored),
                "skipped_chunks": embedded.failed,
                "word_count": word_count,
            }
    finally:
        await engine.dispose()


async def _mark_failed(
    document_id: uuid.UUID,
    org_id: uuid.UUID,
    message: str,
    *,
    stage: str = "unknown",
    attempts: int = 0,
    detail: str | None = None,
) -> None:
    engine, session = await _open_session()
    try:
        async with session:
            await session.begin()
            await set_tenant_claims(session, org_id=org_id, svc=_INGESTION_SVC)
            await use_tenant_role(session)
            await _set_status(
                session,
                document_id=document_id,
                org_id=org_id,
                status="failed",
                error_message=message,
                processing_completed_at=datetime.now(UTC),
                retry_count=attempts,
            )
            await _record_failure(
                session,
                document_id=document_id,
                org_id=org_id,
                stage=stage,
                reason=detail or message,
                attempts=attempts,
            )
            await session.commit()
    finally:
        await engine.dispose()


@celery_app.task(bind=True, name="app.workers.ingestion.ingest_document", max_retries=2)
def ingest_document(self: Any, document_id: str, org_id: str) -> dict[str, Any]:
    """Ingest one uploaded document. Identifiers are passed as strings for JSON transport."""
    doc_uuid = uuid.UUID(document_id)
    org_uuid = uuid.UUID(org_id)
    attempt = self.request.retries
    # Which tenant and which document, as opaque ids, so an ingestion error in Sentry can be
    # scoped to one organization without the filename or any content travelling with it.
    set_tag("org_id", org_id)
    set_tag("document_id", document_id)
    try:
        return asyncio.run(_ingest(doc_uuid, org_uuid, attempt=attempt))
    except SoftTimeLimitExceeded:
        # The hard limit is two minutes out, which is enough to record the outcome.
        asyncio.run(
            _mark_failed(
                doc_uuid,
                org_uuid,
                "Processing timed out.",
                stage="unknown",
                attempts=attempt + 1,
                detail="The task exceeded its soft time limit and was stopped.",
            )
        )
        raise
    except Exception as exc:
        logger.opt(exception=exc).error("Ingestion crashed for document {doc}", doc=document_id)
        # Reported here rather than left to Celery's own integration, which only sees a task
        # that *finished* failing. An attempt that ends in `self.retry` raises `Retry` — a
        # normal outcome to Celery — so a document that fails twice and then succeeds would
        # otherwise leave no record of the two failures. Sentry's dedupe drops the duplicate
        # on the final attempt, so this is one event per failure either way.
        capture_exception(exc, subsystem="ingestion", attempt=str(attempt))
        if attempt >= self.max_retries:
            # Final attempt: leave a terminal status so the UI stops showing a spinner, and
            # a dead-letter row so the reason survives past this worker's stdout.
            asyncio.run(
                _mark_failed(
                    doc_uuid,
                    org_uuid,
                    "Processing failed after repeated attempts.",
                    stage="unknown",
                    attempts=attempt + 1,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        # Jitter: a broker outage queues every document at once, and without it every
        # retry would land in the same instant and knock the recovering dependency over
        # again. Randomizing the tail spreads them out. Not a security decision, so the
        # standard generator is the right one.
        backoff = 30 * (2**attempt)
        countdown = backoff + random.uniform(0, backoff / 2)  # noqa: S311
        raise self.retry(exc=exc, countdown=countdown) from exc


__all__ = ["ingest_document"]
