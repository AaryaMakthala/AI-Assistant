"""Scheduled maintenance for the ingestion pipeline.

Retries cover a task that *fails*. Nothing covers a task that simply stops existing — a
worker killed by an OOM killer, a container evicted mid-run, a broker that lost the
message. In all three the document row is left saying `processing` with no worker behind
it, and because `task_acks_late` only redelivers while a connection is alive, nothing ever
comes back to correct it. The UI then spins forever on a job that will never finish.

The reaper is the backstop: anything that has been `processing` longer than any legitimate
run could take is marked failed, with a dead-letter row saying so. It is deliberately
conservative — the cutoff is derived from the task's own hard time limit, so a slow but
live job is never killed by it.

`pending` rows are handled too, but differently: a document queued while the broker was
down was never picked up by anyone, so it is re-queued rather than failed.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Document
from app.workers.celery_app import celery_app
from app.workers.ingestion import _open_session, _record_failure

#: How far past a task's hard time limit a job must be before it is presumed dead. The
#: margin covers queue wait and a worker that started late — without it, a document that
#: sat in the queue during a burst would be reaped while its worker was still starting.
_ABANDON_MARGIN = timedelta(minutes=5)

#: Documents re-queued or failed in one pass. A bound rather than a full-table sweep: if
#: something has gone systemically wrong there may be thousands, and a single transaction
#: holding all of them is its own outage.
_BATCH_LIMIT = 100


def _abandoned_before() -> datetime:
    """The timestamp before which a `processing` document is presumed dead."""
    limit = timedelta(seconds=celery_app.conf.task_time_limit or 720)
    return datetime.now(UTC) - limit - _ABANDON_MARGIN


async def _reap(session: AsyncSession) -> dict[str, int]:
    """Fail abandoned jobs and re-queue ones that were never picked up.

    **On RLS.** This is the one path in the codebase that deliberately does not go through
    `tenant_session`, because maintenance spans every organization by definition and there
    is no principal to scope it to. It therefore neither sets claims nor drops to
    `app_tenant`, and relies on the application's login role holding BYPASSRLS — which is
    what Supabase's `postgres` role has, and what `app/security/rls.py` documents as the
    reason the tenant path must switch roles at all.

    That makes the safety argument structural rather than incidental, and it rests on three
    things: this task never returns document *content*, only ids and statuses; it is
    reachable only from the beat schedule, never from a request handler; and the only
    columns it writes are status and progress. If the login role is ever changed to one
    without BYPASSRLS, this sweep silently finds zero rows rather than misbehaving — it
    fails closed, which is the right direction to fail in.
    """
    cutoff = _abandoned_before()

    abandoned = (
        await session.execute(
            select(Document.id, Document.org_id, Document.retry_count)
            .where(
                Document.status == "processing",
                Document.processing_started_at.is_not(None),
                Document.processing_started_at < cutoff,
            )
            .limit(_BATCH_LIMIT)
        )
    ).all()

    for document_id, org_id, retry_count in abandoned:
        logger.warning(
            "Reaping document {doc}: processing since before {cutoff} with no worker",
            doc=document_id,
            cutoff=cutoff,
        )
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status="failed",
                error_message="Processing stopped unexpectedly. Try uploading the file again.",
                processing_completed_at=datetime.now(UTC),
            )
        )
        await _record_failure(
            session,
            document_id=document_id,
            org_id=org_id,
            stage="abandoned",
            reason=(
                "No worker reported an outcome before the abandonment cutoff; the task was "
                "presumed lost (worker killed, evicted, or the message was never redelivered)."
            ),
            attempts=(retry_count or 0) + 1,
        )

    # A document still `pending` well past its enqueue was never claimed — the usual cause
    # is the broker having been down at upload (see the enqueue failure path in
    # app/api/documents.py). Re-queued rather than failed: nothing has been attempted yet.
    stale_pending = (
        await session.execute(
            select(Document.id, Document.org_id)
            .where(
                Document.status == "pending",
                Document.created_at < datetime.now(UTC) - _ABANDON_MARGIN,
            )
            .limit(_BATCH_LIMIT)
        )
    ).all()

    requeued = 0
    for document_id, org_id in stale_pending:
        if _requeue(document_id, org_id):
            requeued += 1

    return {"failed": len(abandoned), "requeued": requeued}


def _requeue(document_id: uuid.UUID, org_id: uuid.UUID) -> bool:
    """Re-enqueue one document, reporting whether the broker accepted it."""
    from app.workers.ingestion import ingest_document

    try:
        ingest_document.delay(str(document_id), str(org_id))
    except Exception as exc:
        # The broker is still down. Left `pending` so the next pass tries again — this is
        # the one state it is correct to leave a document in indefinitely, because no work
        # has been lost, only delayed.
        logger.opt(exception=exc).warning(
            "Could not re-queue document {doc}; leaving it pending", doc=document_id
        )
        return False
    logger.info("Re-queued document {doc} that was never picked up", doc=document_id)
    return True


async def _run() -> dict[str, int]:
    engine, session = await _open_session()
    try:
        async with session:
            await session.begin()
            result = await _reap(session)
            await session.commit()
            return result
    finally:
        await engine.dispose()


@celery_app.task(name="app.workers.maintenance.reap_stuck_documents")
def reap_stuck_documents() -> dict[str, Any]:
    """Beat-scheduled sweep for ingestion jobs that will never report back."""
    if not get_settings().ingestion_reaper_enabled:
        return {"failed": 0, "requeued": 0, "skipped": True}
    return dict(asyncio.run(_run()))


__all__ = ["reap_stuck_documents"]
