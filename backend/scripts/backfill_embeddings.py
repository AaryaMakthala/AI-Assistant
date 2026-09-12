"""Re-embed every chunk with the configured hosted provider after a model change.

Migration 0017 (384->768) and 0018 (768->1024) null every chunk embedding (the
dimension is part of the pgvector type and old embeddings cannot be reused after
the model swap), so this script is the bridge that restores searchability.  It
must run after the migration; until it does, every retrieval returns 0.0
similarity and the assistant refuses honestly rather than answering ungrounded.

Locked decisions implemented here:

* Re-embeds only chunks with ``embedding IS NULL``, so re-runs resume where a
  previous run stopped (idempotent; the migration guarantees the null state).
* Uses the same resilient path as ingestion (``embed_passages_resilient``): a
  poisonous chunk is dropped, never escalates to failing the whole workspace.
* Verifies every returned vector's dimension against the configured model before
  writing — a wrong-dimension vector would silently poison the index (CLAUDE.md
  Risk 1), so a mismatch aborts rather than corrupting the corpus.
* Dropped chunks stay NULL and are counted, so the operator can see partial state.
* After a successful run (zero chunks still NULL, zero dimension mismatches) the
  script seals the database invariant itself: it restores ``embedding NOT NULL``.
  The constraint is never restored by a migration, because the migration must
  stay nullable until re-embedding happens and any follow-up migration would run
  before backfill on ``upgrade head``.  Idempotent: re-running after success is
  a no-op (nothing pending).
* Requires a BYPASSRLS role — updating every chunk in the table would otherwise be
  denied by RLS for a non-owner writer (same prerequisite as migration 0008).

Usage (from ``backend/``):

    python scripts/backfill_embeddings.py [--database-url URL] [--batch-size N]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field

from loguru import logger
from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings
from app.rag.embeddings import embed_passages_resilient

_BYPASS_CHECK = text(
    "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
)
_PENDING_CHUNKS = text(
    "SELECT id, document_id, content FROM document_chunks "
    "WHERE embedding IS NULL ORDER BY created_at, id"
)
_NULL_CHUNKS = text(
    "SELECT count(*) FROM document_chunks WHERE embedding IS NULL"
)
_DIM_MISMATCH = text(
    "SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL "
    "AND cardinality(embedding::real[]) <> :dim"
)
_RESTORE_NOT_NULL = text(
    "ALTER TABLE document_chunks ALTER COLUMN embedding SET NOT NULL"
)
# Vector() bind type makes asyncpg receive the pgvector text form; the explicit
# CAST keeps the current column dimension unambiguous.
_UPDATE_EMBEDDING = (
    text(
        "UPDATE document_chunks SET embedding = CAST(:vector AS vector) WHERE id = :id"
    ).bindparams(bindparam("vector", type_=Vector))
)

_SEAL_LOG = "backfill sealed: NOT NULL restored on document_chunks.embedding"
_SEAL_SKIP_LOG = (
    "backfill NOT sealed: {n} chunks lack a valid {dim}-dim embedding — "
    "column stays nullable"
)


def _normalize_async_database_url(url: str) -> str:
    """Map bare PostgreSQL URLs onto the async driver SQLAlchemy's async engine needs.

    ``create_async_engine`` selects its dialect from the URL scheme; bare
    ``postgresql://``/``postgres://`` resolve to the sync psycopg2 driver, which this
    project does not install (it uses asyncpg). URLs that already name an explicit
    driver (e.g. ``postgresql+asyncpg://``) are left unchanged.
    """
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    return url


@dataclass
class BackfillReport:
    """What one run did, for the operator."""

    total: int = 0
    reembedded: int = 0
    dropped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def problems(self) -> int:
        return self.dropped + len(self.errors)

    def summary(self) -> str:
        return (
            f"backfill: {self.reembedded}/{self.total} chunks re-embedded, "
            f"{self.dropped} dropped, {len(self.errors)} errors"
        )


async def _require_bypass_role(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        bypass = (await conn.execute(_BYPASS_CHECK)).scalar()
    if bypass is not True:
        raise RuntimeError(
            "backfill_embeddings requires a BYPASSRLS role (superuser). "
            "Run it as the same role that applied migration 0017."
        )


async def backfill_embeddings(
    *, database_url: str | None = None, batch_size: int = 32, delay_seconds: float = 0.0
) -> BackfillReport:
    report = BackfillReport()
    engine = create_async_engine(
        _normalize_async_database_url(database_url or str(get_settings().database_url))
    )
    try:
        await _require_bypass_role(engine)

        async with engine.connect() as conn:
            rows = list((await conn.execute(_PENDING_CHUNKS)).mappings())
        report.total = len(rows)
        logger.info("backfill starting: {n} embeddings pending", n=report.total)
        if not rows:
            logger.info("backfill: nothing to do")

        documents: dict = {}
        for row in rows:
            documents.setdefault(row["document_id"], []).append(row)

        for index, (_document_id, doc_rows) in enumerate(documents.items(), start=1):
            chunk_ids = [row["id"] for row in doc_rows]
            contents = [row["content"] for row in doc_rows]

            # Pace the provider between documents: Voyage free-tier accounts (no
            # payment method) run at 3 RPM, so backfills against them need an
            # explicit floor between requests or the reduced rate limit aborts
            # the whole run (CLAUDE.md Risk 3).
            if delay_seconds > 0 and index > 1:
                await asyncio.sleep(delay_seconds)

            # Same resilient path ingestion uses: failures are per-chunk, logged,
            # and the rest of the document still gets embedded.
            result = embed_passages_resilient(contents, batch_size=batch_size)
            report.reembedded += sum(
                1 for vector in result.vectors if vector is not None
            )
            report.dropped += result.failed
            for chunk_id, failed_index in zip(
                [chunk_ids[i] for i in result.failed_indices], result.failed_indices,
                strict=True,
            ):
                report.errors.append(f"chunk {chunk_id} (index {failed_index})")

            # Verify dimensions before writing anything: a provider returning the
            # wrong dimension would silently poison the index (CLAUDE.md Risk 1),
            # so abort the run rather than persist a corrupt vector.
            expected_dim = get_settings().embedding_dim
            for chunk_id, vector in zip(chunk_ids, result.vectors, strict=True):
                if vector is not None and len(vector) != expected_dim:
                    raise RuntimeError(
                        f"chunk {chunk_id}: provider returned {len(vector)}-dim "
                        f"vector, expected {expected_dim} — aborting"
                    )

            updates = [
                (chunk_id, vector)
                for chunk_id, vector in zip(chunk_ids, result.vectors, strict=True)
                if vector is not None
            ]
            async with engine.begin() as conn:
                for chunk_id, vector in updates:
                    await conn.execute(
                        _UPDATE_EMBEDDING, {"id": chunk_id, "vector": vector}
                    )
            logger.info(
                "backfill progress: {done}/{n} documents, {chunks} chunks re-embedded",
                done=index,
                n=len(documents),
                chunks=report.reembedded,
            )

        # Seal the invariant only now: every chunk must have a valid, correctly
        # dimensioned vector (or none must remain to fill). Restoring NOT NULL
        # while rows are still NULL or wrong-dimensioned would fail the ALTER or
        # poison the corpus; skipping the seal keeps retrieval safe (NULL rows
        # score 0.0 and can never ground an answer).
        expected_dim = get_settings().embedding_dim
        async with engine.connect() as conn:
            remaining = (await conn.execute(_NULL_CHUNKS)).scalar()
            mismatched = (await conn.execute(_DIM_MISMATCH, {"dim": expected_dim})).scalar()
        if remaining == 0 and mismatched == 0:
            async with engine.begin() as conn:
                await conn.execute(_RESTORE_NOT_NULL)
            logger.info(_SEAL_LOG)
        else:
            logger.warning(_SEAL_SKIP_LOG, n=(remaining or 0) + (mismatched or 0), dim=expected_dim)

        logger.info("backfill complete: {summary}", summary=report.summary())
        return report
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-embed all chunks with the configured hosted provider "
        "after a dimension migration. Run after 0018; idempotent."
    )
    parser.add_argument(
        "--database-url", default=None,
        help="Override the configured DATABASE_URL.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Chunks per embedding batch (provider caps at 128).",
    )
    parser.add_argument(
        "--delay-seconds", type=float, default=0.0,
        help="Sleep between documents (Voyage free tier: 3 RPM, so use >= 22s "
        "until a payment method is on the account).",
    )
    args = parser.parse_args()

    try:
        report = asyncio.run(
            backfill_embeddings(
                database_url=args.database_url,
                batch_size=args.batch_size,
                delay_seconds=args.delay_seconds,
            )
        )
    except RuntimeError as exc:
        logger.error("{message}", message=exc)
        sys.exit(1)
    if report.problems:
        logger.warning(
            "backfill finished with problems — {problems} chunks need attention: "
            "{dropped} dropped, {errors} errors.",
            problems=report.problems,
            dropped=report.dropped,
            errors=len(report.errors),
        )


if __name__ == "__main__":
    main()