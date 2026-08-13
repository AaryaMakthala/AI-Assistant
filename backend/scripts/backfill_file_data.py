"""Host-side Phase 1C byte backfill: ``storage_key`` -> ``file_data`` + ``checksum``.

Migration 0008 creates the canonical ``documents`` table with ``file_data``/``checksum``
nullable and retains ``storage_key``; the legacy bytes live on the *application host* at
``<upload_dir>/<storage_key>``, which the database cannot read. This script is the bridge:
it must run between 0008 and 0009, and 0009 refuses to apply until every row is filled.

Locked Phase 1C decisions implemented here:

* SHA-256 is computed from the actual file bytes.
* A missing file makes the document ``FAILED`` with ``'migration: raw bytes missing'``.
* A size mismatch against ``file_size`` is treated as corrupt bytes and fails the row.
* A checksum already present in the same workspace makes this row ``FAILED`` with
  ``'migration: duplicate checksum'``; the deterministic first row (``created_at, id``
  order) keeps its bytes. The losing row's bytes are cleared so the
  ``UNIQUE(workspace_id, checksum)`` constraint cannot be poisoned.
* Any row the script marks ``FAILED`` also loses its chunks, holding the section 5
  invariant that non-READY documents are never chunked.
* Idempotent: only rows with ``file_data IS NULL`` are touched, so a re-run resumes
  where a previous run stopped.

The script requires a BYPASSRLS role (the same prerequisite as migration 0008) because
it flips statuses on rows whose RLS policies would otherwise deny a non-owner writer.

Usage (from ``backend/``):

    python scripts/backfill_file_data.py [--upload-dir PATH] [--database-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings

_ERROR_MISSING = "migration: raw bytes missing"
_ERROR_SIZE_MISMATCH = "migration: file size mismatch"
_ERROR_DUPLICATE = "migration: duplicate checksum"

_BYPASS_CHECK = text(
    "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
)
_STORAGE_KEY_PRESENT = text(
    "SELECT count(*) FROM information_schema.columns "
    "WHERE table_name = 'documents' AND column_name = 'storage_key'"
)
_PENDING_ROWS = text(
    "SELECT id, workspace_id, storage_key, file_size FROM documents "
    "WHERE file_data IS NULL ORDER BY created_at, id"
)
_DUPLICATE_CHECK = text(
    "SELECT 1 FROM documents WHERE workspace_id = :workspace_id "
    "AND checksum = :checksum AND id <> :id LIMIT 1"
)
_MARK_FAILED = text(
    "UPDATE documents SET status = 'FAILED', error_message = :reason, "
    "file_data = NULL, checksum = NULL WHERE id = :id"
)
_DELETE_CHUNKS = text("DELETE FROM document_chunks WHERE document_id = :id")
_FILL_BYTES = text(
    "UPDATE documents SET file_data = :file_data, checksum = :checksum WHERE id = :id"
)


@dataclass
class BackfillReport:
    """What one run did, for the operator and for the migration test."""

    total: int = 0
    filled: int = 0
    missing: int = 0
    size_mismatch: int = 0
    duplicates: int = 0
    failed_rows: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def problems(self) -> int:
        return self.missing + self.size_mismatch + self.duplicates

    def summary(self) -> str:
        return (
            f"backfill: {self.filled}/{self.total} filled, "
            f"{self.missing} missing, {self.size_mismatch} size-mismatched, "
            f"{self.duplicates} duplicate, {self.failed_rows} rows marked FAILED"
        )


async def _require_bypass_role(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        bypass = (await conn.execute(_BYPASS_CHECK)).scalar()
    if bypass is not True:
        raise RuntimeError(
            "backfill_file_data requires a BYPASSRLS role (superuser). "
            "Run it as the same role that applied migration 0008."
        )


async def _fail_row(engine: AsyncEngine, document_id, reason: str) -> None:
    """Mark a row FAILED with the reason, drop its bytes, and clear its chunks."""
    async with engine.begin() as conn:
        await conn.execute(_DELETE_CHUNKS, {"id": document_id})
        await conn.execute(_MARK_FAILED, {"id": document_id, "reason": reason})


async def backfill_file_data(
    upload_dir: Path, *, database_url: str | None = None
) -> BackfillReport:
    """Read every legacy file, write ``file_data``/``checksum``, and log progress."""
    settings = get_settings()
    url = database_url or str(settings.database_url)
    upload_dir = upload_dir.resolve()

    report = BackfillReport()
    engine = create_async_engine(url)
    try:
        await _require_bypass_role(engine)

        async with engine.connect() as conn:
            has_key = (await conn.execute(_STORAGE_KEY_PRESENT)).scalar()
        if not has_key:
            raise RuntimeError(
                "documents.storage_key no longer exists — this script must run "
                "between migrations 0008 and 0009."
            )

        async with engine.connect() as conn:
            rows = list((await conn.execute(_PENDING_ROWS)).mappings())
        report.total = len(rows)
        logger.info("backfill starting: {n} documents awaiting bytes", n=report.total)

        for index, row in enumerate(rows, start=1):
            document_id = row["id"]
            path = upload_dir / str(row["storage_key"])

            if not path.is_file():
                report.missing += 1
                report.failed_rows += 1
                await _fail_row(engine, document_id, _ERROR_MISSING)
                logger.warning(
                    "[{i}/{n}] {doc} -> FAILED (missing {path})",
                    i=index, n=report.total, doc=document_id, path=path,
                )
                continue

            data = path.read_bytes()
            if len(data) != row["file_size"]:
                report.size_mismatch += 1
                report.failed_rows += 1
                await _fail_row(engine, document_id, _ERROR_SIZE_MISMATCH)
                logger.warning(
                    "[{i}/{n}] {doc} -> FAILED (size {got} != {want})",
                    i=index, n=report.total, doc=document_id,
                    got=len(data), want=row["file_size"],
                )
                continue

            digest = hashlib.sha256(data).hexdigest()
            async with engine.connect() as conn:
                duplicate = (
                    await conn.execute(
                        _DUPLICATE_CHECK,
                        {
                            "workspace_id": row["workspace_id"],
                            "checksum": digest,
                            "id": document_id,
                        },
                    )
                ).scalar()

            if duplicate:
                report.duplicates += 1
                report.failed_rows += 1
                await _fail_row(engine, document_id, _ERROR_DUPLICATE)
                logger.warning(
                    "[{i}/{n}] {doc} -> FAILED (duplicate checksum {sha})",
                    i=index, n=report.total, doc=document_id, sha=digest[:12],
                )
                continue

            async with engine.begin() as conn:
                await conn.execute(
                    _FILL_BYTES,
                    {"id": document_id, "file_data": data, "checksum": digest},
                )
            report.filled += 1
            if index % 25 == 0 or index == report.total:
                logger.info(
                    "backfill progress: {done}/{n} ({pct:.0f}%)",
                    done=index, n=report.total, pct=100 * index / max(report.total, 1),
                )

        logger.info("backfill complete: {summary}", summary=report.summary())
        return report
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill canonical documents.file_data/checksum from the legacy "
        "on-disk uploads. Run between Alembic 0008 and 0009."
    )
    parser.add_argument(
        "--upload-dir", type=Path, default=None,
        help="Directory holding the legacy <storage_key> files "
        "(default: settings.upload_dir).",
    )
    parser.add_argument(
        "--database-url", default=None,
        help="Override the configured DATABASE_URL.",
    )
    args = parser.parse_args()

    upload_dir = args.upload_dir or get_settings().upload_dir
    try:
        report = asyncio.run(backfill_file_data(upload_dir, database_url=args.database_url))
    except RuntimeError as exc:
        logger.error("{message}", message=exc)
        sys.exit(1)
    if report.problems:
        logger.warning(
            "backfill finished with problems — {problems} rows are FAILED; "
            "0009 will still apply as long as every file_data/checksum is filled.",
            problems=report.problems,
        )


if __name__ == "__main__":
    main()
