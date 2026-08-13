"""Phase 1C finalize: enforce file_data/checksum NOT NULL, drop storage_key.

Revision ID: 0009_phase1c_finalize_bytes
Revises: 0008_phase1c_canonical_schema

0008 created the canonical ``documents`` table with ``file_data``/``checksum`` nullable
and kept the legacy ``storage_key`` column so the host-side backfill script
(``scripts/backfill_file_data.py``) could locate the raw bytes on the application host.
This migration is the other side of that contract: it refuses to apply until every row
has bytes, then enforces the canonical NOT NULL shape (CLAUDE.md section 6 — the
database is the source of truth) and drops the temporary column. Checksum uniqueness is
already guaranteed by ``uq_documents_workspace_checksum`` from 0008.

Unlike 0008 this migration has a simple, supported downgrade: it only relaxes
nullability and restores the temporary column.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0009_phase1c_finalize_bytes"
down_revision: str | None = "0008_phase1c_canonical_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ASSERT_COMPLETE = """
DO $$
DECLARE
    n integer;
    offenders text;
BEGIN
    SELECT count(*), coalesce(string_agg(id::text, ', ' ORDER BY created_at, id), '')
    INTO n, offenders
    FROM documents WHERE file_data IS NULL OR checksum IS NULL;
    IF n > 0 THEN
        -- Decision 8 keeps missing-byte rows FAILED with the reason persisted; the
        -- canonical schema (section 6) then demands file_data NOT NULL, so the two
        -- cannot both hold. The operator must resolve these rows explicitly before
        -- finalizing: restore the files and re-run the backfill, or delete the rows.
        RAISE EXCEPTION
            '% document(s) still lack file_data/checksum: %. Run '
            'backend/scripts/backfill_file_data.py between 0008 and 0009, then restore '
            'or delete any rows whose bytes are irrecoverably missing.', n, offenders;
    END IF;
END
$$
"""


def upgrade() -> None:
    op.execute(_ASSERT_COMPLETE.strip())
    op.alter_column("documents", "file_data", nullable=False)
    op.alter_column("documents", "checksum", nullable=False)
    op.drop_column("documents", "storage_key")


def downgrade() -> None:
    op.add_column("documents", sa.Column("storage_key", postgresql.UUID(as_uuid=True)))
    op.alter_column("documents", "checksum", nullable=True)
    op.alter_column("documents", "file_data", nullable=True)
