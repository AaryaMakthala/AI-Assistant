"""Ingestion progress columns and a dead-letter table for jobs that gave up.

Revision ID: 0005_ingestion_hardening
Revises: 0004_auth_provisioning

Phase 10 hardens background processing. Two things were missing, both of which made a
failure invisible rather than merely unfortunate:

* **No record of progress.** A document row said `processing` and nothing else — not when
  it started, not which attempt it was on, not how many chunks came out. A job whose
  worker was killed mid-run was indistinguishable from one that was merely slow, so the UI
  spun forever and no operator could tell the two apart. `processing_started_at` is what
  makes the reaper possible: it is the only evidence that a job was ever picked up.
* **No record of the give-up.** When retries were exhausted the user got a one-line
  message and the detail was lost to the worker's stdout. `ingestion_failures` keeps the
  operator's copy.

`chunk_count` is denormalized rather than counted from `document_chunks` on demand. The
status endpoint is polled every couple of seconds per in-flight document, and a count over
a growing table is the wrong shape of query to put on that path.

The dead-letter table cascades from its document rather than outliving it. That is
deliberate and worth stating, because the neighbouring `sql_query_audit` does the
opposite: an audit log is a security record that must survive the actor, whereas this is
operational detail *about* a document, meaningless once the document is gone. Phase 10
requires a delete to remove every related resource, and a surviving failure row would be
exactly the kind of orphan that requirement exists to prevent.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0005_ingestion_hardening"
down_revision: str | None = "0004_auth_provisioning"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_NEW_UUID = sa.text("gen_random_uuid()")
_NOW = sa.text("now()")
_TIMESTAMP = postgresql.TIMESTAMP(timezone=True)

#: Mirrors INGESTION_STAGES in app/db/models.py.
_STAGES = ("extraction", "chunking", "embedding", "storage", "abandoned", "unknown")


#: Built fresh per call rather than held as module constants: `op.add_column` binds a
#: Column to a Table, so reusing one instance across calls raises on the second.
def _document_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column("chunk_count", sa.Integer),
        sa.Column("word_count", sa.Integer),
        # Celery task ids are UUID hex by default; 155 leaves room for a custom id without
        # being unbounded text on a hot table.
        sa.Column("task_id", sa.String(155)),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("processing_started_at", _TIMESTAMP),
        sa.Column("processing_completed_at", _TIMESTAMP),
    )


_POLICY = """
CREATE POLICY ingestion_failures_org_isolation ON ingestion_failures
    FOR ALL
    USING (org_id = app.current_org_id())
    WITH CHECK (org_id = app.current_org_id())
"""

_GRANTS = (
    # Supabase's default ACL hands `anon` and `authenticated` full DML on every new table
    # in `public`. RLS would still filter them, but a grant that contradicts the intent is
    # how a future policy added for another purpose silently opens a table up.
    "REVOKE ALL ON ingestion_failures FROM anon, authenticated",
    # SELECT and DELETE for tenants: they read their own failures in the UI, and a document
    # delete has to take its failure rows with it. No INSERT or UPDATE — only the worker
    # writes here, and it does so through the same tenant role, so withholding INSERT would
    # break it. It is granted below for that reason and no other.
    "GRANT SELECT, INSERT, DELETE ON ingestion_failures TO app_tenant",
)


def upgrade() -> None:
    for column in _document_columns():
        op.add_column("documents", column)

    op.create_check_constraint(
        "ck_documents_chunk_count_nonneg", "documents", "chunk_count IS NULL OR chunk_count >= 0"
    )
    op.create_check_constraint(
        "ck_documents_word_count_nonneg", "documents", "word_count IS NULL OR word_count >= 0"
    )
    # Partial: the reaper only ever scans non-terminal rows, and on a table that is mostly
    # `ready` a full index would be almost entirely dead weight.
    op.execute(
        "CREATE INDEX ix_documents_status_started ON documents (status, processing_started_at)"
        " WHERE status IN ('pending', 'processing')"
    )

    op.create_table(
        "ingestion_failures",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        # Composite FK for the same reason as document_chunks: a failure row cannot claim
        # an org its document does not belong to.
        sa.ForeignKeyConstraint(
            ["document_id", "org_id"],
            ["documents.id", "documents.org_id"],
            name="fk_ingestion_failures_document_same_org",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "stage IN (" + ", ".join(f"'{stage}'" for stage in _STAGES) + ")",
            name="ck_ingestion_failures_stage",
        ),
    )
    op.create_index(
        "ix_ingestion_failures_org_created", "ingestion_failures", ["org_id", "created_at"]
    )

    # FORCE as well as ENABLE — without it the table owner bypasses every policy, and the
    # owner is exactly the role a migration or a maintenance script connects as.
    op.execute("ALTER TABLE ingestion_failures ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE ingestion_failures FORCE ROW LEVEL SECURITY")
    op.execute(_POLICY.strip())
    for statement in _GRANTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("REVOKE ALL ON ingestion_failures FROM app_tenant")
    op.drop_index("ix_ingestion_failures_org_created", table_name="ingestion_failures")
    op.drop_table("ingestion_failures")

    op.drop_index("ix_documents_status_started", table_name="documents")
    op.drop_constraint("ck_documents_word_count_nonneg", "documents", type_="check")
    op.drop_constraint("ck_documents_chunk_count_nonneg", "documents", type_="check")
    for column in reversed(_document_columns()):
        op.drop_column("documents", column.name)
