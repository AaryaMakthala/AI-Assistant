"""ORM models.

Every org-scoped table carries `org_id` directly instead of relying on a join to its
parent. RLS policies then filter on a single indexed column, and no policy can be
defeated by a row whose parent moved. Composite foreign keys keep those denormalized
columns honest — a chunk cannot claim an org its document does not belong to.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# BAAI/bge-small-en-v1.5. Pinned: changing this invalidates every stored vector and
# requires a full re-embed of the index, never a mix (CLAUDE.md section 7, Risk 1).
EMBEDDING_DIM = 384

USER_ROLES = ("owner", "admin", "member")
DOCUMENT_STATUSES = ("pending", "processing", "ready", "failed")
#: Terminal statuses. A document in one of these is not going to change on its own, which
#: is what lets the UI stop polling and the reaper skip it.
TERMINAL_DOCUMENT_STATUSES = ("ready", "failed")
MESSAGE_ROLES = ("user", "assistant", "system", "tool")
SQL_AUDIT_STATUSES = ("accepted", "rejected", "failed")
#: Pipeline stages a dead-letter row can be attributed to. `abandoned` is the reaper's:
#: nobody reported a failure, the job simply stopped existing.
INGESTION_STAGES = ("extraction", "chunking", "embedding", "storage", "abandoned", "unknown")

_NEW_UUID = text("gen_random_uuid()")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class User(Base):
    """Mirrors a Supabase `auth.users` row; `id` is the JWT `sub` claim.

    Deliberately no FK to `auth.users`: that schema is Supabase-managed and absent on a
    plain Postgres instance, so the application keeps the two in step instead.
    """

    __tablename__ = "users"
    __table_args__ = (
        # Lets child tables composite-FK on (user_id, org_id), so a user's rows can never
        # be attributed to an org that user does not belong to.
        UniqueConstraint("id", "org_id", name="uq_users_id_org"),
        CheckConstraint(_in_list("role", USER_ROLES), name="ck_users_role"),
        Index("ix_users_org_id", "org_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    full_name: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), nullable=False, server_default="member")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("id", "org_id", name="uq_documents_id_org"),
        CheckConstraint(_in_list("status", DOCUMENT_STATUSES), name="ck_documents_status"),
        CheckConstraint("size_bytes >= 0", name="ck_documents_size_nonneg"),
        CheckConstraint(
            "chunk_count IS NULL OR chunk_count >= 0", name="ck_documents_chunk_count_nonneg"
        ),
        CheckConstraint(
            "word_count IS NULL OR word_count >= 0", name="ck_documents_word_count_nonneg"
        ),
        Index("ix_documents_org_created", "org_id", "created_at"),
        # Partial, matching migration 0005: the reaper only scans non-terminal rows, and on
        # a table that is mostly `ready` a full index would be almost entirely dead weight.
        Index(
            "ix_documents_status_started",
            "status",
            "processing_started_at",
            postgresql_where=text("status IN ('pending', 'processing')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    # Display name only — never used to build a filesystem path (CLAUDE.md 4.2).
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    # The generated name the bytes are actually stored under.
    storage_key: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, server_default=_NEW_UUID
    )
    mime_type: Mapped[str] = mapped_column(String(200), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    #: Chunks actually stored. Denormalized from document_chunks so the status endpoint
    #: answers "is this indexed yet" without counting rows on every poll.
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    #: Words in the extracted text, recorded once during ingestion.
    word_count: Mapped[int | None] = mapped_column(Integer)
    #: The Celery task currently responsible for this document. Kept so a delete can
    #: revoke work that is queued but not yet started.
    task_id: Mapped[str | None] = mapped_column(String(155))
    #: Attempts made so far. Surfaced to the UI because "still retrying" and "failed for
    #: good" look identical otherwise.
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    #: Set when a worker picks the document up, cleared conceptually by a terminal status.
    #: The reaper uses it to find jobs whose worker died mid-run.
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "org_id"],
            ["documents.id", "documents.org_id"],
            name="fk_chunks_document_same_org",
            ondelete="CASCADE",
        ),
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("ix_document_chunks_org_id", "org_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class IngestionFailure(Base):
    """Dead-letter record for an ingestion job that exhausted its retries.

    A failed document already carries a short `error_message` for the user. This table is
    the operator's view of the same event: the full reason, which attempt it died on, and
    how it ended. Kept separate because the two have different audiences and different
    lifetimes — the user's copy disappears when they delete the document, and so should
    this one, which is why it cascades rather than outliving its parent.

    Rows are written by the worker, never by a request. The `reason` is an exception
    string, so it is treated as untrusted text: it is truncated and only ever rendered
    as data.
    """

    __tablename__ = "ingestion_failures"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "org_id"],
            ["documents.id", "documents.org_id"],
            name="fk_ingestion_failures_document_same_org",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            _in_list("stage", INGESTION_STAGES),
            name="ck_ingestion_failures_stage",
        ),
        Index("ix_ingestion_failures_org_created", "org_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    #: Which part of the pipeline gave up, so a systemic failure is greppable by stage.
    stage: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (
        UniqueConstraint("id", "org_id", "user_id", name="uq_chat_sessions_id_org_user"),
        ForeignKeyConstraint(
            ["user_id", "org_id"],
            ["users.id", "users.org_id"],
            name="fk_chat_sessions_user_same_org",
            ondelete="CASCADE",
        ),
        Index("ix_chat_sessions_org_user", "org_id", "user_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    title: Mapped[str | None] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "org_id", "user_id"],
            ["chat_sessions.id", "chat_sessions.org_id", "chat_sessions.user_id"],
            name="fk_chat_messages_session_same_org_user",
            ondelete="CASCADE",
        ),
        CheckConstraint(_in_list("role", MESSAGE_ROLES), name="ck_chat_messages_role"),
        Index("ix_chat_messages_session_created", "session_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # Owner of the parent session, denormalized so the RLS policy stays a column
    # comparison instead of a subquery against chat_sessions.
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    token_usage: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class SqlQueryAudit(Base):
    """Every query the SQL agent generated, accepted or not (CLAUDE.md 4.3).

    Append-only by design: migration 0003 gives tenants an INSERT policy and no SELECT,
    UPDATE or DELETE policy at all. Rejected attempts are recorded too — a log of only the
    queries that ran would omit exactly the entries worth reviewing.

    Deliberately no FK to `users`: an audit row must outlive the account that caused it.
    """

    __tablename__ = "sql_query_audit"
    __table_args__ = (
        CheckConstraint(_in_list("status", SQL_AUDIT_STATUSES), name="ck_sql_audit_status"),
        CheckConstraint(
            "row_count IS NULL OR row_count >= 0", name="ck_sql_audit_row_count_nonneg"
        ),
        Index("ix_sql_query_audit_org_created", "org_id", "created_at"),
        Index("ix_sql_query_audit_user_created", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    #: Null when the model produced nothing parseable; `rejection_reason` carries the detail.
    generated_sql: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    row_count: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


#: Tables holding org-scoped data; all must have RLS enabled and forced.
ORG_SCOPED_TABLES = (
    "organizations",
    "users",
    "documents",
    "document_chunks",
    "ingestion_failures",
    "chat_sessions",
    "chat_messages",
    "sql_query_audit",
)

__all__ = [
    "DOCUMENT_STATUSES",
    "EMBEDDING_DIM",
    "INGESTION_STAGES",
    "MESSAGE_ROLES",
    "ORG_SCOPED_TABLES",
    "SQL_AUDIT_STATUSES",
    "TERMINAL_DOCUMENT_STATUSES",
    "USER_ROLES",
    "ChatMessage",
    "ChatSession",
    "Document",
    "DocumentChunk",
    "IngestionFailure",
    "Organization",
    "SqlQueryAudit",
    "User",
]
