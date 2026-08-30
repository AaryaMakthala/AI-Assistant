"""ORM models — the canonical target schema (CLAUDE.md section 7).

Multi-tenant by construction: every tenant-owned table carries ``workspace_id`` and every
query that touches these tables filters on it server-side — the frontend is never trusted
with the boundary (CLAUDE.md section 4). ``document_chunks`` denormalizes ``workspace_id``
on purpose so retrieval filters on a single indexed column with no join, and a composite
foreign key keeps that column honest: a chunk cannot claim a workspace its document does
not belong to.

Users live in Supabase's ``auth.users`` (CLAUDE.md section 2), not a local table, so
references to them are plain FK columns with no ORM relationship. The retired org-centric
models live in ``app.db.legacy_models`` on a separate metadata and are scheduled for
deletion once their consumers are rebuilt in later phases.

Index notes (deliberate departures from section 7's literal list, each with a reason):

- ``members``: section 7 lists UNIQUE(workspace_id, user_id) *and* INDEX on the same
  columns. Postgres backs every UNIQUE constraint with an index, so the separate index
  would be pure write amplification; the UNIQUE backing index serves the membership
  lookup on every request.
- ``document_chunks``: section 7 lists INDEX(document_id) alongside
  UNIQUE(document_id, chunk_index). The UNIQUE backing index already serves
  ``WHERE document_id = ?`` through its leftmost column, so a standalone index is
  redundant. The ``workspace_id`` index is kept because it is not a prefix of any other
  index and is the retrieval filter.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# BAAI/bge-small-en-v1.5. Pinned: changing this invalidates every stored vector and
# requires a full re-embed of the index, never a mix (CLAUDE.md sections 2 and 13).
# MUST match EMBEDDING_DIMENSION in app/config.py; Phase 1C should assert the match.
EMBEDDING_DIM = 384

#: Workspace-level roles (CLAUDE.md section 4). Two roles only — deliberately no ADMIN.
WORKSPACE_MEMBER_ROLES = ("OWNER", "MEMBER")
#: Lifecycle of a membership row (CLAUDE.md section 4).
MEMBER_STATUSES = ("INVITED", "ACTIVE", "REMOVED")
#: Document lifecycle (CLAUDE.md section 5). Only READY documents are ever searchable.
DOCUMENT_STATUSES = ("PENDING", "READY", "REJECTED", "FAILED")
#: Statuses that will not change on their own (CLAUDE.md section 5).
TERMINAL_DOCUMENT_STATUSES = ("READY", "FAILED")
MESSAGE_ROLES = ("user", "assistant")
INVITATION_STATUSES = ("PENDING", "ACCEPTED", "EXPIRED")

_NEW_UUID = text("gen_random_uuid()")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


class Workspace(Base):
    """A tenant. One deployment hosts many independent workspaces (CLAUDE.md section 4)."""

    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("auth.users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    members: Mapped[list[Member]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_sessions: Mapped[list[ChatSession]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )
    invitations: Mapped[list[Invitation]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )


class Member(Base):
    """Membership of a user in a workspace, with role and lifecycle (CLAUDE.md section 4)."""

    __tablename__ = "members"
    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id", name="uq_members_workspace_user"),
        CheckConstraint(
            _in_list("role", WORKSPACE_MEMBER_ROLES), name="ck_members_role"
        ),
        CheckConstraint(_in_list("status", MEMBER_STATUSES), name="ck_members_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("auth.users.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    workspace: Mapped[Workspace] = relationship(back_populates="members")


class Document(Base):
    """A file uploaded to a workspace (CLAUDE.md sections 5 and 6).

    The raw bytes live in ``file_data`` (BYTEA) — the database is the source of truth.
    ``checksum`` is the SHA-256 of the bytes, and UNIQUE(workspace_id, checksum) makes
    duplicate detection within a workspace a database guarantee rather than a code path.
    """

    __tablename__ = "documents"
    __table_args__ = (
        # Required so document_chunks can composite-FK on (document_id, workspace_id).
        UniqueConstraint("id", "workspace_id", name="uq_documents_id_workspace"),
        UniqueConstraint("workspace_id", "checksum", name="uq_documents_workspace_checksum"),
        CheckConstraint(_in_list("status", DOCUMENT_STATUSES), name="ck_documents_status"),
        CheckConstraint("file_size >= 0", name="ck_documents_file_size_nonneg"),
        Index("ix_documents_workspace_status", "workspace_id", "status"),
        Index("ix_documents_workspace_uploaded_by", "workspace_id", "uploaded_by"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("auth.users.id"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(200), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    #: SHA-256 hex digest of the raw bytes (64 characters).
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    file_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    workspace: Mapped[Workspace] = relationship(back_populates="documents")


class DocumentChunk(Base):
    """One searchable unit of a document's text (CLAUDE.md sections 7 and 8).

    ``workspace_id`` is denormalized from the parent document on purpose — every
    retrieval query filters on it directly, with no join to enforce tenant isolation —
    and the composite FK below makes lying about it impossible.

    ``content_tsv`` is a generated column: Postgres maintains ``to_tsvector('english',
    content)`` automatically, so the keyword index can never drift from the text.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        # Keeps the denormalized workspace_id honest (Phase 1B security review,
        # "inconsistent workspace IDs"): a chunk cannot claim a workspace its document
        # does not belong to, and deleting a document takes its chunks with it.
        ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            name="fk_document_chunks_document_workspace",
            ondelete="CASCADE",
        ),
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
        Index("ix_document_chunks_workspace_id", "workspace_id"),
        Index(
            "ix_document_chunks_content_tsv_gin",
            "content_tsv",
            postgresql_using="gin",
        ),
        # Section 7 also mandates `INDEX USING ivfflat (embedding vector_cosine_ops)`.
        # It is deliberately NOT declared here: pgvector requires `WITH (lists = N)`, and
        # a model-declared index would make Phase 1C's autogenerate emit DDL without it
        # (which Postgres rejects). Following the convention the legacy migration used for
        # its HNSW index, Phase 1C creates the vector index in the migration with tuned
        # parameters — or chooses HNSW, which needs no training step on an empty table.
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    workspace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(String(500))
    #: Column name is `metadata` per section 7; the attribute is renamed because
    #: `metadata` is a reserved name on SQLAlchemy declarative classes.
    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class ChatSession(Base):
    """A conversation, scoped to a workspace and owned by a user (CLAUDE.md section 7)."""

    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("ix_chat_sessions_workspace_user", "workspace_id", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("auth.users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )
    workspace: Mapped[Workspace] = relationship(back_populates="chat_sessions")


class ChatMessage(Base):
    """One turn of a conversation (CLAUDE.md section 7)."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(_in_list("role", MESSAGE_ROLES), name="ck_chat_messages_role"),
        Index("ix_chat_messages_session_id", "session_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    #: Backend-constructed citations for assistant turns (CLAUDE.md 8.4). Null for user
    #: turns and for assistant turns before the grounding pipeline lands.
    sources: Mapped[list | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class Invitation(Base):
    """A pending invitation for a user to join a workspace (CLAUDE.md section 7)."""

    __tablename__ = "invitations"
    __table_args__ = (
        UniqueConstraint("workspace_id", "email", name="uq_invitations_workspace_email"),
        CheckConstraint(_in_list("status", INVITATION_STATUSES), name="ck_invitations_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    invited_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("auth.users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    workspace: Mapped[Workspace] = relationship(back_populates="invitations")


__all__ = [
    "WORKSPACE_MEMBER_ROLES",
    "MEMBER_STATUSES",
    "DOCUMENT_STATUSES",
    "TERMINAL_DOCUMENT_STATUSES",
    "MESSAGE_ROLES",
    "INVITATION_STATUSES",
    "EMBEDDING_DIM",
    "Workspace",
    "Member",
    "Document",
    "DocumentChunk",
    "ChatSession",
    "ChatMessage",
    "Invitation",
]
