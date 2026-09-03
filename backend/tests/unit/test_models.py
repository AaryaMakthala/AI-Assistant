"""Phase 1B model tests — the canonical CLAUDE.md section 7 schema.

These tests inspect the ORM metadata directly and never touch a database: no engine, no
create_all, no migrations. They pin the parts of the target schema that later phases and
Phase 1C's migration depend on — tenant paths, cascades, the 384-dimension vector column,
the generated tsvector column, and the constraint set.
"""

from __future__ import annotations

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

from app.db.base import Base
from app.db.legacy_models import LegacyBase, Organization
from app.db.legacy_models import User as LegacyUser
from app.db.models import (
    DOCUMENT_STATUSES,
    EMBEDDING_DIM,
    INVITATION_STATUSES,
    MEMBER_STATUSES,
    MESSAGE_ROLES,
    TERMINAL_DOCUMENT_STATUSES,
    WORKSPACE_MEMBER_ROLES,
    ChatMessage,
    ChatSession,
    Document,
    DocumentChunk,
    Invitation,
    Member,
    Workspace,
)

#: Exactly the tables of CLAUDE.md section 7 — nothing more, nothing less.
CANONICAL_TABLES = frozenset(
    {
        "workspaces",
        "members",
        "documents",
        "document_chunks",
        "chat_sessions",
        "chat_messages",
        "invitations",
    }
)

#: Stub table registered in Base.metadata so SQLAlchemy can resolve
#: ForeignKey("auth.users.id") at flush time.  Supabase owns the real
#: ``auth.users`` table; this is a minimal echo for ORM FK resolution only.
AUTH_USERS_STUB = "auth.users"

#: Tables from the retired org-centric schema that must never enter the canonical metadata.
LEGACY_ONLY_TABLES = frozenset(
    {
        "organizations",
        "users",
        "ingestion_failures",
        "sql_query_audit",
        "workspace_members",
        "conversation_summaries",
    }
)

ALL_TARGET_MODELS = (
    Workspace,
    Member,
    Document,
    DocumentChunk,
    ChatSession,
    ChatMessage,
    Invitation,
)


def _table(name: str):
    return Base.metadata.tables[name]


# ---------------------------------------------------------------------------
# Model import & metadata
# ---------------------------------------------------------------------------


def test_all_target_models_import() -> None:
    assert all(model.__name__ for model in ALL_TARGET_MODELS)


def test_canonical_metadata_constructs() -> None:
    assert Base.metadata is not None
    for name in CANONICAL_TABLES:
        assert name in Base.metadata.tables, f"canonical table {name} missing from metadata"


def test_canonical_metadata_contains_exactly_target_tables() -> None:
    """One authoritative mapping per target table — no strays from the legacy schema."""
    # The stub auth.users table is expected alongside the canonical tables.
    assert set(Base.metadata.tables) == CANONICAL_TABLES | {AUTH_USERS_STUB}


def test_no_duplicate_table_definitions() -> None:
    for name in CANONICAL_TABLES:
        assert len([t for t in Base.metadata.tables if t == name]) == 1
    # auth.users is a stub — one extra table beyond the canonical seven.
    assert len(Base.metadata.tables) == len(CANONICAL_TABLES) + 1


def test_legacy_tables_are_not_in_canonical_metadata() -> None:
    for name in LEGACY_ONLY_TABLES:
        assert name not in Base.metadata.tables


def test_legacy_models_isolated_on_their_own_metadata() -> None:
    """The legacy and canonical layers share no metadata object (Phase 1B report)."""
    assert LegacyBase.metadata is not Base.metadata
    # The legacy layer still holds its own tables, including the five colliding names —
    # as *distinct* Table objects from the canonical ones.
    assert "organizations" in LegacyBase.metadata.tables
    assert "workspaces" in LegacyBase.metadata.tables
    assert LegacyBase.metadata.tables["workspaces"] is not _table("workspaces")
    # Legacy models remain importable so their consumers keep working until rebuilt.
    assert Organization.__name__ == "Organization"
    assert LegacyUser.__name__ == "User"


# ---------------------------------------------------------------------------
# Columns & nullability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "column", "nullable"),
    [
        ("workspaces", "name", False),
        ("workspaces", "owner_id", False),
        ("workspaces", "created_at", False),
        ("members", "workspace_id", False),
        ("members", "user_id", False),
        ("members", "role", False),
        ("members", "status", False),
        ("documents", "workspace_id", False),
        ("documents", "uploaded_by", False),
        ("documents", "filename", False),
        ("documents", "mime_type", False),
        ("documents", "file_size", False),
        ("documents", "checksum", False),
        ("documents", "file_data", False),
        ("documents", "status", False),
        ("documents", "error_message", True),
        ("documents", "approved_at", True),
        ("document_chunks", "document_id", False),
        ("document_chunks", "workspace_id", False),
        ("document_chunks", "chunk_index", False),
        ("document_chunks", "content", False),
        ("document_chunks", "content_tsv", False),
        ("document_chunks", "embedding", False),
        ("document_chunks", "page_number", True),
        ("document_chunks", "section_title", True),
        ("document_chunks", "metadata", False),
        ("chat_sessions", "workspace_id", False),
        ("chat_sessions", "user_id", False),
        ("chat_messages", "session_id", False),
        ("chat_messages", "role", False),
        ("chat_messages", "content", False),
        ("chat_messages", "sources", True),
        ("invitations", "workspace_id", False),
        ("invitations", "email", False),
        ("invitations", "status", False),
        ("invitations", "invited_by", False),
    ],
)
def test_column_nullability(table: str, column: str, nullable: bool) -> None:
    assert _table(table).c[column].nullable is nullable


def test_documents_use_section7_column_names() -> None:
    """The §7 document shape, not the legacy storage_key/visibility shape."""
    cols = set(_table("documents").c.keys())
    assert {"file_data", "file_size", "checksum", "approved_at"} <= cols
    assert "storage_key" not in cols
    assert "visibility" not in cols


def test_chunk_metadata_column_is_named_metadata() -> None:
    """§7 calls the JSONB column `metadata`; the ORM attribute is renamed to avoid
    SQLAlchemy's reserved declarative name, but the physical column must stay `metadata`."""
    assert "metadata" in _table("document_chunks").c.keys()
    assert DocumentChunk.chunk_metadata.name == "metadata"


# ---------------------------------------------------------------------------
# Foreign keys & tenant ownership
# ---------------------------------------------------------------------------


def _fk_targets(table: str, column: str) -> list[tuple[str, str | None]]:
    """(referred table, ondelete) for a column's inline foreign keys.

    ``auth.users`` is now registered as a stub table in Base.metadata so
    SQLAlchemy can resolve FK targets at flush time.  We can safely use
    ``fk.column`` here, but ``target_fullname`` is kept for consistency
    with the rest of the test suite.
    """
    return [
        (fk.target_fullname, fk.ondelete) for fk in _table(table).c[column].foreign_keys
    ]


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("workspaces", "owner_id"),
        ("members", "user_id"),
        ("documents", "uploaded_by"),
        ("chat_sessions", "user_id"),
        ("invitations", "invited_by"),
    ],
)
def test_user_references_point_to_auth_users(table: str, column: str) -> None:
    """Users live in Supabase's auth.users — never a local table (CLAUDE.md section 2)."""
    assert _fk_targets(table, column) == [("auth.users.id", None)]


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("members", "workspace_id"),
        ("documents", "workspace_id"),
        ("chat_sessions", "workspace_id"),
        ("invitations", "workspace_id"),
    ],
)
def test_workspace_ownership_fks_cascade(table: str, column: str) -> None:
    assert _fk_targets(table, column) == [("workspaces.id", "CASCADE")]


def test_chunks_composite_fk_keeps_workspace_id_honest() -> None:
    """document_chunks(workspace_id) must equal its document's workspace — enforced by the
    database, not by application code (Phase 1B security review)."""
    constraints = [
        c for c in _table("document_chunks").constraints if isinstance(c, ForeignKeyConstraint)
    ]
    assert len(constraints) == 1
    fk = constraints[0]
    assert set(fk.columns.keys()) == {"document_id", "workspace_id"}
    assert fk.elements[0].target_fullname == "documents.id"
    assert fk.elements[1].target_fullname == "documents.workspace_id"
    assert fk.ondelete == "CASCADE"


def test_chat_message_session_fk_cascades() -> None:
    assert _fk_targets("chat_messages", "session_id") == [("chat_sessions.id", "CASCADE")]


def test_every_tenant_table_resolves_to_a_workspace() -> None:
    """Multi-tenancy path: every tenant-owned row either carries workspace_id directly
    or (chat_messages) reaches one through its parent session."""
    # workspaces is the tenant root and correctly has no workspace_id of its own.
    direct = {
        "members",
        "documents",
        "document_chunks",
        "chat_sessions",
        "invitations",
    }
    for name in CANONICAL_TABLES:
        if name in direct:
            assert "workspace_id" in _table(name).c, f"{name} must carry workspace_id"
    assert "workspace_id" not in _table("chat_messages").c
    assert "workspace_id" not in _table("workspaces").c
    assert _fk_targets("chat_messages", "session_id") == [("chat_sessions.id", "CASCADE")]


# ---------------------------------------------------------------------------
# Relationships & cascades
# ---------------------------------------------------------------------------


def _relationship_keys(model) -> set[str]:
    return set(model.__mapper__.relationships.keys())


def test_relationships_configured() -> None:
    assert _relationship_keys(Workspace) == {
        "members",
        "documents",
        "chat_sessions",
        "invitations",
    }
    assert _relationship_keys(Member) == {"workspace"}
    assert _relationship_keys(Document) == {"chunks", "workspace"}
    assert _relationship_keys(DocumentChunk) == {"document"}
    assert _relationship_keys(ChatSession) == {"messages", "workspace"}
    assert _relationship_keys(ChatMessage) == {"session"}
    assert _relationship_keys(Invitation) == {"workspace"}


def test_child_relationships_delete_orphans() -> None:
    for model, child in (
        (Workspace, "members"),
        (Workspace, "documents"),
        (Workspace, "chat_sessions"),
        (Workspace, "invitations"),
        (Document, "chunks"),
        (ChatSession, "messages"),
    ):
        rel = model.__mapper__.relationships[child]
        # CascadeOptions expands `all` into its parts; assert the boolean flags.
        assert rel.cascade.delete is True
        assert rel.cascade.delete_orphan is True


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _unique_constraints(table: str) -> list[frozenset[str]]:
    return [
        frozenset(c.columns.keys())
        for c in _table(table).constraints
        if isinstance(c, UniqueConstraint)
    ]


def _check_constraints(table: str) -> list[str]:
    return [
        str(c.sqltext)
        for c in _table(table).constraints
        if isinstance(c, CheckConstraint)
    ]


def test_unique_constraints() -> None:
    assert frozenset({"workspace_id", "user_id"}) in _unique_constraints("members")
    assert frozenset({"workspace_id", "checksum"}) in _unique_constraints("documents")
    # Required as the composite FK target for chunks.
    assert frozenset({"id", "workspace_id"}) in _unique_constraints("documents")
    assert frozenset({"document_id", "chunk_index"}) in _unique_constraints("document_chunks")
    assert frozenset({"workspace_id", "email"}) in _unique_constraints("invitations")


@pytest.mark.parametrize(
    ("table", "column", "allowed"),
    [
        ("members", "role", WORKSPACE_MEMBER_ROLES),
        ("members", "status", MEMBER_STATUSES),
        ("documents", "status", DOCUMENT_STATUSES),
        ("chat_messages", "role", MESSAGE_ROLES),
        ("invitations", "status", INVITATION_STATUSES),
    ],
)
def test_check_constraints_cover_the_enums(
    table: str, column: str, allowed: tuple[str, ...]
) -> None:
    checks = "\n".join(_check_constraints(table))
    for value in allowed:
        assert value in checks, f"{table}.{column} must allow {value}"
    assert f"{column} IN" in checks


def test_document_status_lifecycle_matches_section5() -> None:
    assert set(DOCUMENT_STATUSES) == {"PENDING", "READY", "REJECTED", "FAILED"}
    assert set(TERMINAL_DOCUMENT_STATUSES) <= set(DOCUMENT_STATUSES)
    assert TERMINAL_DOCUMENT_STATUSES == ("READY", "FAILED")


def test_file_size_is_non_negative() -> None:
    assert any("file_size >= 0" in c for c in _check_constraints("documents"))


# ---------------------------------------------------------------------------
# Vector, full-text search, indexes
# ---------------------------------------------------------------------------


def test_embedding_column_is_384_dimension_pgvector() -> None:
    embedding = _table("document_chunks").c["embedding"]
    assert isinstance(embedding.type, Vector)
    # pgvector 0.3.x stores the dimension on `dim`, not `dimensions`.
    assert embedding.type.dim == EMBEDDING_DIM == 384


def test_only_one_embedding_column_and_it_matches_config_dimension() -> None:
    """No mixing of embedding models/dimensions (CLAUDE.md sections 2 and 13)."""
    vector_columns = [
        name
        for name, col in _table("document_chunks").c.items()
        if isinstance(col.type, Vector)
    ]
    assert vector_columns == ["embedding"]


def test_content_tsv_is_a_generated_stored_column() -> None:
    col = _table("document_chunks").c["content_tsv"]
    assert col.computed is not None
    assert col.computed.persisted is True
    assert "to_tsvector('english', content)" in str(col.computed.sqltext)


def _indexes(table: str) -> dict[str, object]:
    return {idx.name: idx for idx in _table(table).indexes}


def test_indexes_support_tenant_scoped_access_patterns() -> None:
    docs = _indexes("documents")
    assert set(docs) == {"ix_documents_workspace_status", "ix_documents_workspace_uploaded_by"}
    assert tuple(docs["ix_documents_workspace_status"].columns.keys()) == ("workspace_id", "status")
    assert tuple(docs["ix_documents_workspace_uploaded_by"].columns.keys()) == (
        "workspace_id",
        "uploaded_by",
    )

    chunks = _indexes("document_chunks")
    assert {"ix_document_chunks_workspace_id", "ix_document_chunks_content_tsv_gin"} <= set(
        chunks
    )
    assert tuple(chunks["ix_document_chunks_workspace_id"].columns.keys()) == ("workspace_id",)

    sessions = _indexes("chat_sessions")
    assert tuple(sessions["ix_chat_sessions_workspace_user"].columns.keys()) == (
        "workspace_id",
        "user_id",
    )

    messages = _indexes("chat_messages")
    assert tuple(messages["ix_chat_messages_session_id"].columns.keys()) == ("session_id",)


def test_fts_index_is_gin_over_the_generated_column() -> None:
    idx = _indexes("document_chunks")["ix_document_chunks_content_tsv_gin"]
    assert idx.dialect_options["postgresql"]["using"] == "gin"
    assert tuple(idx.columns.keys()) == ("content_tsv",)


def test_vector_index_is_deliberately_not_declared_in_the_model() -> None:
    """§7 mandates an ivfflat index, but pgvector needs `WITH (lists = N)` and a model-
    declared index would make Phase 1C autogenerate emit DDL Postgres rejects. Following
    the legacy convention (HNSW created in migration 0001 via raw SQL), Phase 1C owns the
    vector index. Asserting its absence guards against autogenerate producing broken DDL.
    """
    using = [
        idx.dialect_options["postgresql"].get("using") for idx in _table("document_chunks").indexes
    ]
    assert "ivfflat" not in using
    assert "hnsw" not in using
    assert "gin" in using


def test_membership_lookup_is_covered_by_the_unique_index() -> None:
    """UNIQUE(workspace_id, user_id) backs the §7 membership-lookup index, so no
    redundant standalone index is declared (documented in models.py)."""
    assert frozenset({"workspace_id", "user_id"}) in _unique_constraints("members")
    assert _indexes("members") == {}


# ---------------------------------------------------------------------------
# auth.users stub — FK resolution
# ---------------------------------------------------------------------------


def test_auth_users_stub_registered_in_metadata() -> None:
    """A stub Table for auth.users must exist so SQLAlchemy can resolve
    ForeignKey("auth.users.id") at flush time without NoReferencedTableError."""
    assert AUTH_USERS_STUB in Base.metadata.tables
    stub = Base.metadata.tables[AUTH_USERS_STUB]
    assert stub.schema == "auth"
    assert "id" in stub.c


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("workspaces", "owner_id"),
        ("members", "user_id"),
        ("documents", "uploaded_by"),
        ("chat_sessions", "user_id"),
        ("invitations", "invited_by"),
    ],
)
def test_auth_users_fk_resolves_without_error(table: str, column: str) -> None:
    """Accessing fk.column must not raise NoReferencedTableError — the stub
    table makes the target resolvable."""
    tbl = _table(table)
    fks = [fk for fk in tbl.c[column].foreign_keys if fk.target_fullname == "auth.users.id"]
    assert len(fks) == 1, f"expected exactly one auth.users FK on {table}.{column}"
    # This line raises NoReferencedTableError before the fix.
    assert fks[0].column.table.name == "users"
    assert fks[0].column.table.schema == "auth"
