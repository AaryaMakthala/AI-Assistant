"""Initial schema: orgs, users, documents, chunks, chat, plus RLS.

Revision ID: 0001_initial_schema
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 384

_TIMESTAMP = sa.DateTime(timezone=True)
_NEW_UUID = sa.text("gen_random_uuid()")
_NOW = sa.text("now()")

# Claims are injected per-transaction by the API layer as a local GUC. Reading them
# through a function keeps every policy identical to Supabase's own `auth.jwt()`,
# which is defined over the same setting — so the same policies hold whether a query
# arrives via the backend pool or via Supabase's REST layer.
_CLAIM_FUNCTIONS = """
CREATE SCHEMA IF NOT EXISTS app;

CREATE OR REPLACE FUNCTION app.current_claims()
RETURNS jsonb
LANGUAGE sql
STABLE
SET search_path = pg_catalog
AS $$
    SELECT COALESCE(
        NULLIF(current_setting('request.jwt.claims', true), '')::jsonb,
        '{}'::jsonb
    )
$$;

-- Returns NULL when the claim is absent or not a UUID. NULL never equals a stored
-- org_id, so a request that forgot to set claims sees zero rows rather than all rows.
CREATE OR REPLACE FUNCTION app.current_org_id()
RETURNS uuid
LANGUAGE plpgsql
STABLE
SET search_path = pg_catalog
AS $$
BEGIN
    RETURN (app.current_claims() ->> 'org_id')::uuid;
EXCEPTION WHEN invalid_text_representation THEN
    RETURN NULL;
END
$$;

CREATE OR REPLACE FUNCTION app.current_user_id()
RETURNS uuid
LANGUAGE plpgsql
STABLE
SET search_path = pg_catalog
AS $$
BEGIN
    RETURN (app.current_claims() ->> 'sub')::uuid;
EXCEPTION WHEN invalid_text_representation THEN
    RETURN NULL;
END
$$;
"""

_ORG_SCOPED_TABLES = (
    "organizations",
    "users",
    "documents",
    "document_chunks",
    "chat_sessions",
    "chat_messages",
)

# FORCE as well as ENABLE: without it the table owner silently bypasses every policy,
# which is exactly the role a migration or a careless script is most likely to use.
_ENABLE_RLS = "\n".join(
    f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;\nALTER TABLE {t} FORCE ROW LEVEL SECURITY;"
    for t in _ORG_SCOPED_TABLES
)

_POLICIES = """
-- Read-only for tenants; org and user provisioning happens out of band.
CREATE POLICY org_read_own ON organizations
    FOR SELECT USING (id = app.current_org_id());

CREATE POLICY users_read_own_org ON users
    FOR SELECT USING (org_id = app.current_org_id());

CREATE POLICY users_update_self ON users
    FOR UPDATE
    USING (id = app.current_user_id() AND org_id = app.current_org_id())
    WITH CHECK (id = app.current_user_id() AND org_id = app.current_org_id());

CREATE POLICY documents_org_isolation ON documents
    FOR ALL
    USING (org_id = app.current_org_id())
    WITH CHECK (org_id = app.current_org_id());

CREATE POLICY document_chunks_org_isolation ON document_chunks
    FOR ALL
    USING (org_id = app.current_org_id())
    WITH CHECK (org_id = app.current_org_id());

-- Chat is private to its author, not merely to the org.
CREATE POLICY chat_sessions_owner_only ON chat_sessions
    FOR ALL
    USING (org_id = app.current_org_id() AND user_id = app.current_user_id())
    WITH CHECK (org_id = app.current_org_id() AND user_id = app.current_user_id());

CREATE POLICY chat_messages_owner_only ON chat_messages
    FOR ALL
    USING (org_id = app.current_org_id() AND user_id = app.current_user_id())
    WITH CHECK (org_id = app.current_org_id() AND user_id = app.current_user_id());
"""

_APP_ROLE_GRANTS = """
-- Group role the application's login role must be a member of. It is deliberately
-- NOLOGIN and NOBYPASSRLS: a superuser connection ignores RLS entirely, FORCE included.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_tenant') THEN
        CREATE ROLE app_tenant NOLOGIN NOBYPASSRLS;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA app, public TO app_tenant;
GRANT EXECUTE ON FUNCTION
    app.current_claims(), app.current_org_id(), app.current_user_id() TO app_tenant;
GRANT SELECT ON organizations, users TO app_tenant;
GRANT SELECT, INSERT, UPDATE, DELETE ON
    documents, document_chunks, chat_sessions, chat_messages TO app_tenant;
"""


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(_CLAIM_FUNCTIONS)

    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.Column("updated_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.UniqueConstraint("slug", name="uq_organizations_slug"),
    )

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("full_name", sa.String(200)),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.Column("updated_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["org_id"], ["organizations.id"], name="fk_users_org", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.UniqueConstraint("id", "org_id", name="uq_users_id_org"),
        sa.CheckConstraint("role IN ('owner', 'admin', 'member')", name="ck_users_role"),
    )
    op.create_index("ix_users_org_id", "users", ["org_id"])

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("uploaded_by", postgresql.UUID(as_uuid=True)),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column(
            "storage_key", postgresql.UUID(as_uuid=True), nullable=False, server_default=_NEW_UUID
        ),
        sa.Column("mime_type", sa.String(200), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text),
        sa.Column("page_count", sa.Integer),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.Column("updated_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["org_id"], ["organizations.id"], name="fk_documents_org", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["uploaded_by"], ["users.id"], name="fk_documents_uploaded_by", ondelete="SET NULL"
        ),
        sa.UniqueConstraint("storage_key", name="uq_documents_storage_key"),
        sa.UniqueConstraint("id", "org_id", name="uq_documents_id_org"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')", name="ck_documents_status"
        ),
        sa.CheckConstraint("size_bytes >= 0", name="ck_documents_size_nonneg"),
    )
    op.create_index("ix_documents_org_created", "documents", ["org_id", "created_at"])

    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("page", sa.Integer),
        sa.Column("token_count", sa.Integer),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "chunk_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["document_id", "org_id"],
            ["documents.id", "documents.org_id"],
            name="fk_chunks_document_same_org",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )
    op.create_index("ix_document_chunks_org_id", "document_chunks", ["org_id"])
    # HNSW over cosine distance: no training step, so it works on an empty table and
    # stays correct as rows arrive. Cosine matches how bge-small embeddings are compared.
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "chat_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(300)),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.Column("updated_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["user_id", "org_id"],
            ["users.id", "users.org_id"],
            name="fk_chat_sessions_user_same_org",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("id", "org_id", "user_id", name="uq_chat_sessions_id_org_user"),
    )
    op.create_index(
        "ix_chat_sessions_org_user", "chat_sessions", ["org_id", "user_id", "updated_at"]
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "citations", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("token_usage", postgresql.JSONB),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["session_id", "org_id", "user_id"],
            ["chat_sessions.id", "chat_sessions.org_id", "chat_sessions.user_id"],
            name="fk_chat_messages_session_same_org_user",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')", name="ck_chat_messages_role"
        ),
    )
    op.create_index(
        "ix_chat_messages_session_created", "chat_messages", ["session_id", "created_at"]
    )

    op.execute(_ENABLE_RLS)
    op.execute(_POLICIES)
    op.execute(_APP_ROLE_GRANTS)


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("users")
    op.drop_table("organizations")
    op.execute("DROP FUNCTION IF EXISTS app.current_user_id()")
    op.execute("DROP FUNCTION IF EXISTS app.current_org_id()")
    op.execute("DROP FUNCTION IF EXISTS app.current_claims()")
    op.execute("DROP SCHEMA IF EXISTS app")
    # app_tenant and the extensions are left in place: other databases or migrations on
    # the same cluster may depend on them, and dropping a role fails if anything does.
