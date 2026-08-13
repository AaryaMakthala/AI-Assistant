"""Phase 1C: replace the org-centric schema with the canonical workspace schema.

Revision ID: 0008_phase1c_canonical_schema
Revises: 0007_document_visibility

This is the deliberate, one-way rebuild (CLAUDE.md section 7) from the retired
org-centric architecture to the canonical multi-tenant workspace architecture:

* ``organizations``/``users`` disappear; the tenant root becomes ``workspaces`` and
  identity lives in Supabase's ``auth.users`` (CLAUDE.md section 2).
* The five colliding table names (workspaces, documents, document_chunks,
  chat_sessions, chat_messages) are renamed to ``*_legacy``, the canonical tables are
  created exactly as ``app/db/models.py`` defines them, data is backfilled in
  dependency order, and the legacy tables are dropped last.
* Statuses move from lowercase ``pending/processing/ready/failed`` to uppercase
  ``PENDING/READY/REJECTED/FAILED``, and the legacy ``visibility`` concept folds into
  the section 5 approval lifecycle: ``ready + org -> READY``, ``ready + personal ->
  PENDING``, everything else pending-like. ``REJECTED`` has no legacy source.
* ``storage_key`` -> ``file_data`` is a *hybrid* step: the raw bytes live on the
  application host, so ``scripts/backfill_file_data.py`` reads them and fills
  ``file_data``/``checksum`` between this migration and 0009. 0008 therefore keeps
  ``storage_key`` and leaves the two columns nullable; 0009 enforces NOT NULL and
  drops ``storage_key``.
* Vector index: section 7 literally names ivfflat, but pgvector's ivfflat needs
  ``WITH (lists = N)`` and a training pass, while migration 0001 already established
  HNSW ("no training step, works on an empty table") and ``models.py`` explicitly
  permits it. This migration deliberately follows that precedent:
  ``USING hnsw (embedding vector_cosine_ops)``. Documented deviation.
* Auth provisioning: the 0004/0006 trigger created an org + owner user + default
  workspace. It is replaced here with a workspace + OWNER membership (CLAUDE.md
  section 4: the creator becomes the owner), and the ``org_id`` claim becomes
  ``workspace_id``. The replacement happens *before* the legacy tables are dropped so
  a signup can never hit the retired path.
* RLS is rebuilt workspace-scoped (``app.current_workspace_id()``) for all seven
  canonical tables; the ``app_sql_agent`` role is removed (SQL agent is out of
  canonical scope, CLAUDE.md section 11).

**One-way.** ``downgrade()`` raises: restoring the org-centric schema in place is not
practical, and the safe path is the pre-migration backup.

**Prerequisite.** Must run as a BYPASSRLS role (Supabase's ``postgres``, or the local
image's superuser). The migration verifies this and refuses otherwise.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0008_phase1c_canonical_schema"
down_revision: str | None = "0007_document_visibility"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 384

_TIMESTAMP = sa.DateTime(timezone=True)
_NEW_UUID = sa.text("gen_random_uuid()")
_NOW = sa.text("now()")

_CANONICAL_TABLES = (
    "workspaces",
    "members",
    "documents",
    "document_chunks",
    "chat_sessions",
    "chat_messages",
    "invitations",
)

#: Legacy tables whose names collide with the canonical schema; renamed aside so the
#: canonical names are free, then dropped at the end.
_COLLIDING = (
    "workspaces",
    "workspace_members",
    "documents",
    "document_chunks",
    "chat_sessions",
    "chat_messages",
)

def _execute_all(statements: tuple[str, ...]) -> None:
    for statement in statements:
        op.execute(statement.strip())


# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------

_PREFLIGHT = """
DO $$
DECLARE
    orphans integer;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname = current_user AND (rolsuper OR rolbypassrls)
    ) THEN
        RAISE EXCEPTION
            '0008 must run as a BYPASSRLS role (Supabase postgres, or the local '
            'superuser). RLS is FORCE-enabled on the legacy tables and would block '
            'the data move otherwise.';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'pgvector extension is missing (created by 0001).';
    END IF;
    IF to_regclass('auth.users') IS NULL THEN
        RAISE EXCEPTION
            'auth.users is missing. On local/CI apply backend/scripts/'
            'dev_auth_schema.sql first; on Supabase it already exists.';
    END IF;
    -- Every legacy user id that will become a canonical FK must exist in auth.users.
    SELECT count(*) INTO orphans
    FROM users u
    WHERE NOT EXISTS (SELECT 1 FROM auth.users a WHERE a.id = u.id);
    IF orphans > 0 THEN
        RAISE EXCEPTION
            '% legacy users have no auth.users row. Resolve these before migrating: '
            'a canonical FK to auth.users would fail.', orphans;
    END IF;
END
$$
"""

# ---------------------------------------------------------------------------
# 2. Unreachable legacy rows (orgs with no users) — same honesty rule as 0007
# ---------------------------------------------------------------------------

# 0007 deleted documents whose uploader could not be backfilled because their org had
# no users: "no one can authenticate into that org". The same reachability argument
# applies here to the whole org: without a single user it can never be accessed, and a
# canonical workspace with no electable owner would violate workspaces.owner_id NOT
# NULL. These rows are deleted with the count surfaced for the operator.
_UNREACHABLE_CLEANUP = """
DO $$
DECLARE
    before_count integer;
    after_count integer;
BEGIN
    SELECT count(*) INTO before_count FROM organizations;

    DELETE FROM ingestion_failures f
    USING documents_legacy d
    WHERE f.document_id = d.id
      AND NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = d.org_id);

    DELETE FROM document_chunks_legacy cl
    USING documents_legacy d
    WHERE cl.document_id = d.id
      AND NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = d.org_id);

    DELETE FROM documents_legacy d
    WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = d.org_id);

    DELETE FROM chat_messages_legacy cm
    USING chat_sessions_legacy cs
    WHERE cm.session_id = cs.id
      AND NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = cs.org_id);

    DELETE FROM chat_sessions_legacy cs
    WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = cs.org_id);

    DELETE FROM workspace_members_legacy wm
    WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = wm.org_id);

    DELETE FROM workspaces_legacy w
    WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = w.org_id);

    DELETE FROM sql_query_audit a
    WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = a.org_id);

    DELETE FROM conversation_summaries s
    WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = s.org_id);

    DELETE FROM organizations o
    WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.org_id = o.id);

    SELECT count(*) INTO after_count FROM organizations;
    RAISE NOTICE '0008: removed % unreachable legacy orgs (no users remain)',
        before_count - after_count;
END
$$
"""


# ---------------------------------------------------------------------------
# 3. Canonical tables — exactly app/db/models.py
# ---------------------------------------------------------------------------

_CREATE_CANONICAL = (
    (
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("auth.users.id"),
            nullable=False,
        ),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
    ),
    (
        "members",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("auth.users.id"), nullable=False
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.UniqueConstraint("workspace_id", "user_id", name="uq_members_workspace_user"),
        sa.CheckConstraint("role IN ('OWNER', 'MEMBER')", name="ck_members_role"),
        sa.CheckConstraint(
            "status IN ('INVITED', 'ACTIVE', 'REMOVED')", name="ck_members_status"
        ),
    ),
    (
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "uploaded_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("auth.users.id"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("mime_type", sa.String(200), nullable=False),
        sa.Column("file_size", sa.Integer, nullable=False),
        # Nullable until the host-side backfill script runs; NOT NULL in 0009.
        # SHA-256 hex digest of the raw bytes (64 characters).
        sa.Column("checksum", sa.String(64)),
        sa.Column("file_data", sa.LargeBinary),
        # TEMPORARY — the legacy on-disk name the backfill script reads; dropped in 0009.
        sa.Column("storage_key", postgresql.UUID(as_uuid=True)),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_message", sa.Text),
        sa.Column("approved_at", _TIMESTAMP),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        # Required so document_chunks can composite-FK on (document_id, workspace_id).
        sa.UniqueConstraint("id", "workspace_id", name="uq_documents_id_workspace"),
        sa.UniqueConstraint(
            "workspace_id", "checksum", name="uq_documents_workspace_checksum"
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'READY', 'REJECTED', 'FAILED')", name="ck_documents_status"
        ),
        sa.CheckConstraint("file_size >= 0", name="ck_documents_file_size_nonneg"),
        sa.Index("ix_documents_workspace_status", "workspace_id", "status"),
        sa.Index("ix_documents_workspace_uploaded_by", "workspace_id", "uploaded_by"),
    ),
    (
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Denormalized on purpose (CLAUDE.md section 7); the composite FK keeps it honest.
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("page_number", sa.Integer),
        sa.Column("section_title", sa.String(500)),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            name="fk_document_chunks_document_workspace",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
        sa.Index("ix_document_chunks_workspace_id", "workspace_id"),
        sa.Index(
            "ix_document_chunks_content_tsv_gin", "content_tsv", postgresql_using="gin"
        ),
    ),
    (
        "chat_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("auth.users.id"), nullable=False
        ),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.Index("ix_chat_sessions_workspace_user", "workspace_id", "user_id"),
    ),
    (
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        # Backend-constructed citations for assistant turns (CLAUDE.md 8.4).
        sa.Column("sources", postgresql.JSONB),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_messages_role"),
        sa.Index("ix_chat_messages_session_id", "session_id"),
    ),
    (
        "invitations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column(
            "invited_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("auth.users.id"),
            nullable=False,
        ),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.UniqueConstraint("workspace_id", "email", name="uq_invitations_workspace_email"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'ACCEPTED', 'EXPIRED')", name="ck_invitations_status"
        ),
    ),
)


# ---------------------------------------------------------------------------
# 4. Auth provisioning: workspace + OWNER membership instead of org + user
# ---------------------------------------------------------------------------

_PROVISION_FN = """
CREATE OR REPLACE FUNCTION app.provision_auth_user(
    auth_user_id uuid,
    auth_email text,
    auth_metadata jsonb
)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    existing_ws uuid;
    new_ws uuid;
    display_name text;
    ws_name text;
BEGIN
    -- Idempotent: an account that already has a membership is left alone (this is what
    -- lets the backfill below run over pre-existing auth.users without duplicating).
    SELECT w.id INTO existing_ws
    FROM members m JOIN workspaces w ON w.id = m.workspace_id
    WHERE m.user_id = auth_user_id
    ORDER BY w.created_at ASC
    LIMIT 1;
    IF existing_ws IS NOT NULL THEN
        RETURN existing_ws;
    END IF;

    -- User-supplied, so a display string only: trimmed, capped, never used to join
    -- an existing workspace.
    display_name := nullif(trim(auth_metadata ->> 'full_name'), '');
    ws_name := nullif(trim(auth_metadata ->> 'org_name'), '');
    IF ws_name IS NULL THEN
        ws_name := coalesce(display_name, split_part(coalesce(auth_email, 'workspace'), '@', 1));
    END IF;
    ws_name := left(ws_name, 200);

    INSERT INTO public.workspaces (name, owner_id)
    VALUES (ws_name, auth_user_id)
    RETURNING id INTO new_ws;

    INSERT INTO public.members (workspace_id, user_id, role, status)
    VALUES (new_ws, auth_user_id, 'OWNER', 'ACTIVE');

    RETURN new_ws;
END
$$
"""

_TRIGGER_FN = """
CREATE OR REPLACE FUNCTION app.handle_new_auth_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    ws uuid;
BEGIN
    ws := app.provision_auth_user(
        NEW.id, NEW.email, coalesce(NEW.raw_user_meta_data, '{}'::jsonb)
    );

    UPDATE auth.users
    SET raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb)
        || jsonb_build_object('workspace_id', ws::text)
    WHERE id = NEW.id;

    RETURN NEW;
END
$$
"""

# Repairs accounts that predate this migration or signed up while the trigger was
# absent. Runs after the legacy workspaces/members backfills, so it only fills gaps:
# provision_auth_user short-circuits on an existing membership.
_PROVISION_BACKFILL = """
DO $$
DECLARE
    account record;
    ws uuid;
BEGIN
    FOR account IN SELECT id, email, raw_user_meta_data FROM auth.users LOOP
        ws := app.provision_auth_user(
            account.id, account.email, coalesce(account.raw_user_meta_data, '{}'::jsonb)
        );
        UPDATE auth.users
        SET raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb)
            || jsonb_build_object('workspace_id', ws::text)
        WHERE id = account.id
          AND coalesce(raw_app_meta_data ->> 'workspace_id', '') IS DISTINCT FROM ws::text;
    END LOOP;
END
$$
"""

_PROVISION_DROP_LEGACY = (
    "DROP FUNCTION IF EXISTS app.unique_org_slug(text)",
    "REVOKE ALL ON FUNCTION app.provision_auth_user(uuid, text, jsonb) FROM PUBLIC",
    "REVOKE ALL ON FUNCTION app.handle_new_auth_user() FROM PUBLIC",
    "DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users",
    """
    CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION app.handle_new_auth_user()
    """,
)


# ---------------------------------------------------------------------------
# 5. Data backfills (dependency order)
# ---------------------------------------------------------------------------

# workspaces: 1:1 from the legacy workspaces (ids preserved, so documents and chats
# keep their assignments), owner elected the same way 0006/0007 elect it.
_BACKFILL_WORKSPACES = (
    """
INSERT INTO workspaces (id, name, owner_id, created_at)
SELECT
    w.id,
    w.name,
    COALESCE(w.owner_id, (
        SELECT u.id FROM users u
        WHERE u.org_id = w.org_id
        ORDER BY (u.role = 'owner') DESC, u.created_at ASC
        LIMIT 1
    )),
    w.created_at
FROM workspaces_legacy w
""",
    # Defensive: an org with no workspace at all (should not exist after 0006).
    """
INSERT INTO workspaces (id, name, owner_id, created_at)
SELECT
    gen_random_uuid(),
    o.name,
    (
        SELECT u.id FROM users u
        WHERE u.org_id = o.id
        ORDER BY (u.role = 'owner') DESC, u.created_at ASC
        LIMIT 1
    ),
    o.created_at
FROM organizations o
WHERE NOT EXISTS (SELECT 1 FROM workspaces_legacy w WHERE w.org_id = o.id)
""",
)

# members: role map owner/admin -> OWNER, editor/viewer -> MEMBER; every existing
# membership is ACTIVE (no legacy concept of invitation state).
_BACKFILL_MEMBERS_BASE = """
INSERT INTO members (id, workspace_id, user_id, role, status, created_at)
SELECT
    wm.id,
    wm.workspace_id,
    wm.user_id,
    CASE WHEN wm.role IN ('owner', 'admin') THEN 'OWNER' ELSE 'MEMBER' END,
    'ACTIVE',
    wm.created_at
FROM workspace_members_legacy wm
"""

# Completeness, run AFTER the documents and chat_sessions backfills: workspace owners,
# document uploaders and chat owners must all be members of the workspace their data
# lives in, or canonical access breaks. Reading the canonical tables means workspace_id
# is already resolved and NOT NULL, so no legacy NULL can leak in.
_BACKFILL_MEMBERS_COMPLETENESS = """
INSERT INTO members (id, workspace_id, user_id, role, status, created_at)
SELECT DISTINCT
    gen_random_uuid(),
    t.workspace_id,
    t.user_id,
    'OWNER',
    'ACTIVE',
    now()
FROM (
    SELECT id AS workspace_id, owner_id AS user_id FROM workspaces
    UNION
    SELECT workspace_id, uploaded_by FROM documents WHERE uploaded_by IS NOT NULL
    UNION
    SELECT workspace_id, user_id FROM chat_sessions
) t
WHERE NOT EXISTS (
    SELECT 1 FROM members m
    WHERE m.workspace_id = t.workspace_id AND m.user_id = t.user_id
)
"""

# documents: the visibility concept folds into the section 5 lifecycle. Statuses are
# computed from (legacy status, legacy visibility); org-wide 'ready' documents are the
# owner-published READY ones, personal 'ready' documents await owner approval.
_BACKFILL_DOCUMENTS = """
INSERT INTO documents (
    id, workspace_id, uploaded_by, filename, mime_type, file_size,
    checksum, file_data, storage_key, status, error_message, approved_at, created_at
)
SELECT
    dl.id,
    COALESCE(dl.workspace_id, (
        SELECT w.id FROM workspaces_legacy w
        WHERE w.org_id = dl.org_id
        ORDER BY (w.slug = 'default') DESC, w.created_at ASC
        LIMIT 1
    )),
    dl.uploaded_by,
    dl.filename,
    dl.mime_type,
    dl.size_bytes::integer,
    NULL,
    NULL,
    dl.storage_key,
    CASE
        WHEN dl.status = 'failed'                              THEN 'FAILED'
        WHEN dl.status = 'ready' AND dl.visibility = 'org'     THEN 'READY'
        WHEN dl.status = 'ready' AND dl.visibility = 'personal' THEN 'PENDING'
        ELSE 'PENDING'
    END,
    dl.error_message,
    CASE
        WHEN dl.status = 'ready' AND dl.visibility = 'org' THEN dl.created_at
        ELSE NULL
    END,
    dl.created_at
FROM documents_legacy dl
"""

# Chunks: only READY documents keep chunks — the section 5 invariant is structural, not
# a query-time filter. The embeddings carry over untouched (same pinned model and
# dimension, so no re-embedding). The defensive DELETE then guarantees the invariant
# even for rows that reached the canonical table through another path.
_BACKFILL_CHUNKS = """
INSERT INTO document_chunks (
    id, document_id, workspace_id, chunk_index, content,
    page_number, section_title, metadata, embedding, created_at
)
SELECT
    cl.id,
    cl.document_id,
    d.workspace_id,
    cl.chunk_index,
    cl.content,
    cl.page,
    NULL,
    cl.chunk_metadata,
    cl.embedding,
    cl.created_at
FROM document_chunks_legacy cl
JOIN documents d ON d.id = cl.document_id
WHERE d.status = 'READY'
"""

_CHUNK_INVARIANT = (
    "DELETE FROM document_chunks WHERE document_id IN "
    "(SELECT id FROM documents WHERE status <> 'READY')",
    # A READY document with no chunks cannot exist (section 5). Such rows come from a
    # legacy 'ready' document that had none — convert, never leave the invariant broken.
    """
UPDATE documents
SET status = 'FAILED',
    error_message = 'migration: ready document had no chunks',
    approved_at = NULL
WHERE status = 'READY'
  AND NOT EXISTS (SELECT 1 FROM document_chunks c WHERE c.document_id = documents.id)
""",
)

# chat_sessions: workspace_id was nullable in legacy (0006); NULLs are attributed to
# the org's default workspace, exactly as 0006 backfilled them originally.
_BACKFILL_SESSIONS = """
INSERT INTO chat_sessions (id, workspace_id, user_id, created_at)
SELECT
    cs.id,
    COALESCE(cs.workspace_id, (
        SELECT w.id FROM workspaces_legacy w
        WHERE w.org_id = cs.org_id
        ORDER BY (w.slug = 'default') DESC, w.created_at ASC
        LIMIT 1
    )),
    cs.user_id,
    cs.created_at
FROM chat_sessions_legacy cs
"""

# chat_messages: legacy had 4 roles, canonical has 2. Locked decision: system/tool
# turns are model-generated, so they are coerced to 'assistant' with their content
# preserved. citations -> sources (same shape).
_BACKFILL_MESSAGES = """
INSERT INTO chat_messages (id, session_id, role, content, sources, created_at)
SELECT
    cm.id,
    cm.session_id,
    CASE WHEN cm.role IN ('system', 'tool') THEN 'assistant' ELSE cm.role END,
    cm.content,
    cm.citations,
    cm.created_at
FROM chat_messages_legacy cm
"""

# ---------------------------------------------------------------------------
# 6. RLS: workspace-scoped, FORCE-enabled on all seven canonical tables
# ---------------------------------------------------------------------------

_CLAIM_FUNCTIONS = (
    """
CREATE OR REPLACE FUNCTION app.current_workspace_id()
RETURNS uuid
LANGUAGE plpgsql
STABLE
SET search_path = pg_catalog
AS $$
BEGIN
    RETURN (app.current_claims() ->> 'workspace_id')::uuid;
EXCEPTION WHEN invalid_text_representation THEN
    RETURN NULL;
END
$$
""",
    # Reads the role from members, never from a claim. SECURITY DEFINER so the SQL
    # agent-less policy path needs no extra grant on members.
    """
CREATE OR REPLACE FUNCTION app.current_workspace_role()
RETURNS text
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT m.role
    FROM members m
    WHERE m.user_id = app.current_user_id()
      AND m.workspace_id = app.current_workspace_id()
$$
""",
    # Membership/ownership checks for an ARBITRARY workspace. These must be SECURITY
    # DEFINER functions, not inline subqueries: a policy on `members` may not query
    # `members` (Postgres: "infinite recursion detected in policy"), and every policy
    # on `workspaces`/`invitations` that inlined a members subquery re-entered the
    # members policies the same way. Running as the table owner sidesteps RLS inside
    # the helper, so no recursion is possible.
    """
CREATE OR REPLACE FUNCTION app.is_workspace_member(ws uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT EXISTS (
        SELECT 1 FROM members m
        WHERE m.workspace_id = ws AND m.user_id = app.current_user_id()
    )
$$
""",
    """
CREATE OR REPLACE FUNCTION app.is_workspace_owner(ws uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT EXISTS (
        SELECT 1 FROM members m
        WHERE m.workspace_id = ws
          AND m.user_id = app.current_user_id()
          AND m.role = 'OWNER'
    )
$$
""",
    # The ONLY way to create a workspace (no INSERT policy exists on the table). SECURITY
    # DEFINER so the workspace and its OWNER membership are created atomically — a
    # non-definer INSERT could not create the membership afterwards, because
    # members_insert requires is_workspace_owner, which needs the membership.
    """
CREATE OR REPLACE FUNCTION app.create_workspace(p_name text)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    new_ws uuid;
    owner_id uuid;
BEGIN
    owner_id := app.current_user_id();
    IF owner_id IS NULL THEN
        RAISE EXCEPTION 'app.create_workspace requires an authenticated sub claim';
    END IF;

    INSERT INTO public.workspaces (name, owner_id)
    VALUES (left(p_name, 200), owner_id)
    RETURNING id INTO new_ws;

    INSERT INTO public.members (workspace_id, user_id, role, status)
    VALUES (new_ws, owner_id, 'OWNER', 'ACTIVE');

    RETURN new_ws;
END
$$
""",
)

# The OWNER predicate, expressed through the SECURITY DEFINER helper above (see the
# recursion note on is_workspace_owner).
_OWNER_OF = "app.is_workspace_owner({ws})"

_POLICIES = (
    # --- workspaces ---------------------------------------------------------
    """
CREATE POLICY workspaces_select ON workspaces FOR SELECT USING (
    app.is_workspace_member(workspaces.id)
)
""",
    # Workspace creation has NO table-level INSERT policy: a raw INSERT would leave a
    # workspace with no OWNER membership (the follow-up members_insert check needs
    # is_workspace_owner, which needs the membership that does not exist yet — a
    # chicken-and-egg), and its RETURNING would be invisible because workspaces_select
    # requires membership. Creation therefore goes through the SECURITY DEFINER
    # app.create_workspace() helper below, which atomically creates the workspace and
    # its OWNER membership (section 4: the creator becomes the owner), exactly like
    # the provisioning trigger.

    f"""
CREATE POLICY workspaces_update ON workspaces FOR UPDATE
USING ({_OWNER_OF.format(ws='workspaces.id')})
WITH CHECK ({_OWNER_OF.format(ws='workspaces.id')})
""",
    # --- members ------------------------------------------------------------
    """
CREATE POLICY members_select ON members FOR SELECT USING (
    app.is_workspace_member(members.workspace_id)
)
""",
    f"""
CREATE POLICY members_insert ON members FOR INSERT WITH CHECK (
    {_OWNER_OF.format(ws='members.workspace_id')}
)
""",
    f"""
CREATE POLICY members_update ON members FOR UPDATE
USING ({_OWNER_OF.format(ws='members.workspace_id')})
WITH CHECK ({_OWNER_OF.format(ws='members.workspace_id')})
""",
    f"""
CREATE POLICY members_delete ON members FOR DELETE USING (
    {_OWNER_OF.format(ws='members.workspace_id')}
)
""",
    # --- documents ----------------------------------------------------------
    # Section 5's lifecycle is enforced structurally, not by prompting: only the OWNER
    # (or the ingestion worker) may create or flip a document to READY, so a member can
    # never self-publish. Members may keep their own rows PENDING only.
    """
CREATE POLICY documents_select ON documents FOR SELECT USING (
    workspace_id = app.current_workspace_id() AND (
        status = 'READY'
        OR uploaded_by = app.current_user_id()
        OR app.current_workspace_role() = 'OWNER'
    )
)
""",
    """
CREATE POLICY documents_insert ON documents FOR INSERT WITH CHECK (
    workspace_id = app.current_workspace_id() AND (
        uploaded_by = app.current_user_id()
        OR app.current_workspace_role() = 'OWNER'
        OR app.is_ingestion_worker()
    )
    AND (
        app.current_workspace_role() = 'OWNER'
        OR app.is_ingestion_worker()
        OR status = 'PENDING'
    )
)
""",
    """
CREATE POLICY documents_update ON documents FOR UPDATE
USING (
    workspace_id = app.current_workspace_id() AND (
        uploaded_by = app.current_user_id()
        OR app.current_workspace_role() = 'OWNER'
        OR app.is_ingestion_worker()
    )
)
WITH CHECK (
    workspace_id = app.current_workspace_id() AND (
        uploaded_by = app.current_user_id()
        OR app.current_workspace_role() = 'OWNER'
        OR app.is_ingestion_worker()
    )
    AND (
        app.current_workspace_role() = 'OWNER'
        OR app.is_ingestion_worker()
        OR status = 'PENDING'
    )
)
""",
    """
CREATE POLICY documents_delete ON documents FOR DELETE USING (
    workspace_id = app.current_workspace_id() AND (
        uploaded_by = app.current_user_id()
        OR app.current_workspace_role() = 'OWNER'
    )
)
""",
    # --- document_chunks ----------------------------------------------------
    # Chunks are searchable content: readable for any READY document, but writable
    # ONLY by the ingestion worker (or the OWNER acting for it) on a READY document.
    # A member can never write chunks — not for their own PENDING upload either, which
    # is what keeps the section 5 "no chunks on non-READY documents" invariant
    # structural rather than a query-time filter.
    #
    # Ordering note for Phase 3: because this policy reads the CURRENT status, the
    # approval/ingestion transaction must set the document to READY before inserting
    # chunks (same-transaction visibility), not after.
    """
CREATE POLICY document_chunks_select ON document_chunks FOR SELECT USING (
    workspace_id = app.current_workspace_id() AND EXISTS (
        SELECT 1 FROM documents d
        WHERE d.id = document_chunks.document_id
          AND d.workspace_id = app.current_workspace_id()
          AND (
              d.status = 'READY'
              OR d.uploaded_by = app.current_user_id()
              OR app.current_workspace_role() = 'OWNER'
          )
    )
)
""",
    """
CREATE POLICY document_chunks_write ON document_chunks FOR ALL
USING (
    workspace_id = app.current_workspace_id() AND EXISTS (
        SELECT 1 FROM documents d
        WHERE d.id = document_chunks.document_id
          AND d.workspace_id = app.current_workspace_id()
          AND d.status = 'READY'
          AND (
              app.current_workspace_role() = 'OWNER'
              OR app.is_ingestion_worker()
          )
    )
)
WITH CHECK (
    workspace_id = app.current_workspace_id() AND EXISTS (
        SELECT 1 FROM documents d
        WHERE d.id = document_chunks.document_id
          AND d.workspace_id = app.current_workspace_id()
          AND d.status = 'READY'
          AND (
              app.current_workspace_role() = 'OWNER'
              OR app.is_ingestion_worker()
          )
    )
)
""",
    # --- chat_sessions / chat_messages --------------------------------------
    """
CREATE POLICY chat_sessions_owner_only ON chat_sessions FOR ALL
USING (workspace_id = app.current_workspace_id() AND user_id = app.current_user_id())
WITH CHECK (workspace_id = app.current_workspace_id() AND user_id = app.current_user_id())
""",
    """
CREATE POLICY chat_messages_owner_only ON chat_messages FOR ALL
USING (
    EXISTS (
        SELECT 1 FROM chat_sessions s
        WHERE s.id = chat_messages.session_id
          AND s.workspace_id = app.current_workspace_id()
          AND s.user_id = app.current_user_id()
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1 FROM chat_sessions s
        WHERE s.id = chat_messages.session_id
          AND s.workspace_id = app.current_workspace_id()
          AND s.user_id = app.current_user_id()
    )
)
""",
    # --- invitations --------------------------------------------------------
    """
CREATE POLICY invitations_member_read ON invitations FOR SELECT USING (
    workspace_id = app.current_workspace_id()
    AND app.is_workspace_member(invitations.workspace_id)
)
""",
    f"""
CREATE POLICY invitations_owner_insert ON invitations FOR INSERT WITH CHECK (
    {_OWNER_OF.format(ws='invitations.workspace_id')}
)
""",
    f"""
CREATE POLICY invitations_owner_update ON invitations FOR UPDATE
USING ({_OWNER_OF.format(ws='invitations.workspace_id')})
WITH CHECK ({_OWNER_OF.format(ws='invitations.workspace_id')})
""",
    f"""
CREATE POLICY invitations_owner_delete ON invitations FOR DELETE USING (
    {_OWNER_OF.format(ws='invitations.workspace_id')}
)
""",
)

_RLS_ENABLE = tuple(
    statement
    for table in _CANONICAL_TABLES
    for statement in (
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
    )
)

_GRANTS = (
    f"REVOKE ALL ON {', '.join(_CANONICAL_TABLES)} FROM anon, authenticated",
    f"GRANT SELECT, INSERT, UPDATE, DELETE ON {', '.join(_CANONICAL_TABLES)} TO app_tenant",
    # New functions default to PUBLIC EXECUTE; these SECURITY DEFINER helpers (one of
    # which writes data) must be callable only by app_tenant. Policies evaluate as the
    # querying role, so app_tenant's explicit grant below keeps them working after the
    # PUBLIC revoke.
    """
REVOKE ALL ON FUNCTION
    app.current_workspace_id(), app.current_workspace_role(),
    app.is_workspace_member(uuid), app.is_workspace_owner(uuid),
    app.create_workspace(text) FROM PUBLIC
""",
    """
GRANT EXECUTE ON FUNCTION
    app.current_claims(), app.current_user_id(), app.current_workspace_id(),
    app.current_workspace_role(), app.is_workspace_member(uuid),
    app.is_workspace_owner(uuid), app.create_workspace(text),
    app.is_ingestion_worker() TO app_tenant
""",
)

# The SQL agent is out of canonical scope (CLAUDE.md section 11). Its column-level
# grants die with the legacy tables, so the role must only be dropped AFTER those
# tables are gone; revoke what remains first.
_DROP_SQL_AGENT_ROLE = (
    "REVOKE app_sql_agent FROM CURRENT_USER",
    "REVOKE ALL ON SCHEMA app, public FROM app_sql_agent",
    """
REVOKE ALL ON FUNCTION app.current_claims(), app.current_user_id(), app.current_org_id(),
    app.current_org_role(), app.is_ingestion_worker() FROM app_sql_agent
""",
    "DROP ROLE IF EXISTS app_sql_agent",
)

# The org-scoped helpers have no remaining callers once the legacy tables (and their
# policies) are gone, so these run after the legacy drop.
_DROP_LEGACY_FUNCTIONS = (
    "DROP FUNCTION IF EXISTS app.current_org_id()",
    "DROP FUNCTION IF EXISTS app.current_org_role()",
)


def _upgrade_preflight() -> None:
    op.execute(_PREFLIGHT.strip())


def _rename_legacy_tables() -> None:
    for name in _COLLIDING:
        op.execute(f"ALTER TABLE {name} RENAME TO {name}_legacy")


# The cleanup references the *_legacy names, so it runs after the rename.
def _cleanup_unreachable() -> None:
    op.execute(_UNREACHABLE_CLEANUP.strip())


def _create_canonical_tables() -> None:
    for table in _CREATE_CANONICAL:
        op.create_table(*table)


def _replace_provisioning() -> None:
    op.execute(_PROVISION_FN.strip())
    op.execute(_TRIGGER_FN.strip())
    _execute_all(_PROVISION_DROP_LEGACY)


def _backfill() -> None:
    _execute_all(_BACKFILL_WORKSPACES)
    op.execute(_BACKFILL_MEMBERS_BASE.strip())
    op.execute(_PROVISION_BACKFILL.strip())
    op.execute(_BACKFILL_DOCUMENTS.strip())
    op.execute(_BACKFILL_CHUNKS.strip())
    _execute_all(_CHUNK_INVARIANT)
    op.execute(_BACKFILL_SESSIONS.strip())
    op.execute(_BACKFILL_MESSAGES.strip())
    # After documents and sessions, so workspace_id is always resolved and non-null.
    op.execute(_BACKFILL_MEMBERS_COMPLETENESS.strip())


def _create_vector_index() -> None:
    # Deviation from section 7's literal ivfflat, documented at the top of this file:
    # HNSW needs no training pass and matches the convention migration 0001 set.
    # The legacy table's index (0001) keeps this exact name through the rename, so it
    # is dropped first — its table is being dropped later anyway.
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def _rebuild_rls() -> None:
    _execute_all(_CLAIM_FUNCTIONS)
    _execute_all(_POLICIES)
    _execute_all(_RLS_ENABLE)
    _execute_all(_GRANTS)


#: The section 5 invariant, checked at the end of the migration.
_FINAL_INVARIANTS = """
DO $$
DECLARE
    n integer;
BEGIN
    SELECT count(*) INTO n FROM documents d
    WHERE d.status <> 'READY'
      AND EXISTS (SELECT 1 FROM document_chunks c WHERE c.document_id = d.id);
    IF n > 0 THEN
        RAISE EXCEPTION 'invariant broken: % non-READY documents have chunks', n;
    END IF;

    SELECT count(*) INTO n FROM documents d
    WHERE d.status = 'READY'
      AND NOT EXISTS (SELECT 1 FROM document_chunks c WHERE c.document_id = d.id);
    IF n > 0 THEN
        RAISE EXCEPTION 'invariant broken: % READY documents have no chunks', n;
    END IF;

    SELECT count(*) INTO n FROM document_chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.workspace_id <> d.workspace_id;
    IF n > 0 THEN
        RAISE EXCEPTION 'invariant broken: % chunks claim a workspace their document '
                        'does not belong to', n;
    END IF;

    SELECT count(*) INTO n FROM documents d JOIN workspaces w ON w.id = d.workspace_id
    WHERE NOT EXISTS (SELECT 1 FROM members m WHERE m.workspace_id = w.id);
    IF n > 0 THEN
        RAISE EXCEPTION 'invariant broken: % documents sit in a workspace with no members', n;
    END IF;

    RAISE NOTICE '0008: final invariant checks passed';
END
$$
"""


def _drop_legacy() -> None:
    # Children before parents; CASCADE is a backstop only.
    for table in (
        "ingestion_failures",
        "conversation_summaries",
        "chat_messages_legacy",
        "chat_sessions_legacy",
        "document_chunks_legacy",
        "documents_legacy",
        "workspace_members_legacy",
        "workspaces_legacy",
        "sql_query_audit",
        "users",
        "organizations",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


def upgrade() -> None:
    _upgrade_preflight()
    _rename_legacy_tables()
    _cleanup_unreachable()
    _create_canonical_tables()
    _replace_provisioning()
    _backfill()
    _create_vector_index()
    _rebuild_rls()
    op.execute(_FINAL_INVARIANTS.strip())
    # Legacy tables (and the policies/grants that travel with them) must be gone before
    # the org-scoped helpers and the SQL-agent role can be dropped.
    _drop_legacy()
    _execute_all(_DROP_SQL_AGENT_ROLE)
    _execute_all(_DROP_LEGACY_FUNCTIONS)


def downgrade() -> None:
    raise RuntimeError(
        "Migration 0008 is a one-way rebuild (org-centric -> workspace-centric). "
        "It cannot be rolled back in place: restore the pre-migration backup "
        "(pg_dump / Supabase backup taken before upgrading to 0008) instead."
    )
