"""Workspaces and conversation summaries.

Revision ID: 0006_workspaces
Revises: 0005_ingestion_hardening
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0006_workspaces"
down_revision: str | None = "0005_ingestion_hardening"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_NEW_UUID = sa.text("gen_random_uuid()")
_NOW = sa.text("now()")
_TIMESTAMP = postgresql.TIMESTAMP(timezone=True)


def _execute_all(statements: tuple[str, ...]) -> None:
    for statement in statements:
        op.execute(statement.strip())


_TABLES = (
    "workspaces",
    "workspace_members",
    "conversation_summaries",
)

_ENABLE_RLS = tuple(
    statement
    for table in _TABLES
    for statement in (
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
    )
)

#: Org-level isolation is the right grain for the workspace tables: membership is what
#: narrows them further, and that is checked in the API layer.
_ORG_SCOPED_TABLES = ("workspaces", "workspace_members")

_POLICIES = tuple(
    f"""
CREATE POLICY {table}_org_isolation ON {table}
    FOR ALL
    USING (org_id = app.current_org_id())
    WITH CHECK (org_id = app.current_org_id())
"""
    for table in _ORG_SCOPED_TABLES
) + (
    # A summary is a compression of the conversation itself, so it is exactly as sensitive
    # as the messages it replaces. Chat is private to its author, not merely to the org
    # (see the chat_sessions policy in 0001) — an org-only rule here would let a colleague
    # read a paraphrase of a conversation they cannot read directly.
    """
CREATE POLICY conversation_summaries_owner_only ON conversation_summaries
    FOR ALL
    USING (
        org_id = app.current_org_id()
        AND EXISTS (
            SELECT 1 FROM chat_sessions s
            WHERE s.id = conversation_summaries.session_id
              AND s.org_id = app.current_org_id()
              AND s.user_id = app.current_user_id()
        )
    )
    WITH CHECK (
        org_id = app.current_org_id()
        AND EXISTS (
            SELECT 1 FROM chat_sessions s
            WHERE s.id = conversation_summaries.session_id
              AND s.org_id = app.current_org_id()
              AND s.user_id = app.current_user_id()
        )
    )
""",
)

_GRANTS = (
    f"REVOKE ALL ON {', '.join(_TABLES)} FROM anon, authenticated",
    f"GRANT SELECT, INSERT, UPDATE, DELETE ON {', '.join(_TABLES)} TO app_tenant",
)

# Existing rows predate workspaces entirely. Without a backfill every org would have zero
# workspaces, and since the API only lists workspaces the caller is a member of, every
# existing user would see an empty list and lose access to their own documents and chats.
# The owner is the org's earliest 'owner' user, falling back to its earliest user.
_BACKFILL = (
    """
INSERT INTO workspaces (org_id, name, slug, owner_id)
SELECT
    o.id,
    'Default',
    'default',
    (
        SELECT u.id FROM users u
        WHERE u.org_id = o.id
        ORDER BY (u.role = 'owner') DESC, u.created_at ASC
        LIMIT 1
    )
FROM organizations o
""",
    # Everyone in the org joins its default workspace. Org owners/admins carry that
    # standing across; everyone else starts as an editor so existing upload and chat
    # behaviour keeps working — a viewer could not upload, which would be a regression.
    """
INSERT INTO workspace_members (workspace_id, user_id, org_id, role)
SELECT w.id, u.id, u.org_id,
       CASE WHEN u.role IN ('owner', 'admin') THEN u.role ELSE 'editor' END
FROM workspaces w
JOIN users u ON u.org_id = w.org_id
WHERE w.slug = 'default'
""",
    """
UPDATE documents d
SET workspace_id = w.id
FROM workspaces w
WHERE w.org_id = d.org_id AND w.slug = 'default' AND d.workspace_id IS NULL
""",
    """
UPDATE chat_sessions c
SET workspace_id = w.id
FROM workspaces w
WHERE w.org_id = c.org_id AND w.slug = 'default' AND c.workspace_id IS NULL
""",
)


# A new signup provisions an org and its owner user. Workspaces have to be created on the
# same path: otherwise every org created after this migration starts with none, and its
# owner sees an empty workspace list. Replaces the 0004 definition rather than adding a
# second trigger, so provisioning stays one function with one rule.
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
    existing_org uuid;
    new_org uuid;
    new_workspace uuid;
    display_name text;
    org_name text;
BEGIN
    SELECT org_id INTO existing_org FROM public.users WHERE id = auth_user_id;
    IF existing_org IS NOT NULL THEN
        RETURN existing_org;
    END IF;

    display_name := nullif(trim(auth_metadata ->> 'full_name'), '');
    org_name := nullif(trim(auth_metadata ->> 'org_name'), '');
    IF org_name IS NULL THEN
        org_name := coalesce(display_name, split_part(coalesce(auth_email, 'workspace'), '@', 1));
    END IF;
    org_name := left(org_name, 200);

    INSERT INTO public.organizations (name, slug)
    VALUES (org_name, app.unique_org_slug(org_name))
    RETURNING id INTO new_org;

    INSERT INTO public.users (id, org_id, email, full_name, role)
    VALUES (auth_user_id, new_org, auth_email, display_name, 'owner');

    INSERT INTO public.workspaces (org_id, name, slug, owner_id)
    VALUES (new_org, 'Default', 'default', auth_user_id)
    RETURNING id INTO new_workspace;

    INSERT INTO public.workspace_members (workspace_id, user_id, org_id, role)
    VALUES (new_workspace, auth_user_id, new_org, 'owner');

    RETURN new_org;
END
$$
"""

# The 0004 body, restored on downgrade so the function does not outlive the tables it
# writes to.
_PROVISION_FN_V4 = """
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
    existing_org uuid;
    new_org uuid;
    display_name text;
    org_name text;
BEGIN
    SELECT org_id INTO existing_org FROM public.users WHERE id = auth_user_id;
    IF existing_org IS NOT NULL THEN
        RETURN existing_org;
    END IF;

    display_name := nullif(trim(auth_metadata ->> 'full_name'), '');
    org_name := nullif(trim(auth_metadata ->> 'org_name'), '');
    IF org_name IS NULL THEN
        org_name := coalesce(display_name, split_part(coalesce(auth_email, 'workspace'), '@', 1));
    END IF;
    org_name := left(org_name, 200);

    INSERT INTO public.organizations (name, slug)
    VALUES (org_name, app.unique_org_slug(org_name))
    RETURNING id INTO new_org;

    INSERT INTO public.users (id, org_id, email, full_name, role)
    VALUES (auth_user_id, new_org, auth_email, display_name, 'owner');

    RETURN new_org;
END
$$
"""


def upgrade() -> None:
    # 1. Create workspaces table
    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True)),
        sa.Column("settings", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.Column("updated_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["org_id"], ["organizations.id"], name="fk_workspaces_org_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_workspaces_owner_id", ondelete="SET NULL"
        ),
        sa.UniqueConstraint("org_id", "slug", name="uq_workspaces_org_slug"),
    )
    op.create_index("ix_workspaces_org_id", "workspaces", ["org_id"])

    # 2. Create workspace_members table
    op.create_table(
        "workspace_members",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="viewer"),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_workspace_members_workspace_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_workspace_members_user_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["org_id"], ["organizations.id"], name="fk_workspace_members_org_id", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_members_workspace_user"),
        sa.CheckConstraint(
            "role IN ('owner', 'admin', 'editor', 'viewer')", name="ck_workspace_members_role"
        ),
    )
    op.create_index("ix_workspace_members_user_id", "workspace_members", ["user_id"])
    op.create_index("ix_workspace_members_org_id", "workspace_members", ["org_id"])

    # 3. Create conversation_summaries table
    op.create_table(
        "conversation_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("summary_text", sa.Text, nullable=False),
        sa.Column("message_range_start", sa.Integer, nullable=False),
        sa.Column("message_range_end", sa.Integer, nullable=False),
        sa.Column("token_count", sa.Integer),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["chat_sessions.id"],
            name="fk_conversation_summaries_session_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_conversation_summaries_session_id", "conversation_summaries", ["session_id"]
    )

    # 4 & 5. Add workspace_id columns
    op.add_column("documents", sa.Column("workspace_id", postgresql.UUID(as_uuid=True)))
    op.create_foreign_key(
        "fk_documents_workspace_id",
        "documents",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.add_column("chat_sessions", sa.Column("workspace_id", postgresql.UUID(as_uuid=True)))
    op.create_foreign_key(
        "fk_chat_sessions_workspace_id",
        "chat_sessions",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 6. Backfill before RLS is enabled — this migration runs as the privileged login
    # role, but keeping the data move ahead of the policies avoids depending on that.
    _execute_all(_BACKFILL)

    # 7 & 8. Enable RLS and Policies
    _execute_all(_ENABLE_RLS)
    _execute_all(_POLICIES)

    # 9. Grants and Revokes
    _execute_all(_GRANTS)

    # 10. Newly provisioned orgs get a default workspace from now on.
    op.execute(_PROVISION_FN.strip())


def downgrade() -> None:
    op.execute(_PROVISION_FN_V4.strip())
    op.execute(f"REVOKE ALL ON {', '.join(_TABLES)} FROM app_tenant")

    op.drop_constraint("fk_chat_sessions_workspace_id", "chat_sessions", type_="foreignkey")
    op.drop_column("chat_sessions", "workspace_id")

    op.drop_constraint("fk_documents_workspace_id", "documents", type_="foreignkey")
    op.drop_column("documents", "workspace_id")

    op.drop_table("conversation_summaries")
    op.drop_table("workspace_members")
    op.drop_table("workspaces")
