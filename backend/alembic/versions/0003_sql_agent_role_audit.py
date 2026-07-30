"""SQL agent: read-only role scoped to the allowlist, plus the query audit log.

Revision ID: 0003_sql_agent_role_audit
Revises: 0002_tenant_role_set

Two things are established here, both required by CLAUDE.md 4.3.

**`app_sql_agent`** is the role every generated query runs as. It holds *column-level*
`SELECT` on exactly the columns in `app/sql_agent/allowlist.py` and nothing else — no
write privilege on any table, no access to `users`, the chat tables, the chunk text, or
the withheld `documents` columns (`org_id`, `uploaded_by`, `storage_key`,
`error_message`). So even if the sqlglot validation layer were bypassed entirely, the
database itself refuses a `DROP TABLE`, an `UPDATE`, a `SELECT *`, or a read of another
table. It is NOBYPASSRLS, so org isolation continues to apply on top of that.

Column-level grants rather than a view: a view would need its own RLS handling and would
present the agent with a schema that no longer matches what the rest of the app sees,
which is how the schema tool and the grant drift apart.

**`sql_query_audit`** records every query the agent generates, accepted or rejected, with
the user who caused it. It is deliberately append-only from the application's side and
carries no SELECT policy or grant at all: an audit log a tenant can read is a list of
other people's questions, and one a tenant can amend is not an audit log. Reading it is an
operator action, performed with the administrative role.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003_sql_agent_role_audit"
down_revision: str | None = "0002_tenant_role_set"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP = sa.DateTime(timezone=True)
_NEW_UUID = sa.text("gen_random_uuid()")
_NOW = sa.text("now()")

SQL_AGENT_ROLE = "app_sql_agent"

#: Must stay identical to DOCUMENTS.columns in app/sql_agent/allowlist.py. The schema tool
#: describes this list to the model; this grant is what enforces it.
_DOCUMENTS_COLUMNS = (
    "id",
    "filename",
    "mime_type",
    "size_bytes",
    "status",
    "page_count",
    "created_at",
    "updated_at",
)

AUDIT_STATUSES = ("accepted", "rejected", "failed")

_ROLE_SETUP = (
    # noqa S608: role and column names are module constants defined above, never user
    # input, and `CREATE ROLE`/`GRANT` cannot accept bind parameters.
    f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{SQL_AGENT_ROLE}') THEN
        CREATE ROLE {SQL_AGENT_ROLE} NOLOGIN NOBYPASSRLS;
    END IF;
END
$$
""",  # noqa: S608
    # Schema visibility only. Without USAGE on `app` the RLS policies from migration 0001
    # cannot call their claim functions, and every query would fail rather than filter.
    f"GRANT USAGE ON SCHEMA app, public TO {SQL_AGENT_ROLE}",
    f"""
GRANT EXECUTE ON FUNCTION
    app.current_claims(), app.current_org_id(), app.current_user_id() TO {SQL_AGENT_ROLE}
""",
    # The whole read surface of the SQL agent, in one statement. `org_id` is withheld even
    # though RLS filters on it: the policy evaluates server-side and needs no grant, while
    # granting it would let a query group by an org the caller cannot read and so learn
    # which org_ids exist.
    f"""
GRANT SELECT ({", ".join(_DOCUMENTS_COLUMNS)}) ON documents TO {SQL_AGENT_ROLE}
""",
    # The application's login role must be able to SET ROLE to it. INHERIT FALSE so these
    # privileges are not silently held during ordinary requests.
    f"GRANT {SQL_AGENT_ROLE} TO CURRENT_USER WITH SET TRUE, INHERIT FALSE",
)

_AUDIT_POLICY = (
    "ALTER TABLE sql_query_audit ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE sql_query_audit FORCE ROW LEVEL SECURITY",
    # Supabase's default ACL for the public schema grants `anon` and `authenticated` full
    # DML on every new table, so this one arrived writable and readable by both. RLS
    # already denies those commands (no SELECT/UPDATE/DELETE policy exists, and neither
    # role has BYPASSRLS), but an append-only audit log should not depend on a policy
    # absence for its integrity — a future `FOR ALL` policy added for another purpose
    # would silently re-open it. Revoked so the grant and the intent agree.
    "REVOKE ALL ON sql_query_audit FROM anon, authenticated",
    # INSERT only, and only for the caller's own org and user. There is deliberately no
    # SELECT, UPDATE or DELETE policy: absent a policy, RLS denies the command outright.
    """
CREATE POLICY sql_query_audit_insert_own ON sql_query_audit
    FOR INSERT
    WITH CHECK (org_id = app.current_org_id() AND user_id = app.current_user_id())
""",
    # Matching grant. INSERT alone — a tenant role that could SELECT this table could read
    # every other user's questions, and the withheld columns quoted in their SQL.
    #
    # No SELECT even for the inserting role, which is why `app/sql_agent/audit.py` supplies
    # its own primary key: RETURNING requires SELECT, so letting the server default the id
    # would make every audit write fail on a permission error.
    "GRANT INSERT ON sql_query_audit TO app_tenant",
)


def _execute_all(statements: tuple[str, ...]) -> None:
    for statement in statements:
        op.execute(statement.strip())


def upgrade() -> None:
    op.create_table(
        "sql_query_audit",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=_NEW_UUID),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        # Null when the model produced nothing parseable at all; the rejection reason then
        # carries the detail.
        sa.Column("generated_sql", sa.Text),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("rejection_reason", sa.Text),
        sa.Column("row_count", sa.Integer),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("created_at", _TIMESTAMP, nullable=False, server_default=_NOW),
        # No FK to users: an audit row must outlive the account that caused it. A cascade
        # here would let deleting a user erase the record of what they ran.
        sa.ForeignKeyConstraint(
            ["org_id"], ["organizations.id"], name="fk_sql_audit_org", ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "status IN ('accepted', 'rejected', 'failed')", name="ck_sql_audit_status"
        ),
        sa.CheckConstraint(
            "row_count IS NULL OR row_count >= 0", name="ck_sql_audit_row_count_nonneg"
        ),
    )
    op.create_index("ix_sql_query_audit_org_created", "sql_query_audit", ["org_id", "created_at"])
    op.create_index("ix_sql_query_audit_user_created", "sql_query_audit", ["user_id", "created_at"])

    _execute_all(_ROLE_SETUP)
    _execute_all(_AUDIT_POLICY)


def downgrade() -> None:
    op.execute(f"REVOKE ALL ON documents FROM {SQL_AGENT_ROLE}")
    op.execute(f"REVOKE ALL ON SCHEMA app, public FROM {SQL_AGENT_ROLE}")
    op.execute(
        "REVOKE ALL ON FUNCTION app.current_claims(), app.current_org_id(), "
        f"app.current_user_id() FROM {SQL_AGENT_ROLE}"
    )
    op.execute(f"REVOKE {SQL_AGENT_ROLE} FROM CURRENT_USER")
    op.drop_table("sql_query_audit")
    # The role itself is left in place, matching 0001's treatment of app_tenant: dropping a
    # role fails if any object in any database on the cluster still depends on it.
