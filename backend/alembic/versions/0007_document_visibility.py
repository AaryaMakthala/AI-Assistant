"""Document visibility: org-wide versus personal uploads.

Revision ID: 0007_document_visibility
Revises: 0006_workspaces

Owner-uploaded files are readable by the whole organization; a member's uploads are
readable only by that member and by owners/admins (CLAUDE.md 4.6).

**Ordering in `upgrade()` is load-bearing.** `ADD COLUMN ... DEFAULT 'personal'` would apply
that default to existing rows as well, which would hide every document uploaded before this
migration from everyone but one backfilled owner. The column is therefore added nullable,
existing rows are set to 'org' (they predate the distinction and were always org-wide), and
only then does the default arrive for future inserts.

**On the ingestion claim.** `app/workers/ingestion.py` acts for an organization, not a
person, so it sets claims with `org_id` alone and `app.current_user_id()` is NULL inside it.
Under a policy requiring `uploaded_by = app.current_user_id()` a personal document matches no
branch: the worker's SELECT returns nothing and its status UPDATE silently affects zero rows,
leaving the document `pending` forever while the reaper re-queues it on every tick. The
policies below therefore admit an explicit `svc` claim. It cannot be forged from a token —
`set_tenant_claims` builds the claims object from typed keyword arguments, and
`principal_from_claims` only ever reads `sub`, `org_id`, `org_role` and `email`.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007_document_visibility"
down_revision: str | None = "0006_workspaces"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def _execute_all(statements: tuple[str, ...]) -> None:
    for statement in statements:
        op.execute(statement.strip())


# Role is read from `users`, not from a claim. Supabase's own tokens carry a top-level `role`
# claim meaning the *database* role ('authenticated'), which would collide with the org role
# this application keeps in `org_role` — and a token-asserted role is never checked against
# the users table anyway. STABLE, not IMMUTABLE: it reads a table.
_ORG_ROLE_FN = """
CREATE OR REPLACE FUNCTION app.current_org_role()
RETURNS text
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT u.role
    FROM public.users u
    WHERE u.id = app.current_user_id()
      AND u.org_id = app.current_org_id()
$$
"""

# The worker's identity. A separate function so the policies read declaratively and the
# claim name lives in exactly one place.
_IS_INGESTION_FN = """
CREATE OR REPLACE FUNCTION app.is_ingestion_worker()
RETURNS boolean
LANGUAGE sql
STABLE
SET search_path = pg_catalog
AS $$
    SELECT app.current_claims() ->> 'svc' = 'ingestion'
$$
"""

_GRANT_FNS = """
GRANT EXECUTE ON FUNCTION
    app.current_org_role(), app.is_ingestion_worker() TO app_tenant
"""

# `SECURITY DEFINER` on current_org_role() means it reads `users` as the migration's owner,
# so the SQL agent does not need its own grant on that table to evaluate the policy.
_GRANT_FNS_SQL_AGENT = """
GRANT EXECUTE ON FUNCTION
    app.current_org_role(), app.is_ingestion_worker() TO app_sql_agent
"""

# Existing rows predate the distinction: they were uploaded when every document was visible
# org-wide, so marking them 'org' preserves exactly the access people already had. Doing this
# before the default is applied is the whole point of the ordering.
_BACKFILL = (
    "UPDATE documents SET visibility = 'org' WHERE visibility IS NULL",
    # `uploaded_by` is about to become NOT NULL but is nullable today (and was set to NULL by
    # the old ON DELETE SET NULL). Attribute orphans to the org's owner — the same
    # owner-election used by 0006's workspace backfill.
    """
UPDATE documents d
SET uploaded_by = (
    SELECT u.id FROM users u
    WHERE u.org_id = d.org_id
    ORDER BY (u.role = 'owner') DESC, u.created_at ASC
    LIMIT 1
)
WHERE d.uploaded_by IS NULL
""",
    # An org with no users at all would leave the column NULL and abort the SET NOT NULL
    # below. Such a row is unreachable — no one can authenticate into that org — so deleting
    # it is the honest outcome. Chunks follow via ON DELETE CASCADE.
    "DELETE FROM documents WHERE uploaded_by IS NULL",
)

_CONSTRAINTS = (
    "ALTER TABLE documents ALTER COLUMN visibility SET NOT NULL",
    "ALTER TABLE documents ALTER COLUMN visibility SET DEFAULT 'personal'",
    """
ALTER TABLE documents
    ADD CONSTRAINT ck_documents_visibility CHECK (visibility IN ('org', 'personal'))
""",
    "ALTER TABLE documents ALTER COLUMN uploaded_by SET NOT NULL",
)

# ON DELETE SET NULL is incompatible with NOT NULL — deleting a user would violate the
# constraint. CASCADE is wrong too: the bytes on disk are removed by _unlink_stored_file in
# the API, so a database cascade would delete rows and orphan every file. Deferred NO ACTION
# lets an org deletion cascade documents and users within one statement without the check
# firing mid-way, while still refusing to leave a document pointing at a deleted user.
_FK_SWAP = (
    "ALTER TABLE documents DROP CONSTRAINT fk_documents_uploaded_by",
    """
ALTER TABLE documents
    ADD CONSTRAINT fk_documents_uploaded_by
    FOREIGN KEY (uploaded_by) REFERENCES users(id)
    DEFERRABLE INITIALLY DEFERRED
""",
)

_FK_SWAP_DOWN = (
    "ALTER TABLE documents DROP CONSTRAINT fk_documents_uploaded_by",
    """
ALTER TABLE documents
    ADD CONSTRAINT fk_documents_uploaded_by
    FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL
""",
)

# The single FOR ALL policy from 0001 is replaced by per-command policies: reading is scoped
# by visibility, while writing stays org-scoped plus the ingestion claim. Splitting them is
# what lets a member update their own document without also being able to read a colleague's.
_POLICIES = (
    "DROP POLICY IF EXISTS documents_org_isolation ON documents",
    """
CREATE POLICY documents_select ON documents
    FOR SELECT
    USING (
        org_id = app.current_org_id()
        AND (
            visibility = 'org'
            OR uploaded_by = app.current_user_id()
            OR app.current_org_role() IN ('owner', 'admin')
            OR app.is_ingestion_worker()
        )
    )
""",
    # Only an owner/admin may create an org-wide document. The API returns 403 for this case
    # rather than downgrading silently; this policy is the half that cannot be forgotten.
    """
CREATE POLICY documents_insert ON documents
    FOR INSERT
    WITH CHECK (
        org_id = app.current_org_id()
        AND (
            (
                uploaded_by = app.current_user_id()
                AND (
                    visibility = 'personal'
                    OR app.current_org_role() IN ('owner', 'admin')
                )
            )
            OR app.is_ingestion_worker()
        )
    )
""",
    # The worker branch is what allows status transitions on a personal document. USING and
    # WITH CHECK both carry it: USING selects the row to update, WITH CHECK validates the
    # result, and omitting either would fail the write in a different place.
    """
CREATE POLICY documents_update ON documents
    FOR UPDATE
    USING (
        org_id = app.current_org_id()
        AND (
            uploaded_by = app.current_user_id()
            OR app.current_org_role() IN ('owner', 'admin')
            OR app.is_ingestion_worker()
        )
    )
    WITH CHECK (
        org_id = app.current_org_id()
        AND (
            uploaded_by = app.current_user_id()
            OR app.current_org_role() IN ('owner', 'admin')
            OR app.is_ingestion_worker()
        )
    )
""",
    # Deliberately narrower than SELECT: a member may read an org-wide document but may not
    # delete one they did not upload.
    """
CREATE POLICY documents_delete ON documents
    FOR DELETE
    USING (
        org_id = app.current_org_id()
        AND (
            uploaded_by = app.current_user_id()
            OR app.current_org_role() IN ('owner', 'admin')
        )
    )
""",
    # Chunks carry the document's text, so they are exactly as sensitive as the document and
    # must not rely on callers remembering to join. Mirrored in the policy as well as at
    # query time (CLAUDE.md 4.6) — read_document in the MCP server queries chunks directly.
    "DROP POLICY IF EXISTS document_chunks_org_isolation ON document_chunks",
    """
CREATE POLICY document_chunks_select ON document_chunks
    FOR SELECT
    USING (
        org_id = app.current_org_id()
        AND EXISTS (
            SELECT 1 FROM documents d
            WHERE d.id = document_chunks.document_id
              AND d.org_id = app.current_org_id()
              AND (
                  d.visibility = 'org'
                  OR d.uploaded_by = app.current_user_id()
                  OR app.current_org_role() IN ('owner', 'admin')
                  OR app.is_ingestion_worker()
              )
        )
    )
""",
    # Writes are the worker's job; a chunk is never authored by a user directly.
    """
CREATE POLICY document_chunks_write ON document_chunks
    FOR ALL
    USING (
        org_id = app.current_org_id()
        AND EXISTS (
            SELECT 1 FROM documents d
            WHERE d.id = document_chunks.document_id
              AND d.org_id = app.current_org_id()
              AND (
                  d.uploaded_by = app.current_user_id()
                  OR app.current_org_role() IN ('owner', 'admin')
                  OR app.is_ingestion_worker()
              )
        )
    )
    WITH CHECK (
        org_id = app.current_org_id()
        AND EXISTS (
            SELECT 1 FROM documents d
            WHERE d.id = document_chunks.document_id
              AND d.org_id = app.current_org_id()
              AND (
                  d.uploaded_by = app.current_user_id()
                  OR app.current_org_role() IN ('owner', 'admin')
                  OR app.is_ingestion_worker()
              )
        )
    )
""",
)

_POLICIES_DOWN = (
    "DROP POLICY IF EXISTS documents_select ON documents",
    "DROP POLICY IF EXISTS documents_insert ON documents",
    "DROP POLICY IF EXISTS documents_update ON documents",
    "DROP POLICY IF EXISTS documents_delete ON documents",
    "DROP POLICY IF EXISTS document_chunks_select ON document_chunks",
    "DROP POLICY IF EXISTS document_chunks_write ON document_chunks",
    """
CREATE POLICY documents_org_isolation ON documents
    FOR ALL
    USING (org_id = app.current_org_id())
    WITH CHECK (org_id = app.current_org_id())
""",
    """
CREATE POLICY document_chunks_org_isolation ON document_chunks
    FOR ALL
    USING (org_id = app.current_org_id())
    WITH CHECK (org_id = app.current_org_id())
""",
)

#: Kept in step with _DOCUMENTS_COLUMNS in 0003 and DOCUMENTS.columns in
#: app/sql_agent/allowlist.py. The schema tool describes the column; this grant is what
#: actually permits reading it.
_GRANT_VISIBILITY = "GRANT SELECT (visibility) ON documents TO app_sql_agent"


def upgrade() -> None:
    # 1. Nullable and defaultless, so existing rows are untouched for now.
    op.add_column("documents", sa.Column("visibility", sa.String(20)))

    # 2. Claim helpers the policies depend on, before any policy references them.
    op.execute(_ORG_ROLE_FN.strip())
    op.execute(_IS_INGESTION_FN.strip())
    op.execute(_GRANT_FNS.strip())
    op.execute(_GRANT_FNS_SQL_AGENT.strip())

    # 3. Backfill: existing documents become org-wide, orphaned uploads get an owner.
    _execute_all(_BACKFILL)

    # 4. Only now are NOT NULL, the default and the check safe to apply.
    _execute_all(_CONSTRAINTS)
    _execute_all(_FK_SWAP)

    # 5. Replace the org-only policies with visibility-aware ones.
    _execute_all(_POLICIES)

    op.execute(_GRANT_VISIBILITY)
    op.create_index(
        "ix_documents_org_visibility_uploaded_by",
        "documents",
        ["org_id", "visibility", "uploaded_by"],
    )


def downgrade() -> None:
    op.drop_index("ix_documents_org_visibility_uploaded_by", table_name="documents")
    op.execute("REVOKE SELECT (visibility) ON documents FROM app_sql_agent")

    _execute_all(_POLICIES_DOWN)
    _execute_all(_FK_SWAP_DOWN)

    # uploaded_by stays populated; only the constraint is lifted, so a downgrade loses no
    # attribution.
    op.execute("ALTER TABLE documents ALTER COLUMN uploaded_by DROP NOT NULL")
    op.execute("ALTER TABLE documents DROP CONSTRAINT ck_documents_visibility")
    op.drop_column("documents", "visibility")

    op.execute("DROP FUNCTION IF EXISTS app.is_ingestion_worker()")
    op.execute("DROP FUNCTION IF EXISTS app.current_org_role()")
