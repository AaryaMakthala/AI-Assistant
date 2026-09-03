"""Add DELETE policy on workspaces so owners can delete their organization.

Without this policy, RLS blocks DELETE operations on the workspaces table,
making the delete-workspace endpoint return 404 even for valid owners.

Revision ID: 0016_add_workspace_delete_policy
Revises: 0015_revert_to_one_org_per_email
"""

from alembic import op

revision = "0016_add_workspace_delete_policy"
down_revision = "0015_revert_to_one_org_per_email"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE POLICY workspaces_delete ON workspaces FOR DELETE
        USING (app.is_workspace_owner(workspaces.id))
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS workspaces_delete ON workspaces")
