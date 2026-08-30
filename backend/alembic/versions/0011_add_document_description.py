"""Add description column to documents for auto-generated summaries.

Revision ID: 0011_add_document_description
Revises: 0010_phase2_accept_invitation
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0011_add_document_description"
down_revision = "0010_phase2_accept_invitation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("description", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "description")
