"""Let the application role drop into app_tenant so RLS actually applies.

Revision ID: 0002_tenant_role_set
Revises: 0001_initial_schema

Managed Postgres providers hand out an administrative login role — on Supabase that is
`postgres`, which carries BYPASSRLS. A connection using it ignores every policy from
migration 0001, FORCE included, so the database half of the defense in CLAUDE.md 4.6 is
inert exactly where it matters most.

Rather than depend on every deployment provisioning a separate login role correctly, the
application switches to `app_tenant` for the duration of each tenant transaction. That
role is NOBYPASSRLS, so the policies bind. This migration grants the membership needed to
make that switch, with SET TRUE (permission to `SET ROLE`) and INHERIT FALSE (so the
privileges are not silently absorbed the rest of the time).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_tenant_role_set"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # CURRENT_USER, not a hardcoded name: the login role differs by provider, and this
    # migration runs as whichever role the application is configured to connect with.
    op.execute("GRANT app_tenant TO CURRENT_USER WITH SET TRUE, INHERIT FALSE")


def downgrade() -> None:
    op.execute("REVOKE app_tenant FROM CURRENT_USER")
