"""Organization-level verification and multi-organization support.

This migration:
1. Adds verification_token and verified_at columns to workspaces for
   organization-level email verification (not user-level Supabase verification).
2. Replaces the one-organization-per-email limit with a duplicate-name check:
   owners can create multiple organizations, but not with duplicate names.
3. Adds a case-insensitive unique constraint on (owner_id, normalized_name)
   to prevent duplicate organization names at the database level.

Revision ID: 0014_workspace_verification
Revises: 0013_one_org_per_email
"""

from alembic import op
import sqlalchemy as sa

revision = "0014_workspace_verification"
down_revision = "0013_one_org_per_email"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add verification columns to workspaces.
    op.add_column(
        "workspaces",
        sa.Column("verification_token", sa.String(64), nullable=True),
    )
    op.add_column(
        "workspaces",
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workspaces",
        sa.Column("normalized_name", sa.String(200), nullable=True),
    )

    # 2. Backfill normalized_name for existing workspaces.
    op.execute("""
        UPDATE workspaces
        SET normalized_name = lower(trim(name))
        WHERE normalized_name IS NULL
    """)

    # 3. Make normalized_name NOT NULL after backfill.
    op.alter_column("workspaces", "normalized_name", nullable=False)

    # 4. Add unique constraint: one owner cannot have two orgs with the same name.
    op.create_unique_constraint(
        "uq_workspaces_owner_normalized_name",
        "workspaces",
        ["owner_id", "normalized_name"],
    )

    # 5. Replace the create_workspace function: remove one-org limit,
    #    add duplicate-name check, and set normalized_name.
    op.execute(_CREATE_WORKSPACE_MULTI_ORG)


_CREATE_WORKSPACE_MULTI_ORG = """\
CREATE OR REPLACE FUNCTION app.create_workspace(p_name text)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    new_ws uuid;
    owner_id uuid;
    norm_name text;
BEGIN
    owner_id := app.current_user_id();
    IF owner_id IS NULL THEN
        RAISE EXCEPTION 'app.create_workspace requires an authenticated sub claim';
    END IF;

    -- Normalize name: trim and lowercase for duplicate detection.
    norm_name := lower(trim(p_name));

    -- Check if the owner already has an organization with this normalized name.
    IF EXISTS (
        SELECT 1 FROM public.workspaces
        WHERE owner_id = app.current_user_id()
          AND normalized_name = norm_name
    ) THEN
        RAISE EXCEPTION 'This email already has an organization with this name. Please try a different organization name.'
            USING ERRCODE = 'W1007';
    END IF;

    INSERT INTO public.workspaces (name, owner_id, normalized_name)
    VALUES (left(p_name, 200), owner_id, norm_name)
    RETURNING id INTO new_ws;

    INSERT INTO public.members (workspace_id, user_id, role, status)
    VALUES (new_ws, owner_id, 'OWNER', 'ACTIVE');

    RETURN new_ws;
END
$$
"""


def downgrade() -> None:
    # Restore the one-org-per-email limit from migration 0013.
    op.execute(_CREATE_WORKSPACE_ONE_ORG)

    op.drop_constraint(
        "uq_workspaces_owner_normalized_name",
        "workspaces",
        type_="unique",
    )
    op.drop_column("workspaces", "normalized_name")
    op.drop_column("workspaces", "verified_at")
    op.drop_column("workspaces", "verification_token")


_CREATE_WORKSPACE_ONE_ORG = """\
CREATE OR REPLACE FUNCTION app.create_workspace(p_name text)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    new_ws uuid;
    owner_id uuid;
    owner_ws_count integer;
BEGIN
    owner_id := app.current_user_id();
    IF owner_id IS NULL THEN
        RAISE EXCEPTION 'app.create_workspace requires an authenticated sub claim';
    END IF;

    SELECT count(*) INTO owner_ws_count
    FROM public.members
    WHERE user_id = owner_id
      AND role = 'OWNER'
      AND status = 'ACTIVE';

    IF owner_ws_count >= 1 THEN
        RAISE EXCEPTION 'This email is already registered with an organization. Please try again with a different email.'
            USING ERRCODE = 'W1006';
    END IF;

    INSERT INTO public.workspaces (name, owner_id)
    VALUES (left(p_name, 200), owner_id)
    RETURNING id INTO new_ws;

    INSERT INTO public.members (workspace_id, user_id, role, status)
    VALUES (new_ws, owner_id, 'OWNER', 'ACTIVE');

    RETURN new_ws;
END
$$
"""
