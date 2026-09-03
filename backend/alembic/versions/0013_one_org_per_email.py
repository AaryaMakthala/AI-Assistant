"""Replace 4-organization limit with one-organization-per-email limit.

Each user (each email) may own exactly one organization. A user who already owns
an organization is blocked from creating another with a clear error message.

Revision ID: 0013_one_org_per_email
Revises: 0012_org_creation_limit
"""

from alembic import op

revision = "0013_one_org_per_email"
down_revision = "0012_org_creation_limit"
branch_labels = None
depends_on = None

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

    -- Count workspaces this user already OWNS (not just membership).
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

# Downgrade: restore the 4-org limit from migration 0012.
_CREATE_WORKSPACE_FOUR_ORG_LIMIT = """\
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

    -- Count workspaces this user already OWNS (not just membership).
    SELECT count(*) INTO owner_ws_count
    FROM public.members
    WHERE user_id = owner_id
      AND role = 'OWNER'
      AND status = 'ACTIVE';

    IF owner_ws_count >= 4 THEN
        RAISE EXCEPTION 'You can create a maximum of 4 organizations per account.'
            USING ERRCODE = 'W1005';
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


def upgrade() -> None:
    op.execute(_CREATE_WORKSPACE_ONE_ORG)


def downgrade() -> None:
    op.execute(_CREATE_WORKSPACE_FOUR_ORG_LIMIT)
