"""Add 4-organization-per-user creation limit.

The limit is enforced inside ``app.create_workspace()`` so it runs in the same
transaction as the INSERT — concurrent requests cannot bypass it.

Revision ID: 0012_org_creation_limit
Revises: 0011_add_document_description
"""

from alembic import op

revision = "0012_org_creation_limit"
down_revision = "0011_add_document_description"
branch_labels = None
depends_on = None

_CREATE_WORKSPACE_LIMITED = """\
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

_CREATE_WORKSPACE_ORIGINAL = """\
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
"""


def upgrade() -> None:
    op.execute(_CREATE_WORKSPACE_LIMITED)


def downgrade() -> None:
    op.execute(_CREATE_WORKSPACE_ORIGINAL)
