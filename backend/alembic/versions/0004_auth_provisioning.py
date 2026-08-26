"""Give every Supabase signup an organization, a users row, and the claims to prove it.

Revision ID: 0004_auth_provisioning
Revises: 0003_sql_agent_role_audit

Before this migration the two halves of identity never met. Supabase Auth would happily
register a user and mint a token, but nothing created the `organizations` and `users` rows
the application is built around, and nothing wrote `org_id` into the token. The result was
a signup that succeeded and then failed every subsequent request with "Token is missing a
valid user or organization claim" — `principal_from_claims` has no org to derive.

Provisioning runs as an `AFTER INSERT` trigger on `auth.users` rather than in application
code, for three reasons:

* It cannot be skipped. Signup happens inside Supabase's auth service, not in this
  codebase, so any application-side hook would only cover the paths this repo knows about
  — a magic-link login, an OAuth provider or an admin-created user would each bypass it.
* There is no window in which a valid token exists without claims. The metadata is written
  in the same transaction as the auth row, so the first token ever minted already carries
  `org_id`.
* `app_tenant` holds `SELECT` only on `organizations` and `users` (migration 0001, where
  the comment notes provisioning "happens out of band"). Insert privilege has to come from
  somewhere outside the tenant path, and `SECURITY DEFINER` is that somewhere.

**On SECURITY DEFINER.** The function runs with the privileges of its owner, so its body
is a privilege boundary and is written like one: `search_path` is pinned so a caller
cannot resolve `organizations` to something they control, and every value it writes comes
from the `auth.users` row being inserted — never from a claim, a parameter, or anything a
client can set. The one thing a signup *could* influence is the org name, which is read
from `raw_user_meta_data` (user-controlled) and therefore only ever treated as a display
string: it is trimmed, length-capped, and never used to pick an existing org.

**Every new user gets their own organization, as its owner.** That is the conservative
reading — joining an existing org is an authorization decision, and inferring it from
something a signup controls (an email domain, say) would let anyone with a matching
address into someone else's data. Invitations are a later, deliberate feature; until then
the only way into an existing org is an operator action.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_auth_provisioning"
down_revision: str | None = "0003_sql_agent_role_audit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Derives a unique, URL-safe slug from a display name. Called by the trigger below and by
# the backfill, which is the only reason it is a separate function.
_SLUGIFY = """
CREATE OR REPLACE FUNCTION app.unique_org_slug(display_name text)
RETURNS text
LANGUAGE plpgsql
VOLATILE
SET search_path = pg_catalog, public
AS $$
DECLARE
    base_slug text;
    candidate text;
    suffix integer := 0;
BEGIN
    base_slug := lower(regexp_replace(coalesce(display_name, ''), '[^a-zA-Z0-9]+', '-', 'g'));
    base_slug := trim(both '-' from base_slug);
    base_slug := left(nullif(base_slug, ''), 80);
    IF base_slug IS NULL THEN
        base_slug := 'org';
    END IF;

    candidate := base_slug;
    -- organizations.slug is UNIQUE. Two people signing up with the same company name is
    -- ordinary, not exceptional, so the collision is resolved rather than raised.
    WHILE EXISTS (SELECT 1 FROM public.organizations WHERE slug = candidate) LOOP
        suffix := suffix + 1;
        candidate := left(base_slug, 80 - length(suffix::text) - 1) || '-' || suffix::text;
    END LOOP;

    RETURN candidate;
END
$$
"""


# The provisioning body, shared by the trigger and the backfill so the two cannot drift.
# Idempotent on `users.id`: re-running it for an existing user is a no-op that still
# repairs missing claims.
_PROVISION = """
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

    -- User-supplied, so treated as a display string and nothing more: trimmed, capped,
    -- and never used to look up or join an existing organization.
    display_name := nullif(trim(auth_metadata ->> 'full_name'), '');
    org_name := nullif(trim(auth_metadata ->> 'org_name'), '');
    IF org_name IS NULL THEN
        org_name := coalesce(display_name, split_part(coalesce(auth_email, 'workspace'), '@', 1));
    END IF;
    org_name := left(org_name, 200);

    INSERT INTO public.organizations (name, slug)
    VALUES (org_name, app.unique_org_slug(org_name))
    RETURNING id INTO new_org;

    -- id is the JWT `sub`: models.py keeps the two in step deliberately, with no FK to the
    -- Supabase-managed auth schema.
    INSERT INTO public.users (id, org_id, email, full_name, role)
    VALUES (auth_user_id, new_org, auth_email, display_name, 'owner');

    RETURN new_org;
END
$$
"""


# What actually puts the claims in the token. Supabase copies `raw_app_meta_data` into
# every access token it mints; `raw_user_meta_data` is excluded on purpose, because the
# user can edit it and auth.py reads `app_metadata` only.
_TRIGGER_FN = """
CREATE OR REPLACE FUNCTION app.handle_new_auth_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    org uuid;
BEGIN
    org := app.provision_auth_user(
        NEW.id, NEW.email, coalesce(NEW.raw_user_meta_data, '{}'::jsonb)
    );

    UPDATE auth.users
    SET raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb)
        || jsonb_build_object('org_id', org::text, 'org_role', 'owner')
    WHERE id = NEW.id;

    RETURN NEW;
END
$$
"""


# Repairs accounts that predate this migration, and any signup that happened while the
# trigger was absent. Same function as the live path, so there is one provisioning rule.
_BACKFILL = """
DO $$
DECLARE
    account record;
    org uuid;
BEGIN
    FOR account IN SELECT id, email, raw_user_meta_data FROM auth.users LOOP
        org := app.provision_auth_user(
            account.id, account.email, coalesce(account.raw_user_meta_data, '{}'::jsonb)
        );
        UPDATE auth.users
        SET raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb)
            || jsonb_build_object('org_id', org::text, 'org_role', 'owner')
        WHERE id = account.id
          AND coalesce(raw_app_meta_data ->> 'org_id', '') IS DISTINCT FROM org::text;
    END LOOP;
END
$$
"""


def upgrade() -> None:
    op.execute(_SLUGIFY)
    op.execute(_PROVISION)
    op.execute(_TRIGGER_FN)

    # The functions stay owned by the migration role, which owns `organizations`/`users`
    # and already holds UPDATE on `auth.users`. That is exactly the privilege set the body
    # needs and no more — reassigning ownership to `supabase_auth_admin` would both widen
    # it and require a SET ROLE this role does not have.
    #
    # PUBLIC gets EXECUTE on new functions by default; on a SECURITY DEFINER function that
    # is an open door to inserting organizations, so it is closed explicitly. The trigger
    # itself still fires — trigger execution is not an EXECUTE privilege check.
    op.execute("REVOKE ALL ON FUNCTION app.provision_auth_user(uuid, text, jsonb) FROM PUBLIC")
    op.execute("REVOKE ALL ON FUNCTION app.handle_new_auth_user() FROM PUBLIC")
    op.execute("REVOKE ALL ON FUNCTION app.unique_org_slug(text) FROM PUBLIC")

    op.execute("DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users")
    op.execute(
        """
        CREATE TRIGGER on_auth_user_created
        AFTER INSERT ON auth.users
        FOR EACH ROW EXECUTE FUNCTION app.handle_new_auth_user()
        """
    )

    op.execute(_BACKFILL)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users")
    op.execute("DROP FUNCTION IF EXISTS app.handle_new_auth_user()")
    op.execute("DROP FUNCTION IF EXISTS app.provision_auth_user(uuid, text, jsonb)")
    op.execute("DROP FUNCTION IF EXISTS app.unique_org_slug(text)")
