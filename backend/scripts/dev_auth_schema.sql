-- DEV/CI-ONLY bootstrap for a plain local Postgres. Do NOT run this on Supabase —
-- Supabase already provides the auth schema, auth.users, and the anon/authenticated
-- roles. Idempotent: safe to re-run.
--
-- Why it exists: the canonical schema (CLAUDE.md section 7) declares foreign keys to
-- Supabase's auth.users, and the auth-provisioning triggers (migrations 0004/0006 and
-- the Phase 1C replacement) write to it. A local docker-compose Postgres has none of
-- that, so migrations 0001..0009 cannot apply until this runs.
--
-- docker-compose mounts this into /docker-entrypoint-initdb.d so a fresh volume gets
-- it automatically; for an existing volume, apply it once manually:
--   psql "$DATABASE_URL" -f backend/scripts/dev_auth_schema.sql

CREATE SCHEMA IF NOT EXISTS auth;

CREATE TABLE IF NOT EXISTS auth.users (
    id                 uuid PRIMARY KEY,
    email              text,
    raw_user_meta_data jsonb,
    raw_app_meta_data  jsonb,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now()
);

-- Migrations 0003/0005/0006 and Phase 1C 0008 REVOKE privileges from these two roles,
-- which only exist on Supabase. Without them the legacy migrations fail on plain PG.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        CREATE ROLE anon NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        CREATE ROLE authenticated NOLOGIN;
    END IF;
END
$$;
