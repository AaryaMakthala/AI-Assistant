"""Phase 2: invitation acceptance via a SECURITY DEFINER function.

Revision ID: 0010_phase2_accept_invitation
Revises: 0009_phase1c_finalize_bytes

**Why a definer function is required.** Migration 0008's RLS policies gate every
tenant-facing operation on membership — ``invitations_member_read`` only admits a
reader who is already a member of the invitation's workspace, and ``members_insert``
requires ``app.is_workspace_owner()``. The accepting user is, by definition, *not yet*
a member and *not yet* the owner, so neither the read nor the membership insert can
ever pass under RLS. That is not a policy bug to paper over with a weaker policy; it is
the same structural gap 0008 already solved for workspace creation, which has no INSERT
policy at all and is reachable only through the SECURITY DEFINER
``app.create_workspace()``. This migration follows that exact precedent:
``app.accept_invitation()`` runs as the table owner, validates the invitation, and
creates the membership — atomically — without weakening any policy.

**Security boundaries** (mirrors ``app.create_workspace()``):

* ``SECURITY DEFINER`` with ``SET search_path = pg_catalog, public`` — the same
  hardening 0008 applies; tables are referenced schema-qualified so nothing outside
  ``pg_catalog``/``public`` can be resolved during execution.
* The invoking *identity* (``sub``) is read from the RLS claims via
  ``app.current_user_id()`` — never taken as a parameter, so a caller cannot name a
  different user. The endpoint may only pass the email from the caller's own verified
  JWT; the function cross-checks it against the invitation.
* Validation is exhaustive inside the function: invitation exists, is PENDING, the
  email matches, and no membership row already exists for (workspace, user) — the
  last check exists because ``uq_members_workspace_user`` would otherwise reject the
  insert with a bare constraint violation.
* The membership is written exactly as ``OWNER``-accepted: ``role = 'MEMBER'``,
  ``status = 'ACTIVE'`` (canonical enums from ``app/db/models.py``), and the
  invitation flips to ``ACCEPTED`` — both in the same transaction.
* ``REVOKE ... FROM PUBLIC`` / ``GRANT ... TO app_tenant`` matches 0008's grant
  pattern: the function is callable only by the NOBYPASSRLS application role, and
  policies are untouched — no policy is created, modified, or dropped.

Failures raise with distinct SQLSTATEs so the API layer can map them to HTTP statuses
without parsing message text: ``P0002`` not found, ``W1001`` missing ``sub`` claim,
``W1002`` invitation not PENDING, ``W1003`` email mismatch, ``W1004`` duplicate
membership.

One-way consideration: this is additive and has a simple, supported downgrade (drop
the function). Unlike 0008 there is no data to lose.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0010_phase2_accept_invitation"
down_revision: str | None = "0009_phase1c_finalize_bytes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FUNCTION = """
CREATE OR REPLACE FUNCTION app.accept_invitation(p_invitation_id uuid, p_email text)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_invitation public.invitations%ROWTYPE;
    v_user_id uuid;
    v_member_id uuid;
BEGIN
    -- The invoking identity comes from the RLS claims (sub), never from a parameter.
    v_user_id := app.current_user_id();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'accept_invitation requires an authenticated sub claim'
            USING ERRCODE = 'W1001';
    END IF;

    SELECT * INTO v_invitation
    FROM public.invitations
    WHERE id = p_invitation_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'invitation not found'
            USING ERRCODE = 'P0002';
    END IF;

    IF v_invitation.status <> 'PENDING' THEN
        RAISE EXCEPTION 'invitation is %, not PENDING', v_invitation.status
            USING ERRCODE = 'W1002';
    END IF;

    -- The email arrives from the caller's verified JWT; it must be the invitation's
    -- target address. NULL never matches (no silent bypass for email-less tokens).
    IF p_email IS NULL OR lower(p_email) <> lower(v_invitation.email) THEN
        RAISE EXCEPTION 'email does not match this invitation'
            USING ERRCODE = 'W1003';
    END IF;

    -- Any existing membership row (ACTIVE, INVITED or REMOVED) would violate
    -- uq_members_workspace_user, so the check is on existence, not just ACTIVE.
    IF EXISTS (
        SELECT 1 FROM public.members
        WHERE workspace_id = v_invitation.workspace_id AND user_id = v_user_id
    ) THEN
        RAISE EXCEPTION 'you are already a member of this workspace'
            USING ERRCODE = 'W1004';
    END IF;

    INSERT INTO public.members (workspace_id, user_id, role, status)
    VALUES (v_invitation.workspace_id, v_user_id, 'MEMBER', 'ACTIVE')
    RETURNING id INTO v_member_id;

    UPDATE public.invitations
    SET status = 'ACCEPTED'
    WHERE id = p_invitation_id;

    RETURN v_member_id;
END
$$
"""


def upgrade() -> None:
    op.execute(_FUNCTION.strip())
    # New functions default to PUBLIC EXECUTE; the definer writes data, so it must be
    # callable only by app_tenant — the same pattern 0008 applies to its helpers.
    op.execute("REVOKE ALL ON FUNCTION app.accept_invitation(uuid, text) FROM PUBLIC")
    op.execute("GRANT EXECUTE ON FUNCTION app.accept_invitation(uuid, text) TO app_tenant")


def downgrade() -> None:
    op.execute("REVOKE ALL ON FUNCTION app.accept_invitation(uuid, text) FROM app_tenant")
    op.execute("DROP FUNCTION IF EXISTS app.accept_invitation(uuid, text)")
