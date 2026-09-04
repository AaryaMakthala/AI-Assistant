"""JWT verification: identity comes from a verified token and nowhere else (4.6).

Phase 2: the Principal carries `user_id` and `workspace_id`, and the workspace *role*
(OWNER/MEMBER) is deliberately NOT read from the token — it comes from the `members`
table at request time (CLAUDE.md section 4). These tests pin the first half (verified
claims -> Principal) and the second half (a role claim in the token is ignored).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import pytest
from fastapi import HTTPException

from app.config import get_settings
from app.security.auth import JWT_AUDIENCE, decode_token, principal_from_claims

pytestmark = pytest.mark.usefixtures("valid_env")


def _token(claims: dict[str, Any], *, secret: str | None = None, expires_in: int = 3600) -> str:
    payload = {
        "aud": JWT_AUDIENCE,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        **claims,
    }
    key = secret if secret is not None else get_settings().jwt_secret.get_secret_value()
    return jwt.encode(payload, key, algorithm="HS256")


def test_valid_token_yields_the_principal() -> None:
    user_id, workspace_id = uuid.uuid4(), uuid.uuid4()

    claims = decode_token(_token({"sub": str(user_id), "workspace_id": str(workspace_id)}))
    principal = principal_from_claims(claims)

    assert principal.user_id == user_id
    assert principal.workspace_id == workspace_id


def test_workspace_id_is_read_from_app_metadata() -> None:
    """Supabase puts admin-set custom claims in app_metadata, not at the top level."""
    user_id, workspace_id = uuid.uuid4(), uuid.uuid4()

    claims = decode_token(
        _token(
            {
                "sub": str(user_id),
                "app_metadata": {"workspace_id": str(workspace_id), "org_role": "admin"},
            }
        )
    )
    principal = principal_from_claims(claims)

    assert principal.workspace_id == workspace_id


def test_user_metadata_cannot_supply_a_workspace() -> None:
    """user_metadata is user-editable — trusting it would let anyone pick their tenant."""
    claims = decode_token(
        _token({"sub": str(uuid.uuid4()), "user_metadata": {"workspace_id": str(uuid.uuid4())}})
    )

    with pytest.raises(HTTPException) as exc:
        principal_from_claims(claims)
    assert exc.value.status_code == 403


def test_a_role_claim_is_never_trusted() -> None:
    """Phase 2: roles come from the members table, so a token role claim is ignored.

    The token may still carry a legacy role claim — Supabase projects migrating from the
    org-era schema keep one in app_metadata — but the Principal must not expose it,
    because nothing downstream may base a decision on it.
    """
    claims = decode_token(
        _token(
            {
                "sub": str(uuid.uuid4()),
                "workspace_id": str(uuid.uuid4()),
                "app_metadata": {"org_role": "owner"},
            }
        )
    )
    principal = principal_from_claims(claims)

    assert not hasattr(principal, "role")
    assert not hasattr(principal, "org_role")


def test_token_signed_with_the_wrong_secret_is_rejected() -> None:
    token = _token(
        {"sub": str(uuid.uuid4()), "workspace_id": str(uuid.uuid4())}, secret="not-the-secret"
    )

    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_expired_token_is_rejected() -> None:
    token = _token({"sub": str(uuid.uuid4()), "workspace_id": str(uuid.uuid4())}, expires_in=-60)

    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_token_issued_a_few_seconds_in_the_future_is_accepted() -> None:
    """Clock-skew tolerance (jwt_leeway_seconds).

    Supabase signs tokens with its own clock, which can run a second or two ahead
    of the host's. With zero leeway those fresh tokens were rejected with
    ImmatureSignatureError on a machine whose clock has no NTP source — observed
    live against real Supabase tokens. A token whose iat is within the leeway
    window must verify.
    """
    user_id, workspace_id = uuid.uuid4(), uuid.uuid4()
    now = int(datetime.now(timezone.utc).timestamp())
    token = _token(
        {"sub": str(user_id), "workspace_id": str(workspace_id), "iat": now + 5}
    )

    claims = decode_token(token)
    assert claims["sub"] == str(user_id)


def test_token_issued_far_in_the_future_is_rejected() -> None:
    """Leeway is tolerance, not a hole: iat beyond the window is still refused."""
    now = int(datetime.now(timezone.utc).timestamp())
    token = _token({"sub": str(uuid.uuid4()), "iat": now + 300})

    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_token_without_expiry_is_rejected() -> None:
    """A token that never expires is a permanent credential if it ever leaks."""
    payload = {
        "aud": JWT_AUDIENCE,
        "sub": str(uuid.uuid4()),
        "workspace_id": str(uuid.uuid4()),
    }
    secret = get_settings().jwt_secret.get_secret_value()

    with pytest.raises(HTTPException):
        decode_token(jwt.encode(payload, secret, algorithm="HS256"))


def test_token_for_another_audience_is_rejected() -> None:
    secret = get_settings().jwt_secret.get_secret_value()
    payload = {
        "aud": "some-other-service",
        "sub": str(uuid.uuid4()),
        "workspace_id": str(uuid.uuid4()),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }

    with pytest.raises(HTTPException):
        decode_token(jwt.encode(payload, secret, algorithm="HS256"))


def test_unsigned_token_is_rejected() -> None:
    """The `none` algorithm is the classic JWT bypass; PyJWT must not accept it."""
    payload = {
        "aud": JWT_AUDIENCE,
        "sub": str(uuid.uuid4()),
        "workspace_id": str(uuid.uuid4()),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    forged = jwt.encode(payload, key="", algorithm="none")

    with pytest.raises(HTTPException) as exc:
        decode_token(forged)
    assert exc.value.status_code == 401


def test_token_without_a_workspace_claim_is_forbidden() -> None:
    claims = decode_token(_token({"sub": str(uuid.uuid4())}))

    with pytest.raises(HTTPException) as exc:
        principal_from_claims(claims)
    assert exc.value.status_code == 403


def test_malformed_workspace_claim_is_forbidden() -> None:
    claims = decode_token(_token({"sub": str(uuid.uuid4()), "workspace_id": "not-a-uuid"}))

    with pytest.raises(HTTPException) as exc:
        principal_from_claims(claims)
    assert exc.value.status_code == 403
