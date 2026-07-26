"""JWT verification and the request principal (CLAUDE.md 4.6).

`org_id` and `user_id` are derived from a verified token and nowhere else. A client that
sends its own org_id in a body or query string is ignored — that value is a request, not
a fact. The same claims are then handed to Postgres for RLS, so the API check and the
database check agree on who is asking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

#: Supabase issues access tokens with this audience.
JWT_AUDIENCE = "authenticated"
JWT_ALGORITHMS = ["HS256"]

_bearer = HTTPBearer(auto_error=False)

_INVALID = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing or invalid authentication credentials.",
    headers={"WWW-Authenticate": "Bearer"},
)


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Every org-scoped query is built from these values."""

    user_id: uuid.UUID
    org_id: uuid.UUID
    role: str
    email: str | None = None


def _claim(claims: dict[str, Any], name: str) -> Any:
    """Read a claim from the top level or from Supabase's app_metadata.

    Custom claims set through Supabase's admin API land in `app_metadata`; a hook-issued
    token can put them at the top level. Accepting both keeps the two paths equivalent.
    Only app_metadata is consulted — never user_metadata, which the user can edit.
    """
    if name in claims:
        return claims[name]
    metadata = claims.get("app_metadata")
    return metadata.get(name) if isinstance(metadata, dict) else None


def principal_from_claims(claims: dict[str, Any]) -> Principal:
    """Build a Principal from verified claims, rejecting anything malformed."""
    try:
        user_id = uuid.UUID(str(claims["sub"]))
        org_id = uuid.UUID(str(_claim(claims, "org_id")))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token is missing a valid user or organization claim.",
        ) from exc

    role = _claim(claims, "org_role") or "member"
    email = claims.get("email")
    return Principal(user_id=user_id, org_id=org_id, role=str(role), email=email)


def decode_token(token: str) -> dict[str, Any]:
    """Verify a token's signature, expiry and audience, or raise 401."""
    try:
        return jwt.decode(
            token,
            get_settings().jwt_secret.get_secret_value(),
            algorithms=JWT_ALGORITHMS,
            audience=JWT_AUDIENCE,
            options={"require": ["exp", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        raise _INVALID from exc


async def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """FastAPI dependency: the verified caller, or 401."""
    if credentials is None or not credentials.credentials:
        raise _INVALID
    return principal_from_claims(decode_token(credentials.credentials))


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]

__all__ = [
    "JWT_ALGORITHMS",
    "JWT_AUDIENCE",
    "CurrentPrincipal",
    "Principal",
    "decode_token",
    "get_principal",
    "principal_from_claims",
]
