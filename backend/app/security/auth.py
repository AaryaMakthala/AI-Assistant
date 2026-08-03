"""JWT verification and the request principal (CLAUDE.md 4.6).

`org_id` and `user_id` are derived from a verified token and nowhere else. A client that
sends its own org_id in a body or query string is ignored — that value is a request, not
a fact. The same claims are then handed to Postgres for RLS, so the API check and the
database check agree on who is asking.

Two signing schemes are accepted. Supabase now signs session tokens with rotatable
asymmetric keys (ES256/RS256) published at the project's JWKS endpoint; older projects
sign with a shared HS256 secret. The algorithm is chosen from the token's own header only
after the corresponding key has been located, never by trusting the header outright — that
is what stops the classic confusion attack where a token declares `alg: HS256` and is
verified against a public key that the attacker also holds.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from loguru import logger

from app.config import get_settings
from app.observability.context import bind_principal

#: Supabase issues access tokens with this audience.
JWT_AUDIENCE = "authenticated"

#: Symmetric algorithms are only ever applied to the configured shared secret.
_SYMMETRIC_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})

_bearer = HTTPBearer(auto_error=False)

_INVALID = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing or invalid authentication credentials.",
    headers={"WWW-Authenticate": "Bearer"},
)

_jwks_client: PyJWKClient | None = None
_jwks_url: str | None = None
_jwks_lock = threading.Lock()
_jwks_failed_until: float = 0.0

#: After a JWKS fetch fails, wait this long before trying again, so every request during
#: an auth-service outage does not add its own outbound call to the pile.
_JWKS_RETRY_COOLDOWN = 30.0


def _get_jwks_client() -> PyJWKClient | None:
    """The signing-key client for this project, or None if keys cannot be fetched."""
    global _jwks_client, _jwks_url, _jwks_failed_until

    settings = get_settings()
    url = settings.jwks_url

    with _jwks_lock:
        if _jwks_client is not None and _jwks_url == url:
            return _jwks_client
        if time.monotonic() < _jwks_failed_until:
            return None
        try:
            client = PyJWKClient(
                url,
                cache_keys=True,
                lifespan=settings.jwks_cache_seconds,
                timeout=5,
            )
            _jwks_client = client
            _jwks_url = url
            return client
        except Exception as exc:
            _jwks_failed_until = time.monotonic() + _JWKS_RETRY_COOLDOWN
            logger.warning("Could not initialise JWKS client for {url}: {err}", url=url, err=exc)
            return None


def reset_jwks_cache() -> None:
    """Drop the cached signing keys. For tests and for a deliberate reconfiguration."""
    global _jwks_client, _jwks_url, _jwks_failed_until
    with _jwks_lock:
        _jwks_client = None
        _jwks_url = None
        _jwks_failed_until = 0.0


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


def _resolve_key(token: str, algorithm: str) -> Any:
    """Find the verification key for `token`, or raise 401.

    The token's declared algorithm selects *which store* to look in, and each store only
    ever yields keys for its own class — a symmetric `alg` can never be satisfied by a
    JWKS public key, and an asymmetric one can never be satisfied by the shared secret.
    That separation is the whole defense against algorithm confusion.
    """
    if algorithm in _SYMMETRIC_ALGORITHMS:
        return get_settings().jwt_secret.get_secret_value()

    client = _get_jwks_client()
    if client is None:
        raise _INVALID
    try:
        return client.get_signing_key_from_jwt(token).key
    except Exception as exc:
        # An unknown kid is the expected shape of a forged or newly rotated token.
        raise _INVALID from exc


def decode_token(token: str) -> dict[str, Any]:
    """Verify a token's signature, expiry and audience, or raise 401."""
    allowed = get_settings().jwt_algorithms
    try:
        algorithm = jwt.get_unverified_header(token).get("alg")
    except jwt.InvalidTokenError as exc:
        raise _INVALID from exc

    # Checked before the header is used for anything else: an attacker controls this
    # value, so it may only ever select from a list the server already approved.
    if algorithm not in allowed:
        raise _INVALID

    key = _resolve_key(token, algorithm)

    try:
        return jwt.decode(
            token,
            key,
            # Single-element list, not the full allowlist: the token is verified with
            # exactly the algorithm its key was chosen for.
            algorithms=[algorithm],
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
    principal = principal_from_claims(decode_token(credentials.credentials))
    # Tag this request's error reports and traces with who made it — after verification, so
    # the tags reflect claims the server checked rather than ones the client asserted. Only
    # opaque ids travel; see app/observability/context.py.
    bind_principal(user_id=principal.user_id, org_id=principal.org_id, role=principal.role)
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]

__all__ = [
    "JWT_AUDIENCE",
    "CurrentPrincipal",
    "Principal",
    "decode_token",
    "get_principal",
    "principal_from_claims",
    "reset_jwks_cache",
]
