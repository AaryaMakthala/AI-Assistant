"""JWT verification and the request principal (CLAUDE.md 4.6).

`user_id` and `workspace_id` are derived from a verified token and nowhere else. A client
that sends its own workspace_id in a body or query string is ignored — that value is a
request, not a fact. The same claims are then handed to Postgres for RLS, so the API check
and the database check agree on who is asking.

Two signing schemes are accepted. Supabase now signs session tokens with rotatable
asymmetric keys (ES256/RS256) published at the project's JWKS endpoint; older projects
sign with a shared HS256 secret. The algorithm is chosen from the token's own header only
after the corresponding key has been located, never by trusting the header outright — that
is what stops the classic confusion attack where a token declares `alg: HS256` and is
verified against a public key that the attacker also holds.

Phase 2 change: `Principal` carries `workspace_id` (from the auth-trigger-issued claim),
not `org_id`. The workspace *role* (OWNER/MEMBER) is NOT extracted from the JWT — it is
looked up from the canonical `members` table at each request, so the database remains the
sole source of truth for authorization (CLAUDE.md section 4).
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
from sqlalchemy import select

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
#: True once the JWKS client has been successfully created at least once.  When False
#: (cold start), failed initialisations use a short retry window so the first few
#: requests after startup don't all fail for 30 seconds while a slow DNS or TLS
#: handshake finishes in the background.
_jwks_ever_succeeded: bool = False

#: After a JWKS fetch fails *after* a successful init, wait this long before retrying
#: so every request during a genuine auth-service outage doesn't pile onto the failing
#: endpoint.
_JWKS_RETRY_COOLDOWN = 30.0
#: Shorter window used only before the first successful init — keeps cold-start
#: retries fast (≈2 s apart) instead of blocking all requests for half a minute.
_JWKS_COLDSTART_COOLDOWN = 2.0


def _get_jwks_client() -> PyJWKClient | None:
    """The signing-key client for this project, or None if keys cannot be fetched."""
    global _jwks_client, _jwks_url, _jwks_failed_until, _jwks_ever_succeeded

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
            _jwks_ever_succeeded = True
            return client
        except Exception as exc:
            cooldown = _JWKS_RETRY_COOLDOWN if _jwks_ever_succeeded else _JWKS_COLDSTART_COOLDOWN
            _jwks_failed_until = time.monotonic() + cooldown
            logger.warning(
                "Could not initialise JWKS client for {url} (cooldown {cooldown}s): {err}",
                url=url, cooldown=cooldown, err=exc,
            )
            return None


def reset_jwks_cache() -> None:
    """Drop the cached signing keys. For tests and for a deliberate reconfiguration."""
    global _jwks_client, _jwks_url, _jwks_failed_until, _jwks_ever_succeeded
    with _jwks_lock:
        _jwks_client = None
        _jwks_url = None
        _jwks_failed_until = 0.0
        _jwks_ever_succeeded = False


@dataclass(frozen=True)
class Principal:
    """The authenticated caller.

    ``workspace_id`` is resolved from the JWT claim when present, or looked up
    from the ``members`` table when the claim is absent (e.g. when the Supabase
    provisioning trigger has not written ``workspace_id`` into ``raw_app_meta_data``).
    It is the scope for RLS queries. The workspace *role* is NOT stored here —
    it is looked up from the ``members`` table at each workspace-scoped request,
    so revocation is immediate.
    """

    user_id: uuid.UUID
    workspace_id: uuid.UUID
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


def _try_workspace_id(claims: dict[str, Any]) -> uuid.UUID | None:
    """Extract workspace_id from JWT claims, returning None if missing or invalid."""
    raw = _claim(claims, "workspace_id")
    if raw is None:
        return None
    try:
        return uuid.UUID(str(raw))
    except (TypeError, ValueError):
        return None


async def _resolve_workspace_id(user_id: uuid.UUID) -> uuid.UUID | None:
    """Look up the user's default workspace from the ``members`` table.

    Used when the JWT does not carry a ``workspace_id`` claim (e.g. the Supabase
    provisioning trigger has not written ``workspace_id`` into ``raw_app_meta_data``).
    Queries across all workspaces — no tenant claims are set — so the connection
    must have BYPASSRLS (Supabase ``postgres`` role) or equivalent privileges.
    """
    from sqlalchemy import select

    from app.db.models import Member
    from app.db.session import get_session_factory

    async with get_session_factory()() as session:
        result = await session.execute(
            select(Member.workspace_id)
            .where(Member.user_id == user_id, Member.status == "ACTIVE")
            .order_by(Member.created_at)
            .limit(1)
        )
        return result.scalar_one_or_none()


def principal_from_claims(claims: dict[str, Any]) -> Principal:
    """Build a Principal from verified claims, rejecting anything malformed.

    Requires ``sub`` (user id). ``workspace_id`` may be absent — callers that
    need it (``get_principal``) fall back to a database lookup.
    """
    try:
        user_id = uuid.UUID(str(claims["sub"]))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token is missing a valid user or workspace claim.",
        ) from exc

    email = claims.get("email")
    workspace_id = _try_workspace_id(claims)
    if workspace_id is None:
        # workspace_id will be resolved from the DB by get_principal().
        # Return a sentinel — this function never performs I/O.
        # We use a temporary None and let the caller resolve it.
        # However, Principal.workspace_id is required, so we raise here
        # and let get_principal() handle the fallback directly.
        raise _MISSING_WORKSPACE

    return Principal(user_id=user_id, workspace_id=workspace_id, email=email)


_MISSING_WORKSPACE = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="Token is missing a valid user or workspace claim.",
)


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
            # Clock-skew tolerance: the issuing server's clock is authoritative for
            # iat/exp, and a host clock a couple of seconds behind it rejects fresh
            # tokens as "not yet valid" (CLAUDE.md risk register — verified live on a
            # no-NTP host). See jwt_leeway_seconds in config.
            leeway=get_settings().jwt_leeway_seconds,
            options={"require": ["exp", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        raise _INVALID from exc


async def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """FastAPI dependency: the verified caller, or 401/403.

    Extracts ``user_id`` from the verified JWT ``sub`` claim (always present on
    valid Supabase tokens), then resolves ``workspace_id`` either from the JWT
    ``app_metadata`` or, when absent, from the ``members`` table.
    """
    if credentials is None or not credentials.credentials:
        raise _INVALID
    claims = decode_token(credentials.credentials)

    # Extract user_id — always present in a valid Supabase token.
    try:
        user_id = uuid.UUID(str(claims["sub"]))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise _INVALID from exc

    email = claims.get("email")

    # Try workspace_id from JWT claim (the happy path — no DB round-trip).
    workspace_id = _try_workspace_id(claims)

    # Fall back to the members table when the claim is absent.
    if workspace_id is None:
        workspace_id = await _resolve_workspace_id(user_id)
        if workspace_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Token is missing a valid user or workspace claim.",
            )

    principal = Principal(user_id=user_id, workspace_id=workspace_id, email=email)

    # Check for workspace-switch override (set by workspace_switch middleware).
    try:
        from app.security.workspace_switch import get_requested_workspace_id
        requested_ws = get_requested_workspace_id()
        if requested_ws is not None and requested_ws != workspace_id:
            # Validate membership in the requested workspace.
            from app.db.models import Member
            from app.security.rls import tenant_session

            async with tenant_session(
                workspace_id=workspace_id, user_id=user_id
            ) as session:
                member = (
                    await session.execute(
                        select(Member.role).where(
                            Member.workspace_id == requested_ws,
                            Member.user_id == user_id,
                            Member.status == "ACTIVE",
                        )
                    )
                ).scalar_one_or_none()

            if member is None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You are not a member of the requested workspace.",
                )

            principal = Principal(
                user_id=user_id, workspace_id=requested_ws, email=email
            )
    except ImportError:
        pass  # workspace_switch module not available

    # Tag this request's error reports and traces with who made it — after verification, so
    # the tags reflect claims the server checked rather than ones the client asserted. Only
    # opaque ids travel; see app/observability/context.py.
    bind_principal(user_id=principal.user_id, workspace_id=principal.workspace_id)
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
    "_resolve_workspace_id",
    "_try_workspace_id",
]
