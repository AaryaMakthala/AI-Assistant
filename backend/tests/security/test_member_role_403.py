"""MEMBER-role authorization surface (CLAUDE.md section 4).

Proves, at the HTTP layer, that a MEMBER-role user is blocked from every
owner-only action, exactly as the live demo-guest verification observed:

* DELETE of another user's document       -> 403
* POST /workspaces/{id}/invitations       -> 403
* PATCH /workspaces/{id}/members/{id}     -> 403
* DELETE /workspaces/{id}/members/{id}    -> 403
* POST /documents/{id}/approve            -> 403 (owner-only, Phase 4)

Member upload is deliberately NOT 403: the product design (CLAUDE.md sections
1, 4, 5) lets a MEMBER upload a document that stays PENDING — structurally
unsearchable because it has zero chunks until an OWNER approves it. The test
asserts the actual contract (201 + status=PENDING + no chunks) rather than a
403, so a regression to the authorization boundary still fails loudly.

Role is looked up from the canonical ``members`` table at request time (never
the token). These tests fake that lookup with a MEMBER row and let the real
dependency / endpoint logic run.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.security.auth import JWT_AUDIENCE

pytestmark = pytest.mark.usefixtures("valid_env")


# ---------------------------------------------------------------------------
# Test doubles — fake Member row + fake tenant session
# ---------------------------------------------------------------------------


class _FakeMember:
    """Minimal stand-in for a ``members`` row: the role check only reads ``role``."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.status = "ACTIVE"


class _FakeResult:
    def __init__(
        self,
        scalar: Any = None,
        scalar_one: Any = None,
        rows: list[Any] | None = None,
    ) -> None:
        self._scalar = scalar
        self._scalar_one = scalar_one
        self._rows = rows or []

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def scalar_one(self) -> Any:
        return self._scalar_one

    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list[Any]:
        return self._rows

    def first(self) -> Any:
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Async context manager returning scripted results per execute call."""

    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = list(results)
        self._i = 0

    async def execute(self, _stmt: Any) -> _FakeResult:
        result = (
            self._results[self._i] if self._i < len(self._results) else _FakeResult()
        )
        self._i += 1
        return result

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The real app with the membership lookup forced to a MEMBER row.

    ``get_workspace_member`` is the single funnel every workspace authorization
    check goes through (both the ``WorkspaceOwner`` dependency and
    ``assert_workspace_role``), so forcing it to a MEMBER drives the real
    dependency logic below it.
    """
    import app.api.workspace_deps as ws_deps

    async def _member_lookup(_workspace_id: uuid.UUID, _principal: Any) -> _FakeMember:
        return _FakeMember(role="MEMBER")

    monkeypatch.setattr(ws_deps, "get_workspace_member", _member_lookup)

    # documents_v2 imports assert_workspace_role directly into its own namespace.
    import app.api.documents_v2 as docs_module

    async def _member_role(
        _workspace_id: uuid.UUID, _principal: Any, *allowed: str
    ) -> str:
        # Mirrors the real assert_workspace_role: an owner-only call site passes
        # the required role(s), and a MEMBER fails it with 403; member-permitted
        # call sites pass nothing and get the role back.
        from fastapi import HTTPException
        from fastapi import status as http_status

        if allowed:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Your workspace role does not permit this action.",
            )
        return "MEMBER"

    monkeypatch.setattr(docs_module, "assert_workspace_role", _member_role)

    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def _auth_header(sub: uuid.UUID | None = None) -> dict[str, str]:
    payload = {
        "aud": JWT_AUDIENCE,
        "sub": str(sub or uuid.uuid4()),
        "workspace_id": str(uuid.uuid4()),
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    token = jwt.encode(
        payload, get_settings().jwt_secret.get_secret_value(), algorithm="HS256"
    )
    return {"Authorization": f"Bearer {token}"}


def _fake_document(
    document_id: uuid.UUID, workspace_id: uuid.UUID, uploaded_by: uuid.UUID, status: str = "PENDING"
) -> Any:
    """An ORM-shaped row accepted by DocumentResponse.model_validate."""
    return type(
        "Row",
        (),
        {
            "id": document_id,
            "workspace_id": workspace_id,
            "uploaded_by": uploaded_by,
            "filename": "guest_notes.csv",
            "mime_type": "text/csv",
            "file_size": 16,
            "checksum": "abc123",
            "status": status,
            "error_message": None,
            "description": None,
            "approved_at": None,
            "created_at": datetime.now(UTC),
        },
    )()


# ---------------------------------------------------------------------------
# Owner-only actions must be 403 for a MEMBER
# ---------------------------------------------------------------------------


def test_member_delete_of_another_users_document_is_403(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DELETE /documents/{id} as MEMBER on someone else's upload -> 403."""
    import app.api.documents_v2 as docs_module

    owner_uploaded = uuid.uuid4()

    session = _FakeSession(
        [_FakeResult(scalar=owner_uploaded)]  # select uploaded_by of the target doc
    )
    monkeypatch.setattr(docs_module, "tenant_session", lambda **kw: session)

    response = client.delete(f"/documents/{uuid.uuid4()}", headers=_auth_header())
    assert response.status_code == 403


def test_member_create_invitation_is_403(client: TestClient) -> None:
    """POST /workspaces/{id}/invitations as MEMBER -> 403."""
    response = client.post(
        f"/workspaces/{uuid.uuid4()}/invitations",
        json={"email": "someone@example.com"},
        headers=_auth_header(),
    )
    assert response.status_code == 403


def test_member_change_member_role_is_403(client: TestClient) -> None:
    """PATCH /workspaces/{id}/members/{id} as MEMBER -> 403."""
    response = client.patch(
        f"/workspaces/{uuid.uuid4()}/members/{uuid.uuid4()}",
        json={"role": "OWNER"},
        headers=_auth_header(),
    )
    assert response.status_code == 403


def test_member_remove_member_is_403(client: TestClient) -> None:
    """DELETE /workspaces/{id}/members/{id} as MEMBER -> 403."""
    response = client.delete(
        f"/workspaces/{uuid.uuid4()}/members/{uuid.uuid4()}",
        headers=_auth_header(),
    )
    assert response.status_code == 403


def test_member_approve_pending_document_is_403(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /documents/{id}/approve as MEMBER -> 403 (Phase 4, owner-only)."""
    import app.api.documents_v2 as docs_module

    session = _FakeSession([_FakeResult(scalar=None)])
    monkeypatch.setattr(docs_module, "tenant_session", lambda **kw: session)

    response = client.post(
        f"/documents/{uuid.uuid4()}/approve", headers=_auth_header()
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Designed member behaviors (not 403 — documented contract)
# ---------------------------------------------------------------------------


def test_member_upload_is_pending_with_no_chunks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /documents as MEMBER -> 201 PENDING, chunk_count None.

    This is the designed flow (CLAUDE.md sections 1, 4, 5): a member upload is
    stored but never ingested — zero chunks means it is structurally
    unsearchable until an OWNER approves it. A 403 here would be a regression;
    so would a READY status or a nonzero chunk_count.
    """
    import app.api.documents_v2 as docs_module

    doc_id = uuid.uuid4()
    principal_id = uuid.uuid4()

    pending = _fake_document(doc_id, uuid.uuid4(), principal_id, status="PENDING")
    session = _FakeSession(
        [
            _FakeResult(scalar=None),  # _ensure_unique_in_workspace: no duplicate
            _FakeResult(scalar_one=pending),  # insert ... returning(Document)
        ]
    )
    monkeypatch.setattr(docs_module, "tenant_session", lambda **kw: session)

    response = client.post(
        "/documents",
        files={"file": ("guest_notes.csv", b"name,value\nx,1\n", "text/csv")},
        headers=_auth_header(),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["document"]["status"] == "PENDING"
    assert body["chunk_count"] is None


def test_member_delete_of_own_document_is_204(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DELETE /documents/{id} as MEMBER on their own upload -> 204 (designed)."""
    import app.api.documents_v2 as docs_module

    principal_id = uuid.uuid4()

    session = _FakeSession(
        [
            _FakeResult(scalar=principal_id),  # uploaded_by == caller
            _FakeResult(),  # chunk delete
            _FakeResult(),  # document delete
        ]
    )
    monkeypatch.setattr(docs_module, "tenant_session", lambda **kw: session)
    monkeypatch.setattr(docs_module, "_invalidate_workspace_caches", lambda _ws: None)

    response = client.delete(
        f"/documents/{uuid.uuid4()}", headers=_auth_header(sub=principal_id)
    )
    assert response.status_code == 204