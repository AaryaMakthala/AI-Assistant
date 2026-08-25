"""The upload endpoint rejects bad requests before any database or worker is involved.

Only rejection paths are exercised here: a successful upload writes a row, which needs a
live database. That path is covered by tests/integration/test_ingestion.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.security.auth import JWT_AUDIENCE

pytestmark = pytest.mark.usefixtures("valid_env")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import app.api.documents_v2 as docs_module

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()

    async def _fake_role(
        workspace_id: uuid.UUID,
        principal: Any,
        *allowed: str,
    ) -> str:
        return "OWNER"

    monkeypatch.setattr(docs_module, "assert_workspace_role", _fake_role)

    import app.api.workspace_deps as ws_deps

    original = ws_deps.assert_workspace_role
    ws_deps.assert_workspace_role = _fake_role  # type: ignore[assignment]
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as test_client:
            yield test_client
    finally:
        ws_deps.assert_workspace_role = original
        get_settings.cache_clear()


def _auth_header() -> dict[str, str]:
    payload = {
        "aud": JWT_AUDIENCE,
        "sub": str(uuid.uuid4()),
        # Phase 2: the Principal requires a workspace_id claim (CLAUDE.md section 4).
        "workspace_id": str(uuid.uuid4()),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    token = jwt.encode(payload, get_settings().jwt_secret.get_secret_value(), algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def test_upload_without_a_token_is_unauthorized(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 401


def test_upload_with_a_forged_token_is_unauthorized(client: TestClient) -> None:
    forged = jwt.encode({"sub": str(uuid.uuid4())}, "wrong-secret", algorithm="HS256")

    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {forged}"},
    )

    assert response.status_code == 401


def test_disallowed_file_type_is_refused(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("payload.exe", b"MZ\x90\x00", "application/octet-stream")},
        headers=_auth_header(),
    )

    assert response.status_code == 415


def test_declared_content_type_does_not_override_the_bytes(client: TestClient) -> None:
    """A client claiming application/pdf proves nothing; the leading bytes decide."""
    response = client.post(
        "/documents",
        files={"file": ("invoice.pdf", b"MZ\x90\x00not a pdf", "application/pdf")},
        headers=_auth_header(),
    )

    assert response.status_code == 415


def test_oversized_file_is_refused(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Override max_upload_size_mb to 0 so any non-empty file exceeds the cap
    original = get_settings()
    monkeypatch.setattr(original, "max_upload_size_mb", 0)

    response = client.post(
        "/documents",
        files={"file": ("big.txt", b"A" * 5000, "text/plain")},
        headers=_auth_header(),
    )

    assert response.status_code == 413


def test_listing_documents_requires_authentication(client: TestClient) -> None:
    assert client.get("/documents").status_code == 401
    assert client.get(f"/documents/{uuid.uuid4()}").status_code == 401
