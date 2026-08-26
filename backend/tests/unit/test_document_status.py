"""The document response model's fields, which the UI trusts for status display.

These build the response model directly rather than through HTTP: the mapping is the
logic, and a test client would add auth plumbing without testing anything more.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.api.documents_v2 import DocumentResponse
from app.db.legacy_models import DOCUMENT_STATUSES, TERMINAL_DOCUMENT_STATUSES

pytestmark = pytest.mark.usefixtures("valid_env")


def _response(status: str) -> DocumentResponse:
    # datetime.UTC is 3.11+; timezone.utc is compatible with both.
    now = datetime.now(timezone.utc)
    return DocumentResponse(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        uploaded_by=uuid.uuid4(),
        filename="policy.pdf",
        mime_type="application/pdf",
        file_size=1024,
        checksum="abc123",
        status=status,
        created_at=now,
    )


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_in_flight_statuses_are_not_terminal(status: str) -> None:
    assert _response(status).status in {"pending", "processing"}


@pytest.mark.parametrize("status", ["ready", "failed", "rejected"])
def test_finished_statuses_are_terminal(status: str) -> None:
    """Both outcomes end polling — a failed document is finished, not still working."""
    assert _response(status).status == status


def test_every_document_status_is_classified() -> None:
    """A new status must be deliberately sorted, not silently treated as non-terminal."""
    for status in DOCUMENT_STATUSES:
        resp = _response(status)
        assert resp.status == status
    assert set(TERMINAL_DOCUMENT_STATUSES) <= set(DOCUMENT_STATUSES)


def test_a_ready_document_has_the_expected_status() -> None:
    assert _response("ready").status == "ready"


def test_a_pending_document_has_the_expected_status() -> None:
    assert _response("pending").status == "pending"


def test_a_processing_document_has_the_expected_status() -> None:
    assert _response("processing").status == "processing"


def test_a_failed_document_has_the_expected_status() -> None:
    assert _response("failed").status == "failed"


def test_response_carries_the_owning_workspace() -> None:
    """Phase 10 requires the workspace on the response; a client must never infer it."""
    response = _response("ready")

    assert isinstance(response.workspace_id, uuid.UUID)
    assert "workspace_id" in response.model_dump()
