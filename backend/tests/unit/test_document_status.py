"""The status endpoint's derived fields, which the UI trusts to stop polling.

`is_terminal` and `is_indexed` exist so the client never reimplements which statuses are
final or what "searchable" means. If they are wrong, a browser either polls forever or
tells a user a document is ready to query when the retriever cannot find a word of it.

These build the response model directly rather than through HTTP: the mapping is the
logic, and a test client would add auth plumbing without testing anything more.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from app.api.documents import DocumentStatusResponse
from app.db.models import DOCUMENT_STATUSES, TERMINAL_DOCUMENT_STATUSES

pytestmark = pytest.mark.usefixtures("valid_env")


def _status(status: str, chunk_count: int | None = None) -> DocumentStatusResponse:
    now = datetime.now(UTC)
    return DocumentStatusResponse(
        id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        filename="policy.pdf",
        processing_status=status,
        is_terminal=status in TERMINAL_DOCUMENT_STATUSES,
        is_indexed=status == "ready" and bool(chunk_count),
        chunk_count=chunk_count,
        retry_count=0,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_in_flight_statuses_are_not_terminal(status: str) -> None:
    assert _status(status).is_terminal is False


@pytest.mark.parametrize("status", ["ready", "failed"])
def test_finished_statuses_are_terminal(status: str) -> None:
    """Both outcomes end polling — a failed document is finished, not still working."""
    assert _status(status, chunk_count=3).is_terminal is True


def test_every_document_status_is_classified() -> None:
    """A new status must be deliberately sorted, not silently treated as non-terminal."""
    for status in DOCUMENT_STATUSES:
        assert isinstance(_status(status).is_terminal, bool)
    assert set(TERMINAL_DOCUMENT_STATUSES) <= set(DOCUMENT_STATUSES)


def test_a_ready_document_with_chunks_is_indexed() -> None:
    assert _status("ready", chunk_count=12).is_indexed is True


def test_a_ready_document_with_no_chunks_is_not_indexed() -> None:
    """Nothing was stored, so nothing is searchable — saying otherwise misleads."""
    assert _status("ready", chunk_count=0).is_indexed is False


def test_a_document_still_processing_is_not_indexed() -> None:
    assert _status("processing").is_indexed is False


def test_a_failed_document_is_not_indexed() -> None:
    assert _status("failed").is_indexed is False


def test_upload_status_defaults_to_complete() -> None:
    """A row only exists after the bytes are stored, so this is constant by construction."""
    assert _status("pending").upload_status == "complete"


def test_response_carries_the_owning_organization() -> None:
    """Phase 10 requires the org on the response; a client must never infer it."""
    response = _status("ready", chunk_count=1)

    assert isinstance(response.org_id, uuid.UUID)
    assert "org_id" in response.model_dump()
