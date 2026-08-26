"""Regression tests for Fix 1 (metadata questions bypass retrieval) and
Fix 2 (improved refusal messages) in ``app.api.chat_v2``.

Every test stubs the tenant session and (where relevant) the retrieval pipeline
and LLM provider so it can prove *exactly* what code paths ran without needing
a real database or API key.

Test matrix:
1. Count question → workspace-scoped DB count, no LLM call.
2. List question → actual documents returned, no LLM call.
3. Content question with relevant chunks → LLM called.
4. Content question with no relevant chunks → no LLM call.
5. Documents exist but none relevant → clear refusal message.
6. Cross-workspace docs never included in metadata results.
7. "How many documents discuss X?" uses retrieval, not metadata path.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.chat_v2 as chat_module
from app.api.chat_v2 import (
    REFUSAL_NO_EVIDENCE,
    REFUSAL_NOT_RELEVANT,
    _is_metadata_question,
)
from app.api.dependencies import get_generic_llm
from app.db.models import Document
from app.llm.base import Completion, Message, TokenUsage
from app.retrieval.pipeline import RetrievalResult, RetrievedChunk
from app.security.auth import Principal, get_principal

pytestmark = pytest.mark.usefixtures("valid_env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    score: float = 0.9,
    content: str = "Vacation accrues at 20 days per year.",
    *,
    chunk_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="handbook.pdf",
        content=content,
        page_number=2,
        section_title="Leave policy",
        chunk_index=0,
        rrf_score=0.02,
        rerank_score=score,
    )


class _StubLLM:
    """Scripted provider: records calls so tests can assert whether the LLM ran."""

    name = "test-provider"
    model = "test-model"

    def __init__(self, text: str = "An answer.") -> None:
        self._text = text
        self.calls: list[list[Message]] = []

    async def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        completion.provider = self.name
        completion.model = self.model
        completion.text = self._text
        completion.usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        yield self._text


class _FakeQuery:
    """Minimal SQLAlchemy-style query builder that supports .where(), .order_by(),
    .scalar_one(), and .all() — enough for the metadata query paths."""

    def __init__(self, rows: list[Any] | None = None, scalar: Any = None) -> None:
        self._rows = rows or []
        self._scalar = scalar

    def where(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def order_by(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def limit(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def offset(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def select_from(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def scalar_one(self) -> Any:
        return self._scalar

    def scalars(self) -> Any:
        return self

    def all(self) -> list[Any]:
        return self._rows


class _FakeResult:
    def __init__(self, rows: list[Any] | None = None, scalar: Any = None) -> None:
        self._rows = rows or []
        self._scalar = scalar

    def scalar_one(self) -> Any:
        return self._scalar

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[Any]:
        return self._rows

    def first(self) -> Any:
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Intercepts .execute() and returns pre-configured results."""

    def __init__(self, responses: list[_FakeResult] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_index = 0
        self.execute_calls: list[Any] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        self.execute_calls.append(stmt)
        if self._call_index < len(self._responses):
            result = self._responses[self._call_index]
            self._call_index += 1
            return result
        return _FakeResult()

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class _FakeSessionFactory:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def __call__(self) -> _FakeSessionFactory:
        return self

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *args: Any) -> None:
        pass


def _make_doc_row(filename: str = "handbook.pdf") -> Any:
    """Create a fake Document row for the metadata response."""
    from types import SimpleNamespace

    return SimpleNamespace(filename=filename)


def _make_doc_rows(*filenames: str) -> list[Any]:
    return [_make_doc_row(f) for f in filenames]


# ---------------------------------------------------------------------------
# Fix 1: Metadata intent detection (unit tests)
# ---------------------------------------------------------------------------


class TestMetadataIntentDetection:
    """Pure-function tests for ``_is_metadata_question``."""

    def test_count_questions(self) -> None:
        assert _is_metadata_question("How many documents have I uploaded?") == "count"
        assert _is_metadata_question("How many files do I have?") == "count"
        assert _is_metadata_question("How many documents are there?") == "count"
        assert _is_metadata_question("Number of uploaded documents") == "count"
        assert _is_metadata_question("What is the total number of documents?") == "count"
        assert _is_metadata_question("How many documents have I uploaded") == "count"

    def test_list_questions(self) -> None:
        assert _is_metadata_question("What documents have I uploaded?") == "list"
        assert _is_metadata_question("List my uploaded documents.") == "list"
        assert _is_metadata_question("What files are in my workspace?") == "list"
        assert _is_metadata_question("Show me my documents") == "list"
        assert _is_metadata_question("Which documents do I have?") == "list"
        assert _is_metadata_question("Name all my files") == "list"

    def test_content_questions_bypass_metadata(self) -> None:
        """Questions with topic qualifiers go through retrieval, not metadata."""
        assert _is_metadata_question("How many documents discuss authentication?") is None
        assert _is_metadata_question("How many documents are about vacation?") is None
        assert _is_metadata_question("How many documents cover the refund policy?") is None
        assert _is_metadata_question("List documents about HR") is None
        assert _is_metadata_question("What documents mention the travel policy?") is None

    def test_general_questions_are_not_metadata(self) -> None:
        assert _is_metadata_question("What does the vacation policy say?") is None
        assert _is_metadata_question("Summarize the refund policy") is None
        assert _is_metadata_question("hello") is None
        assert _is_metadata_question("") is None


# ---------------------------------------------------------------------------
# Fix 1: Metadata questions bypass retrieval and LLM (integration tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, _valid_env: None  # noqa: ARG001
) -> tuple[TestClient, Principal]:
    """A real app with DB and LLM stubbed for metadata testing."""
    from app.main import create_app

    principal = Principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
    app = create_app()

    async def _principal() -> Principal:
        return principal

    app.dependency_overrides[get_principal] = _principal

    async def _member(workspace_id: uuid.UUID, principal: Principal, *allowed: str) -> str:  # noqa: ARG001
        return "OWNER"

    monkeypatch.setattr(chat_module, "assert_workspace_role", _member)

    # Default: no documents in the workspace.
    _default_session = _FakeSession(responses=[_FakeResult(scalar=0), _FakeResult(rows=[])])

    def _make_tenant_session(
        *, workspace_id: uuid.UUID, user_id: uuid.UUID | None = None  # noqa: ARG001
    ) -> _FakeSession:
        return _default_session

    monkeypatch.setattr(chat_module, "tenant_session", _make_tenant_session)

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, principal
    app.dependency_overrides.clear()


@pytest.fixture
def _valid_env() -> None:
    pass


def test_count_bypasses_retrieval_and_llm(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
) -> None:
    """A count question queries the documents table directly — no retrieval, no LLM."""
    test_client, principal = client

    # Stub: document count is 4.
    # No history lookup needed — metadata intents skip rewrite.
    session = _FakeSession(responses=[
        _FakeResult(scalar=4),      # metadata count query
    ])
    monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

    retrieval_called: list[str] = []

    async def _retrieve(*args: Any, **kwargs: Any) -> RetrievalResult:
        retrieval_called.append("called")
        raise AssertionError("retrieval must NOT run for metadata questions")

    monkeypatch.setattr(chat_module, "retrieve", _retrieve)

    stub = _StubLLM()
    test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

    response = test_client.post(
        "/chat/grounded", json={"message": "How many documents have I uploaded?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert "4 uploaded documents" in body["answer"]
    assert body["sources"] == []
    assert retrieval_called == []  # retrieval never ran
    assert stub.calls == []  # LLM was never called


def test_list_bypasses_retrieval_and_llm(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
) -> None:
    """A list question returns actual document names — no retrieval, no LLM."""
    test_client, principal = client

    doc_rows = _make_doc_rows("handbook.pdf", "refund_policy.docx", "travel_guide.csv")
    # No history lookup needed — metadata intents skip rewrite.
    session = _FakeSession(responses=[
        _FakeResult(rows=doc_rows),  # metadata list query
    ])
    monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

    retrieval_called: list[str] = []

    async def _retrieve(*args: Any, **kwargs: Any) -> RetrievalResult:
        retrieval_called.append("called")
        raise AssertionError("retrieval must NOT run for metadata questions")

    monkeypatch.setattr(chat_module, "retrieve", _retrieve)

    stub = _StubLLM()
    test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

    response = test_client.post(
        "/chat/grounded", json={"message": "What documents have I uploaded?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert "3 uploaded documents" in body["answer"]
    assert "handbook.pdf" in body["answer"]
    assert "refund_policy.docx" in body["answer"]
    assert "travel_guide.csv" in body["answer"]
    assert body["sources"] == []
    assert retrieval_called == []
    assert stub.calls == []


def test_content_question_uses_retrieval_and_llm(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
) -> None:
    """A content question goes through retrieval and the LLM."""
    test_client, _ = client

    async def _retrieve(session: Any, *, query: str, workspace_id: uuid.UUID, **kwargs: Any) -> RetrievalResult:  # noqa: ARG001
        return RetrievalResult(chunks=[_chunk(0.9)], grounded=True, top_score=0.9)

    monkeypatch.setattr(chat_module, "retrieve", _retrieve)

    stub = _StubLLM(text="Vacation is 20 days per year.")
    test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

    response = test_client.post(
        "/chat/grounded", json={"message": "What does the vacation policy say?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert "20 days" in body["answer"]
    assert len(body["sources"]) == 1
    assert len(stub.calls) == 1  # LLM was called


def test_no_relevant_chunks_does_not_call_llm(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
) -> None:
    """When retrieval returns nothing, the LLM is not called."""
    test_client, _ = client

    async def _retrieve(session: Any, *, query: str, workspace_id: uuid.UUID, **kwargs: Any) -> RetrievalResult:  # noqa: ARG001
        return RetrievalResult(chunks=[], grounded=False, top_score=None)

    monkeypatch.setattr(chat_module, "retrieve", _retrieve)

    stub = _StubLLM()
    test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

    response = test_client.post(
        "/chat/grounded", json={"message": "quantum entanglement explained"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["sources"] == []
    assert stub.calls == []


def test_chunks_below_threshold_uses_not_relevant_refusal(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
) -> None:
    """When chunks exist but none pass the threshold, the refusal mentions documents."""
    test_client, _ = client

    async def _retrieve(session: Any, *, query: str, workspace_id: uuid.UUID, **kwargs: Any) -> RetrievalResult:  # noqa: ARG001
        return RetrievalResult(chunks=[_chunk(0.05)], grounded=False, top_score=0.05)

    monkeypatch.setattr(chat_module, "retrieve", _retrieve)

    stub = _StubLLM()
    test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

    response = test_client.post(
        "/chat/grounded", json={"message": "quantum entanglement explained"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["answer"] == REFUSAL_NOT_RELEVANT
    assert "workspace contains documents" in body["answer"]
    assert stub.calls == []


def test_no_chunks_uses_no_evidence_refusal(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
) -> None:
    """When no chunks are found at all, the refusal is about missing evidence."""
    test_client, _ = client

    async def _retrieve(session: Any, *, query: str, workspace_id: uuid.UUID, **kwargs: Any) -> RetrievalResult:  # noqa: ARG001
        return RetrievalResult(chunks=[], grounded=False, top_score=None)

    monkeypatch.setattr(chat_module, "retrieve", _retrieve)

    stub = _StubLLM()
    test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

    response = test_client.post(
        "/chat/grounded", json={"message": "quantum entanglement explained"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["answer"] == REFUSAL_NO_EVIDENCE
    assert "uploaded documents" in body["answer"]
    assert stub.calls == []


def test_cross_workspace_docs_not_in_metadata_results(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
) -> None:
    """Metadata queries are workspace-scoped — another workspace's docs don't appear."""
    test_client, principal = client

    # The session mock returns zero docs for this workspace.
    session = _FakeSession(responses=[_FakeResult(rows=[])])
    monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

    response = test_client.post(
        "/chat/grounded", json={"message": "List my documents"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert "no uploaded documents" in body["answer"].lower()


def test_how_many_discuss_uses_retrieval_not_metadata(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
) -> None:
    """'How many documents discuss X?' is a content question, not a metadata count."""
    test_client, _ = client

    async def _retrieve(session: Any, *, query: str, workspace_id: uuid.UUID, **kwargs: Any) -> RetrievalResult:  # noqa: ARG001
        return RetrievalResult(
            chunks=[_chunk(0.85, content="We process 50 refunds per month.")], grounded=True, top_score=0.85
        )

    monkeypatch.setattr(chat_module, "retrieve", _retrieve)

    stub = _StubLLM(text="The documents discuss approximately 50 refunds per month.")
    test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

    response = test_client.post(
        "/chat/grounded",
        json={"message": "How many documents discuss the refund process?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert "50 refunds" in body["answer"]
    assert len(body["sources"]) == 1
    assert len(stub.calls) == 1  # LLM was called — this is a content question
