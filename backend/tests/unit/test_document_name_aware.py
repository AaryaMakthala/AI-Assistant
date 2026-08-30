"""Tests for document-name-aware routing (Problem B).

Verifies:
1. \"what are doucuments names\" routes to METADATA (document listing), not retrieval.
2. \"what does devops contain\" routes to DOCUMENT_CONTENT with doc targeting.
3. \"what theu sau about\" / \"what each document contains\" are handled properly.
4. Regression: \"what documents you haeve\" still lists documents.
5. Regression: typo filename match (\"aarya\") still works.
6. Regression: \"capital of India\" is still rejected.

All external LLM/provider calls are mocked.  Never hits a real API.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.chat_v2 as chat_module
from app.api.dependencies import get_generic_llm
from app.retrieval.intent import IntentCategory, classify_intent_regex
from app.retrieval.pipeline import RetrievalResult, RetrievedChunk
from app.security.auth import Principal, get_principal
from tests.unit.conftest import (
    FakeResult,
    FakeSession,
    StubLLM,
    session_with_history,
    smart_mock_route,
)

pytestmark = pytest.mark.usefixtures("valid_env")


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    from app.retrieval.routing_cache import _cache
    from app.retrieval.workspace_knowledge import _cache as _wk_cache
    _cache.clear()
    _wk_cache.clear()
    yield
    _cache.clear()
    _wk_cache.clear()


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, _valid_env: None  # noqa: ARG001
) -> tuple[TestClient, Principal]:
    from app.main import create_app

    principal = Principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
    app = create_app()

    async def _principal() -> Principal:
        return principal

    app.dependency_overrides[get_principal] = _principal

    async def _member(
        workspace_id: uuid.UUID, principal: Principal, *allowed: str  # noqa: ARG001
    ) -> str:
        return "OWNER"

    monkeypatch.setattr(chat_module, "assert_workspace_role", _member)

    default_session = FakeSession(responses=[
        FakeResult(scalar=None),
    ])

    def _make_tenant_session(
        *, workspace_id: uuid.UUID, user_id: uuid.UUID | None = None  # noqa: ARG001
    ) -> FakeSession:
        return default_session

    monkeypatch.setattr(chat_module, "tenant_session", _make_tenant_session)
    monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", smart_mock_route)

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, principal
    app.dependency_overrides.clear()


@pytest.fixture
def _valid_env() -> None:
    pass


def _track_retrieval(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    async def _tracking_retrieve(*args: Any, **kwargs: Any) -> RetrievalResult:
        calls.append("called")
        raise AssertionError("retrieval must NOT be called for this message")

    monkeypatch.setattr(chat_module, "retrieve", _tracking_retrieve)
    return calls


# ---------------------------------------------------------------------------
# PART 1: Problem B — document-name-aware routing
# ---------------------------------------------------------------------------


class TestDocumentNameAwareRouting:
    """Verify document-name-aware routing for the 3 Problem B failing cases."""

    def test_doucuments_names_routes_to_metadata(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'what are doucuments names' → DOCUMENT_LIST (metadata), not retrieval."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        from types import SimpleNamespace
        doc_rows = [
            SimpleNamespace(filename="resume.pdf"),
            SimpleNamespace(filename="devops.docx"),
        ]
        session = session_with_history(FakeResult(rows=doc_rows))
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        response = test_client.post(
            "/chat/grounded", json={"message": "what are doucuments names"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == [], "retrieval must NOT be called for metadata"
        assert stub.calls == [], "LLM must NOT be called for metadata"
        # Answer should list documents
        answer_lower = body["answer"].lower()
        assert "resume" in answer_lower or "devops" in answer_lower

    def test_devops_contain_routes_to_document_content(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'what does devops contain' → DOCUMENT_CONTENT with retrieval."""
        test_client, _ = client
        captured_queries: list[str] = []

        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="DevOps 4-1 AIML QUESTION BANK.docx",
            content="Kanban is a visual workflow management method.",
            page_number=1,
            section_title="Kanban",
            chunk_index=0,
            rrf_score=0.02,
            rerank_score=0.9,
        )

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID, **kw: Any
        ) -> RetrievalResult:
            captured_queries.append(query)
            return RetrievalResult(chunks=[chunk], grounded=True, top_score=0.9)

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        class _SpyLLM:
            name = "spy"
            model = "spy"

            async def stream(self, messages: Any, *, completion: Any) -> Any:
                completion.text = "The DevOps document covers Kanban methodology."
                completion.provider = "spy"
                completion.model = "spy"
                from app.llm.base import TokenUsage
                completion.usage = TokenUsage(prompt_tokens=50, completion_tokens=10)
                yield "The DevOps document covers Kanban methodology."

        monkeypatch.setattr("app.api.chat_v2.get_generic_llm", lambda: _SpyLLM())

        response = test_client.post(
            "/chat/grounded", json={"message": "what does devops contain"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert len(body["sources"]) >= 1
        # Retrieval was called (not short-circuited by relevance gate)
        assert len(captured_queries) >= 1

    def test_vague_question_asked_for_clarification(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'what theu sau about' (very vague) → either NEEDS_CLARIFICATION or
        DOCUMENT_CONTENT with retrieval.  Must NOT return a flat 'no relevant
        information' refusal."""
        test_client, _ = client

        # Let retrieval run and return empty (no relevant chunks)
        async def _retrieve(*args: Any, **kw: Any) -> RetrievalResult:
            return RetrievalResult(chunks=[], grounded=False, top_score=0.0)

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded", json={"message": "what theu sau about"}
        )
        assert response.status_code == 200
        body = response.json()
        # Must NOT be a flat "no relevant information" refusal.
        # Either it's grounded (some answer) or it asks for clarification.
        answer_lower = body["answer"].lower()
        # The answer should mention documents or ask for clarification,
        # not just say "no relevant information".
        assert not (
            "couldn't find" in answer_lower and "relevant information" in answer_lower
        ), f"Got flat refusal: {body['answer']}"


# ---------------------------------------------------------------------------
# PART 2: Regression checks
# ---------------------------------------------------------------------------


class TestRegressionChecks:
    """Verify existing working cases still work after the changes."""

    def test_documents_you_haeve_still_lists(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'what documents you haeve' still correctly lists documents."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        from types import SimpleNamespace
        doc_rows = [
            SimpleNamespace(filename="handbook.pdf"),
            SimpleNamespace(filename="policy.docx"),
        ]
        session = session_with_history(FakeResult(rows=doc_rows))
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        response = test_client.post(
            "/chat/grounded", json={"message": "what documents you haeve"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []
        answer_lower = body["answer"].lower()
        assert "handbook" in answer_lower or "policy" in answer_lower

    def test_aarya_typo_still_matches_filename(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """Typo 'abput aarya' still filename-matches the resume."""
        test_client, _ = client
        captured_queries: list[str] = []

        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="Makthala Aarya Resume.pdf",
            content="Aarya is a software engineer with 3 years of experience.",
            page_number=1,
            section_title="Profile",
            chunk_index=0,
            rrf_score=0.02,
            rerank_score=0.9,
        )

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID, **kw: Any
        ) -> RetrievalResult:
            captured_queries.append(query)
            return RetrievalResult(chunks=[chunk], grounded=True, top_score=0.9)

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        class _SpyLLM:
            name = "spy"
            model = "spy"

            async def stream(self, messages: Any, *, completion: Any) -> Any:
                completion.text = "Aarya is a software engineer."
                completion.provider = "spy"
                completion.model = "spy"
                from app.llm.base import TokenUsage
                completion.usage = TokenUsage(prompt_tokens=50, completion_tokens=10)
                yield "Aarya is a software engineer."

        monkeypatch.setattr("app.api.chat_v2.get_generic_llm", lambda: _SpyLLM())

        response = test_client.post(
            "/chat/grounded", json={"message": "do you have details abput aarya"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert len(body["sources"]) >= 1
        assert body["sources"][0]["filename"] == "Makthala Aarya Resume.pdf"

    def test_capital_of_india_still_rejected(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'capital of India' is still correctly rejected as out-of-scope."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded", json={"message": "capital of India"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []
        answer_lower = body["answer"].lower()
        assert any(kw in answer_lower for kw in ("outside", "scope", "workspace", "can't help"))


# ---------------------------------------------------------------------------
# PART 3: Regex fast-path unit tests for new patterns
# ---------------------------------------------------------------------------


class TestDocListTypoPattern:
    """Pure-function tests for the typo-tolerant document list pattern."""

    def test_doucuments_names(self) -> None:
        i = classify_intent_regex("what are doucuments names")
        assert i.category == IntentCategory.DOCUMENT_LIST

    def test_docs_names(self) -> None:
        i = classify_intent_regex("what are docs names")
        assert i.category == IntentCategory.DOCUMENT_LIST

    def test_original_pattern_still_works(self) -> None:
        i = classify_intent_regex("what documents you haeve")
        assert i.category == IntentCategory.DOCUMENT_LIST

    def test_original_pattern_with_the(self) -> None:
        i = classify_intent_regex("what are the documents names")
        assert i.category == IntentCategory.DOCUMENT_LIST

    def test_list_all_documents(self) -> None:
        i = classify_intent_regex("list all documents")
        assert i.category == IntentCategory.DOCUMENT_LIST

    def test_show_me_files(self) -> None:
        i = classify_intent_regex("show me the files")
        assert i.category == IntentCategory.DOCUMENT_LIST

    def test_not_false_positive_on_content(self) -> None:
        """'what is our leave policy' should NOT match doc list."""
        i = classify_intent_regex("what is our leave policy")
        assert i.category == IntentCategory.DOCUMENT_CONTENT

    def test_not_false_positive_on_out_of_scope(self) -> None:
        """'capital of India' should NOT match doc list."""
        i = classify_intent_regex("capital of India")
        assert i.category == IntentCategory.OUT_OF_SCOPE
