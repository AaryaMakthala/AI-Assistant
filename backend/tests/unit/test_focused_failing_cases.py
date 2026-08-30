"""Focused tests for specific routing failures and regression checks.

Tests the 5 previously failing messages that incorrectly reached the
relevance gate and returned \"I couldn't find any relevant information...\":
  1. \"who are you\"          → IDENTITY_ASSISTANT (direct)
  2. \"how manu docs there\"  → METADATA/DOC_COUNT (direct)
  3. \"i have an doubt\"      → GENERAL_CONVERSATION (direct)
  4. \"write a pyathon code\" → OUT_OF_SCOPE (direct)
  5. \"hi my name is aarya\"  → IDENTITY_USER (direct)

Plus 3 already-working messages (regression check):
  6. \"hey\"                           → GREETING (direct)
  7. \"what is you doing\"             → DOCUMENT_CONTENT (via LLM router)
  8. \"what are the documents available\" → DOCUMENT_LIST (direct)

And 1 retrieval-path case:
  9. \"what is our leave policy?\" → DOCUMENT_CONTENT with retrieval + grounding

All external LLM/provider calls are mocked.  Never hits a real API.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.chat_v2 as chat_module
from app.api.dependencies import get_generic_llm
from app.retrieval.intent import (
    IntentCategory,
    classify_intent_regex,
)
from app.retrieval.llm_router import RouteResult
from app.retrieval.pipeline import RetrievalResult, RetrievedChunk
from app.security.auth import Principal, get_principal
from tests.unit.conftest import FakeResult, FakeSession, StubLLM, session_with_history

pytestmark = pytest.mark.usefixtures("valid_env")


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Reset caches between tests."""
    from app.retrieval.routing_cache import _cache
    from app.retrieval.workspace_knowledge import _cache as _wk_cache
    _cache.clear()
    _wk_cache.clear()
    yield
    _cache.clear()
    _wk_cache.clear()


# ---------------------------------------------------------------------------
# Client fixture (shared across integration-level tests in this file)
# ---------------------------------------------------------------------------


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, _valid_env: None  # noqa: ARG001
) -> tuple[TestClient, Principal]:
    """A real app with DB and LLM stubbed for testing."""
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

    # Default: no session in history.
    default_session = FakeSession(responses=[
        FakeResult(scalar=None),  # _load_recent_history: no session found
    ])

    def _make_tenant_session(
        *, workspace_id: uuid.UUID, user_id: uuid.UUID | None = None  # noqa: ARG001
    ) -> FakeSession:
        return default_session

    monkeypatch.setattr(chat_module, "tenant_session", _make_tenant_session)

    # Default LLM router mock — tests that need specific routes override it.
    monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _smart_route_mock)

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, principal
    app.dependency_overrides.clear()


@pytest.fixture
def _valid_env() -> None:
    pass


async def _smart_route_mock(
    *, query: str, history: list | None = None, **kw: Any
) -> RouteResult:
    """Smart mock that returns routes for common patterns."""
    from app.retrieval.intent import normalize_for_classification
    import re
    q = normalize_for_classification(query)

    if re.match(r"^(?:hi+|hello+|hey+|hola|bonjour|namaste|ciao|bye|thank)", q):
        return RouteResult(route="GREETING", confidence=0.95, reasoning="mock")
    if re.search(r"who\s+(?:are|is)\s+you", q):
        return RouteResult(route="IDENTITY_ASSISTANT", confidence=0.95, reasoning="mock")
    if re.search(r"(?:my\s+name\s+is|what\s+is\s+my\s+(?:name|info|email))", q):
        return RouteResult(route="IDENTITY_USER", confidence=0.9, reasoning="mock")
    if re.search(r"(?:capital\s+of|weather|joke|python|javascript|\d\s+\d)", q):
        return RouteResult(route="OUT_OF_SCOPE", confidence=0.95, reasoning="mock")
    if re.search(r"(?:who\s+can|can\s+(?:i|we|members?)\s+(?:upload|add|invite|approve|delete))", q):
        return RouteResult(route="PERMISSIONS", confidence=0.9, reasoning="mock")
    if re.search(r"(?:monitored|tracked|watched|logging)", q):
        return RouteResult(route="APP_HELP", confidence=0.8, reasoning="mock")
    if re.search(r"(?:how\s+(?:do|can|should)\s+(?:i|we)\s+(?:invite|add|onboard)\s+(?:a\s+)?(?:member|user|person|colleague|someone))", q):
        return RouteResult(route="APP_HELP", confidence=0.9, reasoning="mock")
    if re.search(r"(?:what\s+(?:are|were)\s+(?:the\s+)?(?:questions?|things?)|what\s+(?:did|was)\s+(?:my|the)\s+(?:previous|last)|what\s+did\s+(?:i|we)\s+(?:ask|say)|what\s+did\s+you\s+(?:just\s+)?(?:answer|say)|what\s+have\s+(?:i|we)\s+(?:been|discussed|talked)|show\s+(?:me\s+)?(?:my\s+)?(?:previous|recent|last))", q):
        return RouteResult(route="CONVERSATION_HISTORY", confidence=0.9, reasoning="mock")
    if re.search(r"(?:about|discuss|cover|mention|regarding)\b", q):
        return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="mock")
    if re.search(r"(?:how\s+many|number\s+of)\s+(?:uploaded\s+)?(?:my\s+|the\s+|this\s+)?(?:own\s+)?(?:documents?|files?|uploaded)\b", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if re.search(r"(?:list|show)\s+(?:are\s+the\s+)?(?:me\s+)?(?:all\s+)?(?:my\s+|the\s+|this\s+)?(?:uploaded\s+)?(?:documents?|files?)\b", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if re.search(r"^what\s+(?:are|is)\s+(?:the\s+|my\s+)?(?:names?\s+(?:of\s+)?)?(?:uploaded\s+)?(?:documents?|files?)\s*$", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if re.search(r"^what\s+(?:the\s+|my\s+|this\s+)?(?:uploaded\s+)?(?:documents?|files?)\s+(?:have|are|did)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if re.search(r"^(?:list|show)\s+(?:all\s+)?(?:the\s+|my\s+)?(?:uploaded\s+)?(?:documents?|files?)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if re.search(r"^(?:how\s+many|number\s+of)\s+(?:\w+\s+)?(?:members?|people|users?|employees?)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if re.search(r"^how\s+many\s+are\s+(?:invited|pending|active|confirmed|removed)\s*$", q):
        return RouteResult(route="METADATA", confidence=0.85, reasoning="mock")
    if re.search(r"^who\s+(?:is|are)\s+(?:invited|pending|active)\s*$", q):
        return RouteResult(route="METADATA", confidence=0.85, reasoning="mock")
    if re.search(r"(?:list|show)\s+(?:all\s+)?(?:the\s+|my\s+)?(?:\w+\s+)?(?:members?|people|users?|employees?)\b", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if re.search(r"(?:what\s+(?:is|are)\s+my|my)\s+(?:role|access|permission)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if re.search(r"(?:how\s+many|number\s+of|total)\s+\w*\s*(?:pages?|sheets?)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Default: document content (RAG path)
    return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="mock")


# ---------------------------------------------------------------------------
# PART 1: The 5 previously failing cases — must NOT reach retrieval
# ---------------------------------------------------------------------------


class TestPreviouslyFailingCases:
    """These messages must be caught by the regex fast-path and never
    reach the retrieval/relevance pipeline."""

    def test_who_are_you(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'who are you' → IDENTITY_ASSISTANT, no retrieval."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post("/chat/grounded", json={"message": "who are you"})
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == [], "retrieval must NOT be called"
        assert stub.calls == [], "LLM must NOT be called"
        answer_lower = body["answer"].lower()
        assert any(kw in answer_lower for kw in ("assistant", "knowledge", "help"))

    def test_how_manu_docs_there(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'how manu docs there' (typo) → METADATA/DOC_COUNT, no retrieval."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        # Session with history + doc count response.
        session = session_with_history(FakeResult(scalar=5))
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        response = test_client.post("/chat/grounded", json={"message": "how manu docs there"})
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "5" in body["answer"]
        assert body["sources"] == []
        assert retrieval_called == [], "retrieval must NOT be called"
        assert stub.calls == [], "LLM must NOT be called"

    def test_i_have_an_doubt(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'i have an doubt' → GENERAL_CONVERSATION, no retrieval."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post("/chat/grounded", json={"message": "i have an doubt"})
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == [], "retrieval must NOT be called"
        assert stub.calls == [], "LLM must NOT be called"
        answer_lower = body["answer"].lower()
        assert any(kw in answer_lower for kw in ("help", "doubt", "question", "assistant"))

    def test_write_a_pyathon_code(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'write a pyathon code' → OUT_OF_SCOPE, no retrieval."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post("/chat/grounded", json={"message": "write a pyathon code"})
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == [], "retrieval must NOT be called"
        assert stub.calls == [], "LLM must NOT be called"
        answer_lower = body["answer"].lower()
        assert any(kw in answer_lower for kw in ("outside", "scope", "workspace", "can't help"))

    def test_hi_my_name_is_aarya(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'hi my name is aarya' → IDENTITY_USER, no retrieval."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post("/chat/grounded", json={"message": "hi my name is aarya"})
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == [], "retrieval must NOT be called"
        assert stub.calls == [], "LLM must NOT be called"
        answer_lower = body["answer"].lower()
        assert any(kw in answer_lower for kw in ("name", "workspace", "member", "profile"))


# ---------------------------------------------------------------------------
# PART 2: Regression checks — 3 already-working cases
# ---------------------------------------------------------------------------


class TestRegressionWorkingCases:
    """These messages must continue to work correctly after the changes."""

    def test_hey_is_greeting(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'hey' → GREETING, no retrieval."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post("/chat/grounded", json={"message": "hey"})
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []
        answer_lower = body["answer"].lower()
        assert any(kw in answer_lower for kw in ("hello", "help", "knowledge"))

    def test_what_is_you_doing(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'what is you doing' → DOCUMENT_CONTENT via LLM router, retrieval runs."""
        test_client, _ = client

        # Track retrieval but don't block it.
        captured_queries: list[str] = []

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID, **kw: Any
        ) -> RetrievalResult:
            captured_queries.append(query)
            return RetrievalResult(chunks=[], grounded=False, top_score=0.0)

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post("/chat/grounded", json={"message": "what is you doing"})
        assert response.status_code == 200
        body = response.json()
        # May be grounded or not depending on retrieval results — the key
        # check is that retrieval WAS called (not short-circuited).
        assert len(captured_queries) >= 1, "retrieval should have been called"

    def test_documents_available_is_metadata(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'what are the documents available' → DOCUMENT_LIST, no retrieval."""
        test_client, _ = client
        retrieval_called = _track_retrieval(monkeypatch)
        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        from types import SimpleNamespace
        doc_rows = [SimpleNamespace(filename="handbook.pdf"), SimpleNamespace(filename="policy.docx")]
        session = session_with_history(FakeResult(rows=doc_rows))
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        response = test_client.post("/chat/grounded", json={"message": "what are the documents available"})
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "handbook.pdf" in body["answer"]
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []


# ---------------------------------------------------------------------------
# PART 3: Retrieval path — end-to-end grounding still works
# ---------------------------------------------------------------------------


class TestRetrievalPathStillWorks:
    """Verify the retrieval path (RAG) still works end-to-end for
    legitimate document-content questions."""

    def test_leave_policy_retrieval_and_grounding(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """'what is our leave policy?' → retrieval + grounding + citations."""
        test_client, _ = client

        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="handbook.pdf",
            content="Vacation accrues at 20 days per year for full-time employees.",
            page_number=14,
            section_title="Leave Policy",
            chunk_index=0,
            rrf_score=0.02,
            rerank_score=0.9,
        )

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID, **kw: Any
        ) -> RetrievalResult:
            return RetrievalResult(chunks=[chunk], grounded=True, top_score=0.9)

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        captured_messages: list[Any] = []

        class _SpyLLM:
            name = "spy"
            model = "spy"

            async def stream(self, messages: Any, *, completion: Any) -> Any:
                captured_messages.extend(messages)
                completion.text = "You get 20 vacation days per year."
                completion.provider = "spy"
                completion.model = "spy"
                from app.llm.base import TokenUsage
                completion.usage = TokenUsage(prompt_tokens=50, completion_tokens=10)
                yield "You get 20 vacation days per year."

        monkeypatch.setattr("app.api.chat_v2.get_generic_llm", lambda: _SpyLLM())

        response = test_client.post("/chat/grounded", json={"message": "what is our leave policy?"})
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert len(body["sources"]) == 1
        assert body["sources"][0]["filename"] == "handbook.pdf"
        assert body["sources"][0]["page_number"] == 14
        # The LLM was called with the chunk content in context.
        assert len(captured_messages) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track_retrieval(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Monkeypatch retrieve to track calls. Returns the list that records calls."""
    calls: list[str] = []

    async def _tracking_retrieve(*args: Any, **kwargs: Any) -> RetrievalResult:
        calls.append("called")
        raise AssertionError("retrieval must NOT be called for this message")

    monkeypatch.setattr(chat_module, "retrieve", _tracking_retrieve)
    return calls


# ---------------------------------------------------------------------------
# Regex fast-path unit tests (no HTTP)
# ---------------------------------------------------------------------------


class TestRegexFastPath:
    """Pure-function tests for classify_intent_regex with the new patterns."""

    def test_who_are_you_fast_path(self) -> None:
        intent = classify_intent_regex("who are you")
        assert intent.category == IntentCategory.IDENTITY_ASSISTANT
        assert intent.reason == "identity_bot_fast_path"

    def test_what_are_you_fast_path(self) -> None:
        intent = classify_intent_regex("what are you")
        assert intent.category == IntentCategory.IDENTITY_ASSISTANT
        assert intent.reason == "identity_bot_fast_path"

    def test_how_manu_docs_typo_fast_path(self) -> None:
        intent = classify_intent_regex("how manu docs there")
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.reason == "doc_count_typo"

    def test_how_manyy_docs_typo_fast_path(self) -> None:
        intent = classify_intent_regex("how manyy documents are there")
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.reason == "doc_count_typo"

    def test_i_have_a_doubt_fast_path(self) -> None:
        intent = classify_intent_regex("i have a doubt")
        assert intent.category == IntentCategory.GENERAL_CONVERSATION

    def test_i_have_an_doubt_fast_path(self) -> None:
        intent = classify_intent_regex("i have an doubt")
        assert intent.category == IntentCategory.GENERAL_CONVERSATION

    def test_can_you_help_me_fast_path(self) -> None:
        intent = classify_intent_regex("can you help me")
        assert intent.category == IntentCategory.GENERAL_CONVERSATION

    def test_write_a_pyathon_code_fast_path(self) -> None:
        intent = classify_intent_regex("write a pyathon code")
        assert intent.category == IntentCategory.OUT_OF_SCOPE
        assert intent.reason == "code_request"

    def test_create_a_script_fast_path(self) -> None:
        intent = classify_intent_regex("create a javascript script")
        assert intent.category == IntentCategory.OUT_OF_SCOPE

    def test_hi_my_name_is_fast_path(self) -> None:
        intent = classify_intent_regex("hi my name is aarya")
        assert intent.category == IntentCategory.IDENTITY_USER
        assert intent.reason == "greeting_name_statement"

    def test_hey_my_name_is_fast_path(self) -> None:
        intent = classify_intent_regex("hey my name is john")
        assert intent.category == IntentCategory.IDENTITY_USER

    def test_existing_greeting_still_works(self) -> None:
        """'hey' without 'my name is' should still be GREETING."""
        intent = classify_intent_regex("hey")
        assert intent.category == IntentCategory.GREETING

    def test_existing_out_of_scope_still_works(self) -> None:
        intent = classify_intent_regex("capital of japan")
        assert intent.category == IntentCategory.OUT_OF_SCOPE

    def test_existing_metadata_still_works(self) -> None:
        intent = classify_intent_regex("how many documents are there")
        assert intent.category == IntentCategory.WORKSPACE_METADATA

    def test_document_content_still_works(self) -> None:
        intent = classify_intent_regex("what is our leave policy")
        assert intent.category == IntentCategory.DOCUMENT_CONTENT
