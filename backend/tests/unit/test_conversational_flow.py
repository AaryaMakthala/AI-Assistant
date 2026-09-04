"""Focused conversational flow integration tests (Tests A–G).

All external LLM/provider calls are mocked.  Never hits a real API.
Tests verify the actual chat endpoint behavior for follow-up scenarios,
ensuring the rewritten query is the canonical query used everywhere
downstream: metadata routing, document targeting, retrieval, and LLM.

Rewriting now lives in the LLM router: a RouteResult with needs_rewrite=True
and query=<rewritten> becomes Intent.rewritten_query inside classify_intent.
These tests drive that router seam and verify chat_v2 consumes
intent.rewritten_query consistently.

Test matrix:
A — Simple follow-up (pronoun resolution)
B — Document follow-up (document targeting + citation)
C — Metadata follow-up (member count → invited count)
D — Month follow-up (document count → last month)
E — Ambiguous follow-up (clarification requested)
F — Follow-up after refusal (clarification incorporated)
G — Rewrite provider failure (graceful degradation)
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.api.chat_v2 as chat_module
from app.api.dependencies import get_generic_llm
from app.llm.base import Completion, LLMError, Message, TokenUsage
from app.retrieval.intent import normalize_for_classification
from app.retrieval.llm_router import RouteResult
from app.retrieval.pipeline import RetrievedChunk, RetrievalResult
from app.security.auth import Principal, get_principal

pytestmark = pytest.mark.usefixtures("valid_env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    score: float = 0.9,
    content: str = "Kanban is a visual workflow management method.",
    *,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    filename: str = "DevOps Question Bank.docx",
    page_number: int | None = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        filename=filename,
        content=content,
        page_number=page_number,
        section_title="Kanban",
        chunk_index=0,
        rrf_score=0.02,
        rerank_score=score,
    )


def _make_rewrite_router(source: str, rewritten: str) -> Any:
    """Build a route_with_llm stand-in that rewrites one specific follow-up.

    The first call with ``source`` returns a needs_rewrite RouteResult
    carrying ``rewritten`` — which classify_intent turns into
    ``Intent.rewritten_query``, the seam chat_v2 now consumes.  Any other
    query (including the rewritten one during re-classification) is plain
    DOCUMENT_CONTENT.
    """

    async def _router(
        *, query: str, history: list | None = None, **kw: Any
    ) -> RouteResult:
        if normalize_for_classification(query) == normalize_for_classification(source):
            return RouteResult(
                route="DOCUMENT_CONTENT",
                needs_rewrite=True,
                query=rewritten,
                confidence=0.95,
                reasoning="test",
            )
        return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="test")

    return _router


class _StubLLM:
    """Scripted provider: returns fixed text, records calls."""

    name = "test-provider"
    model = "test-model"

    def __init__(self, text: str = "An answer.") -> None:
        self._text = text
        self._error: Exception | None = None
        self.calls: list[list[Message]] = []

    def fail_with(self, error: Exception) -> None:
        self._error = error

    async def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        completion.provider = self.name
        completion.model = self.model
        if self._error is not None:
            raise self._error
        completion.text = self._text
        completion.usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        yield self._text


class _FakeResult:
    def __init__(self, rows: list[Any] | None = None, scalar: Any = None) -> None:
        self._rows = rows or []
        self._scalar = scalar

    def scalar_one(self) -> Any:
        return self._scalar

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    def __init__(self, responses: list[_FakeResult] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_index = 0

    async def execute(self, stmt: Any) -> _FakeResult:
        if self._call_index < len(self._responses):
            result = self._responses[self._call_index]
            self._call_index += 1
            return result
        return _FakeResult()

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class _FakeRewriteProvider:
    """Fake LLM provider for the rewrite step."""

    def __init__(self, response_text: str) -> None:
        self._response = response_text
        self.name = "test-rewrite"
        self.model = "test-model"

    async def stream(
        self, messages: list[Any], *, completion: Any
    ) -> AsyncIterator[str]:
        completion.text = self._response
        completion.provider = self.name
        completion.model = self.model
        yield self._response


class _FailingRewriteProvider:
    """Fake LLM provider that always raises."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.name = "test-rewrite"
        self.model = "test-model"

    async def stream(
        self, messages: list[Any], *, completion: Any  # noqa: ARG002
    ) -> AsyncIterator[str]:
        raise self._error
        yield  # make it a generator  # noqa: RET503


def _make_rewrite_response(
    rewritten: str,
    *,
    needs_clarification: bool = False,
    confidence: float = 0.9,
) -> str:
    return json.dumps({
        "rewritten_query": rewritten,
        "needs_clarification": needs_clarification,
        "confidence": confidence,
    })


# ---------------------------------------------------------------------------
# Client fixture
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
    default_session = _FakeSession(responses=[
        _FakeResult(scalar=None),  # _load_recent_history: no session found
    ])

    def _make_tenant_session(
        *, workspace_id: uuid.UUID, user_id: uuid.UUID | None = None  # noqa: ARG001
    ) -> _FakeSession:
        return default_session

    monkeypatch.setattr(chat_module, "tenant_session", _make_tenant_session)

    # Mock the LLM router to avoid real API calls for intent classification.
    # Returns DOCUMENT_CONTENT by default (most conversational queries are doc queries).
    from app.retrieval.llm_router import RouteResult as _RouteResult

    async def _mock_route(*, query: str, history: list | None = None, **kw: Any) -> _RouteResult:
        from app.retrieval.intent import normalize_for_classification
        import re as _re
        q = normalize_for_classification(query)
        if _re.search(r"(?:about|discuss|cover|mention|regarding)\b", q):
            return _RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="test")
        if _re.search(r"(?:how\s+many|number\s+of)\s+(?:uploaded\s+)?(?:my\s+|the\s+|this\s+)?(?:own\s+)?(?:members?|documents?|files?|uploaded)", q):
            return _RouteResult(route="METADATA", confidence=0.9, reasoning="test")
        if _re.search(r"(?:list|show)\s+(?:are\s+the\s+)?(?:me\s+)?(?:all\s+)?(?:my\s+|the\s+|this\s+)?(?:uploaded\s+)?(?:members?|documents?|files?)\b", q):
            return _RouteResult(route="METADATA", confidence=0.9, reasoning="test")
        if _re.search(r"^what\s+(?:are|is)\s+(?:the\s+|my\s+)?(?:uploaded\s+)?(?:documents?|files?)\s*$", q):
            return _RouteResult(route="METADATA", confidence=0.9, reasoning="test")
        if _re.search(r"^how\s+many\s+are\s+(?:invited|pending|active|confirmed|removed)\s*$", q):
            return _RouteResult(route="METADATA", confidence=0.85, reasoning="test")
        if _re.search(r"^who\s+(?:is|are)\s+(?:invited|pending|active)\s*$", q):
            return _RouteResult(route="METADATA", confidence=0.85, reasoning="test")
        return _RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="test")

    monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _mock_route)

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, principal
    app.dependency_overrides.clear()


@pytest.fixture
def _valid_env() -> None:
    pass


# ===========================================================================
# TEST A — SIMPLE FOLLOW-UP
# ===========================================================================

class TestASimpleFollowUp:
    """Turn 1: 'What is Kanban?' -> Turn 2: 'What about its benefits?'

    Verify: rewrite occurs, relevance uses rewritten query,
    retrieval uses rewritten query, answer uses canonical query.
    """

    def test_a_simple_followup(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        test_client, _ = client

        # Track what query reaches retrieval.
        captured_queries: list[str] = []

        kanban_chunk = _chunk(
            0.9,
            content="Kanban benefits include improved visibility and reduced lead times.",
            filename="DevOps Question Bank.docx",
        )

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID,  # noqa: ARG001
            **kwargs: Any
        ) -> RetrievalResult:
            captured_queries.append(query)
            return RetrievalResult(
                chunks=[kanban_chunk], grounded=True, top_score=0.9
            )

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        # Router attaches the rewritten follow-up to the Intent.
        monkeypatch.setattr(
            "app.retrieval.llm_router.route_with_llm",
            _make_rewrite_router(
                "What about its benefits?",
                "What are the benefits of Kanban?",
            ),
        )

        stub = _StubLLM(text="Kanban benefits include improved visibility.")
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What about its benefits?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True

        # Verify rewrite was called and the resolved query reached retrieval.
        assert len(captured_queries) == 1
        assert "Kanban" in captured_queries[0]
        assert "benefit" in captured_queries[0].lower()
        # The LLM should have received the rewritten query.
        assert len(stub.calls) == 1
        user_msg = stub.calls[0][-1].content
        assert "Kanban" in user_msg
        assert "benefit" in user_msg.lower()


# ===========================================================================
# TEST B — DOCUMENT FOLLOW-UP
# ===========================================================================

class TestDocumentFollowUp:
    """Turn 1: 'Tell me about the DevOps document.' ->
    Turn 2: 'What questions about it mention Kanban?'

    Verify: document targeting identifies DevOps document, document_id filter
    reaches retrieval, citation belongs to DevOps document.
    """

    def test_b_document_followup(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        test_client, _ = client

        devops_doc_id = uuid.uuid4()

        devops_chunk = _chunk(
            0.85,
            content="Kanban boards visualize work in progress.",
            document_id=devops_doc_id,
            filename="DevOps Question Bank.docx",
            page_number=3,
        )

        captured_queries: list[str] = []

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID,  # noqa: ARG001
            **kwargs: Any
        ) -> RetrievalResult:
            captured_queries.append(query)
            return RetrievalResult(
                chunks=[devops_chunk], grounded=True, top_score=0.85
            )

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        # Router rewrites the bare follow-up into a DevOps-scoped question.
        monkeypatch.setattr(
            "app.retrieval.llm_router.route_with_llm",
            _make_rewrite_router(
                "What questions about it mention Kanban?",
                "What questions about Kanban are present in the DevOps document?",
            ),
        )

        stub = _StubLLM(
            text="The DevOps document contains questions about Kanban boards."
        )
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What questions about it mention Kanban?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True

        # Verify rewrite resolved the reference.
        assert len(captured_queries) == 1
        assert "DevOps" in captured_queries[0]
        assert "Kanban" in captured_queries[0]

        # Citation must belong to the DevOps document.
        assert len(body["sources"]) == 1
        assert body["sources"][0]["filename"] == "DevOps Question Bank.docx"
        assert body["sources"][0]["document_id"] == str(devops_doc_id)


# ===========================================================================
# TEST C — METADATA FOLLOW-UP
# ===========================================================================

class TestMetadataFollowUp:
    """Turn 1: 'How many members are in the workspace?' ->
    Turn 2: 'How many are invited?'

    Verify: metadata routing is used, retrieval is NOT called,
    generation is NOT called.
    """

    def test_c_metadata_followup(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        test_client, _ = client

        # No rewrite needed — metadata intents skip rewrite in the new code path.
        # "How many are invited?" is classified directly as member_count+INVITED.

        # Set up tenant_session with member count response.
        # Only one call: _answer_metadata_question (member_count query).
        session = _FakeSession(responses=[
            _FakeResult(scalar=None),  # _load_recent_history: no session found
            _FakeResult(scalar=3),     # member_count query result
        ])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        # Verify retrieval is NOT called.
        retrieval_called: list[str] = []

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID,  # noqa: ARG001
            **kwargs: Any
        ) -> RetrievalResult:
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run for metadata question")

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = _StubLLM(text="should not be called")
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "How many are invited?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        # The answer should contain the count (3) and mention invited members.
        assert "3" in body["answer"]
        assert body["sources"] == []
        # Retrieval was NOT called.
        assert retrieval_called == []
        # LLM was NOT called.
        assert stub.calls == []


# ===========================================================================
# TEST D — MONTH FOLLOW-UP
# ===========================================================================

class TestMonthFollowUp:
    """Turn 1: 'How many documents were uploaded this month?' ->
    Turn 2: 'What about last month?'

    Verify: metadata routing or existing date-query path handles it.
    The literal query 'What about last month?' must NOT be treated as
    unrelated merely because it contains little semantic content.
    """

    def test_d_month_followup(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        test_client, _ = client

        # Router attaches a rewritten doc-count query; re-classifying the
        # rewritten text hits the metadata regex fast-path (no router call).
        monkeypatch.setattr(
            "app.retrieval.llm_router.route_with_llm",
            _make_rewrite_router(
                "What about last month?",
                "How many documents were uploaded last month?",
            ),
        )

        # Set up tenant_session for metadata count.
        # "What about last month?" starts as DOCUMENT_CONTENT (no metadata match),
        # so history loading runs first, then rewrite produces a metadata query.
        session = _FakeSession(responses=[
            _FakeResult(scalar=None),  # _load_recent_history: no session
            _FakeResult(scalar=5),     # metadata count query
        ])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called: list[str] = []

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID,  # noqa: ARG001
            **kwargs: Any
        ) -> RetrievalResult:
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run for metadata question")

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = _StubLLM(text="should not be called")
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What about last month?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "5" in body["answer"]
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []


# ===========================================================================
# TEST E — AMBIGUOUS FOLLOW-UP
# ===========================================================================

class TestAmbiguousFollowUp:
    """Turn 1: 'Tell me about the DevOps document and the Security Policy.' ->
    Turn 2: 'How many are there?'

    Verify: no arbitrary document chosen, no false not_relevant refusal,
    no retrieval based on fabricated interpretation, user receives
    clarification request.
    """

    def test_e_ambiguous_followup_clarification(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        test_client, _ = client

        # Ambiguity now surfaces from the router: a NEEDS_CLARIFICATION
        # RouteResult maps to Intent.needs_clarification, and chat_v2 replies
        # with a clarification request (no retrieval, no LLM).
        async def _ambiguous_router(
            *, query: str, history: list | None = None, **kw: Any
        ) -> RouteResult:
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.9,
                reasoning="test",
            )

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _ambiguous_router)

        # Verify retrieval is NOT called.
        retrieval_called: list[str] = []

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID,  # noqa: ARG001
            **kwargs: Any
        ) -> RetrievalResult:
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run for ambiguous query")

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = _StubLLM(text="should not be called")
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "How many are there?"},
        )
        assert response.status_code == 200
        body = response.json()
        # Should get a clarification response.
        assert "clarif" in body["answer"].lower()
        assert body["sources"] == []
        # Retrieval was NOT called.
        assert retrieval_called == []
        # LLM was NOT called.
        assert stub.calls == []


# ===========================================================================
# TEST F — FOLLOW-UP AFTER REFUSAL
# ===========================================================================

class TestFollowUpAfterRefusal:
    """Turn 1: 'How many members are in this?' (ambiguous) ->
    Turn 2: 'I mean the workspace.'

    Verify: the second request is correctly routed to metadata handling.
    The clarification 'the workspace' resolves the reference.
    """

    def test_f_followup_after_refusal(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        test_client, _ = client

        # Router rewrites the clarification into a member-count query; the
        # rewritten text re-classifies via the metadata regex fast-path.
        monkeypatch.setattr(
            "app.retrieval.llm_router.route_with_llm",
            _make_rewrite_router(
                "I mean the workspace.",
                "How many members are in this workspace?",
            ),
        )

        session = _FakeSession(responses=[
            _FakeResult(scalar=None),  # _load_recent_history: no session
            _FakeResult(scalar=7),     # member_count query
        ])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called: list[str] = []

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID,  # noqa: ARG001
            **kwargs: Any
        ) -> RetrievalResult:
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run for metadata question")

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = _StubLLM(text="should not be called")
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "I mean the workspace."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "7" in body["answer"]
        assert "member" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []


# ===========================================================================
# TEST G — REWRITE PROVIDER FAILURE
# ===========================================================================

class TestRewriteProviderFailure:
    """Force the router (which now carries rewriting) to fail.  Use a query
    with an obvious follow-up pronoun.

    Expected: original query is preserved, request continues,
    no internal error exposed, no false not_relevant decision solely
    because the router degraded.
    """

    def test_g_rewrite_failure_graceful(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        test_client, _ = client

        # Rewriting now lives inside the LLM router.  Simulate the router
        # failing outright: a degraded RouteResult makes classify_intent fall
        # back to regex (Stage 2), which leaves this follow-up as plain
        # DOCUMENT_CONTENT with no rewritten query — the original question
        # must flow through unchanged, with no error and no false refusal.
        async def _degraded_router(
            *, query: str, history: list | None = None, **kw: Any
        ) -> RouteResult:
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.0,
                reasoning="llm_error: provider unreachable",
                status="degraded",
            )

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _degraded_router)

        # Track what query reaches retrieval — should be the ORIGINAL query.
        captured_queries: list[str] = []

        kanban_chunk = _chunk(
            0.9,
            content="Kanban is a workflow method.",
            filename="DevOps Question Bank.docx",
        )

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID,  # noqa: ARG001
            **kwargs: Any
        ) -> RetrievalResult:
            captured_queries.append(query)
            return RetrievalResult(
                chunks=[kanban_chunk], grounded=True, top_score=0.9
            )

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = _StubLLM(text="Kanban is a workflow management method.")
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What about its benefits?"},
        )
        # Should NOT return an error — graceful degradation.
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True

        # The original query should have been used (rewrite failed -> fallback).
        assert len(captured_queries) == 1
        assert captured_queries[0] == "What about its benefits?"

        # LLM should still be called with the original query.
        assert len(stub.calls) == 1
        user_msg = stub.calls[0][-1].content
        assert "What about its benefits?" in user_msg

    def test_g_rewrite_failure_no_false_not_relevant(
        self, monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, Principal]
    ) -> None:
        """No rewrite signal from the router must not cause a refusal.

        The fixture router returns plain DOCUMENT_CONTENT (no needs_rewrite),
        so intent.rewritten_query is None and the original question proceeds
        through retrieval to a grounded answer — not a not_relevant refusal.
        """
        test_client, _ = client

        kanban_chunk = _chunk(0.9, content="Kanban is a workflow method.")

        async def _retrieve(
            session: Any, *, query: str, workspace_id: uuid.UUID,  # noqa: ARG001
            **kwargs: Any
        ) -> RetrievalResult:
            return RetrievalResult(
                chunks=[kanban_chunk], grounded=True, top_score=0.9
            )

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = _StubLLM(text="Kanban is a workflow method.")
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What does it say about Kanban?"},
        )
        assert response.status_code == 200
        body = response.json()
        # Must NOT be refused — the rewrite failure is not a relevance issue.
        assert body["grounded"] is True
        assert len(body["sources"]) == 1
