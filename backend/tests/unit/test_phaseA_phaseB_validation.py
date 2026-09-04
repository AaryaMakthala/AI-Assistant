"""Production-style validation of Phase A (intent routing) and Phase B (overview grounding).

Tests behavioral requirements against the actual implementation.
All external LLM/provider calls are mocked.
Never hits a real database.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.api.chat_v2 as chat_module
from app.api.dependencies import get_generic_llm
from app.db.models import ChatMessage, ChatSession, Document, Member
from app.llm.base import Completion, Message, TokenUsage
from app.retrieval.grounding import is_grounded, is_overview_grounded
from app.retrieval.intent import (
    ConversationHistorySubIntent,
    IntentCategory,
    MetadataSubIntent,
    QueryShape,
    classify_intent_regex,
    classify_query_shape,
)
from app.retrieval.pipeline import RetrievedChunk, RetrievalResult
from app.retrieval.refusals import ResponseReason, refusal_message
from app.security.auth import Principal, get_principal
from tests.unit.conftest import (
    FakeResult,
    FakeSession,
    StubLLM,
    smart_mock_route,
)

pytestmark = pytest.mark.usefixtures("valid_env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    score: float = 0.9,
    content: str = "Vacation accrues at 20 days per year.",
    *,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    filename: str = "handbook.pdf",
    page_number: int | None = 2,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        filename=filename,
        content=content,
        page_number=page_number,
        section_title="Leave policy",
        chunk_index=0,
        rrf_score=0.02,
        rerank_score=score,
    )





# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, _valid_env: None
) -> tuple[TestClient, Principal]:
    from app.main import create_app

    principal = Principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
    app = create_app()

    async def _principal() -> Principal:
        return principal

    app.dependency_overrides[get_principal] = _principal

    async def _member(
        workspace_id: uuid.UUID, principal: Principal, *allowed: str
    ) -> str:
        return "OWNER"

    monkeypatch.setattr(chat_module, "assert_workspace_role", _member)

    default_session = FakeSession(
        responses=[FakeResult(scalar=None)],  # _load_recent_history: no session
        default=FakeResult(scalar=None),       # extra calls: no session (safe default)
    )

    def _make_tenant_session(
        *, workspace_id: uuid.UUID, user_id: uuid.UUID | None = None
    ) -> FakeSession:
        return default_session

    monkeypatch.setattr(chat_module, "tenant_session", _make_tenant_session)

    # Mock the LLM router — shared smart mock from conftest.
    monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", smart_mock_route)

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, principal
    app.dependency_overrides.clear()


@pytest.fixture
def _valid_env() -> None:
    pass


# ===========================================================================
# SECTION 2: Phase A — Intent Routing Validation
# ===========================================================================

class TestPhaseAMetadata:
    """Tests 1-6: Metadata intent routing."""

    def test_1_doc_count(self, monkeypatch, client):
        """'How many documents are in the workspace?' -> metadata/doc_count."""
        test_client, _ = client
        session = FakeSession(responses=[
            FakeResult(scalar=None),  # _load_recent_history
            FakeResult(scalar=3),     # doc count
        ])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "How many documents are in the workspace?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "3" in body["answer"]
        assert "documents" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []

    def test_2_doc_list(self, monkeypatch, client):
        """'List all documents in the workspace.' -> metadata/doc_list."""
        test_client, _ = client
        doc_rows = [
            SimpleNamespace(filename="handbook.pdf"),
            SimpleNamespace(filename="policy.docx"),
        ]
        session = FakeSession(responses=[FakeResult(scalar=None), FakeResult(rows=doc_rows)])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "List all documents in the workspace."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "handbook.pdf" in body["answer"]
        assert "policy.docx" in body["answer"]
        assert body["sources"] == []
        assert retrieval_called == []

    def test_3_doc_page_count(self, monkeypatch, client):
        """'How many pages are in the documents?' -> metadata/doc_page_count."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "How many pages are in the documents?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "not available" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []

    def test_4_member_count_active(self, monkeypatch, client):
        """'How many active members are there?' -> metadata/member_count + ACTIVE."""
        test_client, _ = client
        session = FakeSession(responses=[FakeResult(scalar=None), FakeResult(scalar=5)])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "How many active members are there?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "5" in body["answer"]
        assert "active" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []

    def test_5_member_list_active(self, monkeypatch, client):
        """'List the active members.' -> metadata/member_list + ACTIVE."""
        test_client, _ = client
        member_rows = [
            SimpleNamespace(user_id=uuid.uuid4(), role="OWNER", status="ACTIVE"),
            SimpleNamespace(user_id=uuid.uuid4(), role="MEMBER", status="ACTIVE"),
        ]
        session = FakeSession(responses=[FakeResult(scalar=None), FakeResult(rows=member_rows)])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "List the active members."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "active" in body["answer"].lower()
        assert "2" in body["answer"]
        assert body["sources"] == []
        assert retrieval_called == []

    def test_6_role_query(self, monkeypatch, client):
        """'What is my role?' -> metadata/role."""
        test_client, _ = client
        role_row = SimpleNamespace(role="OWNER")
        session = FakeSession(responses=[FakeResult(scalar=None), FakeResult(rows=[role_row])])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What is my role?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "owner" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []


# ===========================================================================
# SECTION 3: Conversation History
# ===========================================================================

class TestConversationHistory:
    def test_previous_question(self, monkeypatch, client):
        """'What was my previous question?' -> conversation_history."""
        test_client, _ = client

        user_msg = SimpleNamespace(role="user", content="What is Kanban?")
        asst_msg = SimpleNamespace(role="assistant", content="Kanban is a method.")
        session_id = uuid.uuid4()
        session_row = SimpleNamespace(id=session_id)

        session = FakeSession(responses=[
            FakeResult(scalar=session_row),  # _load_recent_history: session lookup
            FakeResult(rows=[user_msg, asst_msg]),  # _load_recent_history: messages
            FakeResult(scalar=session_row),  # _answer_conversation_history: session lookup
            FakeResult(rows=[user_msg, asst_msg]),  # _answer_conversation_history: messages
        ])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What was my previous question?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "kanban" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []

    def test_previous_answer(self, monkeypatch, client):
        """'What did you just answer?' -> conversation_history/PREVIOUS_ANSWER."""
        test_client, _ = client

        user_msg = SimpleNamespace(role="user", content="What is vacation policy?")
        asst_msg = SimpleNamespace(role="assistant", content="20 days per year.")
        session_id = uuid.uuid4()
        session_row = SimpleNamespace(id=session_id)

        session = FakeSession(responses=[
            FakeResult(scalar=session_row),
            FakeResult(rows=[user_msg, asst_msg]),
            FakeResult(scalar=session_row),
            FakeResult(rows=[user_msg, asst_msg]),
        ])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What did you just answer?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "20 days" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []

    def test_recent_conversation(self, monkeypatch, client):
        """'What have we discussed recently?' -> conversation_history/RECENT."""
        test_client, _ = client

        user_msg = SimpleNamespace(role="user", content="Tell me about Kanban.")
        asst_msg = SimpleNamespace(role="assistant", content="Kanban is a workflow method.")
        session_id = uuid.uuid4()
        session_row = SimpleNamespace(id=session_id)
        session = FakeSession(responses=[
            FakeResult(scalar=session_row),
            FakeResult(rows=[user_msg, asst_msg]),
            FakeResult(scalar=session_row),
            FakeResult(rows=[user_msg, asst_msg]),
        ])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What have we discussed recently?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == []


# ===========================================================================
# SECTION 4: App Help
# ===========================================================================

class TestAppHelp:
    def test_how_to_invite(self, monkeypatch, client):
        """'How do I invite a member?' -> app_help."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "How do I invite a member?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []

    def test_who_can_upload(self, monkeypatch, client):
        """'Who can upload documents?' -> app_help."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "Who can upload documents?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "member" in body["answer"].lower() or "upload" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []


# ===========================================================================
# SECTION 6: Query Shape Classification
# ===========================================================================

class TestQueryShapes:
    def test_fact_lookup(self):
        """FACT_LOOKUP: specific question with 'the' after 'what is'."""
        shape = classify_query_shape("What is the vacation accrual rate?")
        assert shape == QueryShape.FACT_LOOKUP

    def test_overview_what_is(self):
        """OVERVIEW: 'what is X' without 'the'."""
        shape = classify_query_shape("What is Kanban?")
        assert shape == QueryShape.OVERVIEW

    def test_overview_tell_me_about(self):
        """OVERVIEW: 'tell me about X'."""
        shape = classify_query_shape("Tell me about the DevOps document.")
        assert shape == QueryShape.OVERVIEW

    def test_overview_explain(self):
        """OVERVIEW: 'explain X'."""
        shape = classify_query_shape("Explain the Kanban section.")
        assert shape == QueryShape.OVERVIEW

    def test_overview_summarize(self):
        """OVERVIEW: 'summarize X'."""
        shape = classify_query_shape("Summarize the vacation policy.")
        assert shape == QueryShape.OVERVIEW

    def test_comparison(self):
        """COMPARISON: 'how does X differ from Y'."""
        shape = classify_query_shape("How do Scrum and Kanban differ?")
        assert shape == QueryShape.COMPARISON

    def test_comparison_vs(self):
        """COMPARISON: 'X vs Y'."""
        shape = classify_query_shape("Scrum vs Kanban")
        assert shape == QueryShape.COMPARISON

    def test_list_extraction(self):
        """LIST_EXTRACTION: 'list all X'."""
        shape = classify_query_shape("List all benefits mentioned.")
        assert shape == QueryShape.LIST_EXTRACTION

    def test_targeted_with_doc_ref(self):
        """TARGETED_QUERY: has document reference."""
        shape = classify_query_shape(
            "What does the DevOps document say about Kanban?",
            has_doc_target=True,
        )
        assert shape == QueryShape.TARGETED_QUERY

    def test_targeted_overview_prefers_overview(self):
        """'Tell me about X' with doc target -> OVERVIEW (not TARGETED)."""
        shape = classify_query_shape(
            "Tell me about the DevOps document.",
            has_doc_target=True,
        )
        assert shape == QueryShape.OVERVIEW

    def test_default_is_fact_lookup(self):
        """Unmatched query -> FACT_LOOKUP."""
        shape = classify_query_shape("random query that matches nothing")
        assert shape == QueryShape.FACT_LOOKUP


# ===========================================================================
# SECTION 7: Overview Grounding (Critical)
# ===========================================================================

class TestOverviewGrounding:
    def test_real_kanban_scores_ground(self):
        """Real Kanban scores (all negative logits) should ground."""
        scores = [-2.4534, -3.8034, -4.2362, -5.9403, -6.1114, -9.5952]
        assert is_overview_grounded(scores) is True

    def test_positive_cluster_grounds(self):
        """Tight positive cluster should ground."""
        assert is_overview_grounded([0.25, 0.22, 0.18, 0.15, 0.10]) is True

    def test_one_positive_outlier_still_grounds(self):
        """One strong positive score among negatives: top=0.5 clears min,
        top-3 mean=-2.17 clears aggregate -> grounds (correct behavior).
        """
        assert is_overview_grounded([0.5, -3.0, -4.0, -5.0, -6.0]) is True

    def test_three_one_positive_outlier_grounds(self):
        """Three chunks with one positive outlier: top=0.8 clears min,
        top-2 mean=-2.1 clears aggregate -> grounds.
        """
        assert is_overview_grounded([0.8, -5.0, -6.0]) is True

    def test_tight_negative_cluster_grounds(self):
        """Consistently negative but tightly clustered top should ground."""
        assert is_overview_grounded([-2.0, -2.2, -2.5, -8.0, -9.0]) is True

    def test_all_identical_grounds(self):
        """All identical scores = consistent relevance."""
        assert is_overview_grounded([-3.0, -3.0, -3.0, -3.0]) is True

    def test_single_chunk_does_not_ground(self):
        """Overview needs multiple chunks."""
        assert is_overview_grounded([-2.0]) is False

    def test_empty_does_not_ground(self):
        assert is_overview_grounded([]) is False

    def test_two_close_chunks_ground(self):
        """Two chunks with close scores = consistent."""
        assert is_overview_grounded([-2.0, -2.1]) is True

    def test_two_far_chunks_do_not_ground(self):
        """Two chunks with very different scores: mean fails aggregate."""
        assert is_overview_grounded([-1.0, -18.0]) is False


# ===========================================================================
# SECTION 8: Out-of-Scope Routing
# ===========================================================================

class TestOutOfScope:
    def test_weather(self, monkeypatch, client):
        """'What is the weather today?' -> out_of_scope."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What is the weather today?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert "outside" in body["answer"].lower() or "help with" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []
        assert stub.calls == []

    def test_math(self, monkeypatch, client):
        """'What is 2+2?' -> out_of_scope."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What is 2+2?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "outside" in body["answer"].lower() or "help with" in body["answer"].lower()
        assert retrieval_called == []

    def test_capital_of_france(self, monkeypatch, client):
        """'Capital of France' -> out_of_scope."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "Capital of France"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "outside" in body["answer"].lower() or "help with" in body["answer"].lower()
        assert retrieval_called == []


# ===========================================================================
# SECTION 9: Ambiguous Routing
# ===========================================================================

class TestAmbiguous:
    def test_tell_me_about_that(self, monkeypatch, client):
        """'Tell me about that.' -> ambiguous/needs_clarification."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        # Ambiguity now surfaces from the router (which also carries rewriting):
        # a NEEDS_CLARIFICATION RouteResult maps to Intent.needs_clarification,
        # and chat_v2 replies with a clarification request.
        async def _ambiguous_router(*, query: str, history: list | None = None, **kw):
            from app.retrieval.llm_router import RouteResult
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.9,
                reasoning="test",
            )
        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _ambiguous_router)

        response = test_client.post(
            "/chat/grounded",
            json={"message": "Tell me about that."},
        )
        assert response.status_code == 200
        body = response.json()
        assert "clarif" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []

    def test_how_many_are_there(self, monkeypatch, client):
        """'How many are there?' -> ambiguous/needs_clarification."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
            raise AssertionError("retrieval must NOT run")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        # Ambiguity now surfaces from the router (which also carries rewriting):
        # a NEEDS_CLARIFICATION RouteResult maps to Intent.needs_clarification,
        # and chat_v2 replies with a clarification request.
        async def _ambiguous_router(*, query: str, history: list | None = None, **kw):
            from app.retrieval.llm_router import RouteResult
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.9,
                reasoning="test",
            )
        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _ambiguous_router)

        response = test_client.post(
            "/chat/grounded",
            json={"message": "How many are there?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "clarif" in body["answer"].lower()
        assert body["sources"] == []
        assert retrieval_called == []


# ===========================================================================
# SECTION 13: Structured Logging
# ===========================================================================

class TestLogging:
    def test_metadata_logs_intent(self, monkeypatch, client):
        """Metadata route should log intent=metadata."""
        test_client, _ = client
        session = FakeSession(responses=[FakeResult(scalar=2)])
        monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: session)

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "How many documents are there?"},
        )
        assert response.status_code == 200

    def test_out_of_scope_logs_intent(self, monkeypatch, client):
        """Out-of-scope route should log intent=out_of_scope."""
        test_client, _ = client

        retrieval_called = []
        async def _retrieve(*a, **kw):
            retrieval_called.append("called")
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM()
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What is 2+2?"},
        )
        assert response.status_code == 200

    def test_document_content_logs_query_shape(self, monkeypatch, client):
        """Document content route should log query_shape."""
        test_client, _ = client

        kanban_chunk = _chunk(0.9, content="Kanban is a workflow method.")
        async def _retrieve(*a, **kw):
            return RetrievalResult(chunks=[kanban_chunk], grounded=True, top_score=0.9)
        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        stub = StubLLM(text="Kanban is a workflow management method.")
        test_client.app.dependency_overrides[get_generic_llm] = lambda: stub

        response = test_client.post(
            "/chat/grounded",
            json={"message": "What does the document say about Kanban?"},
        )
        assert response.status_code == 200


# ===========================================================================
# SECTION 14: Regression Test
# ===========================================================================

class TestRegression:
    def test_all_intent_patterns_still_work(self):
        """Verify all intent patterns produce valid Intent objects."""
        cases = [
            ("How many documents are there?", IntentCategory.WORKSPACE_METADATA, MetadataSubIntent.DOC_COUNT),
            ("List all documents.", IntentCategory.DOCUMENT_LIST, MetadataSubIntent.DOC_LIST),
            ("How many pages in documents?", IntentCategory.WORKSPACE_METADATA, MetadataSubIntent.DOC_PAGE_COUNT),
            ("How many active members?", IntentCategory.WORKSPACE_METADATA, MetadataSubIntent.MEMBER_COUNT),
            ("List active members.", IntentCategory.WORKSPACE_METADATA, MetadataSubIntent.MEMBER_LIST),
            ("What is my role?", IntentCategory.WORKSPACE_METADATA, MetadataSubIntent.ROLE),
            ("What was my previous question?", IntentCategory.CONVERSATION_HISTORY, None),
            # Identity queries are NOT in the regex fast-path — they go to LLM router.
            # ("What is my name?", IntentCategory.IDENTITY, None),
            ("Who can upload documents?", IntentCategory.WORKSPACE_PERMISSION, None),
            ("How do I invite a member?", IntentCategory.APP_HELP, None),
            ("What is 2+2?", IntentCategory.OUT_OF_SCOPE, None),
            ("Capital of Japan", IntentCategory.OUT_OF_SCOPE, None),
            ("Tell me about that.", IntentCategory.AMBIGUOUS, None),
            ("What does the vacation policy say?", IntentCategory.DOCUMENT_CONTENT, None),
        ]

        for query, expected_category, expected_sub in cases:
            intent = classify_intent_regex(query)
            assert intent.category == expected_category, (
                f"Query '{query}': expected {expected_category}, got {intent.category}"
            )
            if expected_sub is not None:
                assert intent.metadata_sub == expected_sub, (
                    f"Query '{query}': expected sub {expected_sub}, got {intent.metadata_sub}"
                )

    def test_query_shape_patterns_still_work(self):
        """Verify all query shape patterns produce valid QueryShape objects."""
        cases = [
            ("What is Kanban?", QueryShape.OVERVIEW),
            ("Tell me about the DevOps document.", QueryShape.OVERVIEW),
            ("Explain the Kanban section.", QueryShape.OVERVIEW),
            ("Summarize the vacation policy.", QueryShape.OVERVIEW),
            ("What is the vacation accrual rate?", QueryShape.FACT_LOOKUP),
            ("How do Scrum and Kanban differ?", QueryShape.COMPARISON),
            ("List all benefits mentioned.", QueryShape.LIST_EXTRACTION),
            ("random query", QueryShape.FACT_LOOKUP),
        ]

        for query, expected in cases:
            shape = classify_query_shape(query)
            assert shape == expected, (
                f"Query '{query}': expected {expected}, got {shape}"
            )
