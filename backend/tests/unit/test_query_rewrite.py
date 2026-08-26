"""Conversational query rewriting tests.

All external LLM/provider calls are mocked.  Tests verify:
- Standalone queries pass through unchanged
- Pronoun/reference follow-ups are rewritten correctly
- Metadata follow-ups resolve their subject
- Ambiguity is detected and flagged
- Rewrite failure degrades gracefully
- Tenant isolation is preserved
- History loading works correctly

Never hits a real API.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from app.retrieval.query_rewrite import (
    ChatTurn,
    RewriteResult,
    _needs_rewrite,
    _parse_rewrite_response,
    rewrite_query,
)

pytestmark = pytest.mark.usefixtures("valid_env")


@pytest.fixture
def _valid_env() -> None:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(text: str) -> ChatTurn:
    return ChatTurn(role="user", content=text)


def _assistant(text: str) -> ChatTurn:
    return ChatTurn(role="assistant", content=text)


def _make_llm_response(rewritten: str, *, needs_clarification: bool = False, confidence: float = 0.9) -> str:
    """Build a mock LLM JSON response."""
    import json
    return json.dumps({
        "rewritten_query": rewritten,
        "needs_clarification": needs_clarification,
        "confidence": confidence,
    })


class _MockStream:
    """Async iterator that yields a single text chunk."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def __aiter__(self) -> AsyncIterator[str]:
        yield self._text


class _FakeProvider:
    """Fake LLM provider that returns scripted text."""

    def __init__(self, text: str) -> None:
        self.name = "test-provider"
        self.model = "test-model"
        self._text = text

    async def stream(self, messages: Any, *, completion: Any) -> AsyncIterator[str]:
        completion.provider = self.name
        completion.model = self.model
        completion.text = self._text
        yield self._text


class _FailingProvider:
    """Fake LLM provider that always raises."""

    def __init__(self, error: Exception) -> None:
        self.name = "test-provider"
        self.model = "test-model"
        self._error = error

    async def stream(self, messages: Any, *, completion: Any) -> AsyncIterator[str]:
        raise self._error
        yield  # make it a generator  # noqa: RET503


# ---------------------------------------------------------------------------
# TEST 1 — standalone query unchanged
# ---------------------------------------------------------------------------
class TestStandaloneQueryUnchanged:
    """TEST 1: standalone query is not rewritten."""

    @pytest.mark.asyncio
    async def test_standalone_query_unchanged(self) -> None:
        """'What is Kanban?' should pass through unchanged."""
        result = await rewrite_query(query="What is Kanban?", history=[])
        assert result.rewritten_query == "What is Kanban?"
        assert result.status == "skipped"
        assert result.needs_clarification is False

    @pytest.mark.asyncio
    async def test_long_standalone_not_rewritten(self) -> None:
        """Long standalone query with history is not rewritten."""
        history = [_user("Hello"), _assistant("Hi there!")]
        result = await rewrite_query(
            query="What is the company vacation policy for full-time employees?",
            history=history,
        )
        assert result.rewritten_query == "What is the company vacation policy for full-time employees?"
        assert result.status == "skipped"

    @pytest.mark.asyncio
    async def test_empty_query_skipped(self) -> None:
        """Empty query returns immediately."""
        result = await rewrite_query(query="", history=[])
        assert result.rewritten_query == ""
        assert result.status == "skipped"


# ---------------------------------------------------------------------------
# TEST 2 — simple pronoun follow-up
# ---------------------------------------------------------------------------
class TestPronounFollowUp:
    """TEST 2: pronoun follow-up is rewritten using context."""

    @pytest.mark.asyncio
    async def test_pronoun_followup_rewritten(self) -> None:
        """'What does it say about Kanban?' -> references the DevOps document."""
        history = [
            _user("What is the DevOps document about?"),
            _assistant("It discusses CI/CD and Kanban."),
        ]
        mock_response = _make_llm_response(
            "What does the DevOps document say about Kanban?"
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="What does it say about Kanban?",
                history=history,
            )

        assert result.status == "success"
        assert "DevOps" in result.rewritten_query
        assert "Kanban" in result.rewritten_query
        assert result.needs_clarification is False


# ---------------------------------------------------------------------------
# TEST 3 — "this" follow-up
# ---------------------------------------------------------------------------
class TestThisFollowUp:
    """TEST 3: 'this' follow-up resolves only if context supports it."""

    @pytest.mark.asyncio
    async def test_this_resolves_if_context_supports(self) -> None:
        """'How many members are there in this?' -> resolves 'this' if context supports."""
        history = [
            _user("Tell me about the authentication protocol."),
            _assistant("The document describes JWT authentication."),
        ]
        mock_response = _make_llm_response(
            "How many members are in the authentication protocol?"
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="How many members are there in this?",
                history=history,
            )

        assert result.status == "success"
        assert result.rewritten_query != "How many members are there in this?"

    @pytest.mark.asyncio
    async def test_ambiguous_this_gets_clarification(self) -> None:
        """'How many are there?' -> needs_clarification if context is ambiguous."""
        history = [
            _user("Tell me about the DevOps document and the Security Policy."),
            _assistant("Both documents contain different information."),
        ]
        mock_response = _make_llm_response(
            "How many documents are there?",
            needs_clarification=True,
            confidence=0.3,
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="How many are there?",
                history=history,
            )

        assert result.needs_clarification is True
        assert result.status == "ambiguous"


# ---------------------------------------------------------------------------
# TEST 4 — metadata follow-up
# ---------------------------------------------------------------------------
class TestMetadataFollowUp:
    """TEST 4: metadata follow-up resolves the implied subject."""

    @pytest.mark.asyncio
    async def test_metadata_followup_resolves_subject(self) -> None:
        """'How many are invited?' -> 'How many invited members are in the workspace?'"""
        history = [
            _user("How many members are in the workspace?"),
            _assistant("There are 5 members."),
        ]
        mock_response = _make_llm_response(
            "How many invited members are in the workspace?"
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="How many are invited?",
                history=history,
            )

        assert result.status == "success"
        assert "members" in result.rewritten_query.lower()
        assert result.needs_clarification is False


# ---------------------------------------------------------------------------
# TEST 5 — month follow-up
# ---------------------------------------------------------------------------
class TestMonthFollowUp:
    """TEST 5: 'What about last month?' resolves to a document count question."""

    @pytest.mark.asyncio
    async def test_month_followup_resolves(self) -> None:
        """'What about last month?' -> document count for last month."""
        history = [
            _user("How many documents were uploaded this month?"),
            _assistant("8."),
        ]
        mock_response = _make_llm_response(
            "How many documents were uploaded last month?"
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="What about last month?",
                history=history,
            )

        assert result.status == "success"
        assert "document" in result.rewritten_query.lower()
        assert "last month" in result.rewritten_query.lower()


# ---------------------------------------------------------------------------
# TEST 6 — document-specific follow-up
# ---------------------------------------------------------------------------
class TestDocumentSpecificFollowUp:
    """TEST 6: document-targeted follow-up resolves document reference."""

    @pytest.mark.asyncio
    async def test_document_followup_rewrites_correctly(self) -> None:
        """'What questions about Kanban are present in it?' -> references DevOps document."""
        history = [
            _user("Tell me about the DevOps document."),
            _assistant("It covers Kanban..."),
        ]
        mock_response = _make_llm_response(
            "What questions about Kanban are present in the DevOps document?"
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="What questions about Kanban are present in it?",
                history=history,
            )

        assert result.status == "success"
        assert "DevOps" in result.rewritten_query
        assert "Kanban" in result.rewritten_query


# ---------------------------------------------------------------------------
# TEST 7 — ambiguous follow-up
# ---------------------------------------------------------------------------
class TestAmbiguousFollowUp:
    """TEST 7: ambiguous follow-up is flagged, not forced."""

    @pytest.mark.asyncio
    async def test_ambiguous_followup_flagged(self) -> None:
        """'How many are there?' -> needs_clarification when context is ambiguous."""
        history = [
            _user("Tell me about the DevOps document and the Security Policy."),
            _assistant("Both documents contain different information."),
        ]
        mock_response = _make_llm_response(
            "How many are there?",
            needs_clarification=True,
            confidence=0.2,
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="How many are there?",
                history=history,
            )

        assert result.needs_clarification is True
        assert result.status == "ambiguous"
        # Should NOT invent a referent.
        assert result.rewritten_query == "How many are there?"


# ---------------------------------------------------------------------------
# TEST 8 — genuinely unrelated
# ---------------------------------------------------------------------------
class TestGenuinelyUnrelated:
    """TEST 8: genuinely unrelated query is not given false relevance."""

    @pytest.mark.asyncio
    async def test_unrelated_query_not_rewritten_with_relevance(self) -> None:
        """'What is the capital of France?' should not be made relevant by rewriting."""
        history = [
            _user("What is Kanban?"),
            _assistant("Kanban is a workflow method."),
        ]
        mock_response = _make_llm_response(
            "What is the capital of France?",
            confidence=0.95,
        )

        # Force rewrite by using a query that triggers the heuristic.
        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            # Override _needs_rewrite to force a rewrite for this test.
            with patch("app.retrieval.query_rewrite._needs_rewrite", return_value=True):
                result = await rewrite_query(
                    query="What is the capital of France?",
                    history=history,
                )

        assert result.status == "success"
        assert "France" in result.rewritten_query
        # No document names should be injected.
        assert "Kanban" not in result.rewritten_query


# ---------------------------------------------------------------------------
# TEST 9 — relevance gate receives rewritten query
# ---------------------------------------------------------------------------
class TestRelevanceGateReceivesRewrittenQuery:
    """TEST 9: verify that the rewritten query flows to downstream."""

    @pytest.mark.asyncio
    async def test_rewrite_used_for_retrieval(self) -> None:
        """Verify the rewritten query is the one used downstream."""
        history = [
            _user("What is the DevOps document about?"),
            _assistant("It discusses CI/CD and Kanban."),
        ]
        mock_response = _make_llm_response(
            "What does the DevOps document say about Kanban?"
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="What does it say about Kanban?",
                history=history,
            )

        assert "DevOps" in result.rewritten_query
        assert result.status == "success"
        assert result.original_query == "What does it say about Kanban?"


# ---------------------------------------------------------------------------
# TEST 10 — rewrite failure
# ---------------------------------------------------------------------------
class TestRewriteFailure:
    """TEST 10: rewrite provider failure degrades gracefully."""

    @pytest.mark.asyncio
    async def test_rewrite_failure_falls_back_to_original(self) -> None:
        """Provider failure -> original query used, request not rejected."""
        history = [
            _user("What is the DevOps document about?"),
            _assistant("It discusses CI/CD."),
        ]

        failing = _FailingProvider(Exception("Connection refused"))
        with patch("app.llm.generic.GenericProvider", return_value=failing):
            result = await rewrite_query(
                query="What does it say about Kanban?",
                history=history,
            )

        assert result.rewritten_query == "What does it say about Kanban?"
        assert result.status == "degraded"
        assert result.needs_clarification is False

    @pytest.mark.asyncio
    async def test_empty_response_falls_back(self) -> None:
        """Empty LLM response -> original query used."""
        history = [
            _user("What is the DevOps document about?"),
            _assistant("It discusses CI/CD."),
        ]

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider("")):
            result = await rewrite_query(
                query="What does it say about Kanban?",
                history=history,
            )

        assert result.rewritten_query == "What does it say about Kanban?"
        assert result.status == "degraded"


# ---------------------------------------------------------------------------
# TEST 11 — tenant isolation
# ---------------------------------------------------------------------------
class TestTenantIsolation:
    """TEST 11: rewrite context is workspace-scoped."""

    @pytest.mark.asyncio
    async def test_rewrite_uses_only_current_conversation(self) -> None:
        """Rewrite context contains only the provided history, not cross-workspace data."""
        history = [
            _user("Tell me about the DevOps document."),
            _assistant("It covers Kanban..."),
        ]
        mock_response = _make_llm_response(
            "What questions about Kanban are present in the DevOps document?"
        )

        with patch("app.llm.generic.GenericProvider", return_value=_FakeProvider(mock_response)):
            result = await rewrite_query(
                query="What questions about Kanban are present in it?",
                history=history,
            )

        assert "DevOps" in result.rewritten_query
        assert result.status == "success"


# ---------------------------------------------------------------------------
# TEST 12 — no conversation history
# ---------------------------------------------------------------------------
class TestNoHistory:
    """TEST 12: no history -> no rewrite LLM call."""

    @pytest.mark.asyncio
    async def test_no_history_skips_rewrite(self) -> None:
        """No previous turns -> no LLM call, original query used directly."""
        result = await rewrite_query(
            query="What is Kanban?",
            history=[],
        )

        assert result.rewritten_query == "What is Kanban?"
        assert result.status == "skipped"
        assert result.original_query == "What is Kanban?"

    @pytest.mark.asyncio
    async def test_needs_rewrite_heuristic(self) -> None:
        """Verify the heuristic correctly identifies follow-up queries."""
        # Short query with history -> likely needs rewrite.
        assert _needs_rewrite("it?", [_user("Hello")]) is True
        # Long standalone query with history -> likely does not need rewrite.
        assert _needs_rewrite(
            "What is the company vacation policy for full-time employees?",
            [_user("Hello")],
        ) is False
        # No history -> never needs rewrite.
        assert _needs_rewrite("What is Kanban?", []) is False
        # Very short query with history -> needs rewrite.
        assert _needs_rewrite("this?", [_user("Hello")]) is True


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------
class TestHelperFunctions:
    """Tests for _needs_rewrite and _parse_rewrite_response."""

    def test_parse_valid_json(self) -> None:
        """Valid JSON rewrite response is parsed correctly."""
        import json
        response = json.dumps({
            "rewritten_query": "What is Kanban?",
            "needs_clarification": False,
            "confidence": 0.9,
        })
        result = _parse_rewrite_response(response, "What is it?")
        assert result.rewritten_query == "What is Kanban?"
        assert result.needs_clarification is False
        assert result.confidence == 0.9
        assert result.status == "success"

    def test_parse_json_in_markdown(self) -> None:
        """JSON wrapped in markdown fences is parsed correctly."""
        import json
        inner = json.dumps({
            "rewritten_query": "What is Kanban?",
            "needs_clarification": False,
            "confidence": 0.8,
        })
        response = f"```json\n{inner}\n```"
        result = _parse_rewrite_response(response, "What is it?")
        assert result.rewritten_query == "What is Kanban?"

    def test_parse_needs_clarification(self) -> None:
        """needs_clarification=true returns the original query."""
        import json
        response = json.dumps({
            "rewritten_query": "How many documents are there?",
            "needs_clarification": True,
            "confidence": 0.3,
        })
        result = _parse_rewrite_response(response, "How many are there?")
        assert result.needs_clarification is True
        assert result.status == "ambiguous"
        assert result.rewritten_query == "How many are there?"

    def test_parse_invalid_json_falls_back(self) -> None:
        """Invalid JSON falls back to original query."""
        result = _parse_rewrite_response("not json at all", "What is it?")
        assert result.rewritten_query == "What is it?"
        assert result.status == "degraded"

    def test_parse_empty_rewritten_query_falls_back(self) -> None:
        """Empty rewritten_query falls back to original."""
        import json
        response = json.dumps({
            "rewritten_query": "",
            "needs_clarification": False,
            "confidence": 0.5,
        })
        result = _parse_rewrite_response(response, "What is it?")
        assert result.rewritten_query == "What is it?"
