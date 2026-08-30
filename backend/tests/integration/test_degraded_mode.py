"""LLM failure / degraded mode tests.

Verifies:
1. LLM router failure → regex fallback (not a 500).
2. LLM router returns degraded status → regex fallback.
3. Both LLM and regex give ambiguous results → user gets clarification.
4. LLM generation failure → safe error event (not a stack trace).

These are unit-level tests that exercise the classify_intent and chat
endpoints with mocked LLM providers.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.retrieval.intent import classify_intent, classify_intent_regex, IntentCategory
from app.retrieval.llm_router import RouteResult

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


# ------------------------------------------------------------------
# 1. LLM router raises exception → regex fallback
# ------------------------------------------------------------------


class TestLLMRouterFailure:
    """When the LLM router fails, classify_intent falls back to regex.

    Note: route_with_llm catches exceptions internally and returns a
    degraded RouteResult. These tests verify the regex fallback path
    works when the router reports degradation.
    """

    @pytest.mark.asyncio
    async def test_llm_connection_error_returns_degraded(self) -> None:
        """route_with_llm catches connection errors and returns degraded."""
        ws = uuid.uuid4()

        async def _failing_route(**kwargs: Any) -> RouteResult:
            # Simulate what route_with_llm does internally on failure.
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.0,
                reasoning="llm_error: ConnectionError: provider unreachable",
                status="degraded",
            )

        with patch(
            "app.retrieval.llm_router.route_with_llm",
            side_effect=_failing_route,
        ):
            intent = await classify_intent("hello", workspace_id=ws)

        # Degraded status triggers regex fallback, which catches "hello".
        assert intent.category == IntentCategory.GREETING

    @pytest.mark.asyncio
    async def test_llm_timeout_returns_degraded(self) -> None:
        """A timeout is reported as degraded, triggering regex fallback."""
        ws = uuid.uuid4()

        async def _timeout_route(**kwargs: Any) -> RouteResult:
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.0,
                reasoning="llm_error: TimeoutError: request timed out",
                status="degraded",
            )

        with patch(
            "app.retrieval.llm_router.route_with_llm",
            side_effect=_timeout_route,
        ):
            intent = await classify_intent("what is 2+2", workspace_id=ws)

        # Regex catches math as out-of-scope.
        assert intent.category == IntentCategory.OUT_OF_SCOPE

    @pytest.mark.asyncio
    async def test_llm_empty_response_returns_degraded(self) -> None:
        """An empty LLM response is reported as degraded."""
        ws = uuid.uuid4()

        async def _empty_route(**kwargs: Any) -> RouteResult:
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.0,
                reasoning="empty_response",
                status="degraded",
            )

        with patch(
            "app.retrieval.llm_router.route_with_llm",
            side_effect=_empty_route,
        ):
            intent = await classify_intent("capital of france", workspace_id=ws)

        # Regex catches geography as out-of-scope.
        assert intent.category == IntentCategory.OUT_OF_SCOPE


# ------------------------------------------------------------------
# 2. LLM router returns degraded status → regex fallback
# ------------------------------------------------------------------


class TestLLMRouterDegraded:
    """When the LLM router returns degraded, regex handles classification."""

    @pytest.mark.asyncio
    async def test_degraded_status_triggers_regex_fallback(self) -> None:
        """A degraded RouteResult must trigger the regex fallback path."""
        ws = uuid.uuid4()

        async def _degraded_route(**kwargs: Any) -> RouteResult:
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.0,
                reasoning="llm_error: provider returned empty response",
                status="degraded",
            )

        with patch(
            "app.retrieval.llm_router.route_with_llm",
            side_effect=_degraded_route,
        ):
            intent = await classify_intent("hello", workspace_id=ws)

        # Regex should have caught "hello" as a greeting.
        assert intent.category == IntentCategory.GREETING

    @pytest.mark.asyncio
    async def test_degraded_not_cached(self) -> None:
        """Degraded results must NOT be cached (low confidence)."""
        ws = uuid.uuid4()

        async def _degraded_route(**kwargs: Any) -> RouteResult:
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.0,
                reasoning="empty_response",
                status="degraded",
            )

        with patch(
            "app.retrieval.llm_router.route_with_llm",
            side_effect=_degraded_route,
        ):
            await classify_intent("hello", workspace_id=ws)

        # The degraded result should not be in the cache.
        from app.retrieval.routing_cache import get_cached_route
        from app.retrieval.intent import normalize_for_classification

        cached = get_cached_route(ws, normalize_for_classification("hello"))
        assert cached is None


# ------------------------------------------------------------------
# 3. Both LLM and regex ambiguous → clarification
# ------------------------------------------------------------------


class TestAmbiguousFallback:
    """When both LLM and regex are uncertain, the regex default applies."""

    @pytest.mark.asyncio
    async def test_unrecognized_query_defaults_to_document_content(self) -> None:
        """An unrecognized query with degraded LLM falls through to regex default.

        The regex fallback defaults to DOCUMENT_CONTENT for unrecognized
        patterns, which then goes through the RAG pipeline and may get
        refused for insufficient evidence.
        """
        ws = uuid.uuid4()

        async def _degraded_route(**kwargs: Any) -> RouteResult:
            return RouteResult(
                route="NEEDS_CLARIFICATION",
                confidence=0.0,
                reasoning="low confidence",
                status="degraded",
            )

        with patch(
            "app.retrieval.llm_router.route_with_llm",
            side_effect=_degraded_route,
        ):
            intent = await classify_intent(
                "asdfghjkl nonsense", workspace_id=ws
            )

        # Regex fallback defaults to DOCUMENT_CONTENT.
        assert intent.category == IntentCategory.DOCUMENT_CONTENT

    @pytest.mark.asyncio
    async def test_low_confidence_llm_result_forces_clarification(self) -> None:
        """Low confidence from the LLM (not degraded) forces AMBIGUOUS."""
        ws = uuid.uuid4()

        async def _low_confidence_route(**kwargs: Any) -> RouteResult:
            return RouteResult(
                route="DOCUMENT_CONTENT",
                confidence=0.2,
                reasoning="uncertain",
                status="success",
            )

        with patch(
            "app.retrieval.llm_router.route_with_llm",
            side_effect=_low_confidence_route,
        ):
            intent = await classify_intent("vague query", workspace_id=ws)

        # Low confidence → AMBIGUOUS.
        assert intent.category == IntentCategory.AMBIGUOUS
        assert intent.needs_clarification is True


# ------------------------------------------------------------------
# 4. LLM generation failure → safe error (streaming endpoint)
# ------------------------------------------------------------------


class TestGenerationFailure:
    """When the generation LLM fails, the user gets a safe error event."""

    def test_streaming_chat_handles_llm_generation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The SSE stream emits an error event, not a 500, on LLM failure."""
        from app.api import chat_v2 as chat_module
        from app.config import get_settings
        from app.llm.base import Completion, LLMError, Message, TokenUsage
        from app.retrieval.pipeline import RetrievedChunk
        from app.security.auth import Principal, get_principal
        from app.main import create_app

        principal = Principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
        app = create_app()

        async def _principal() -> Principal:
            return principal

        app.dependency_overrides[get_principal] = _principal

        async def _member(workspace_id: uuid.UUID, principal: Principal, *allowed: str) -> str:
            return "OWNER"

        monkeypatch.setattr(chat_module, "assert_workspace_role", _member)

        # Mock tenant_session to return a session with no history.
        from contextlib import asynccontextmanager
        from collections.abc import Iterator

        class _FakeSession:
            async def execute(self, stmt: Any) -> Any:
                from tests.unit.conftest import FakeResult
                return FakeResult(scalar=None)
            async def __aenter__(self) -> _FakeSession:
                return self
            async def __aexit__(self, *args: Any) -> None:
                pass

        @asynccontextmanager
        async def _tenant_session(**kw: Any) -> Iterator[_FakeSession]:
            yield _FakeSession()

        monkeypatch.setattr(chat_module, "tenant_session", _tenant_session)

        # Mock LLM router to return DOCUMENT_CONTENT.
        async def _mock_route(**kwargs: Any) -> RouteResult:
            return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="test")

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _mock_route)

        # Mock retrieval to return grounded chunks.
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="policy.pdf",
            content="Refunds within 30 days.",
            page_number=1,
            section_title="Refund Policy",
            chunk_index=0,
            rrf_score=0.1,
            rerank_score=0.9,
        )

        async def _retrieve(*args: Any, **kwargs: Any) -> Any:
            from app.retrieval.pipeline import RetrievalResult
            return RetrievalResult(chunks=[chunk], grounded=True, top_score=0.9)

        monkeypatch.setattr(chat_module, "retrieve", _retrieve)

        # Mock the generation LLM to raise an error.
        class _FailingLLM:
            name = "test-failing"
            model = "test-failing"

            async def stream(self, messages: list[Message], *, completion: Completion) -> Any:
                raise LLMError("Provider is down", provider="test", retryable=True)
                yield  # make it an async generator

        monkeypatch.setattr(
            "app.api.chat_v2.get_generic_llm",
            lambda: _FailingLLM(),
        )

        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/chat/grounded",
                json={"message": "what is the refund policy"},
            )

        # The grounded endpoint surfaces the LLM error as a non-500 response.
        # FastAPI may return 503 for LLMError — the key assertion is that it's
        # NOT a raw 500 Internal Server Error (which would indicate an unhandled
        # exception / stack trace leak).
        assert response.status_code != 500, (
            f"Raw 500 returned — possible stack trace leak: {response.text[:200]}"
        )
        body = response.json()
        detail = body.get("detail", body.get("answer", ""))
        # The response must contain a user-friendly error message.
        assert any(
            kw in detail.lower()
            for kw in ("unavailable", "try again", "language model")
        ), f"Expected user-friendly error in response, got: {detail[:200]}"

        app.dependency_overrides.clear()
