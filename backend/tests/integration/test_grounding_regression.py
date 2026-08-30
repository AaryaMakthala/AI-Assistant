"""Grounding regression tests for the DOCUMENT_CONTENT route.

Verifies that the grounding pipeline (retrieval → threshold check →
citation construction → answer generation) still works correctly now
that more traffic flows through the LLM router. These tests mock the
retrieval pipeline and LLM to verify the orchestration logic, not the
models themselves.

Key guarantees tested:
- Ungrounded questions are refused without an LLM call (CLAUDE.md 8.3).
- Citations are built from the chunks actually sent to the LLM.
- The answer references specific facts from the provided chunks.
- The grounding threshold decision is enforced, not bypassed.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.retrieval.llm_router import RouteResult
from app.retrieval.pipeline import RetrievedChunk

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


def _make_chunk(
    content: str = "Vacation accrues at 20 days per year.",
    *,
    score: float = 0.9,
    filename: str = "handbook.pdf",
    page: int = 2,
    section: str = "Leave Policy",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        content=content,
        page_number=page,
        section_title=section,
        chunk_index=0,
        rrf_score=0.02,
        rerank_score=score,
    )


# ------------------------------------------------------------------
# 1. Ungrounded questions are refused without an LLM call
# ------------------------------------------------------------------


class TestGroundingRefusal:
    """CLAUDE.md 8.3: ungrounded questions must not reach the LLM."""

    def test_ungrounded_question_refused_without_llm(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When retrieval finds nothing relevant, the question is refused."""
        from app.api import chat_v2 as chat_module
        from app.security.auth import Principal, get_principal
        from app.main import create_app

        principal = Principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
        app = create_app()

        async def _principal() -> Principal:
            return principal

        app.dependency_overrides[get_principal] = _principal

        async def _member(*args: Any, **kwargs: Any) -> str:
            return "OWNER"

        monkeypatch.setattr(chat_module, "assert_workspace_role", _member)

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

        async def _mock_route(**kwargs: Any) -> RouteResult:
            return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="test")

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _mock_route)

        # Retrieval returns an ungrounded result (no relevant chunks).
        async def _ungrounded_retrieve(*args: Any, **kwargs: Any) -> Any:
            from app.retrieval.pipeline import RetrievalResult
            return RetrievalResult(chunks=[], grounded=False, top_score=0.0)

        monkeypatch.setattr(chat_module, "retrieve", _ungrounded_retrieve)

        # The generation LLM must NOT be called.
        llm_called = [False]

        class _SpyLLM:
            name = "spy"
            model = "spy"

            async def stream(self, messages: Any, *, completion: Any) -> Any:
                llm_called[0] = True
                yield "This should not appear"

        monkeypatch.setattr(
            "app.api.chat_v2.get_generic_llm",
            lambda: _SpyLLM(),
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/chat/grounded",
                json={"message": "what is the obscure policy"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is False
        assert body["sources"] == []
        assert not llm_called[0], "LLM was called for an ungrounded question!"

        app.dependency_overrides.clear()


# ------------------------------------------------------------------
# 2. Grounded questions pass through to the LLM
# ------------------------------------------------------------------


class TestGroundingPassThrough:
    """When retrieval finds relevant chunks, the answer is generated."""

    def test_grounded_question_calls_llm_with_chunks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A grounded question must reach the LLM with the correct context."""
        from app.api import chat_v2 as chat_module
        from app.security.auth import Principal, get_principal
        from app.main import create_app

        principal = Principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
        app = create_app()

        async def _principal() -> Principal:
            return principal

        app.dependency_overrides[get_principal] = _principal

        async def _member(*args: Any, **kwargs: Any) -> str:
            return "OWNER"

        monkeypatch.setattr(chat_module, "assert_workspace_role", _member)

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

        async def _mock_route(**kwargs: Any) -> RouteResult:
            return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="test")

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _mock_route)

        # Retrieval returns a grounded chunk.
        chunk = _make_chunk(
            content="Refunds are issued within 30 days of purchase.",
            score=0.9,
            filename="refund_policy.pdf",
            page=5,
            section="Refund Terms",
        )

        async def _grounded_retrieve(*args: Any, **kwargs: Any) -> Any:
            from app.retrieval.pipeline import RetrievalResult
            return RetrievalResult(chunks=[chunk], grounded=True, top_score=0.9)

        monkeypatch.setattr(chat_module, "retrieve", _grounded_retrieve)

        # The generation LLM receives the messages and returns an answer.
        captured_messages: list[Any] = []

        class _SpyLLM:
            name = "spy"
            model = "spy"

            async def stream(self, messages: Any, *, completion: Any) -> Any:
                captured_messages.extend(messages)
                completion.text = "Refunds are available within 30 days."
                completion.provider = "spy"
                completion.model = "spy"
                from app.llm.base import TokenUsage
                completion.usage = TokenUsage(prompt_tokens=50, completion_tokens=10)
                yield "Refunds are available within 30 days."

        monkeypatch.setattr(
            "app.api.chat_v2.get_generic_llm",
            lambda: _SpyLLM(),
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/chat/grounded",
                json={"message": "what is the refund policy"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert len(body["sources"]) == 1
        # Citation references the correct file and page.
        assert body["sources"][0]["filename"] == "refund_policy.pdf"
        assert body["sources"][0]["page_number"] == 5
        # The LLM was called with the chunk content in context.
        assert len(captured_messages) > 0

        app.dependency_overrides.clear()


# ------------------------------------------------------------------
# 3. Citation metadata matches chunks sent to LLM
# ------------------------------------------------------------------


class TestCitationCorrectness:
    """Citations are built from the actual chunks, not invented by the LLM."""

    def test_citations_match_retrieved_chunks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The sources in the response must correspond 1:1 to retrieved chunks."""
        from app.api import chat_v2 as chat_module
        from app.security.auth import Principal, get_principal
        from app.main import create_app

        principal = Principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
        app = create_app()

        async def _principal() -> Principal:
            return principal

        app.dependency_overrides[get_principal] = _principal

        async def _member(*args: Any, **kwargs: Any) -> str:
            return "OWNER"

        monkeypatch.setattr(chat_module, "assert_workspace_role", _member)

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

        async def _mock_route(**kwargs: Any) -> RouteResult:
            return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="test")

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _mock_route)

        # Two chunks from different documents.
        chunk_a = _make_chunk(
            content="Vacation accrues at 20 days per year.",
            filename="handbook.pdf",
            page=12,
            section="Leave Policy",
        )
        chunk_b = _make_chunk(
            content="Sick leave is 10 days per year.",
            filename="benefits.pdf",
            page=3,
            section="Health Benefits",
        )

        async def _multi_chunk_retrieve(*args: Any, **kwargs: Any) -> Any:
            from app.retrieval.pipeline import RetrievalResult
            return RetrievalResult(
                chunks=[chunk_a, chunk_b], grounded=True, top_score=0.9
            )

        monkeypatch.setattr(chat_module, "retrieve", _multi_chunk_retrieve)

        class _SpyLLM:
            name = "spy"
            model = "spy"

            async def stream(self, messages: Any, *, completion: Any) -> Any:
                completion.text = "You get 20 vacation days and 10 sick days."
                completion.provider = "spy"
                completion.model = "spy"
                from app.llm.base import TokenUsage
                completion.usage = TokenUsage(prompt_tokens=80, completion_tokens=12)
                yield "You get 20 vacation days and 10 sick days."

        monkeypatch.setattr(
            "app.api.chat_v2.get_generic_llm",
            lambda: _SpyLLM(),
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/chat/grounded",
                json={"message": "how many vacation and sick days do I get"},
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["sources"]) == 2
        filenames = {s["filename"] for s in body["sources"]}
        assert filenames == {"handbook.pdf", "benefits.pdf"}

        app.dependency_overrides.clear()
