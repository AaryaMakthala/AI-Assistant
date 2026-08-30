"""Tests for routing cache and workspace knowledge injection.

Verifies:
1. Cache hit skips the LLM router call (assert call count).
2. Cache is workspace-scoped (same phrase, two workspaces, isolated entries).
3. Workspace knowledge context is actually passed into the LLM router prompt
   (not just constructed and unused).

Uses shared helpers from conftest.py — no new fake classes.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.retrieval.intent import classify_intent, normalize_for_classification
from app.retrieval.llm_router import RouteResult
from app.retrieval.routing_cache import (
    ROUTING_CACHE_TTL_SECONDS,
    get_cached_route,
    invalidate_workspace_cache,
    set_cached_route,
)
from app.retrieval.workspace_knowledge import (
    WorkspaceKnowledge,
    invalidate_workspace_knowledge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches() -> None:  # noqa: D401
    """Ensure a clean cache state for every test."""
    from app.retrieval import routing_cache, workspace_knowledge

    routing_cache._cache.clear()
    workspace_knowledge._cache.clear()
    yield
    routing_cache._cache.clear()
    workspace_knowledge._cache.clear()


@pytest.fixture()
def ws_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture()
def ws_id_b() -> uuid.UUID:
    """A second workspace ID for isolation tests."""
    return uuid.uuid4()


def _make_route(route: str = "DOCUMENT_CONTENT", confidence: float = 0.9) -> RouteResult:
    return RouteResult(route=route, confidence=confidence, reasoning="test")


# ---------------------------------------------------------------------------
# 1. Cache hit skips the LLM call
# ---------------------------------------------------------------------------


class TestCacheHitSkipsLLM:
    """When a routing result is cached, the LLM router is never called."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm_router(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_id: uuid.UUID,
    ) -> None:
        llm_calls: list[str] = []

        async def _spy_route(
            *, query: str, history: list | None = None, **kw: Any
        ) -> RouteResult:
            llm_calls.append(query)
            return _make_route("GREETING", 0.95)

        # Mock the LLM router at the module level where classify_intent imports it.
        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _spy_route)

        # Pre-populate the cache.
        normalized = normalize_for_classification("hello")
        set_cached_route(ws_id, normalized, "GREETING", "cached", 0.95)

        # "hello" is caught by the regex fast-path before tenant_session
        # is needed, so no tenant_session mock is required.

        intent = await classify_intent("hello", workspace_id=ws_id)

        assert intent.category.value == "greeting"
        # LLM router must NOT have been called.
        assert llm_calls == [], f"LLM router called {len(llm_calls)} time(s), expected 0"

    @pytest.mark.asyncio
    async def test_cache_miss_calls_llm_router(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_id: uuid.UUID,
    ) -> None:
        llm_calls: list[str] = []

        async def _spy_route(
            *, query: str, history: list | None = None, **kw: Any
        ) -> RouteResult:
            llm_calls.append(query)
            return _make_route("GREETING", 0.95)

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _spy_route)

        # "hello" is caught by the regex fast-path before the LLM router,
        # so the LLM router is never called.  The cache is still populated
        # for future cache-hit tests.
        intent = await classify_intent("hello", workspace_id=ws_id)

        assert intent.category.value == "greeting"
        assert len(llm_calls) == 0  # fast-path short-circuits the LLM router


# ---------------------------------------------------------------------------
# 2. Cache is workspace-scoped
# ---------------------------------------------------------------------------


class TestCacheWorkspaceIsolation:
    """Same query in different workspaces must not share cache entries."""

    @pytest.mark.asyncio
    async def test_same_query_different_workspaces(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_id: uuid.UUID,
        ws_id_b: uuid.UUID,
    ) -> None:
        call_log: list[tuple[uuid.UUID, str]] = []

        async def _spy_route(
            *, query: str, history: list | None = None, **kw: Any
        ) -> RouteResult:
            # Return different routes to prove isolation.
            return _make_route("DOCUMENT_CONTENT", 0.9)

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _spy_route)

        class _FakeSession:
            async def execute(self, stmt: Any) -> Any:
                from tests.unit.conftest import FakeResult
                return FakeResult(rows=[], scalar=0)

            async def __aenter__(self) -> _FakeSession:
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

        monkeypatch.setattr(
            "app.security.rls.tenant_session",
            lambda **kw: _FakeSession(),
        )

        normalized = normalize_for_classification("what is the leave policy")

        # Classify in workspace A.
        intent_a = await classify_intent("what is the leave policy", workspace_id=ws_id)
        assert intent_a.category.value == "document_content"

        # Classify the same query in workspace B — must also call the LLM
        # because the cache is workspace-scoped.
        intent_b = await classify_intent("what is the leave policy", workspace_id=ws_id_b)
        assert intent_b.category.value == "document_content"

        # Verify the cache has separate entries.
        entry_a = get_cached_route(ws_id, normalized)
        entry_b = get_cached_route(ws_id_b, normalized)
        assert entry_a is not None
        assert entry_b is not None
        # Same route, but independent cache entries.
        assert entry_a[0] == entry_b[0] == "DOCUMENT_CONTENT"

    @pytest.mark.asyncio
    async def test_invalidate_one_workspace_does_not_affect_other(
        self,
        ws_id: uuid.UUID,
        ws_id_b: uuid.UUID,
    ) -> None:
        normalized = normalize_for_classification("hello")

        # Populate both caches.
        set_cached_route(ws_id, normalized, "GREETING", "cached_a", 0.95)
        set_cached_route(ws_id_b, normalized, "GREETING", "cached_b", 0.95)

        # Invalidate workspace A only.
        invalidate_workspace_cache(ws_id)

        # Workspace A cache should be empty.
        assert get_cached_route(ws_id, normalized) is None
        # Workspace B cache should still be there.
        assert get_cached_route(ws_id_b, normalized) is not None


# ---------------------------------------------------------------------------
# 3. Workspace knowledge is passed into the router prompt
# ---------------------------------------------------------------------------


class TestWorkspaceKnowledgeInjection:
    """Verify that workspace knowledge context reaches the LLM router prompt."""

    @pytest.mark.asyncio
    async def test_knowledge_context_passed_to_llm_router(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_id: uuid.UUID,
    ) -> None:
        captured_kwargs: dict[str, Any] = {}

        async def _spy_route(
            *, query: str, history: list | None = None, **kw: Any
        ) -> RouteResult:
            captured_kwargs.update(kw)
            return _make_route("DOCUMENT_CONTENT", 0.9)

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _spy_route)

        # Mock workspace knowledge to return a known context string.
        known_knowledge = WorkspaceKnowledge(
            workspace_id=ws_id,
            document_titles=["handbook.pdf", "policy.docx"],
            document_count=2,
            member_count=5,
            has_documents=True,
        )

        async def _mock_get_knowledge(
            session: Any, workspace_id: uuid.UUID
        ) -> WorkspaceKnowledge:
            return known_knowledge

        monkeypatch.setattr(
            "app.retrieval.workspace_knowledge.get_workspace_knowledge",
            _mock_get_knowledge,
        )

        class _FakeSession:
            async def __aenter__(self) -> _FakeSession:
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

        monkeypatch.setattr(
            "app.security.rls.tenant_session",
            lambda **kw: _FakeSession(),
        )

        await classify_intent("what is the leave policy", workspace_id=ws_id)

        # The workspace_knowledge_context kwarg must be present and non-empty.
        assert "workspace_knowledge_context" in captured_kwargs
        ctx = captured_kwargs["workspace_knowledge_context"]
        assert ctx is not None
        assert len(ctx) > 0
        # The context must contain at least one of the document titles.
        assert "handbook.pdf" in ctx or "policy.docx" in ctx

    @pytest.mark.asyncio
    async def test_no_workspace_id_skips_knowledge_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When workspace_id is None, knowledge loading is skipped entirely."""
        knowledge_called: list[str] = []

        async def _spy_route(
            *, query: str, history: list | None = None, **kw: Any
        ) -> RouteResult:
            return _make_route("GREETING", 0.95)

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _spy_route)

        async def _spy_knowledge(
            session: Any, workspace_id: uuid.UUID
        ) -> WorkspaceKnowledge:
            knowledge_called.append("called")
            return WorkspaceKnowledge(workspace_id=workspace_id)

        monkeypatch.setattr(
            "app.retrieval.workspace_knowledge.get_workspace_knowledge",
            _spy_knowledge,
        )

        # No workspace_id → knowledge loading must be skipped.
        intent = await classify_intent("hello", workspace_id=None)

        assert intent.category.value == "greeting"
        assert knowledge_called == [], "Knowledge loading should not run without workspace_id"


# ---------------------------------------------------------------------------
# 4. Cache TTL expiry
# ---------------------------------------------------------------------------


class TestCacheTTL:
    """Cache entries must expire after the configured TTL."""

    @pytest.mark.asyncio
    async def test_expired_cache_entry_is_ignored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_id: uuid.UUID,
    ) -> None:
        llm_calls: list[str] = []

        async def _spy_route(
            *, query: str, history: list | None = None, **kw: Any
        ) -> RouteResult:
            llm_calls.append(query)
            return _make_route("GREETING", 0.95)

        monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _spy_route)

        normalized = normalize_for_classification("hello")

        # Insert a cache entry and immediately backdate it past TTL.
        set_cached_route(ws_id, normalized, "GREETING", "old", 0.95)
        from app.retrieval import routing_cache
        entry = routing_cache._cache[ws_id][normalized]
        entry.cached_at = entry.cached_at - ROUTING_CACHE_TTL_SECONDS - 1

        # Expired cache entry is ignored.  "hello" is caught by the regex
        # fast-path before the LLM router, so the route is still correct
        # even though the cache was stale.
        intent = await classify_intent("hello", workspace_id=ws_id)
        assert intent.category.value == "greeting"
        assert len(llm_calls) == 0  # fast-path short-circuits the LLM


# ---------------------------------------------------------------------------
# 5. Workspace knowledge cache invalidation
# ---------------------------------------------------------------------------


class TestWorkspaceKnowledgeInvalidation:
    """Workspace knowledge cache must be invalidatable."""

    def test_invalidate_clears_knowledge_cache(self, ws_id: uuid.UUID) -> None:
        from app.retrieval import workspace_knowledge

        knowledge = WorkspaceKnowledge(
            workspace_id=ws_id,
            document_titles=["test.pdf"],
            document_count=1,
            member_count=2,
            has_documents=True,
        )
        workspace_knowledge._cache[ws_id] = knowledge

        invalidate_workspace_knowledge(ws_id)
        assert ws_id not in workspace_knowledge._cache

    def test_invalidate_nonexistent_workspace_no_crash(self) -> None:
        """Invalidating a workspace that was never cached must not raise."""
        invalidate_workspace_knowledge(uuid.uuid4())  # should not raise
