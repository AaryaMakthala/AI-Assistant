"""Workspace isolation tests for the routing cache.

Verifies that the cache key includes workspace_id, so the same query in
two different workspaces produces independent cache entries with no
cross-workspace hit.

These are unit-level tests that exercise the cache module directly — no
database or LLM API required.
"""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.routing_cache import (
    get_cached_route,
    get_cache_stats,
    invalidate_workspace_cache,
    set_cached_route,
)


@pytest.fixture(autouse=True)
def _clear_cache_between_tests() -> None:  # noqa: D401
    """Reset the module-level cache before each test."""
    from app.retrieval.routing_cache import _cache

    _cache.clear()
    yield
    _cache.clear()


# ------------------------------------------------------------------
# 1. Same query, two workspaces → independent cache entries
# ------------------------------------------------------------------


class TestWorkspaceCacheIsolation:
    """Cache entries are keyed by (workspace_id, normalized_message)."""

    def test_same_query_different_workspaces_are_independent(self) -> None:
        ws_a = uuid.uuid4()
        ws_b = uuid.uuid4()
        msg = "what is the vacation policy"

        set_cached_route(ws_a, msg, "DOCUMENT_CONTENT", "reason_a", 0.9)
        set_cached_route(ws_b, msg, "GREETING", "reason_b", 0.7)

        # Each workspace sees only its own route.
        entry_a = get_cached_route(ws_a, msg)
        entry_b = get_cached_route(ws_b, msg)

        assert entry_a is not None
        assert entry_b is not None
        assert entry_a[0] == "DOCUMENT_CONTENT"
        assert entry_b[0] == "GREETING"

    def test_invalidate_one_workspace_does_not_affect_the_other(self) -> None:
        ws_a = uuid.uuid4()
        ws_b = uuid.uuid4()
        msg = "hello"

        set_cached_route(ws_a, msg, "GREETING", "r1", 0.95)
        set_cached_route(ws_b, msg, "GREETING", "r2", 0.95)

        invalidate_workspace_cache(ws_a)

        assert get_cached_route(ws_a, msg) is None
        assert get_cached_route(ws_b, msg) is not None

    def test_cache_stats_show_two_workspaces(self) -> None:
        ws_a = uuid.uuid4()
        ws_b = uuid.uuid4()

        set_cached_route(ws_a, "q1", "GREETING", "r", 0.9)
        set_cached_route(ws_a, "q2", "METADATA", "r", 0.9)
        set_cached_route(ws_b, "q1", "DOCUMENT_CONTENT", "r", 0.9)

        stats = get_cache_stats()
        assert stats["workspaces"] == 2
        assert stats["total_entries"] == 3

    def test_normalized_message_isolation(self) -> None:
        """Cache key is the normalized message string, not the raw input."""
        ws = uuid.uuid4()
        # The cache stores pre-normalized keys. The caller (classify_intent)
        # normalizes before calling the cache.
        normalized_key = "hello world"
        set_cached_route(ws, normalized_key, "GREETING", "r", 0.9)

        # Lookup with the same normalized string — hits.
        assert get_cached_route(ws, normalized_key) is not None
        # Lookup with a different string — miss.
        assert get_cached_route(ws, "goodbye world") is None
        # Lookup with the un-normalized form — miss (caller must normalize).
        assert get_cached_route(ws, "Hello, World!") is None

    def test_cache_persists_across_multiple_lookups(self) -> None:
        ws = uuid.uuid4()
        msg = "test message"

        set_cached_route(ws, msg, "METADATA", "reason", 0.85)

        # Multiple lookups return the same entry.
        for _ in range(5):
            entry = get_cached_route(ws, msg)
            assert entry is not None
            assert entry[0] == "METADATA"

    def test_invalidate_nonexistent_workspace_no_crash(self) -> None:
        """Invalidating a workspace that has no cache must not raise."""
        invalidate_workspace_cache(uuid.uuid4())
        # No assertion needed — just verifying no exception.

    def test_eviction_respects_per_workspace_cap(self) -> None:
        """Exceeding _MAX_ENTRIES_PER_WORKSPACE triggers LRU eviction."""
        from app.retrieval.routing_cache import _MAX_ENTRIES_PER_WORKSPACE

        ws = uuid.uuid4()
        # Fill to the cap.
        for i in range(_MAX_ENTRIES_PER_WORKSPACE + 10):
            set_cached_route(ws, f"msg_{i}", "GREETING", "r", 0.9)

        stats = get_cache_stats()
        assert stats["total_entries"] <= _MAX_ENTRIES_PER_WORKSPACE
