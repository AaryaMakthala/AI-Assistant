"""In-memory routing cache for LLM intent classification.

Caches the result of LLM router calls keyed by (workspace_id, normalized_message).
On cache hit, the LLM call is skipped entirely.  Cache entries expire after a
configurable TTL and are invalidated when the workspace's knowledge file changes.

Storage: plain Python dict.  No Redis, no DB table — this is a process-local
cache that resets on server restart, which is fine for a free-tier portfolio
project where cold starts are expected.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from loguru import logger


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """One cached routing result."""

    route: str
    reasoning: str
    confidence: float
    cached_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

#: How long a routing entry stays valid (seconds).
ROUTING_CACHE_TTL_SECONDS = 600.0  # 10 minutes

#: Maximum entries per workspace before oldest are evicted.
_MAX_ENTRIES_PER_WORKSPACE = 500

# Inner dict is keyed by normalized_message (str).
# Outer dict is keyed by workspace_id (uuid.UUID).
_cache: dict[uuid.UUID, dict[str, _CacheEntry]] = {}


def _evict_if_needed(ws_cache: dict[str, _CacheEntry]) -> None:
    """Evict oldest entries if the workspace cache exceeds the cap."""
    if len(ws_cache) <= _MAX_ENTRIES_PER_WORKSPACE:
        return
    # Sort by cached_at and keep the newest entries.
    sorted_items = sorted(ws_cache.items(), key=lambda kv: kv[1].cached_at)
    to_remove = len(ws_cache) - _MAX_ENTRIES_PER_WORKSPACE
    for key, _ in sorted_items[:to_remove]:
        del ws_cache[key]


def get_cached_route(
    workspace_id: uuid.UUID,
    normalized_message: str,
) -> tuple[str, str, float] | None:
    """Look up a cached routing result.

    Returns (route, reasoning, confidence) on hit, None on miss.
    """
    ws_cache = _cache.get(workspace_id)
    if ws_cache is None:
        return None

    entry = ws_cache.get(normalized_message)
    if entry is None:
        return None

    # Check TTL.
    if (time.monotonic() - entry.cached_at) > ROUTING_CACHE_TTL_SECONDS:
        del ws_cache[normalized_message]
        return None

    logger.debug(
        "Routing cache hit: workspace={ws} route={route} confidence={conf:.2f}",
        ws=workspace_id,
        route=entry.route,
        conf=entry.confidence,
    )
    return entry.route, entry.reasoning, entry.confidence


def set_cached_route(
    workspace_id: uuid.UUID,
    normalized_message: str,
    route: str,
    reasoning: str,
    confidence: float,
) -> None:
    """Store a routing result in the cache."""
    if workspace_id not in _cache:
        _cache[workspace_id] = {}

    ws_cache = _cache[workspace_id]
    ws_cache[normalized_message] = _CacheEntry(
        route=route,
        reasoning=reasoning,
        confidence=confidence,
    )
    _evict_if_needed(ws_cache)


def invalidate_workspace_cache(workspace_id: uuid.UUID) -> None:
    """Clear all cached routes for a workspace.

    Call when the workspace's knowledge file changes (document add/remove/approve).
    """
    _cache.pop(workspace_id, None)
    logger.debug("Invalidated routing cache for workspace {ws}", ws=workspace_id)


def get_cache_stats() -> dict[str, int]:
    """Return cache statistics for monitoring/logging."""
    total_entries = sum(len(ws_cache) for ws_cache in _cache.values())
    return {
        "workspaces": len(_cache),
        "total_entries": total_entries,
    }


__all__ = [
    "ROUTING_CACHE_TTL_SECONDS",
    "get_cached_route",
    "get_cache_stats",
    "invalidate_workspace_cache",
    "set_cached_route",
]
