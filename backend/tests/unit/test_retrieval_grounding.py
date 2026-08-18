"""Layer-1 retrieval-level grounding (CLAUDE.md 8.3).

The threshold is configuration, never hard-coded: these tests exercise the decision
logic against a stubbed settings object so the exact boundary behaviour is pinned
independently of the default value in config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.retrieval.grounding as grounding_module
from app.retrieval.grounding import is_grounded

pytestmark = pytest.mark.usefixtures("valid_env")


@pytest.fixture
def threshold(monkeypatch: pytest.MonkeyPatch) -> float:
    """Pin the threshold to a known value for the duration of a test."""
    settings = SimpleNamespace(retrieval_relevance_threshold=0.3)
    monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
    return 0.3


async def test_grounded_when_evidence_clears_threshold(threshold: float) -> None:
    assert is_grounded(0.9) is True
    assert is_grounded(threshold) is True  # boundary: equal clears


async def test_ungrounded_when_evidence_is_insufficient(threshold: float) -> None:
    assert is_grounded(0.2) is False
    assert is_grounded(0.0) is False
    assert is_grounded(-1.0) is False


async def test_no_evidence_is_never_grounded(threshold: float) -> None:
    """Nothing retrieved at all is the same refusal as weak evidence."""
    assert is_grounded(None) is False
