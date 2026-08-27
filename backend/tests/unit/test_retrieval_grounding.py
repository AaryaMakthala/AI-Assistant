"""Layer-1 retrieval-level grounding (CLAUDE.md 8.3).

The threshold is configuration, never hard-coded: these tests exercise the decision
logic against a stubbed settings object so the exact boundary behaviour is pinned
independently of the default value in config.

Phase B: overview grounding uses relative scoring (not absolute thresholds)
because cross-encoder scores are unbounded logits that can be negative.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.retrieval.grounding as grounding_module
from app.retrieval.grounding import is_grounded, is_overview_grounded

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


# ---------------------------------------------------------------------------
# Phase B: is_overview_grounded — relative scoring tests
# ---------------------------------------------------------------------------


class TestOverviewGrounding:
    """Overview queries use absolute cross-encoder score thresholds (Phase B-2).
    Cross-encoder scores are raw logits (can be negative, range ~[-12, +12]).
    These tests verify the absolute-threshold grounding logic against real and
    synthetic score sets.
    """

    def test_real_kanban_scores_ground(self) -> None:
        """The real Kanban overview scores (all negative logits) should ground
        because the top chunks clear the absolute minimum threshold.
        """
        # Best first: -2.45, -3.80, -4.24, -5.94, -6.11, -9.60
        scores = [-2.4534, -3.8034, -4.2362, -5.9403, -6.1114, -9.5952]
        assert is_overview_grounded(scores) is True

    def test_positive_cluster_grounds(self) -> None:
        """Tight positive cluster should ground."""
        assert is_overview_grounded([0.25, 0.22, 0.18, 0.15, 0.10]) is True

    def test_one_positive_outlier_with_negative_rest_still_grounds(self) -> None:
        """One strong positive score among negatives still grounds with absolute
        thresholds — the top chunk clears the minimum and the top-k mean is
        acceptable.  This is correct: if one chunk is genuinely relevant
        (score 0.5), an overview answer can be grounded.
        """
        assert is_overview_grounded([0.5, -3.0, -4.0, -5.0, -6.0]) is True

    def test_three_one_positive_outlier_grounds(self) -> None:
        """Three chunks with one positive outlier: top=0.8 clears min, top-2
        mean = -2.1 clears aggregate min -> grounds.
        """
        assert is_overview_grounded([0.8, -5.0, -6.0]) is True

    def test_tight_negative_cluster_grounds(self) -> None:
        """Consistently negative but tightly clustered top should ground."""
        assert is_overview_grounded([-2.0, -2.2, -2.5, -8.0, -9.0]) is True

    def test_all_identical_scores_ground(self) -> None:
        """All identical scores = consistent relevance."""
        assert is_overview_grounded([-3.0, -3.0, -3.0, -3.0]) is True

    def test_single_chunk_does_not_ground(self) -> None:
        """Overview needs multiple chunks for diffuse relevance."""
        assert is_overview_grounded([-2.0]) is False

    def test_empty_does_not_ground(self) -> None:
        assert is_overview_grounded([]) is False

    def test_two_close_chunks_ground(self) -> None:
        """Two chunks with close scores = consistent relevance."""
        assert is_overview_grounded([-2.0, -2.1]) is True

    def test_two_far_chunks_do_not_ground(self) -> None:
        """Two chunks with very different scores: top=-1.0 clears min but
        mean = -9.5 fails aggregate -> does not ground.
        """
        assert is_overview_grounded([-1.0, -18.0]) is False

    def test_fact_lookup_still_uses_absolute_threshold(self) -> None:
        """FACT_LOOKUP should still use the absolute threshold, not relative."""
        # A single high-scoring chunk should ground for fact lookup.
        assert is_grounded(0.5) is True
        # But a negative score should not.
        assert is_grounded(-2.0) is False
