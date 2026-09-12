"""Layer-1 retrieval-level grounding (CLAUDE.md 8.3).

The threshold is configuration, never hard-coded: these tests exercise the decision
logic against a stubbed settings object so the exact boundary behaviour is pinned
independently of the default value in config.

Phase B: overview grounding uses absolute thresholds on the cosine-similarity scale
([0, 1]) that the hosted embedder produces, not percentage-based heuristics.
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
# Phase B: is_overview_grounded — absolute cosine-score threshold tests
# ---------------------------------------------------------------------------


class TestOverviewGrounding:
    """Overview queries use absolute cosine-similarity score thresholds (Phase B-2).

    Scores are cosine similarities of the hosted embedder (range [0, 1]; ``0.30`` is
    clearly relevant for a 768-dim Gemini embedding).  These tests verify the
    absolute-threshold grounding logic with defaults from the repo's ``.env.example``
    (``overview_min_score=0.25``, ``overview_aggregate_min=0.20``).
    """

    def test_real_similarity_scores_ground(self) -> None:
        """A realistic overview result set — strong, tightly grouped similarities —
        must ground because the top chunks clear both absolute thresholds.
        """
        # Best first: 0.58, 0.55, 0.52, 0.48, 0.44, 0.40
        scores = [0.5774, 0.5519, 0.5231, 0.4902, 0.4411, 0.4023]
        assert is_overview_grounded(scores) is True

    def test_positive_cluster_grounds(self) -> None:
        """Tight positive cluster should ground."""
        assert is_overview_grounded([0.25, 0.22, 0.18, 0.15, 0.10]) is True

    def test_one_strong_outlier_with_weak_rest_still_grounds(self) -> None:
        """One strong positive score among weak ones still grounds with absolute
        thresholds — the top chunk clears the minimum and the top-k mean is
        acceptable.  This is correct: if one chunk is genuinely relevant
        (score 0.6), an overview answer can be grounded.
        """
        assert is_overview_grounded([0.6, 0.12, 0.10, 0.08]) is True

    def test_three_one_strong_outlier_grounds(self) -> None:
        """Three chunks with one strong outlier: top=0.62 clears min, top-2
        mean = 0.36 clears aggregate min -> grounds.
        """
        assert is_overview_grounded([0.62, 0.10, 0.08]) is True

    def test_tight_on_topic_cluster_grounds(self) -> None:
        """Consistently on-topic but tightly clustered tops should ground."""
        assert is_overview_grounded([0.45, 0.44, 0.43, 0.40, 0.38]) is True

    def test_all_identical_scores_ground(self) -> None:
        """All identical scores = consistent relevance."""
        assert is_overview_grounded([0.3, 0.3, 0.3, 0.3]) is True

    def test_single_chunk_does_not_ground(self) -> None:
        """Overview needs multiple chunks for diffuse relevance."""
        assert is_overview_grounded([0.5]) is False

    def test_empty_does_not_ground(self) -> None:
        assert is_overview_grounded([]) is False

    def test_two_close_chunks_ground(self) -> None:
        """Two chunks with close scores = consistent relevance."""
        assert is_overview_grounded([0.3, 0.29]) is True

    def test_two_far_chunks_do_not_ground(self) -> None:
        """Two chunks with very different scores: top=0.30 clears min but
        mean = 0.15 fails aggregate -> does not ground.
        """
        assert is_overview_grounded([0.3, 0.0]) is False

    def test_fact_lookup_still_uses_absolute_threshold(self) -> None:
        """FACT_LOOKUP should still use the absolute threshold, not relative."""
        # A single high-scoring chunk should ground for fact lookup.
        assert is_grounded(0.5) is True
        # But a weak score should not.
        assert is_grounded(0.0) is False
