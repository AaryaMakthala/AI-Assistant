"""Layer-1 retrieval-level grounding (CLAUDE.md section 8.3).

Two-layer grounding keeps the LLM from answering without evidence:

* **Layer 1 (this module) — retrieval-level.** If the top reranked chunk scores below
  ``RETRIEVAL_RELEVANCE_THRESHOLD``, the caller must skip the LLM call entirely and
  refuse honestly ("I couldn't find that information..."). This is what prevents the
  LLM from ever seeing a question with no real supporting evidence.
* Layer 2 — generation-level — is the strict system prompt in the chat phase, a
  backstop for cases that pass the threshold but are still partial matches.

The check is deliberately a separate decision from the retrieved chunks: the pipeline
always returns what it found, and ``RetrievalResult.grounded`` tells the caller whether
that evidence is strong enough to generate from. An out-of-scope question ("who won the
World Cup") is not a special case — it is simply a question whose top rerank score lands
below the threshold, handled exactly like any other ungrounded question.

Phase B: Overview queries (e.g. "What is Kanban?") have diffuse relevance across many
chunks rather than sharp relevance to one.  ``is_overview_grounded`` considers multiple
relevant chunks, aggregate relevance, and same-document consistency instead of requiring
a single chunk to clear the threshold.
"""

from __future__ import annotations

import statistics

from app.config import get_settings


def is_grounded(top_score: float | None) -> bool:
    """Whether the best rerank score clears the Layer-1 grounding threshold.

    ``None`` means nothing was retrieved at all — the absence of evidence, which is
    the same refusal as evidence that scores too low.
    """
    if top_score is None:
        return False
    threshold = get_settings().retrieval_relevance_threshold
    return top_score >= threshold


def is_overview_grounded(scores: list[float], *, top_k: int = 3) -> bool:
    """Evidence-aware grounding for OVERVIEW queries using RELATIVE scoring.

    Overview queries ("What is Kanban?", "Tell me about X") have diffuse relevance
    across many chunks rather than sharp relevance to one.  Cross-encoder scores
    are unbounded logits (can be negative), so absolute thresholds don't work for
    overview queries.  Instead, this uses relative scoring:

    1. At least 2 of the top-k chunks must score above the median of ALL
       retrieved candidates (i.e. the top chunks are meaningfully better than
       the noise floor).
    2. The gap between the best and the k-th best chunk must be small enough
       to indicate consistent relevance (not one lucky match plus garbage).
    3. There must be at least 2 chunks above the median (evidence coverage).

    Parameters
    ----------
    scores:
        Rerank scores for the retrieved chunks, best first.
    top_k:
        Number of top chunks to consider for aggregate relevance.

    Returns
    -------
    True if the evidence is sufficient for an overview answer.
    """
    if len(scores) < 2:
        return False

    top_scores = scores[:min(top_k, len(scores))]

    # The median of ALL retrieved scores is the noise floor — chunks above
    # it are meaningfully better than random retrieval results.
    all_median = statistics.median(scores)

    # Condition 1: at least 2 of the top-k chunks are at or above the median.
    # Use >= (not >) so identical scores still count.
    above_median = [s for s in top_scores if s >= all_median]
    if len(above_median) < 2:
        # Special case: only 2 chunks total, both must be close together.
        if len(scores) == 2:
            gap = abs(scores[0] - scores[1])
            return gap < 2.0  # Close scores = consistent relevance.
        return False

    # Condition 2: the top chunk is not a statistical outlier.
    # If the best score is more than 2 standard deviations above the mean,
    # it's one lucky match, not diffuse relevance.
    best = top_scores[0]
    if len(scores) >= 3:
        mean_all = statistics.mean(scores)
        std_all = statistics.stdev(scores)
        if std_all > 0 and best > mean_all + std_all * 2:
            return False
    # Also check: the gap between best and k-th best should be bounded
    # relative to the overall score spread.
    kth = top_scores[-1]
    gap = best - kth
    if len(scores) >= 3:
        overall_range = scores[0] - scores[-1]
        if overall_range > 0 and gap > overall_range * 0.6:
            return False

    # Condition 3: the mean of the top-k is at or above the median.
    # This ensures the top chunks collectively represent real signal.
    top_mean = statistics.mean(top_scores)
    if top_mean < all_median:
        return False

    return True


__all__ = ["is_grounded", "is_overview_grounded"]
