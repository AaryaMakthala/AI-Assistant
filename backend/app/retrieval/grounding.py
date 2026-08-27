"""Layer-1 retrieval-level grounding (CLAUDE.md section 8.3).

Two-layer grounding keeps the LLM from answering without evidence:

* **Layer 1 (this module) — retrieval-level.** If the top reranked chunk scores below
  the configured absolute threshold, the caller must skip the LLM call entirely and
  refuse honestly ("I couldn't find that information..."). This is what prevents the
  LLM from ever seeing a question with no real supporting evidence.
* Layer 2 — generation-level — is the strict system prompt in the chat phase, a
  backstop for cases that pass the threshold but are still partial matches.

The check is deliberately a separate decision from the retrieved chunks: the pipeline
always returns what it found, and ``RetrievalResult.grounded`` tells the caller whether
that evidence is strong enough to generate from. An out-of-scope question ("who won the
World Cup") is not a special case — it is simply a question whose top rerank score lands
below the threshold, handled exactly like any other ungrounded question.

Phase B-2: Overview queries use absolute thresholds calibrated to the cross-encoder
score scale (raw logits, range ~[-12, +12]).  High-confidence document targeting
relaxes the grounding floor when supported by at least one retrieved chunk from the
targeted document.
"""

from __future__ import annotations

import statistics

from app.config import get_settings


def is_grounded(
    top_score: float | None,
    *,
    doc_target_high_confidence: bool = False,
    has_target_chunk: bool = False,
    has_filename_match_chunk: bool = False,
) -> bool:
    """Whether the best rerank score clears the Layer-1 grounding threshold.

    ``None`` means nothing was retrieved at all — the absence of evidence, which is
    the same refusal as evidence that scores too low.

    Parameters
    ----------
    top_score:
        Best rerank score across the candidates.
    doc_target_high_confidence:
        Whether the query resolved a high-confidence document target.
    has_target_chunk:
        Whether at least one retrieved chunk belongs to the targeted document.
        Required alongside ``doc_target_high_confidence`` for relaxation.
    has_filename_match_chunk:
        Whether at least one retrieved chunk belongs to a filename-matched document.
        When True, uses a very permissive floor — the filename IS the evidence.
    """
    if top_score is None:
        return False

    settings = get_settings()

    # Filename match with chunks from the matched document: use the permissive
    # floor.  The filename IS the evidence; the reranker score is secondary.
    if has_filename_match_chunk:
        return top_score >= settings.filename_match_relaxed_score

    # High-confidence document targeting with at least one chunk from the target
    # document: use the relaxed absolute threshold.
    if doc_target_high_confidence and has_target_chunk:
        return top_score >= settings.doc_target_relaxed_score

    # Normal fact-lookup: use the global relevance threshold.
    # This threshold is on the [0, 1] scale — valid for positive-logit cases.
    return top_score >= settings.retrieval_relevance_threshold


def is_overview_grounded(
    scores: list[float],
    *,
    top_k: int = 3,
    doc_target_high_confidence: bool = False,
    has_target_chunk: bool = False,
    has_filename_match_chunk: bool = False,
) -> bool:
    """Grounding for OVERVIEW queries using absolute cross-encoder score thresholds.

    Overview queries ("What is Kanban?", "Tell me about X") have diffuse relevance
    across many chunks rather than sharp relevance to one.  Cross-encoder scores
    are raw logits (can be negative, range ~[-12, +12]), so we use absolute
    thresholds calibrated to the actual score scale, not percentage-based heuristics.

    The grounding check:
    1. The top chunk's absolute score must exceed ``overview_min_score``.
    2. The mean of the top-k chunks must exceed ``overview_aggregate_min``.
    3. At least 2 chunks must be present (overview needs diffuse evidence).

    When a high-confidence document target is present with at least one chunk from
    the targeted document, the thresholds are relaxed via ``doc_target_relaxed_score``.

    Parameters
    ----------
    scores:
        Rerank scores for the retrieved chunks, best first.
    top_k:
        Number of top chunks to consider for aggregate relevance.
    doc_target_high_confidence:
        Whether the query resolved a high-confidence document target.
    has_target_chunk:
        Whether at least one retrieved chunk belongs to the targeted document.

    Returns
    -------
    True if the evidence is sufficient for an overview answer.
    """
    if len(scores) < 2:
        return False

    settings = get_settings()

    # Determine which thresholds to use.
    if has_filename_match_chunk:
        min_score = settings.filename_match_relaxed_score
        aggregate_min = settings.filename_match_relaxed_score
    elif doc_target_high_confidence and has_target_chunk:
        min_score = settings.doc_target_relaxed_score
        aggregate_min = settings.doc_target_relaxed_score
    else:
        min_score = settings.overview_min_score
        aggregate_min = settings.overview_aggregate_min

    top_scores = scores[: min(top_k, len(scores))]

    # Condition 1: the top chunk must clear the absolute minimum.
    if top_scores[0] < min_score:
        return False

    # Condition 2: the mean of the top-k chunks must clear the aggregate minimum.
    top_mean = statistics.mean(top_scores)
    if top_mean < aggregate_min:
        return False

    # Condition 3: there must be at least 2 chunks for diffuse evidence.
    # (Already guaranteed by the len(scores) < 2 check above, but
    # explicitly checking top_scores for clarity.)
    if len(top_scores) < 2:
        return False

    return True


__all__ = ["is_grounded", "is_overview_grounded"]
