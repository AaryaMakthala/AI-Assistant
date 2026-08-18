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
"""

from __future__ import annotations

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


__all__ = ["is_grounded"]
