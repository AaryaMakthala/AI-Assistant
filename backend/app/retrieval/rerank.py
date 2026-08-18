"""Local cross-encoder reranking (CLAUDE.md sections 2, 8.2).

The reranker is a small, free, local cross-encoder (``cross-encoder/ms-marco-MiniLM-L-6-v2``,
pinned in config) run via sentence-transformers — no LLM-as-judge, no paid API. It scores
``(query, chunk_text)`` pairs directly, which is what makes it a *cross*-encoder: the query
and the candidate are passed to the model together, so their interaction is modeled, unlike
a bi-encoder (the embedding model) which encodes each side in isolation.

Load is lazy and thread-guarded, exactly like the embedding model
(``app/rag/embeddings.py``): sentence-transformers pulls in torch, and the API process
should never pay that import cost until a retrieval actually needs it.

The score scale matters for the grounding threshold (CLAUDE.md 8.3): ms-marco-MiniLM
returns raw logits, not a probability. Relevant pairs typically score positive, irrelevant
negative, and the configured ``RETRIEVAL_RELEVANCE_THRESHOLD`` is tuned against this exact
model's scale — which is why the model is pinned rather than swappable without re-tuning
(CLAUDE.md section 14, risk register).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger

from app.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_reranker: CrossEncoder | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoder:
    """Load the pinned cross-encoder once per process, guarding concurrent loads."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                settings = get_settings()
                logger.info("Loading reranker model {name}", name=settings.reranker_model)
                _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank_scores(query: str, chunks: list[str]) -> list[float]:
    """Score each ``(query, chunk_text)`` pair; higher means more relevant.

    Pure computation, aligned with the input: ``scores[i]`` belongs to ``chunks[i]``.
    Runs on CPU in the calling thread; callers that must not block the event loop
    wrap this in ``asyncio.to_thread`` (the retrieval pipeline does).
    """
    if not chunks:
        return []
    pairs = [(query, chunk) for chunk in chunks]
    scores = get_reranker().predict(pairs)
    return [float(score) for score in scores]


def reset_reranker() -> None:
    """Drop the cached model. For tests and for a deliberate model change."""
    global _reranker
    with _reranker_lock:
        _reranker = None


__all__ = ["get_reranker", "rerank_scores", "reset_reranker"]
