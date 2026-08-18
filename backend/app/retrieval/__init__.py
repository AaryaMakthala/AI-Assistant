"""Hybrid retrieval + reranking + grounding (CLAUDE.md Phase 5).

Canonical, workspace-scoped retrieval over ``document_chunks``: semantic (pgvector)
and keyword (full-text) search fused with Reciprocal Rank Fusion, a local
cross-encoder rerank, and Layer-1 grounding (CLAUDE.md section 8).
"""

from app.retrieval.pipeline import RetrievalResult, RetrievedChunk, retrieve

__all__ = ["RetrievedChunk", "RetrievalResult", "retrieve"]
