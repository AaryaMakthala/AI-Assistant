"""Hybrid retrieval pipeline: search → fuse → rerank → ground (CLAUDE.md section 8).

Orchestrates the canonical Phase 5 pipeline over workspace-scoped chunks:

    query → embed (bi-encoder, thread) ──────────────┐
           → keyword FTS  ──┐                        │
           → semantic      ─┴─ RRF fuse (top ~15) ──► cross-encoder rerank
                                                      → top ~5–8
                                                      → Layer-1 grounding check

Deterministic, single pipeline — no routing, no agents (CLAUDE.md section 8). There is
exactly one kind of question this system answers.

The caller supplies a session that already carries tenant claims
(:func:`app.security.rls.tenant_session`); the pipeline applies its own explicit
``workspace_id`` predicate on every query on top of RLS (CLAUDE.md section 4). It never
opens its own transaction, so the chat phase can close the database session before the
LLM call — a pooled connection must not be pinned for the duration of a generation.

All CPU-bound model work (embedding the query, reranking candidates) runs in worker
threads via ``asyncio.to_thread`` so the event loop stays responsive, matching the
Phase 3 ingestion pattern.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.rag.embeddings import embed_query
from app.retrieval.grounding import is_grounded
from app.retrieval.hybrid import (
    HybridCandidate,
    keyword_search,
    rrf_merge,
    semantic_search,
)
from app.retrieval.rerank import rerank_scores


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk the LLM may cite, with everything a citation needs (CLAUDE.md 8.4)."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    page_number: int | None
    section_title: str | None
    chunk_index: int
    #: Fused score before reranking (RRF, section 8.1) — for diagnostics.
    rrf_score: float
    #: Cross-encoder relevance score (section 8.2) — the value the grounding
    #: threshold and the final ordering are based on.
    rerank_score: float

    @property
    def citation_label(self) -> str:
        return f"{self.filename} · page {self.page_number}" if self.page_number else self.filename


@dataclass(frozen=True)
class RetrievalResult:
    """What retrieval found and whether it is strong enough to generate from."""

    #: Final reranked chunks, capped at RETRIEVAL_FINAL_COUNT, best first.
    chunks: list[RetrievedChunk]
    #: Whether Layer-1 grounding passed (CLAUDE.md 8.3). When False, the caller
    #: must refuse without calling the LLM.
    grounded: bool
    #: Best rerank score across the candidates; None when nothing was retrieved.
    top_score: float | None

    @property
    def had_evidence(self) -> bool:
        return self.grounded and bool(self.chunks)


def _to_retrieved(chunk: HybridCandidate, rerank_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        content=chunk.content,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        chunk_index=chunk.chunk_index,
        rrf_score=chunk.rrf_score,
        rerank_score=rerank_score,
    )


async def retrieve(
    session: AsyncSession,
    *,
    query: str,
    workspace_id: uuid.UUID,
) -> RetrievalResult:
    """Retrieve the best evidence for `query` inside one workspace.

    `session` must already be tenant-scoped (RLS); `workspace_id` is the pipeline's
    own explicit filter on top of that. Returns the reranked top-K plus the Layer-1
    grounding verdict. Raises nothing on an ungrounded query — an empty/refused
    result is an ordinary outcome, not an error.
    """
    text = query.strip()
    if not text:
        return RetrievalResult(chunks=[], grounded=False, top_score=None)

    settings = get_settings()
    candidate_count = settings.retrieval_candidate_count
    final_count = settings.retrieval_final_count

    # Embed the query in a worker thread: sentence-transformers on CPU is the
    # slowest step before the reranker, and the event loop should not pay for it.
    query_embedding = await asyncio.to_thread(embed_query, text)

    semantic = await semantic_search(
        session,
        query_embedding=query_embedding,
        workspace_id=workspace_id,
        limit=candidate_count,
    )
    keyword = await keyword_search(
        session,
        query=text,
        workspace_id=workspace_id,
        limit=candidate_count,
    )

    # Fuse the two ranked lists; the merged pool is capped at the pre-rerank count
    # (section 8.2: never rerank hundreds of chunks — slow and unnecessary).
    candidates = rrf_merge(semantic, keyword, top_n=candidate_count)
    if not candidates:
        logger.info("No retrieval candidates for workspace {ws}", ws=workspace_id)
        return RetrievalResult(chunks=[], grounded=False, top_score=None)

    reranked = await asyncio.to_thread(
        rerank_scores, text, [candidate.content for candidate in candidates]
    )
    scored = sorted(
        zip(candidates, reranked, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    final = [_to_retrieved(chunk, score) for chunk, score in scored[:final_count]]
    top_score = scored[0][1]
    grounded = is_grounded(top_score)

    logger.info(
        "Retrieved {final}/{fused} chunks for workspace {ws} (grounded={grounded}, "
        "top_score={top_score:.4f})",
        final=len(final),
        fused=len(candidates),
        ws=workspace_id,
        grounded=grounded,
        top_score=top_score,
    )
    return RetrievalResult(chunks=final, grounded=grounded, top_score=top_score)


__all__ = ["RetrievedChunk", "RetrievalResult", "retrieve"]
