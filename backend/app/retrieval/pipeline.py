"""Hybrid retrieval pipeline: search → fuse → rerank → ground (CLAUDE.md section 8).

Orchestrates the canonical Phase 5 pipeline over workspace-scoped chunks:

    query → relevance gate (LLM + heuristics)
             ├── not relevant → skip retrieval, refuse
             └── relevant / ambiguous → continue
           → embed (bi-encoder, thread) ──────────────┐
           → keyword FTS  ──┐                        │
           → semantic      ─┴─ RRF fuse (top ~15) ──► cross-encoder rerank
                                                      → top ~5–8
                                                      → Layer-1 grounding check

Three separate decisions in this pipeline:
1. Query/company relevance — is this about this workspace at all? (relevance gate)
2. Retrieval relevance — did we get any reasonably on-topic chunks? (grounding threshold)
3. Answer grounding — do the top chunks support an answer? (system prompt, Layer 2)

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
import statistics
import uuid
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.rag.embeddings import embed_query
from app.retrieval.doc_targeting import DocumentTargetingResult
from app.retrieval.grounding import is_grounded, is_overview_grounded
from app.retrieval.hybrid import (
    HybridCandidate,
    RRF_K,
    filename_search,
    keyword_search,
    rrf_merge,
    semantic_search,
)
from app.retrieval.intent import QueryShape
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
    """What retrieval found and whether it is strong enough to generate from.

    Carries three independent decisions:
    - relevance_decision: Was the question about this workspace at all? (Part 2)
    - grounded: Did retrieval find reasonably on-topic chunks? (relevance threshold)
    - The LLM's answer grounding is a separate decision in the system prompt.
    """

    #: Final reranked chunks, capped at RETRIEVAL_FINAL_COUNT, best first.
    chunks: list[RetrievedChunk]
    #: Whether Layer-1 grounding passed (CLAUDE.md 8.3). When False, the caller
    #: must refuse without calling the LLM.
    grounded: bool
    #: Best rerank score across the candidates; None when nothing was retrieved.
    top_score: float | None
    #: Why the relevance gate decided as it did (for logging/audit).
    relevance_decision: str = "pass"

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
    query_shape: QueryShape | None = None,
    doc_target_result: DocumentTargetingResult | None = None,
) -> RetrievalResult:
    """Retrieve the best evidence for `query` inside one workspace.

    `session` must already be tenant-scoped (RLS); `workspace_id` is the pipeline's
    own explicit filter on top of that. Returns the reranked top-K plus the Layer-1
    grounding verdict. Raises nothing on an ungrounded query — an empty/refused
    result is an ordinary outcome, not an error.

    The pipeline now includes a relevance gate (Part 2) that runs BEFORE retrieval
    to avoid wasting computation on obviously unrelated questions.

    Phase B: ``query_shape`` controls retrieval breadth and grounding strategy.
    OVERVIEW queries use broader candidate pools and aggregate grounding.

    Phase B-2: ``doc_target_result`` enables filename-aware candidate generation
    and high-confidence document-target grounding relaxation.
    """
    text = query.strip()
    if not text:
        return RetrievalResult(chunks=[], grounded=False, top_score=None)

    settings = get_settings()
    # Phase B: OVERVIEW queries need broader retrieval.
    is_overview = query_shape == QueryShape.OVERVIEW
    candidate_count = settings.retrieval_candidate_count
    if is_overview:
        # For overview, retrieve more candidates to capture diffuse relevance.
        candidate_count = min(candidate_count * 2, 30)
    final_count = settings.retrieval_final_count
    if is_overview:
        # For overview, keep more chunks in the final set for the LLM.
        final_count = min(final_count + 2, 10)

    # --- Relevance gate (Part 2) ---
    # Check if the question is about this workspace's documents before retrieval.
    from app.retrieval.relevance import check_relevance

    relevance = await check_relevance(
        session=session,
        question=text,
        workspace_id=workspace_id,
    )
    if not relevance.relevant:
        logger.info(
            "Relevance gate rejected question for workspace {ws}: reason={reason} "
            "confidence={confidence:.2f} layer={layer}",
            ws=workspace_id,
            reason=relevance.reason,
            confidence=relevance.confidence,
            layer=relevance.layer,
        )
        return RetrievalResult(
            chunks=[],
            grounded=False,
            top_score=None,
            relevance_decision=relevance.reason,
        )

    logger.info(
        "Relevance gate passed for workspace {ws}: reason={reason} "
        "confidence={confidence:.2f} layer={layer}",
        ws=workspace_id,
        reason=relevance.reason,
        confidence=relevance.confidence,
        layer=relevance.layer,
    )

    # --- Document targeting (Phase B-2) ---
    # Use the doc_target_result passed by the caller (resolved in chat_v2.py
    # or by the pipeline itself if not provided).
    if doc_target_result is not None:
        doc_target = doc_target_result
    else:
        from app.retrieval.doc_targeting import resolve_document_target

        doc_target = await resolve_document_target(
            session=session,
            question=text,
            workspace_id=workspace_id,
        )
    target_doc_id = doc_target.matched_document_id
    if target_doc_id is not None:
        logger.info(
            "Document targeting: resolved '{name}' to doc {doc_id} ({filename}, "
            "confidence={confidence:.2f})",
            name=doc_target.detected_name,
            doc_id=target_doc_id,
            filename=doc_target.matched_filename,
            confidence=doc_target.confidence,
        )

    # --- Filename-aware retrieval (Phase B-2) ---
    # Additional candidate source: match query tokens against normalized
    # filenames of READY documents in this workspace.
    filename_matched_docs: list[tuple[uuid.UUID, str]] = []
    filename_candidates = await filename_search(
        session,
        query=text,
        workspace_id=workspace_id,
        limit=candidate_count,
    )
    if filename_candidates:
        # Track which documents were matched by filename.
        seen_docs: set[uuid.UUID] = set()
        for fc in filename_candidates:
            if fc.document_id not in seen_docs:
                filename_matched_docs.append((fc.document_id, fc.filename))
                seen_docs.add(fc.document_id)
        logger.info(
            "Filename search matched {n} document(s) for query='{query}': "
            "docs={docs}",
            n=len(filename_matched_docs),
            query=text[:80],
            docs=[(fid, fn) for fid, fn in filename_matched_docs],
        )

    # --- Hybrid retrieval ---
    # Embed the query in a worker thread: sentence-transformers on CPU is the
    # slowest step before the reranker, and the event loop should not pay for it.
    query_embedding = await asyncio.to_thread(embed_query, text)

    semantic = await semantic_search(
        session,
        query_embedding=query_embedding,
        workspace_id=workspace_id,
        limit=candidate_count,
        document_id=target_doc_id,
    )
    keyword = await keyword_search(
        session,
        query=text,
        workspace_id=workspace_id,
        limit=candidate_count,
        document_id=target_doc_id,
    )

    # Fuse all candidate sources: semantic + keyword + filename.
    # The merged pool is capped at the pre-rerank count
    # (section 8.2: never rerank hundreds of chunks — slow and unnecessary).
    candidates = rrf_merge(semantic, keyword, top_n=candidate_count)

    # Inject filename-matched chunks if not already present.
    existing_chunk_ids = {c.chunk_id for c in candidates}
    for fc in filename_candidates:
        if fc.chunk_id not in existing_chunk_ids and len(candidates) < candidate_count:
            # Give filename-matched chunks an RRF-like score to keep them
            # competitive in the candidate pool.
            candidates.append(HybridCandidate(
                chunk_id=fc.chunk_id,
                document_id=fc.document_id,
                filename=fc.filename,
                content=fc.content,
                page_number=fc.page_number,
                section_title=fc.section_title,
                chunk_index=fc.chunk_index,
                rrf_score=1.0 / (RRF_K + 1),  # Rank-1 equivalent RRF score
            ))
            existing_chunk_ids.add(fc.chunk_id)

    fused_count = len(candidates)
    if not candidates:
        logger.info(
            "No retrieval candidates for workspace {ws} (relevance={reason})",
            ws=workspace_id,
            reason=relevance.reason,
        )
        return RetrievalResult(
            chunks=[], grounded=False, top_score=None,
            relevance_decision=relevance.reason,
        )

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
    second_score = scored[1][1] if len(scored) > 1 else None

    # --- Phase B-2: Compute doc-target grounding parameters ---
    is_high_confidence_target = (
        doc_target is not None
        and doc_target.matched_document_id is not None
        and doc_target.confidence >= settings.doc_target_high_confidence
    )
    has_target_chunk = False
    if is_high_confidence_target and target_doc_id is not None:
        has_target_chunk = any(
            c.document_id == target_doc_id for c in final
        )

    # Check if filename-matched documents have chunks in the final set.
    # When True, grounding uses the permissive filename_match_relaxed_score.
    has_filename_match_chunk = False
    if filename_matched_docs:
        filename_doc_ids = {doc_id for doc_id, _ in filename_matched_docs}
        has_filename_match_chunk = any(
            c.document_id in filename_doc_ids for c in final
        )
        if has_filename_match_chunk and not is_high_confidence_target:
            logger.info(
                "Filename match grounding relaxation: matched_docs={docs} "
                "chunks_in_final=True",
                docs=[fn for _, fn in filename_matched_docs],
            )

    # Filename match info for logging.
    filename_match = bool(filename_matched_docs)
    matched_filename = (
        filename_matched_docs[0][1] if filename_matched_docs else None
    )

    # --- Grounding check ---
    # Use query-shape-aware grounding with doc-target relaxation.
    all_scores = [score for _, score in scored[:final_count]]
    if is_overview and len(all_scores) >= 2:
        # Overview: absolute-threshold aggregate grounding.
        grounded = is_overview_grounded(
            all_scores,
            doc_target_high_confidence=is_high_confidence_target,
            has_target_chunk=has_target_chunk,
            has_filename_match_chunk=has_filename_match_chunk,
        )
        # Diagnostic logging for overview queries.
        all_median = statistics.median(all_scores)
        logger.info(
            "Overview grounding: query='{query}' scores={scores} "
            "median={median:.4f} top_mean={top_mean:.4f} "
            "doc_target_high_conf={dt_hc} has_target_chunk={htc} "
            "grounded={grounded}",
            query=text[:80],
            scores=[round(s, 4) for s in all_scores],
            median=all_median,
            top_mean=statistics.mean(all_scores),
            dt_hc=is_high_confidence_target,
            htc=has_target_chunk,
            grounded=grounded,
        )
    else:
        # Default: single-chunk grounding (fact_lookup, targeted, etc.).
        grounded = is_grounded(
            top_score,
            doc_target_high_confidence=is_high_confidence_target,
            has_target_chunk=has_target_chunk,
            has_filename_match_chunk=has_filename_match_chunk,
        )

    # Collect document metadata for logging.
    selected_doc_ids = list({str(c.document_id) for c in final})
    selected_doc_titles = list({c.filename for c in final})

    logger.info(
        "Retrieved {final}/{fused} chunks for workspace {ws} "
        "(grounded={grounded}, top_score={top_score:.4f}, "
        "second_score={second_score}, "
        "selected_docs={docs}, relevance={reason}, "
        "filename_match={fm}, matched_filename={mf}, "
        "doc_target_confidence={dtc})",
        final=len(final),
        fused=fused_count,
        ws=workspace_id,
        grounded=grounded,
        top_score=top_score,
        second_score=f"{second_score:.4f}" if second_score is not None else "None",
        docs=selected_doc_titles,
        reason=relevance.reason,
        fm=filename_match,
        mf=matched_filename,
        dtc=doc_target.confidence if doc_target else 0.0,
    )
    return RetrievalResult(
        chunks=final,
        grounded=grounded,
        top_score=top_score,
        relevance_decision=relevance.reason,
    )


__all__ = ["RetrievedChunk", "RetrievalResult", "retrieve"]
