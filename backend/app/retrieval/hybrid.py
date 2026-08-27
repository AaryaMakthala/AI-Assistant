"""Hybrid retrieval over canonical chunks: semantic + keyword, fused with RRF.

CLAUDE.md section 8.1 — semantic-only search misses exact identifiers (``HR-004``,
``POL-17``) because embeddings represent meaning, not tokens; keyword-only search
misses paraphrases (``vacation`` vs. ``annual leave``). Phase 5 runs both and merges
them with **Reciprocal Rank Fusion** (RRF):

    score = 1/(k + rank_semantic) + 1/(k + rank_keyword),   k = 60

RRF is chosen over normalized-score blending because cosine distance and ``ts_rank``
are on incomparable scales, and rank-based fusion sidesteps normalizing them
(CLAUDE.md 8.1). Only the *order* each engine returned matters, never the magnitude.

Both searches are workspace-scoped twice on purpose (CLAUDE.md section 4): the query
carries an explicit ``workspace_id`` predicate, and it runs on a session already
under RLS policies that enforce the same rule in the database.

No status filter is applied to ``document_chunks`` — deliberately. Section 5's
invariant makes it unnecessary: only READY documents are ever chunked, so a PENDING
or REJECTED document structurally has zero rows here. A query-time filter would be
papering over a broken invariant rather than enforcing it.

Phase B-2: filename-aware retrieval.  When a user asks "do you have any resume"
or "aarya document", semantic + keyword chunk retrieval may be weak, but the
filename is strong evidence.  This module provides ``filename_search`` which matches
query tokens against normalized document filenames within the workspace, then returns
chunks from matching documents.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk

#: RRF constant (CLAUDE.md 8.1) — a standard, well-documented default. The 60 is
#: deliberate: it is large enough that the exact rank matters only near the top,
#: where the two engines agree, and small enough that a chunk found by only one
#: engine still scores meaningfully.
RRF_K = 60


@dataclass(frozen=True)
class Match:
    """One chunk as returned by a single retrieval engine, with its rank."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    page_number: int | None
    section_title: str | None
    chunk_index: int
    #: 1-based position within this engine's result list.
    rank: int


@dataclass(frozen=True)
class HybridCandidate:
    """A chunk that survived fusion, carrying its combined RRF score."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    page_number: int | None
    section_title: str | None
    chunk_index: int
    rrf_score: float


async def semantic_search(
    session: AsyncSession,
    *,
    query_embedding: list[float],
    workspace_id: uuid.UUID,
    limit: int,
    document_id: uuid.UUID | None = None,
) -> list[Match]:
    """Chunks nearest to `query_embedding` by cosine distance, one workspace only.

    The vector index (``ix_document_chunks_embedding_hnsw``, migration 0008) is a
    cosine-distance index, and ``cosine_distance`` compiles to exactly the operator
    that index serves — ordering by it keeps the planner on the ANN scan.
    """
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.content,
            DocumentChunk.page_number,
            DocumentChunk.section_title,
            DocumentChunk.chunk_index,
            Document.filename,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.workspace_id == workspace_id)
    )
    if document_id is not None:
        stmt = stmt.where(DocumentChunk.document_id == document_id)

    rows = (
        await session.execute(
            stmt.order_by(distance).limit(limit)
        )
    ).all()

    return [
        Match(
            chunk_id=row.id,
            document_id=row.document_id,
            filename=row.filename,
            content=row.content,
            page_number=row.page_number,
            section_title=row.section_title,
            chunk_index=row.chunk_index,
            rank=index,
        )
        for index, row in enumerate(rows, start=1)
    ]


async def keyword_search(
    session: AsyncSession,
    *,
    query: str,
    workspace_id: uuid.UUID,
    limit: int,
    document_id: uuid.UUID | None = None,
) -> list[Match]:
    """Chunks matching `query` by PostgreSQL full-text search, one workspace only.

    ``websearch_to_tsquery`` turns the raw user string into a ``tsquery`` — it is a
    function receiving a bind parameter, never interpolated SQL, so there is no
    injection surface. The keyword index (``ix_document_chunks_content_tsv_gin``,
    migration 0008) serves the ``@@`` operator; ``ts_rank`` orders by relevance.
    """
    ts_query = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank(DocumentChunk.content_tsv, ts_query)

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.content,
            DocumentChunk.page_number,
            DocumentChunk.section_title,
            DocumentChunk.chunk_index,
            Document.filename,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.workspace_id == workspace_id,
            DocumentChunk.content_tsv.op("@@")(ts_query),
        )
    )
    if document_id is not None:
        stmt = stmt.where(DocumentChunk.document_id == document_id)

    rows = (
        await session.execute(
            stmt.order_by(rank.desc()).limit(limit)
        )
    ).all()

    return [
        Match(
            chunk_id=row.id,
            document_id=row.document_id,
            filename=row.filename,
            content=row.content,
            page_number=row.page_number,
            section_title=row.section_title,
            chunk_index=row.chunk_index,
            rank=index,
        )
        for index, row in enumerate(rows, start=1)
    ]


def rrf_merge(
    semantic: list[Match],
    keyword: list[Match],
    *,
    top_n: int,
    k: int = RRF_K,
) -> list[HybridCandidate]:
    """Fuse two ranked lists into one by Reciprocal Rank Fusion, deduplicated.

    A chunk found by both engines keeps both contributions, so agreement pushes it
    up; a chunk found by one engine only still competes via its single term. The
    merge is keyed on ``chunk_id`` and keeps the first row's citation metadata —
    both engines read the same chunk row, so the fields cannot disagree.
    """
    scores: dict[uuid.UUID, float] = {}
    first: dict[uuid.UUID, Match] = {}

    for match in semantic:
        scores[match.chunk_id] = scores.get(match.chunk_id, 0.0) + 1.0 / (k + match.rank)
        first.setdefault(match.chunk_id, match)
    for match in keyword:
        scores[match.chunk_id] = scores.get(match.chunk_id, 0.0) + 1.0 / (k + match.rank)
        first.setdefault(match.chunk_id, match)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [
        HybridCandidate(
            chunk_id=match.chunk_id,
            document_id=match.document_id,
            filename=match.filename,
            content=match.content,
            page_number=match.page_number,
            section_title=match.section_title,
            chunk_index=match.chunk_index,
            rrf_score=score,
        )
        for chunk_id, score in ranked
        if (match := first.get(chunk_id)) is not None
    ]


# ---------------------------------------------------------------------------
# Filename-aware retrieval (Phase B-2)
# ---------------------------------------------------------------------------

# Common words that are too generic to serve as filename evidence.
_FILENAME_STOP_WORDS = frozenset({
    "document", "doc", "file", "the", "a", "an", "have", "has",
    "any", "you", "what", "which", "do", "does", "is", "are",
    "about", "that", "this", "there", "can", "could", "would",
    "should", "tell", "me", "give", "show", "list", "name",
})


def _normalize_filename_for_match(text: str) -> str:
    """Normalize a filename (or query token) for matching.

    Reuses the same normalization strategy as doc_targeting.py:
    strip extension, lowercase, replace punctuation with spaces,
    collapse whitespace.
    """
    import re
    import unicodedata

    t = text.lower().strip()
    # Remove file extensions.
    t = re.sub(r"\.(pdf|docx?|xlsx?|csv|txt|pptx?)$", "", t)
    # Replace punctuation and special characters with spaces.
    t = re.sub(r"[_\-/\\.,;:!?\"'()\[\]{}]", " ", t)
    # Normalize unicode.
    t = unicodedata.normalize("NFKD", t)
    # Collapse whitespace.
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_filename_tokens(query: str) -> set[str]:
    """Extract meaningful tokens from a query for filename matching.

    Strips stop words and generic document-type words that don't help
    distinguish filenames.
    """
    normalized = _normalize_filename_for_match(query)
    tokens = set(normalized.split()) - _FILENAME_STOP_WORDS
    return tokens


def _filename_matches_query(filename: str, query_tokens: set[str]) -> bool:
    """Check if a normalized filename contains meaningful query tokens.

    At least one query token (not a stop word) must appear in the
    normalized filename tokens.
    """
    if not query_tokens:
        return False
    doc_tokens = set(_normalize_filename_for_match(filename).split())
    # Remove stop words from the document tokens too.
    doc_tokens -= _FILENAME_STOP_WORDS
    if not doc_tokens:
        return False
    # At least one query token must overlap with document tokens.
    return bool(query_tokens & doc_tokens)


async def filename_search(
    session: AsyncSession,
    *,
    query: str,
    workspace_id: uuid.UUID,
    limit: int,
) -> list[Match]:
    """Retrieve chunks from documents whose filenames match the query.

    This is a workspace-scoped, READY-only filename search.  It extracts
    meaningful tokens from the query (stripping stop words), normalizes
    document filenames the same way as doc_targeting.py, and returns
    chunks from matching documents.

    Used as an additional candidate source alongside semantic and keyword
    search, to handle queries like "do you have any resume" where chunk
    content may not contain the word "resume" but the filename does.
    """
    query_tokens = _extract_filename_tokens(query)
    if not query_tokens:
        return []

    # Fetch READY document metadata for this workspace.
    doc_rows = (
        await session.execute(
            select(
                Document.id,
                Document.filename,
            ).where(
                Document.workspace_id == workspace_id,
                Document.status == "READY",
            )
        )
    ).all()

    if not doc_rows:
        return []

    # Find documents whose filenames match the query.
    matching_doc_ids: list[uuid.UUID] = []
    for doc_id, filename in doc_rows:
        if _filename_matches_query(filename, query_tokens):
            matching_doc_ids.append(doc_id)

    if not matching_doc_ids:
        return []

    # Retrieve chunks from matching documents (workspace-scoped).
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.content,
            DocumentChunk.page_number,
            DocumentChunk.section_title,
            DocumentChunk.chunk_index,
            Document.filename,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.workspace_id == workspace_id,
            DocumentChunk.document_id.in_(matching_doc_ids),
        )
        .order_by(DocumentChunk.chunk_index)
        .limit(limit)
    )

    rows = (await session.execute(stmt)).all()

    return [
        Match(
            chunk_id=row.id,
            document_id=row.document_id,
            filename=row.filename,
            content=row.content,
            page_number=row.page_number,
            section_title=row.section_title,
            chunk_index=row.chunk_index,
            rank=index,
        )
        for index, row in enumerate(rows, start=1)
    ]


__all__ = [
    "HybridCandidate",
    "Match",
    "RRF_K",
    "filename_search",
    "keyword_search",
    "rrf_merge",
    "semantic_search",
]
