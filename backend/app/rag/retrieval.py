"""Vector retrieval over ingested chunks.

Org scoping is applied twice on purpose (CLAUDE.md 4.6): the query carries an explicit
`org_id` predicate, and the session runs under RLS policies that enforce the same rule in
the database. The redundancy is the point — either one alone is a single point of failure
for cross-tenant leakage.

Retrieval returns chunks with the metadata a citation needs. A cited answer whose citation
cannot be traced back to a real document and page is worse than an uncited one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import Float, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.legacy_models import Document, DocumentChunk
from app.rag.embeddings import embed_query


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk plus everything needed to cite it."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    page: int | None
    chunk_index: int
    #: Cosine distance: 0 is identical, 2 is opposite. Lower is better.
    distance: float

    @property
    def similarity(self) -> float:
        """Distance restated as a 0–1 score, for display only."""
        return max(0.0, 1.0 - self.distance / 2.0)

    @property
    def citation_label(self) -> str:
        return f"{self.filename} · page {self.page}" if self.page else self.filename


async def retrieve_chunks(
    session: AsyncSession,
    *,
    query: str,
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    top_k: int | None = None,
    max_distance: float | None = None,
) -> list[RetrievedChunk]:
    """Find the chunks nearest to `query` within one organization.

    `session` must already carry tenant claims — see :func:`app.security.rls.tenant_session`.

    Results span the organization's shared documents plus the caller's own uploads; a
    colleague's personal document is never searched. `user_id` is required rather than
    optional because a default would silently widen the result set the day a caller forgot
    to pass it.
    """
    text = query.strip()
    if not text:
        return []

    settings = get_settings()
    top_k = top_k if top_k is not None else settings.retrieval_top_k
    max_distance = max_distance if max_distance is not None else settings.retrieval_max_distance

    # Same model that embedded the stored passages, with the query-side instruction
    # prefix bge expects (CLAUDE.md section 7, Risk 1).
    embedding = embed_query(text)
    distance = DocumentChunk.embedding.cosine_distance(embedding)

    rows = (
        await session.execute(
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.content,
                DocumentChunk.page,
                DocumentChunk.chunk_index,
                Document.filename,
                cast(distance, Float).label("distance"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            # Explicit predicate alongside RLS: defense in depth, not decoration. The
            # visibility clause mirrors the documents_select policy from migration 0007 —
            # if the two ever disagree, the narrower one wins and search returns less
            # rather than more.
            .where(
                DocumentChunk.org_id == org_id,
                Document.org_id == org_id,
                or_(
                    Document.visibility == "org",
                    Document.uploaded_by == user_id,
                ),
            )
            .order_by(distance)
            .limit(top_k)
        )
    ).all()

    results = [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            filename=row.filename,
            content=row.content,
            page=row.page,
            chunk_index=row.chunk_index,
            distance=float(row.distance),
        )
        for row in rows
        # Filtered here rather than in SQL: the index orders by distance, and a WHERE on
        # the same expression would push the planner off the HNSW scan.
        if float(row.distance) <= max_distance
    ]

    logger.debug(
        "Retrieved {kept}/{found} chunks for org {org}",
        kept=len(results),
        found=len(rows),
        org=org_id,
    )
    return results


def build_context(
    chunks: list[RetrievedChunk], *, max_chars: int | None = None
) -> list[RetrievedChunk]:
    """Trim the retrieved set to what fits in one prompt, nearest first.

    Truncation drops whole chunks rather than cutting one mid-sentence: a half-quoted
    passage is exactly the kind of thing a model completes from its own priors.
    """
    if max_chars is None:
        max_chars = get_settings().retrieval_max_context_chars

    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = len(chunk.content)
        if kept and used + cost > max_chars:
            break
        kept.append(chunk)
        used += cost
    return kept


__all__ = ["RetrievedChunk", "build_context", "retrieve_chunks"]
