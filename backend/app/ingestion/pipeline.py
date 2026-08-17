"""Synchronous document ingestion pipeline (CLAUDE.md Phase 3).

This is the canonical Phase 3 architecture: **inline, synchronous** ingestion
(CLAUDE.md section 9 — "Document Upload & Synchronous Ingestion"). No Celery, no
Redis, no queue: section 11 explicitly rejects Redis+Celery for this project
("synchronous ingestion with a size cap is simpler and just as correct"). The
10 MB upload cap bounds the work, so blocking the request is the honest cost of a
free-tier single-process deployment.

The pipeline is deliberately split into two halves so the API layer can enforce the
section 7 transaction invariant:

1. :func:`prepare_document` — pure computation (extract → normalize → chunk →
   embed). It never touches the database, so a failure here is just an exception.
2. The API layer persists the document row **and** its chunks in one transaction,
   with the document set READY *before* the chunk inserts — the RLS policy
   ``document_chunks_write`` (migration 0008) only permits chunk writes for a READY
   document, evaluated with same-transaction visibility.

This separation keeps the "never READY with zero chunks / never PENDING with
chunks" invariant structural (CLAUDE.md section 7), because both the status flip
and the chunk inserts happen in the same DB transaction.

The CPU-bound embedding step is run in a worker thread via
:func:`asyncio.to_thread` by the API layer so the event loop is not blocked; the
pipeline itself is plain synchronous code and trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.rag.chunking import chunk_pages
from app.rag.embeddings import embed_passages_resilient
from app.rag.extraction import ExtractionError, count_words, extract_pages
from app.security.uploads import ALLOWED_TYPES

#: mime type -> canonical extension, so the extractor is chosen from validated
#: stored metadata rather than from a filename we never trust on the filesystem.
_EXTENSION_BY_MIME = {allowed.mime_type: allowed.extension for allowed in ALLOWED_TYPES.values()}


class IngestionError(Exception):
    """A document could not be ingested. The message is safe to store and show."""


@dataclass(frozen=True)
class PreparedChunk:
    """One chunk ready to persist, carrying everything retrieval will need."""

    chunk_index: int
    content: str
    embedding: list[float]
    page_number: int | None
    section_title: str | None
    chunk_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedDocument:
    """The full result of extraction + chunking + embedding, ready to persist."""

    chunks: list[PreparedChunk]
    word_count: int
    page_count: int | None


def prepare_document(
    file_data: bytes,
    *,
    mime_type: str,
    filename: str,
) -> PreparedDocument:
    """Extract, normalize, chunk and embed one document's bytes.

    Pure computation: no database access, no I/O beyond the temporary file the
    extractor stages internally. Raises :class:`IngestionError` when the document
    cannot be ingested at all (corrupt file, unsupported type, no indexable text,
    nothing embeddable) — the API layer turns that into a FAILED document row.

    The embedding model is pinned in config and its output dimension is asserted
    against the schema on load (``app/rag/embeddings.py``), so the persisted
    vectors can never silently disagree with the pgvector column (CLAUDE.md 13).
    """
    extension = _EXTENSION_BY_MIME.get(mime_type)
    if extension is None:
        raise IngestionError(f"Unsupported media type '{mime_type}'.")

    try:
        pages = extract_pages(file_data, extension=extension)
    except ExtractionError as exc:
        raise IngestionError(str(exc)) from exc

    word_count = sum(count_words(page.text) for page in pages)

    chunks = chunk_pages(pages)
    if not chunks:
        raise IngestionError("Document produced no indexable text.")

    settings = get_settings()
    embedded = embed_passages_resilient(
        [chunk.content for chunk in chunks],
        batch_size=settings.embedding_batch_size,
        max_attempts=settings.embedding_max_attempts,
    )

    prepared: list[PreparedChunk] = []
    for chunk, vector in zip(chunks, embedded.vectors, strict=True):
        if vector is None:
            # A chunk the model could not embed is dropped, not fatal: the legacy
            # worker stored what it could and so does this pipeline (see
            # app/rag/embeddings.py embed_passages_resilient). Only if *nothing*
            # embedded does the document fail.
            continue
        prepared.append(
            PreparedChunk(
                chunk_index=chunk.index,
                content=chunk.content,
                embedding=vector,
                page_number=chunk.page,
                section_title=chunk.label,
                chunk_metadata={
                    "source": filename,
                    "locator": chunk.label,
                    "page": chunk.page,
                    "chunk_index": chunk.index,
                },
            )
        )

    if not prepared:
        raise IngestionError("No chunk in this document could be embedded.")

    page_count = len({page.page for page in pages if page.page is not None}) or None
    return PreparedDocument(chunks=prepared, word_count=word_count, page_count=page_count)


__all__ = ["IngestionError", "PreparedChunk", "PreparedDocument", "prepare_document"]
