"""Hosted embedding API, with the old local-model interface preserved.

Queries and documents are embedded by the same hosted model that produced the
stored vectors — mixing models silently returns plausible-looking garbage rather
than failing (CLAUDE.md section 7, Risk 1), so the returned dimension is asserted
against config on every call.

The bge asymmetric convention (an instruction prefix on queries) is replaced by
the provider's ``taskType``: queries are tagged ``RETRIEVAL_QUERY`` and indexed
passages ``RETRIEVAL_DOCUMENT``.  Callers see no difference — the two entry
points ``embed_query`` / ``embed_passages`` map onto the two task types.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from app.rag.embedding_provider import (
    RETRIEVAL_DOCUMENT,
    RETRIEVAL_QUERY,
    get_embedding_provider,
)
from app.rag.embedding_provider import reset_embedding_provider as reset_model


def embed_passages(texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
    """Embed document chunks for storage.

    All-or-nothing: any failure propagates. Ingestion uses :func:`embed_passages_resilient`
    instead, which degrades to partial results rather than losing a whole document.
    """
    if not texts:
        return []
    return get_embedding_provider().embed_texts(
        texts,
        task_type=RETRIEVAL_DOCUMENT,
        batch_size=batch_size,
    )


@dataclass(frozen=True)
class EmbeddingResult:
    """Vectors aligned to the input, with `None` where embedding gave up.

    Positional alignment is the contract: ``vectors[i]`` belongs to ``texts[i]`` or is
    ``None``. Returning a compacted list instead would silently renumber the chunks, and a
    chunk stored under another chunk's index is a citation pointing at the wrong text.
    """

    vectors: list[list[float] | None]
    #: Indices that could not be embedded after every attempt.
    failed_indices: list[int]

    @property
    def failed(self) -> int:
        return len(self.failed_indices)


def embed_passages_resilient(
    texts: list[str],
    *,
    batch_size: int = 32,
    max_attempts: int = 2,
) -> EmbeddingResult:
    """Embed in batches, isolating failures instead of losing the document.

    Three levels of degradation, in order: a batch is retried, then split into individual
    texts so one poisonous chunk cannot take its neighbours with it, and only that one
    chunk is finally dropped. A 400-page manual where a single chunk trips the provider
    should still be 99.8% searchable — failing the whole ingestion there would be the
    pipeline choosing nothing over almost everything (CLAUDE.md Phase 10).

    Every failure is logged with its index. Dropped chunks are counted on the document row,
    so a partial index is visible rather than passing for a complete one.
    """
    vectors: list[list[float] | None] = [None] * len(texts)
    failed: list[int] = []
    if not texts:
        return EmbeddingResult(vectors=vectors, failed_indices=failed)

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            for offset, vector in enumerate(_encode_with_retry(batch, max_attempts=max_attempts)):
                vectors[start + offset] = vector
        except Exception:
            # The batch failed even on retry. Fall back to one text at a time so the
            # blast radius is the offending chunk rather than the whole batch.
            logger.warning(
                "Embedding batch at offset {start} failed; retrying its {n} chunks singly",
                start=start,
                n=len(batch),
            )
            for offset, text in enumerate(batch):
                index = start + offset
                try:
                    vectors[index] = _encode_with_retry([text], max_attempts=max_attempts)[0]
                except Exception as exc:
                    logger.opt(exception=exc).error(
                        "Dropping chunk {index}: it could not be embedded", index=index
                    )
                    failed.append(index)

    return EmbeddingResult(vectors=vectors, failed_indices=failed)


def _encode_with_retry(texts: list[str], *, max_attempts: int) -> list[list[float]]:
    """Encode one batch, retrying on transient provider failures.

    The provider itself already retries once on timeout/429/5xx; this loop adds
    headroom for the resilient ingestion path, where a genuinely poisonous chunk
    (which the provider rejects with a 400) surfaces after the provider's own
    retries and falls through to the one-at-a-time split.
    """
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return embed_passages(texts, batch_size=len(texts))
        except Exception as exc:
            last = exc
            logger.warning(
                "Embedding attempt {attempt}/{total} failed for {n} texts: {error}",
                attempt=attempt,
                total=max_attempts,
                n=len(texts),
                error=str(exc)[:200],
            )
    assert last is not None  # noqa: S101 — the loop cannot exit without setting it
    raise last


def embed_query(text: str) -> list[float]:
    """Embed a search query, tagged RETRIEVAL_QUERY for the provider."""
    vector = get_embedding_provider().embed_texts(
        [text],
        task_type=RETRIEVAL_QUERY,
        batch_size=1,
    )[0]
    return vector


__all__ = [
    "EmbeddingResult",
    "embed_passages",
    "embed_passages_resilient",
    "embed_query",
    "reset_model",
]