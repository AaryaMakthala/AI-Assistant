"""Local embedding generation with bge-small-en-v1.5.

The model is pinned in config and loaded once per process. Queries and documents must be
embedded by the same model that produced the stored vectors — mixing models silently
returns plausible-looking garbage rather than failing (CLAUDE.md section 7, Risk 1), so
the dimension is asserted against config on load rather than trusted.

bge models are trained with an asymmetric convention: a short query gets an instruction
prefix, an indexed passage does not. Applying the prefix to both, or neither, measurably
degrades retrieval, which is why the two paths are separate functions here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from app.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

#: Prepended to search queries only, never to stored passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """Load the pinned model once per process, guarding against a concurrent double-load.

    Imported lazily: sentence-transformers pulls in torch, which costs seconds of import
    time that the API process should never pay — only workers embed.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                settings = get_settings()
                logger.info("Loading embedding model {name}", name=settings.embedding_model)
                model = SentenceTransformer(settings.embedding_model)
                actual = model.get_sentence_embedding_dimension()
                if actual != settings.embedding_dim:
                    raise RuntimeError(
                        f"Embedding model {settings.embedding_model} produces {actual}-dim "
                        f"vectors but the schema stores {settings.embedding_dim}. Re-embed "
                        "the whole index before changing models — never mix."
                    )
                _model = model
    return _model


def embed_passages(texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
    """Embed document chunks for storage.

    All-or-nothing: any failure propagates. Ingestion uses :func:`embed_passages_resilient`
    instead, which degrades to partial results rather than losing a whole document.
    """
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        # Cosine distance is what the HNSW index uses; normalizing makes it exact.
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [vector.tolist() for vector in vectors]


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
    chunk is finally dropped. A 400-page manual where a single chunk trips the model
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
    """Encode one batch, retrying on failure.

    The model is local, so a retry is not waiting out a rate limit — it is covering a
    transient allocation failure, which retrying immediately can genuinely clear. There is
    no backoff for that reason: a sleep would only delay the fallback path.
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
                error=exc,
            )
    assert last is not None  # noqa: S101 — the loop cannot exit without setting it
    raise last


def embed_query(text: str) -> list[float]:
    """Embed a search query, with the instruction prefix bge expects."""
    vector = get_model().encode(
        QUERY_INSTRUCTION + text,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vector.tolist()


def reset_model() -> None:
    """Drop the cached model. For tests and for a deliberate model change."""
    global _model
    with _model_lock:
        _model = None


__all__ = [
    "QUERY_INSTRUCTION",
    "EmbeddingResult",
    "embed_passages",
    "embed_passages_resilient",
    "embed_query",
    "get_model",
    "reset_model",
]
