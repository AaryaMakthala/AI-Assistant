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
    """Embed document chunks for storage."""
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


__all__ = ["QUERY_INSTRUCTION", "embed_passages", "embed_query", "get_model", "reset_model"]
