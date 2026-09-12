"""Hosted embedding provider (Voyage voyage-4-lite) — no local ML stack.

Replaces the local sentence-transformers / PyTorch stack with a hosted embedding
API so the process never imports torch and idle memory stays low.

The bge-style asymmetric convention is handled by the API's ``input_type`` field,
not by a text prefix: queries use ``query``, indexed passages use ``document``.
The schema dimension (``app/db/models.py`` ``EMBEDDING_DIM``) is asserted against
every returned vector — mixing models silently returns plausible-looking garbage
rather than failing (CLAUDE.md section 7, Risk 1), so the dimension check stays
even with the model at a Matryoshka-capable provider (voyage-4-lite supports
1024/512/256/2048; config pins one value that must equal the pgvector column).

One process-wide, lazily-created, lock-guarded ``VoyageEmbeddingProvider`` owns one
sync ``voyageai.Client``: connection pooling is reused across requests without the
process opening a client before first use.  Callers already run embedding inside
``asyncio.to_thread``, which is exactly where a sync client belongs.

Failover policy mirrors the LLM chain (CLAUDE.md section 10): timeout/connection
errors, 429, and 5xx are transient — retried in place once, then raised as
retryable so ``embed_passages_resilient`` can degrade gracefully.  A 400-class
payload error or 401/403 auth failure is our own fault (or a configuration
problem), so both surface immediately instead of retrying in place.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

from loguru import logger

from app.config import get_settings

RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"

#: Voyage ``/v1/embeddings`` accepts up to 1000 rows per request; the cap here is
#: a self-imposed blast-radius limit so one poisonous batch cannot hold thousands
#: of chunks hostage inside a single retry. Well under the 1M-token budget.
VOYAGE_BATCH_LIMIT = 128

#: Maps the provider-agnostic task tags onto Voyage's ``input_type`` values.
_TASK_TYPE_TO_INPUT_TYPE = {
    RETRIEVAL_QUERY: "query",
    RETRIEVAL_DOCUMENT: "document",
}


class EmbeddingError(Exception):
    """Embedding failure with a retry hint for the resilient caller."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class EmbeddingProvider(Protocol):
    """Sync embedding contract every provider satisfies.

    ``task_type`` is the provider-level query/passage asymmetric convention
    (``RETRIEVAL_QUERY`` / ``RETRIEVAL_DOCUMENT`` — Voyage maps these onto its
    ``input_type`` field, ``query`` / ``document``).  Results are positionally
    aligned to the input.
    """

    model: str

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str,
        batch_size: int = 32,
    ) -> list[list[float]]: ...


class VoyageEmbeddingProvider:
    """Embed through the Voyage ``/v1/embeddings`` endpoint (official SDK)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.embedding_model
        self._dimension = settings.embedding_dim
        self._timeout = settings.embedding_timeout_seconds

        # The provider reads VOYAGE_API_KEY first; EMBEDDING_API_KEY remains as an
        # optional generic override. No dependency on GEMINI_API_KEY — Gemini is
        # the LLM fallback provider, never an embedding channel.
        api_key = settings.voyage_api_key
        if api_key is None or not api_key.get_secret_value():
            api_key = settings.embedding_api_key
        if api_key is None or not api_key.get_secret_value():
            raise EmbeddingError(
                "No embedding API key configured. Set VOYAGE_API_KEY for provider 'voyage'.",
                retryable=False,
            )
        self._api_key = api_key.get_secret_value()

        try:
            import voyageai
            import voyageai.error as _errors
        except Exception as exc:  # pragma: no cover - packaging failure only
            raise EmbeddingError(
                f"voyageai package unavailable: {exc}",
                retryable=False,
            ) from exc
        self._voyage_error = _errors
        self._client = voyageai.Client(
            api_key=self._api_key,
            max_retries=0,  # our own bounded retry loop is authoritative
            timeout=self._timeout,
        )

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str,
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []
        batch_size = min(max(int(batch_size), 1), VOYAGE_BATCH_LIMIT)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch, task_type=task_type))
        return vectors

    # -- internals ----------------------------------------------------------

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """Embed one <=128-row batch with a single retry on transient failure."""
        input_type = _TASK_TYPE_TO_INPUT_TYPE.get(task_type)
        if input_type is None:
            raise EmbeddingError(
                f"Unknown embedding task_type={task_type!r} "
                f"(want {RETRIEVAL_QUERY} or {RETRIEVAL_DOCUMENT})",
                retryable=False,
            )

        last_error: EmbeddingError | None = None
        for attempt in (1, 2):
            try:
                return self._embed_once(texts, input_type=input_type)
            except EmbeddingError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                logger.warning(
                    "Embedding batch attempt {attempt}/2 failed: {error}",
                    attempt=attempt,
                    error=str(exc)[:200],
                )
                if last_error.status_code == 429:
                    time.sleep(1.0)
        assert last_error is not None  # noqa: S101 — loop guarantees this
        raise last_error

    def _embed_once(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        started = time.perf_counter()
        try:
            result = self._client.embed(
                texts,
                model=self.model,
                input_type=input_type,
                output_dimension=self._dimension,
            )
        except Exception as exc:
            raise self._classify(exc) from exc

        vectors = self._collect_vectors(result, texts)
        logger.info(
            "Embedded {n} texts ({task}) in {elapsed:.2f}s",
            n=len(texts),
            task=input_type,
            elapsed=time.perf_counter() - started,
        )
        return vectors

    def _classify(self, exc: Exception) -> EmbeddingError:
        """Map an SDK exception onto EmbeddingError with a retry hint.

        Rate-limit, service-unavailable, timeout and connection errors are
        transient: retryable.  Auth/config and malformed-request errors cannot be
        fixed by retrying.  Anything else is surfaced as retryable so the
        resilient ingestion path can probe it text-by-text instead of failing the
        whole document.
        """
        err = self._voyage_error
        retryable_types = (
            err.RateLimitError,
            err.ServiceUnavailableError,
            err.ServerError,
            err.Timeout,
            err.APIConnectionError,
        )
        non_retryable_types = (
            err.InvalidRequestError,
            err.MalformedRequestError,
            err.AuthenticationError,
        )
        if isinstance(exc, non_retryable_types):
            return EmbeddingError(
                f"Embedding provider rejected the request: {exc}",
                retryable=False,
                status_code=getattr(exc, "http_status", None),
            )
        if isinstance(exc, retryable_types):
            return EmbeddingError(
                f"Embedding provider error: {exc}",
                retryable=True,
                status_code=getattr(exc, "http_status", None),
            )
        return EmbeddingError(
            f"Embedding provider error: {exc}",
            retryable=True,
            status_code=getattr(exc, "http_status", None),
        )

    def _collect_vectors(self, result: object, texts: list[str]) -> list[list[float]]:
        try:
            items = list(result.embeddings)
        except Exception as exc:
            raise EmbeddingError(
                "Embedding provider returned an unusable response",
                retryable=True,
            ) from exc
        if len(items) != len(texts):
            raise EmbeddingError(
                f"Embedding provider returned {len(items)} vectors for {len(texts)} texts",
                retryable=True,
            )

        # Voyage 0.5.x returns a flat list of vectors in input order; older
        # SDK shapes return objects carrying ``.index``/``.embedding``. Support
        # both, realigning by the explicit ``index`` when it is present so a
        # reordered response cannot renumber a caller's chunks.
        vectors: list[list[float] | None] = [None] * len(texts)
        for i, item in enumerate(items):
            if isinstance(item, (list, tuple)):
                target, values = i, list(item)
            else:
                index = getattr(item, "index", None)
                if not isinstance(index, int) or not 0 <= index < len(texts):
                    raise EmbeddingError(
                        "Embedding provider returned an out-of-range vector index",
                        retryable=True,
                    )
                target, values = index, list(getattr(item, "embedding", None) or [])
            if len(values) != self._dimension:
                raise EmbeddingError(
                    f"Embedding provider returned {len(values)}-dim vectors but "
                    f"the schema stores {self._dimension}. Re-embed the whole "
                    "index before changing models — never mix.",
                    retryable=False,
                )
            vectors[target] = values
        if any(vector is None for vector in vectors):
            raise EmbeddingError(
                "Embedding provider returned an incomplete vector set",
                retryable=True,
            )
        return vectors  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider: EmbeddingProvider | None = None
_provider_lock = threading.Lock()


def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide provider, built once from settings.

    Callers that change settings (tests) must call
    :func:`reset_embedding_provider` so the new values take effect.
    """
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                settings = get_settings()
                if settings.embedding_provider != "voyage":
                    raise EmbeddingError(
                        f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider} "
                        "(only 'voyage' is implemented)",
                        retryable=False,
                    )
                logger.info(
                    "Using embedding provider={provider} model={model} dim={dim}",
                    provider=settings.embedding_provider,
                    model=settings.embedding_model,
                    dim=settings.embedding_dim,
                )
                _provider = VoyageEmbeddingProvider()
    return _provider


def reset_embedding_provider() -> None:
    """Drop the cached provider. For tests and a deliberate model change."""
    global _provider
    with _provider_lock:
        _provider = None


__all__ = [
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "EmbeddingError",
    "EmbeddingProvider",
    "VoyageEmbeddingProvider",
    "get_embedding_provider",
    "reset_embedding_provider",
]
