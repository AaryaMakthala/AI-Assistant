"""Hosted embedding provider (Gemini gemini-embedding-001, Matryoshka-768) — no local ML stack.

Replaces the local sentence-transformers / PyTorch stack with a hosted embedding
API so the process never imports torch and idle memory stays low.

The bge-style asymmetric convention is handled by the API's ``taskType`` field,
not by a text prefix: queries use ``RETRIEVAL_QUERY``, indexed passages use
``RETRIEVAL_DOCUMENT``.  The schema dimension (``app/db/models.py``
``EMBEDDING_DIM``) is asserted against every returned vector — mixing models
silently returns plausible-looking garbage rather than failing (CLAUDE.md section
7, Risk 1), so the dimension check stays even though the model moved to a
provider.

One shared, lazily-created, lock-guarded sync ``httpx.Client`` serves all
embedding calls: connection pooling is reused across requests without the process
opening a client before first use.  Callers already run embedding inside
``asyncio.to_thread``, which is exactly where a sync client belongs.

Failover policy mirrors the LLM chain (CLAUDE.md section 10): timeout/connection
errors, 429, and 5xx are transient — retried once, then raised as retryable so
``embed_passages_resilient`` can degrade gracefully.  A 400 is our own fault (a
payload the API will always reject) and 401/403 is a configuration problem, so
both surface immediately instead of retrying in place.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

import httpx
from loguru import logger

from app.config import get_settings

RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"

#: Gemini ``batchEmbedContents`` accepts up to 100 rows per call.
GEMINI_BATCH_LIMIT = 100


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
    (``RETRIEVAL_QUERY`` / ``RETRIEVAL_DOCUMENT`` for Gemini).  Results are
    positionally aligned to the input.
    """

    model: str

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str,
        batch_size: int = 32,
    ) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Shared sync HTTP client (lazy, lock-guarded)
# ---------------------------------------------------------------------------

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _shared_client(timeout: float) -> httpx.Client:
    """Return the one process-wide sync client, creating it on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(timeout=timeout)
    return _client


def reset_shared_client() -> None:
    """Close and drop the shared client. For tests and deliberate reloads."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


class GeminiEmbeddingProvider:
    """Embed through the Gemini native ``:embedContent`` endpoints."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.embedding_model
        self._dimension = settings.embedding_dim
        self._base_url = settings.embedding_base_url.rstrip("/")
        api_key = settings.embedding_api_key
        if api_key is None and settings.gemini_api_key is not None:
            api_key = settings.gemini_api_key
        if api_key is None or not api_key.get_secret_value():
            raise EmbeddingError(
                "No embedding API key configured. Set EMBEDDING_API_KEY "
                "(or GEMINI_API_KEY) for provider 'gemini'.",
                retryable=False,
            )
        self._api_key = api_key.get_secret_value()
        self._timeout = settings.embedding_timeout_seconds

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str,
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []
        batch_size = min(max(int(batch_size), 1), GEMINI_BATCH_LIMIT)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch, task_type=task_type))
        return vectors

    # -- internals ----------------------------------------------------------

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """Embed one <=100-row batch with a single retry on transient failure."""
        last_error: EmbeddingError | None = None
        for attempt in (1, 2):
            try:
                return self._post_batch(texts, task_type=task_type)
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

    def _post_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        endpoint = f"{self._base_url}/models/{self.model}:batchEmbedContents"
        requests = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
                # Matryoshka-compatible models (gemini-embedding-001) default to
                # their full dimension; pin the schema's EMBEDDING_DIMENSION so the
                # stored vector type matches (asserted again on the response below).
                "outputDimensionality": self._dimension,
            }
            for text in texts
        ]
        payload = {"model": f"models/{self.model}", "requests": requests}
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        try:
            response = _shared_client(self._timeout).post(
                endpoint, json=payload, headers=headers
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding request failed: {exc}",
                retryable=True,
            ) from exc

        if response.status_code >= 400:
            body = response.text[:300]
            if response.status_code in (400, 401, 403):
                # Our own fault (malformed payload) or bad config (auth): retrying
                # in place cannot fix either.
                raise EmbeddingError(
                    f"Embedding provider returned HTTP {response.status_code}: {body}",
                    retryable=False,
                    status_code=response.status_code,
                )
            raise EmbeddingError(
                f"Embedding provider returned HTTP {response.status_code}: {body}",
                retryable=True,
                status_code=response.status_code,
            )

        try:
            data = response.json()
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding provider returned unparseable JSON: {exc}",
                retryable=True,
            ) from exc

        items = data.get("embeddings") or []
        if len(items) != len(texts):
            raise EmbeddingError(
                f"Embedding provider returned {len(items)} vectors for "
                f"{len(texts)} texts",
                retryable=True,
            )

        vectors: list[list[float]] = []
        for item in items:
            values = (item or {}).get("values") or []
            if len(values) != self._dimension:
                raise EmbeddingError(
                    f"Embedding provider returned {len(values)}-dim vectors but "
                    f"the schema stores {self._dimension}. Re-embed the whole "
                    "index before changing models — never mix.",
                    retryable=False,
                )
            vectors.append(values)

        logger.info(
            "Embedded {n} texts ({task}) in {elapsed:.2f}s",
            n=len(texts),
            task=task_type,
            elapsed=time.perf_counter() - started,
        )
        return vectors


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
                if settings.embedding_provider != "gemini":
                    raise EmbeddingError(
                        f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider} "
                        "(only 'gemini' is implemented)",
                        retryable=False,
                    )
                logger.info(
                    "Using embedding provider={provider} model={model} dim={dim}",
                    provider=settings.embedding_provider,
                    model=settings.embedding_model,
                    dim=settings.embedding_dim,
                )
                _provider = GeminiEmbeddingProvider()
    return _provider


def reset_embedding_provider() -> None:
    """Drop the cached provider. For tests and a deliberate model change."""
    global _provider
    with _provider_lock:
        _provider = None
    reset_shared_client()


__all__ = [
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "EmbeddingError",
    "EmbeddingProvider",
    "GeminiEmbeddingProvider",
    "get_embedding_provider",
    "reset_embedding_provider",
]