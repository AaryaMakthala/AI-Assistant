"""Voyage embedding provider: wire-contract tests against a stubbed SDK.

These pin the Voyage path without ever calling the hosted API:

* the configured model (`voyage-4-lite`) and pinned dimension (1024) are sent on
  every request,
* queries use `input_type="query"`, indexed passages `input_type="document"`,
* returned vectors are re-aligned by the provider's explicit index, then asserted
  to be exactly EMBEDDING_DIMENSION wide,
* transient SDK failures retry in place once and only then surface as retryable,
  while 4xx/auth failures surface immediately as non-retryable,
* no key material is needed at test time (VOYAGE_API_KEY is stubbed as a secret).

The provider reads settings through `get_settings()`, so each test re-pins the
embedding section and drops the cached provider before and after.
"""

from __future__ import annotations

import pytest
import voyageai
import voyageai.error as vge
from loguru import logger

from app.config import get_settings
from app.rag import embeddings
from app.rag.embedding_provider import (
    RETRIEVAL_DOCUMENT,
    RETRIEVAL_QUERY,
    EmbeddingError,
    VoyageEmbeddingProvider,
    get_embedding_provider,
    reset_embedding_provider,
)

pytestmark = pytest.mark.usefixtures("voyage_env", "voyage_sdk")


def _vec(length: int = 1024, seed: int = 0) -> list[float]:
    """Deterministic unit vector so cosine math stays stable in tests."""
    value = 1.0 / (length**0.5)
    return [value * (1 if (seed + i) % 2 == 0 else -1) for i in range(length)]


class FakeEmbedding:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class FakeResult:
    def __init__(self, embeddings: list[FakeEmbedding]) -> None:
        self.embeddings = embeddings


class FakeVoyageClient:
    """Scripted stand-in for ``voyageai.Client``.

    ``result_factory`` receives the number of texts and returns the response;
    ``error`` makes every call raise.  Calls are recorded for inspection.
    """

    def __init__(self, result_factory=None, error: Exception | None = None) -> None:
        self.result_factory = result_factory
        self.error = error
        self.instances: list[FakeVoyageClient] = []
        self.calls: list[dict] = []

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        input_type: str | None = None,
        truncation: bool = True,
        output_dtype: str | None = None,
        output_dimension: int | None = None,
    ) -> FakeResult:
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "input_type": input_type,
                "output_dimension": output_dimension,
            }
        )
        if self.error is not None:
            raise self.error
        return (self.result_factory or _aligned_result)(len(texts))


@pytest.fixture
def voyage_env(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
    """Pin the embedding section and provide a test VOYAGE_API_KEY."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-4-lite")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("VOYAGE_API_KEY", "va-test-key")
    get_settings.cache_clear()
    reset_embedding_provider()
    yield
    get_settings.cache_clear()
    reset_embedding_provider()


@pytest.fixture
def voyage_sdk(monkeypatch: pytest.MonkeyPatch) -> FakeVoyageClient:
    """Swap the SDK's Client for a scripted fake; returns the fake."""
    fake = FakeVoyageClient()
    monkeypatch.setattr(voyageai, "Client", lambda **kwargs: fake)
    return fake


def _install(monkeypatch: pytest.MonkeyPatch, fake: object) -> None:
    monkeypatch.setattr(voyageai, "Client", lambda **kwargs: fake)


def _aligned_result(n: int) -> FakeResult:
    embeddings = []
    for index in range(n):
        embeddings.append(FakeEmbedding(index, _vec(seed=index)))
    return FakeResult(embeddings)


# ---------------------------------------------------------------------------
# Wire contract: model, dimension, input_type
# ---------------------------------------------------------------------------


def test_provider_uses_voyage_model_and_dimension(
    voyage_sdk: FakeVoyageClient,
) -> None:
    provider = VoyageEmbeddingProvider()

    assert provider.model == "voyage-4-lite"
    provider.embed_texts(["a", "b"], task_type=RETRIEVAL_DOCUMENT)

    call = voyage_sdk.calls[0]
    assert call["model"] == "voyage-4-lite"
    assert call["output_dimension"] == 1024


def test_document_embedding_uses_input_type_document(
    voyage_sdk: FakeVoyageClient,
) -> None:
    provider = VoyageEmbeddingProvider()
    provider.embed_texts(["doc text"], task_type=RETRIEVAL_DOCUMENT)

    assert voyage_sdk.calls[0]["input_type"] == "document"


def test_query_embedding_uses_input_type_query(
    voyage_sdk: FakeVoyageClient,
) -> None:
    provider = VoyageEmbeddingProvider()
    provider.embed_texts(["question?"], task_type=RETRIEVAL_QUERY)

    assert voyage_sdk.calls[0]["input_type"] == "query"


def test_embed_query_and_embed_passages_route_through_voyage(
    monkeypatch: pytest.MonkeyPatch, voyage_sdk: FakeVoyageClient
) -> None:
    voyage_sdk.result_factory = _aligned_result
    provider = get_embedding_provider()
    assert isinstance(provider, VoyageEmbeddingProvider)

    embeddings.embed_query("what is vacation?")
    embeddings.embed_passages(["handbook paragraph"])

    assert voyage_sdk.calls[0]["input_type"] == "query"
    assert voyage_sdk.calls[1]["input_type"] == "document"


def test_client_instance_is_reused_across_calls(
    voyage_sdk: FakeVoyageClient,
) -> None:
    provider = VoyageEmbeddingProvider()
    provider.embed_texts(["a"], task_type=RETRIEVAL_QUERY)
    provider.embed_texts(["b"], task_type=RETRIEVAL_QUERY)

    # Each call hit the same (fake) client object.
    assert len(voyage_sdk.calls) == 2


def test_unknown_task_type_is_rejected(voyage_sdk: FakeVoyageClient) -> None:
    provider = VoyageEmbeddingProvider()

    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_texts(["a"], task_type="WAT")

    assert exc_info.value.retryable is False
    assert voyage_sdk.calls == []


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_non_retryable(
    monkeypatch: pytest.MonkeyPatch, voyage_env: None
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    with pytest.raises(EmbeddingError) as exc_info:
        VoyageEmbeddingProvider()

    assert exc_info.value.retryable is False
    assert "VOYAGE_API_KEY" in str(exc_info.value)


def test_vectors_are_realigned_by_index(voyage_sdk: FakeVoyageClient) -> None:
    def shuffled(n: int) -> FakeResult:
        items = []
        for index in range(n):
            items.append(FakeEmbedding(index, _vec(seed=index)))
        items.reverse()  # provider reported output in reverse order
        return FakeResult(items)

    voyage_sdk.result_factory = shuffled
    provider = VoyageEmbeddingProvider()

    vectors = provider.embed_texts(["first", "second"], task_type=RETRIEVAL_DOCUMENT)

    assert len(vectors) == 2
    assert vectors[0] == pytest.approx(_vec(seed=0))
    assert vectors[1] == pytest.approx(_vec(seed=1))


def test_wrong_dimension_is_a_hard_error(voyage_env: None, voyage_sdk: FakeVoyageClient) -> None:
    def short(n: int) -> FakeResult:
        return FakeResult([FakeEmbedding(i, [0.5] * 1000) for i in range(n)])

    voyage_sdk.result_factory = short
    provider = VoyageEmbeddingProvider()

    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_texts(["a"], task_type=RETRIEVAL_DOCUMENT)

    assert exc_info.value.retryable is False
    assert "1024" in str(exc_info.value)


def test_missing_vectors_in_response_is_retryable(voyage_sdk: FakeVoyageClient) -> None:
    def gapped(n: int) -> FakeResult:
        return FakeResult([FakeEmbedding(i, _vec()) for i in range(n) if i % 2 == 0])

    voyage_sdk.result_factory = gapped
    provider = VoyageEmbeddingProvider()

    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_texts(["a", "b"], task_type=RETRIEVAL_DOCUMENT)

    assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# Failure classification and bounded retry
# ---------------------------------------------------------------------------


def test_transient_failure_retries_once_then_raises_retryable(
    voyage_sdk: FakeVoyageClient,
) -> None:
    voyage_sdk.error = vge.RateLimitError("slow down")
    provider = VoyageEmbeddingProvider()

    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_texts(["a"], task_type=RETRIEVAL_QUERY)

    assert exc_info.value.retryable is True
    assert len(voyage_sdk.calls) == 2  # attempt + one retry, no more


def test_auth_failure_is_not_retried(voyage_sdk: FakeVoyageClient) -> None:
    voyage_sdk.error = vge.AuthenticationError("bad key")
    provider = VoyageEmbeddingProvider()

    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_texts(["a"], task_type=RETRIEVAL_QUERY)

    assert exc_info.value.retryable is False
    assert len(voyage_sdk.calls) == 1


def test_resilient_path_isolates_a_poisonous_batch(
    monkeypatch: pytest.MonkeyPatch, voyage_env: None
) -> None:
    class FailingClient:
        """Rejects any batch containing the poisonous text, unlike Gemini-era
        retry semantics: a genuine 400 keeps failing, so the resilient layer
        splits the batch and only that one text is dropped."""

        def embed(self, texts, **kwargs) -> FakeResult:  # noqa: ANN001, ANN003
            if any("bad" in text for text in texts):
                raise vge.InvalidRequestError("provider rejects this text")
            return _aligned_result(len(texts))

    _install(monkeypatch, FailingClient())
    # The resilient layer is exercised end-to-end through the real entry point.
    result = embeddings.embed_passages_resilient(["ok", "bad", "ok2"], batch_size=3)

    assert result.failed == 1
    assert result.vectors[0] is not None
    assert result.vectors[1] is None
    assert result.vectors[2] is not None


def test_provider_factory_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch, voyage_env: None
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "some-other")
    get_settings.cache_clear()

    with pytest.raises(EmbeddingError) as exc_info:
        get_embedding_provider()

    assert exc_info.value.retryable is False
    assert "some-other" in str(exc_info.value)


def test_key_is_never_logged(voyage_sdk: FakeVoyageClient) -> None:
    """Regression: retry logs must not echo the API key."""
    import io

    buffer = io.StringIO()
    handler_id = logger.add(buffer, level="WARNING")
    try:
        voyage_sdk.error = vge.ServiceUnavailableError("down")
        provider = VoyageEmbeddingProvider()
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["a"], task_type=RETRIEVAL_QUERY)
    finally:
        logger.remove(handler_id)

    assert "va-test-key" not in buffer.getvalue()
