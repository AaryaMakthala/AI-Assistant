"""Embedding resilience: one bad chunk must not cost a whole document.

The contract these tests pin down is positional. `embed_passages_resilient` returns a list
the same length as its input, with `None` where embedding gave up — because the caller
zips it against the chunk list to build rows. A compacted list would silently shift every
chunk after a failure onto the wrong index, which is a citation pointing at text it did
not come from.

The model is stubbed rather than loaded: these are tests of the retry and fallback logic,
and downloading 130 MB of weights to assert that a counter increments would make them
useless in CI.
"""

from __future__ import annotations

import pytest

from app.rag import embeddings
from app.rag.embeddings import embed_passages_resilient

pytestmark = pytest.mark.usefixtures("valid_env")


class StubModel:
    """Encodes deterministically, failing for any text in `poison`."""

    def __init__(self, poison: frozenset[str] = frozenset(), fail_times: int = 0) -> None:
        self.poison = poison
        #: Batches still to fail before succeeding — models transient failure.
        self.fail_times = fail_times
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003, ANN201
        # sentence-transformers accepts a bare string; the code under test never does.
        batch = list(texts)
        self.calls.append(batch)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("transient allocation failure")
        if any(text in self.poison for text in batch):
            raise RuntimeError("cannot encode this text")
        return [_FakeVector([float(len(text))]) for text in batch]


class _FakeVector:
    """Stands in for a numpy array, which the code only ever calls .tolist() on."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch):
    """Install a stub in place of the real model, per test."""

    def install(model: StubModel) -> StubModel:
        monkeypatch.setattr(embeddings, "get_model", lambda: model)
        return model

    return install


def test_empty_input_produces_empty_output(stub_model) -> None:
    stub_model(StubModel())

    result = embed_passages_resilient([])

    assert result.vectors == []
    assert result.failed == 0


def test_all_chunks_embed_when_nothing_fails(stub_model) -> None:
    stub_model(StubModel())

    result = embed_passages_resilient(["alpha", "beta", "gamma"])

    assert result.failed == 0
    assert all(vector is not None for vector in result.vectors)
    assert len(result.vectors) == 3


def test_a_transient_batch_failure_is_retried(stub_model) -> None:
    """One failure then success: the retry alone should cover it, with nothing dropped."""
    model = stub_model(StubModel(fail_times=1))

    result = embed_passages_resilient(["alpha", "beta"], batch_size=2, max_attempts=2)

    assert result.failed == 0
    assert all(vector is not None for vector in result.vectors)
    assert len(model.calls) == 2


def test_one_poisonous_chunk_does_not_lose_its_batch(stub_model) -> None:
    """The whole point: a batch containing one bad text still yields the good ones."""
    stub_model(StubModel(poison=frozenset({"bad"})))

    result = embed_passages_resilient(["good", "bad", "also good"], batch_size=3)

    assert result.failed == 1
    assert result.vectors[0] is not None
    assert result.vectors[1] is None
    assert result.vectors[2] is not None


def test_failed_indices_align_with_the_input(stub_model) -> None:
    """Positional alignment is what keeps a chunk's stored index truthful."""
    stub_model(StubModel(poison=frozenset({"bad"})))

    texts = ["a", "bad", "c", "d", "bad", "f"]
    result = embed_passages_resilient(texts, batch_size=2)

    assert result.failed_indices == [1, 4]
    assert len(result.vectors) == len(texts)
    for index, vector in enumerate(result.vectors):
        assert (vector is None) == (index in result.failed_indices)


def test_every_chunk_failing_reports_every_index(stub_model) -> None:
    """The caller treats this as a real failure; it must be able to tell."""
    stub_model(StubModel(poison=frozenset({"a", "b"})))

    result = embed_passages_resilient(["a", "b"], batch_size=2)

    assert result.failed == 2
    assert result.vectors == [None, None]


def test_batches_larger_than_the_input_are_handled(stub_model) -> None:
    stub_model(StubModel())

    result = embed_passages_resilient(["only"], batch_size=32)

    assert result.failed == 0
    assert len(result.vectors) == 1


def test_failure_is_isolated_to_its_own_batch(stub_model) -> None:
    """A poisoned second batch must not affect the already-embedded first one."""
    stub_model(StubModel(poison=frozenset({"bad"})))

    result = embed_passages_resilient(["ok1", "ok2", "bad", "ok3"], batch_size=2)

    assert result.failed_indices == [2]
    assert result.vectors[0] is not None
    assert result.vectors[1] is not None
    assert result.vectors[3] is not None
