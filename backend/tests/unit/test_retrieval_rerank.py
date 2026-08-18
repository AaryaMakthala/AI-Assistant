"""Reranker: lazy initialization, config-pinned model, empty-input handling.

The cross-encoder must never load at import time and must never download in tests:
everything here stubs the model. ``get_reranker`` imports ``CrossEncoder`` lazily
inside the function, so the tests inject a fake ``sentence_transformers`` module into
``sys.modules`` — the real package (and its torch import) is never touched.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from app.config import get_settings
from app.retrieval.rerank import get_reranker, rerank_scores, reset_reranker

pytestmark = pytest.mark.usefixtures("valid_env")


class StubCrossEncoder:
    """Records how it was constructed; scores deterministically."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:  # noqa: ANN001
        self.calls.append(pairs)
        # Deterministic score: length of the chunk text. Pairs are (query, text).
        return [float(len(pair[1])) for pair in pairs]


@pytest.fixture(autouse=True)
def _fresh_reranker() -> None:
    """Each test starts with no cached model and restores the pristine state."""
    reset_reranker()
    yield
    reset_reranker()


def _install_fake_sentence_transformers(build):  # noqa: ANN001
    """Put a stub CrossEncoder where the lazy import will find it."""
    fake = ModuleType("sentence_transformers")
    fake.CrossEncoder = build  # type: ignore[attr-defined]
    return patch.dict(sys.modules, {"sentence_transformers": fake})


def test_reranker_is_lazy_and_uses_configured_model() -> None:
    """The model is constructed from config, and only when first needed."""
    constructed: list[str] = []

    def _build(name: str) -> StubCrossEncoder:
        constructed.append(name)
        return StubCrossEncoder(name)

    with _install_fake_sentence_transformers(_build):
        model = get_reranker()

    assert isinstance(model, StubCrossEncoder)
    assert model.model_name == get_settings().reranker_model
    assert constructed == [get_settings().reranker_model]

    # Second call returns the cached instance — no second construction.
    assert get_reranker() is model
    assert len(constructed) == 1


def test_rerank_scores_aligns_with_input() -> None:
    model = StubCrossEncoder("x")

    def _build(name: str) -> StubCrossEncoder:
        return model

    with _install_fake_sentence_transformers(_build):
        chunks = ["a", "much longer chunk", "c"]
        scores = rerank_scores("query", chunks)

    assert scores == [1.0, 17.0, 1.0]  # positional alignment
    assert model.calls == [
        [("query", "a"), ("query", "much longer chunk"), ("query", "c")]
    ]


def test_rerank_scores_empty_input() -> None:
    """Empty input must not touch the model at all."""
    touched = False

    def _build(name: str) -> StubCrossEncoder:
        nonlocal touched
        touched = True
        return StubCrossEncoder(name)

    with _install_fake_sentence_transformers(_build):
        assert rerank_scores("query", []) == []

    assert not touched
