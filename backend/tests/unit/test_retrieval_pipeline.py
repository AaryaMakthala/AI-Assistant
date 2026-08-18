"""Retrieval pipeline: search → fuse → rerank → ground (CLAUDE.md section 8).

The pipeline is orchestration, so its tests stub every leaf: the two searches, the
query embedding and the reranker are all replaced, and the real fusion + grounding
logic runs in between. The properties that matter are structural:

* the caller's ``workspace_id`` is threaded into every search (tenant isolation),
* candidate and final counts come from configuration, not literals,
* grounding is derived from the top rerank score,
* an empty/ungrounded query returns a refused result, never an exception.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

import app.retrieval.pipeline as pipeline_module
from app.retrieval.hybrid import Match
from app.retrieval.pipeline import retrieve

pytestmark = pytest.mark.usefixtures("valid_env")


class FakeSession:
    """Accepts anything; the searches are stubbed, so no SQL is executed."""

    async def execute(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("the pipeline must not touch the session when searches are stubbed")


def _match(cid: uuid.UUID, rank: int, content: str = "chunk content") -> Match:
    return Match(
        chunk_id=cid,
        document_id=uuid.uuid4(),
        filename="handbook.pdf",
        content=content,
        page_number=2,
        section_title="Leave policy",
        chunk_index=rank,
        rank=rank,
    )


@pytest.fixture(autouse=True)
def _stub_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin counts and threshold so assertions don't depend on config defaults."""
    settings = SimpleNamespace(
        retrieval_candidate_count=15,
        retrieval_final_count=3,
        retrieval_relevance_threshold=0.3,
    )
    monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)


async def _empty_semantic(session, **kwargs) -> list[Match]:  # noqa: ANN001, ARG001
    return []


async def _empty_keyword(session, **kwargs) -> list[Match]:  # noqa: ANN001, ARG001
    return []


async def test_retrieve_threads_workspace_into_searches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every search must receive the caller's workspace_id — tenant isolation."""
    ws = uuid.uuid4()
    seen: list[uuid.UUID] = []

    async def _semantic(session, *, query_embedding, workspace_id, limit) -> list[Match]:  # noqa: ANN001
        seen.append(workspace_id)
        return []

    async def _keyword(session, *, query, workspace_id, limit) -> list[Match]:  # noqa: ANN001
        seen.append(workspace_id)
        return []

    monkeypatch.setattr(pipeline_module, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(pipeline_module, "semantic_search", _semantic)
    monkeypatch.setattr(pipeline_module, "keyword_search", _keyword)

    await retrieve(FakeSession(), query="vacation", workspace_id=ws)

    assert seen == [ws, ws]


async def test_retrieve_empty_query_is_refused_without_search() -> None:
    result = await retrieve(FakeSession(), query="   ", workspace_id=uuid.uuid4())
    assert result.chunks == []
    assert result.grounded is False
    assert result.top_score is None


async def test_retrieve_no_candidates_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(pipeline_module, "semantic_search", _empty_semantic)
    monkeypatch.setattr(pipeline_module, "keyword_search", _empty_keyword)

    result = await retrieve(FakeSession(), query="nothing", workspace_id=uuid.uuid4())
    assert result.chunks == []
    assert result.grounded is False
    assert result.top_score is None


async def test_retrieve_reranks_and_caps_at_final_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final chunks are the top `retrieval_final_count` by rerank score, best first.

    The rerank scores are made to INVERT the RRF ordering (highest score for the
    chunk ranked last by fusion), so the test proves the final list is ordered by
    the cross-encoder, not by the pre-rerank RRF order.
    """
    ws = uuid.uuid4()
    chunks = [_match(uuid.uuid4(), i) for i in range(1, 6)]  # RRF order: rank 1..5

    monkeypatch.setattr(pipeline_module, "embed_query", lambda q: [0.0])

    async def _semantic(session, **kwargs):  # noqa: ANN001, ARG001
        return chunks

    monkeypatch.setattr(pipeline_module, "semantic_search", _semantic)
    monkeypatch.setattr(pipeline_module, "keyword_search", _empty_keyword)
    # Invert: later RRF rank ⇒ higher rerank score.
    monkeypatch.setattr(
        pipeline_module,
        "rerank_scores",
        lambda q, texts: [float(index) for index in range(len(texts))],
    )

    result = await retrieve(FakeSession(), query="q", workspace_id=ws)
    assert len(result.chunks) == 3  # final_count, not candidate_count
    # Reranked order must be the inversion of the input RRF order (best last first).
    assert [c.chunk_id for c in result.chunks] == [
        chunks[4].chunk_id,
        chunks[3].chunk_id,
        chunks[2].chunk_id,
    ]
    scores = [c.rerank_score for c in result.chunks]
    assert scores == [4.0, 3.0, 2.0]
    assert result.grounded is True  # top score 4.0 >= threshold 0.3
    assert result.top_score == 4.0


async def test_retrieve_grounding_uses_rerank_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grounding is decided on the rerank score, not the RRF score."""
    ws = uuid.uuid4()
    chunk = _match(uuid.uuid4(), 1)

    monkeypatch.setattr(pipeline_module, "embed_query", lambda q: [0.0])

    async def _semantic(session, **kwargs):  # noqa: ANN001, ARG001
        return [chunk]

    monkeypatch.setattr(pipeline_module, "semantic_search", _semantic)
    monkeypatch.setattr(pipeline_module, "keyword_search", _empty_keyword)
    monkeypatch.setattr(pipeline_module, "rerank_scores", lambda q, texts: [0.05])

    result = await retrieve(FakeSession(), query="q", workspace_id=ws)
    assert result.grounded is False
    assert result.top_score == 0.05
    assert len(result.chunks) == 1  # reported even when refused, for diagnostics
