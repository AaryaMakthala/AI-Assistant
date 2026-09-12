"""Retrieval pipeline: search → fuse → cosine-score → ground (CLAUDE.md section 8).

The pipeline is orchestration, so its tests stub every leaf: the two searches, the
query embedding and the chunk-embedding read (the cosine-distance fill query) are
all replaced, and the real fusion + grounding logic runs in between. The properties
that matter are structural:

* the caller's ``workspace_id`` is threaded into every search (tenant isolation),
* candidate and final counts come from configuration, not literals,
* every fused candidate gets an exact cosine similarity (the fill query), and the
  final ranking + grounding are driven by that similarity — not the RRF order,
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


class _Row:
    """Mimics a SQLAlchemy Row: attribute access to ``.id``, index access for ``[1]``."""

    def __init__(self, chunk_id: uuid.UUID, distance: float) -> None:
        self.id = chunk_id
        self._distance = distance

    def __getitem__(self, index: int) -> object:
        return (self.id, self._distance)[index]


class FakeSession:
    """Serves the cosine-distance fill query from a per-chunk map.

    Anything that is not the fill query (``DocumentChunk.id IN (...)``) returns an
    empty result — the relevance gate treats that as the optimistic fall-through
    (``empty_workspace_fall_through``), exactly like an empty READY document set.
    """

    def __init__(self, distances: dict[uuid.UUID, float] | None = None) -> None:
        #: chunk_id -> cosine distance; sim is 1 - distance in the pipeline.
        self._distances = distances or {}

    async def execute(self, stmt: object) -> SimpleNamespace:
        """Return fill-query rows as ``(chunk_id, distance)`` tuples; else empty."""
        compiled = stmt.compile()  # type: ignore[attr-defined]
        id_list: list[uuid.UUID] | None = None
        for value in compiled.params.values():  # type: ignore[attr-defined]
            if isinstance(value, list) and value and isinstance(value[0], uuid.UUID):
                id_list = value
                break
        if id_list is None:
            return SimpleNamespace(all=lambda: [], scalars=lambda: self)
        rows = [_Row(cid, self._distances[cid]) for cid in id_list if cid in self._distances]
        return SimpleNamespace(all=lambda: rows, scalars=lambda: self)


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
    """Pin counts and thresholds so assertions don't depend on config defaults."""
    settings = SimpleNamespace(
        retrieval_candidate_count=15,
        retrieval_final_count=3,
        retrieval_relevance_threshold=0.3,
        overview_min_score=0.25,
        overview_aggregate_min=0.20,
        doc_target_high_confidence=0.90,
        doc_target_relaxed_score=0.20,
        filename_match_relaxed_score=0.0,
    )
    monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)


async def _empty_semantic(session, **kwargs) -> list[Match]:  # noqa: ANN001, ARG001
    return []


async def _empty_keyword(session, **kwargs) -> list[Match]:  # noqa: ANN001, ARG001
    return []


async def _empty_filename(session, **kwargs) -> list[Match]:  # noqa: ANN001, ARG001
    return []


async def test_retrieve_threads_workspace_into_searches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every search must receive the caller's workspace_id — tenant isolation."""
    ws = uuid.uuid4()
    seen: list[uuid.UUID] = []

    async def _semantic(session, *, query_embedding, workspace_id, limit, document_id=None) -> list[Match]:  # noqa: ANN001
        seen.append(workspace_id)
        return []

    async def _keyword(session, *, query, workspace_id, limit, document_id=None) -> list[Match]:  # noqa: ANN001
        seen.append(workspace_id)
        return []

    monkeypatch.setattr(pipeline_module, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(pipeline_module, "semantic_search", _semantic)
    monkeypatch.setattr(pipeline_module, "keyword_search", _keyword)
    monkeypatch.setattr(pipeline_module, "filename_search", _empty_filename)

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
    monkeypatch.setattr(pipeline_module, "filename_search", _empty_filename)

    result = await retrieve(FakeSession(), query="nothing", workspace_id=uuid.uuid4())
    assert result.chunks == []
    assert result.grounded is False
    assert result.top_score is None


async def test_retrieve_orders_and_caps_by_cosine_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final chunks are the top `retrieval_final_count` by cosine similarity, best first.

    The cosine similarities are made to INVERT the RRF ordering (best similarity for
    the chunk ranked last by fusion), so the test proves the final list is ordered by
    the fill-query similarity, not by the pre-fusion RRF order.
    """
    ws = uuid.uuid4()
    chunks = [_match(uuid.uuid4(), i) for i in range(1, 6)]  # RRF order: rank 1..5
    # Invert: later RRF rank ⇒ higher similarity (lower cosine distance).
    distances = {
        chunks[0].chunk_id: 0.5,
        chunks[1].chunk_id: 0.4,
        chunks[2].chunk_id: 0.3,
        chunks[3].chunk_id: 0.2,
        chunks[4].chunk_id: 0.1,
    }

    monkeypatch.setattr(pipeline_module, "embed_query", lambda q: [0.0])

    async def _semantic(session, **kwargs):  # noqa: ANN001, ARG001
        return chunks

    monkeypatch.setattr(pipeline_module, "semantic_search", _semantic)
    monkeypatch.setattr(pipeline_module, "keyword_search", _empty_keyword)
    monkeypatch.setattr(pipeline_module, "filename_search", _empty_filename)

    result = await retrieve(FakeSession(distances), query="q", workspace_id=ws)
    assert len(result.chunks) == 3  # final_count, not candidate_count
    # Cosine order must be the inversion of the input RRF order (best last first).
    assert [c.chunk_id for c in result.chunks] == [
        chunks[4].chunk_id,
        chunks[3].chunk_id,
        chunks[2].chunk_id,
    ]
    scores = [c.rerank_score for c in result.chunks]
    assert scores == [0.9, 0.8, 0.7]
    assert result.grounded is True  # top score 0.9 >= threshold 0.3
    assert result.top_score == 0.9


async def test_retrieve_grounding_uses_cosine_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grounding is decided on the cosine similarity, not the RRF score."""
    ws = uuid.uuid4()
    chunk = _match(uuid.uuid4(), 1)

    monkeypatch.setattr(pipeline_module, "embed_query", lambda q: [0.0])

    async def _semantic(session, **kwargs):  # noqa: ANN001, ARG001
        return [chunk]

    monkeypatch.setattr(pipeline_module, "semantic_search", _semantic)
    monkeypatch.setattr(pipeline_module, "keyword_search", _empty_keyword)
    monkeypatch.setattr(pipeline_module, "filename_search", _empty_filename)

    result = await retrieve(
        FakeSession({chunk.chunk_id: 0.95}),
        query="q",
        workspace_id=ws,
    )
    assert result.grounded is False
    assert result.top_score == pytest.approx(0.05)
    assert len(result.chunks) == 1  # reported even when refused, for diagnostics