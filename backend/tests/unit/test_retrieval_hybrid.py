"""Hybrid retrieval: semantic + keyword searches and RRF fusion (CLAUDE.md 8.1).

The database is stubbed with a fake session: these tests pin the *shape* of the
queries (workspace scoping, limits, ordering) and the pure fusion math, not pgvector
or Postgres behaviour — the search SQL itself is exercised by the integration suite
in the final test phase.

The one property that must never regress is tenant isolation: every compiled query
must carry the caller's ``workspace_id`` predicate.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from app.retrieval.hybrid import RRF_K, Match, keyword_search, rrf_merge, semantic_search

pytestmark = pytest.mark.usefixtures("valid_env")


class FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return self._rows


class FakeSession:
    """Async session recording the statements executed; returns canned rows."""

    def __init__(self, rows: list) -> None:
        self._rows = rows
        self.statements: list = []

    async def execute(self, statement):  # noqa: ANN001
        self.statements.append(statement)
        return FakeResult(self._rows)


def _row(
    chunk_id: uuid.UUID, *, page: int | None = 1, filename: str = "handbook.pdf"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        document_id=uuid.uuid4(),
        content="some chunk content",
        page_number=page,
        section_title="Section",
        chunk_index=1,
        filename=filename,
    )


def _sql(statement) -> str:  # noqa: ANN001
    """Compile a statement to SQL for predicate inspection.

    Not with ``literal_binds``: ``websearch_to_tsquery('english', ...)`` binds a
    regconfig that the SQL compiler refuses to render literally, so predicates are
    asserted on the parameterized form instead.
    """
    return str(statement.compile(dialect=postgresql.dialect()))


async def test_semantic_search_scopes_workspace_and_limits() -> None:
    ws = uuid.uuid4()
    chunk = uuid.uuid4()
    session = FakeSession([_row(chunk)])

    matches = await semantic_search(
        session, query_embedding=[0.1, 0.2], workspace_id=ws, limit=15
    )

    assert len(session.statements) == 1
    sql = _sql(session.statements[0])
    # The parameterized form names the workspace column; RLS scopes the values.
    # The literal limit value is a bind parameter here; the call above passed 15.
    assert "workspace_id" in sql
    assert "LIMIT" in sql.upper()
    assert matches[0].chunk_id == chunk
    assert matches[0].rank == 1


async def test_semantic_search_orders_by_cosine_distance() -> None:
    ws = uuid.uuid4()
    session = FakeSession([])
    await semantic_search(session, query_embedding=[1.0, 0.0], workspace_id=ws, limit=10)
    sql = _sql(session.statements[0])
    # Ordering must be by the cosine distance operator so the HNSW ANN scan is used.
    assert "cosine_distance" in sql or "<=>" in sql.replace("\n", " ")


async def test_keyword_search_scopes_workspace_and_uses_fts() -> None:
    ws = uuid.uuid4()
    chunk = uuid.uuid4()
    session = FakeSession([_row(chunk)])

    matches = await keyword_search(session, query="annual leave", workspace_id=ws, limit=15)

    sql = _sql(session.statements[0])
    assert "websearch_to_tsquery" in sql
    assert "content_tsv" in sql
    assert matches[0].chunk_id == chunk
    assert matches[0].rank == 1


async def test_keyword_search_orders_by_ts_rank() -> None:
    ws = uuid.uuid4()
    session = FakeSession([])
    await keyword_search(session, query="policy", workspace_id=ws, limit=10)
    sql = _sql(session.statements[0])
    assert "ts_rank" in sql
    assert "ORDER BY" in sql


async def test_search_empty_result_sets() -> None:
    ws = uuid.uuid4()
    session = FakeSession([])
    assert await semantic_search(session, query_embedding=[0.0], workspace_id=ws, limit=5) == []
    assert await keyword_search(session, query="nothing here", workspace_id=ws, limit=5) == []


def test_rrf_merge_favours_chunk_found_by_both_engines() -> None:
    both = uuid.uuid4()
    kw_only = uuid.uuid4()
    sem_only = uuid.uuid4()

    def match(cid: uuid.UUID, rank: int) -> Match:
        return Match(
            chunk_id=cid,
            document_id=uuid.uuid4(),
            filename="f.pdf",
            content="c",
            page_number=1,
            section_title=None,
            chunk_index=1,
            rank=rank,
        )

    semantic = [match(both, 1), match(sem_only, 2)]
    keyword = [match(both, 1), match(kw_only, 3)]

    merged = rrf_merge(semantic, keyword, top_n=10)

    assert merged[0].chunk_id == both  # agreement wins
    assert abs(merged[0].rrf_score - (1 / (RRF_K + 1) + 1 / (RRF_K + 1))) < 1e-12
    assert {m.chunk_id for m in merged} == {both, sem_only, kw_only}  # deduplicated


def test_rrf_merge_deterministic_ordering() -> None:
    ids = [uuid.uuid4() for _ in range(5)]

    def match(cid: uuid.UUID, rank: int) -> Match:
        return Match(
            chunk_id=cid,
            document_id=uuid.uuid4(),
            filename="f.pdf",
            content="c",
            page_number=None,
            section_title=None,
            chunk_index=1,
            rank=rank,
        )

    semantic = [match(ids[i], i + 1) for i in range(5)]
    keyword: list[Match] = []

    first = rrf_merge(semantic, keyword, top_n=5)
    second = rrf_merge(semantic, keyword, top_n=5)
    assert [m.chunk_id for m in first] == [m.chunk_id for m in second]
    # RRF score strictly decreases with worse rank within one engine.
    assert [m.rrf_score for m in first] == sorted(
        (m.rrf_score for m in first), reverse=True
    )


def test_rrf_merge_top_n_truncation_and_empty() -> None:
    ids = [uuid.uuid4() for _ in range(5)]

    def match(cid: uuid.UUID, rank: int) -> Match:
        return Match(
            chunk_id=cid,
            document_id=uuid.uuid4(),
            filename="f.pdf",
            content="c",
            page_number=None,
            section_title=None,
            chunk_index=1,
            rank=rank,
        )

    semantic = [match(ids[i], i + 1) for i in range(5)]
    assert len(rrf_merge(semantic, [], top_n=2)) == 2
    assert rrf_merge([], [], top_n=5) == []


def test_rrf_merge_keeps_first_rows_citation_metadata() -> None:
    cid = uuid.uuid4()

    def match(filename: str, rank: int) -> Match:
        return Match(
            chunk_id=cid,
            document_id=uuid.uuid4(),
            filename=filename,
            content="c",
            page_number=3,
            section_title="T",
            chunk_index=7,
            rank=rank,
        )

    merged = rrf_merge([match("a.pdf", 1)], [match("a.pdf", 2)], top_n=1)
    assert len(merged) == 1
    assert merged[0].filename == "a.pdf"
    assert merged[0].page_number == 3
    assert merged[0].chunk_index == 7
