"""Phase 9 evaluation metrics tests: correctness of metric computation.

These tests verify each metric produces the expected values on carefully
constructed synthetic data, independent of any database or retrieval pipeline.
"""

from __future__ import annotations

import uuid

import pytest

from eval.dataset import EvalCategory, EvalQuestion, ExpectedChunk
from eval.metrics import (
    EvalResult,
    MetricReport,
    RetrievalHit,
    check_workspace_isolation,
    compute_citation_correctness,
    compute_grounded_answer_correctness,
    compute_mrr,
    compute_recall_at_k,
    compute_refusal_correctness,
    generate_report,
)

pytestmark = pytest.mark.usefixtures("valid_env")


def _make_hit(
    *,
    content: str = "chunk content here",
    workspace_id: uuid.UUID | None = None,
    rerank_score: float = 0.9,
    filename: str = "handbook.pdf",
    page_number: int | None = 1,
) -> RetrievalHit:
    """Create a synthetic retrieval hit."""
    return RetrievalHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        workspace_id=workspace_id or uuid.uuid4(),
        filename=filename,
        content=content,
        page_number=page_number,
        rerank_score=rerank_score,
    )


def _make_question(
    *,
    question: str = "test?",
    category: EvalCategory = EvalCategory.SEMANTIC_PARAPHRASE,
    expected_chunks: tuple[ExpectedChunk, ...] = (
        ExpectedChunk(content_keywords=("chunk",)),
    ),
    should_be_grounded: bool = True,
) -> EvalQuestion:
    """Create a synthetic evaluation question."""
    return EvalQuestion(
        id=f"test-{uuid.uuid4().hex[:8]}",
        question=question,
        category=category,
        expected_chunks=expected_chunks,
        should_be_grounded=should_be_grounded,
    )


# --- Recall@K tests --------------------------------------------------------


def test_recall_at_k_perfect_retrieval() -> None:
    """All expected chunks found in top-K → recall = 1.0."""
    question = _make_question()
    hit = _make_hit(content="This chunk has the relevant info")
    result = EvalResult(
        question=question,
        retrieved_chunks=[hit],
        grounded=True,
    )
    assert compute_recall_at_k([result], k=5) == 1.0


def test_recall_at_k_no_retrieval() -> None:
    """Expected chunks exist but nothing retrieved → recall = 0.0."""
    question = _make_question()
    result = EvalResult(
        question=question,
        retrieved_chunks=[],
        grounded=False,
    )
    assert compute_recall_at_k([result], k=5) == 0.0


def test_recall_at_k_partial_retrieval() -> None:
    """Some questions hit, some miss → recall is the fraction that hit."""
    q1 = _make_question(expected_chunks=(ExpectedChunk(content_keywords=("alpha",)),))
    q2 = _make_question(expected_chunks=(ExpectedChunk(content_keywords=("beta",)),))

    results = [
        EvalResult(
            question=q1,
            retrieved_chunks=[_make_hit(content="alpha found")],
            grounded=True,
        ),
        EvalResult(
            question=q2,
            retrieved_chunks=[_make_hit(content="no match here")],
            grounded=True,
        ),
    ]
    assert compute_recall_at_k(results, k=5) == 0.5


def test_recall_at_k_out_of_scope_correct_refusal() -> None:
    """Out-of-scope question with no retrieval → correct, counts as recall hit."""
    question = _make_question(
        category=EvalCategory.OUT_OF_SCOPE,
        expected_chunks=(),
        should_be_grounded=False,
    )
    result = EvalResult(
        question=question,
        retrieved_chunks=[],
        grounded=False,
    )
    assert compute_recall_at_k([result], k=5) == 1.0


def test_recall_at_k_out_of_scope_incorrect_retrieval() -> None:
    """Out-of-scope question but chunks retrieved → counts as recall miss."""
    question = _make_question(
        category=EvalCategory.OUT_OF_SCOPE,
        expected_chunks=(),
        should_be_grounded=False,
    )
    result = EvalResult(
        question=question,
        retrieved_chunks=[_make_hit(content="some irrelevant retrieval")],
        grounded=True,
    )
    assert compute_recall_at_k([result], k=5) == 0.0


def test_recall_at_k_respects_k_value() -> None:
    """Expected chunk is at position 6; recall@5 = 0 but recall@10 = 1."""
    question = _make_question(
        expected_chunks=(ExpectedChunk(content_keywords=("target",)),)
    )
    # Fill positions 1-5 with non-matching, position 6 with matching
    chunks = [_make_hit(content="noise only") for _ in range(5)]
    chunks.append(_make_hit(content="target found here"))
    result = EvalResult(
        question=question,
        retrieved_chunks=chunks,
        grounded=True,
    )
    assert compute_recall_at_k([result], k=5) == 0.0
    assert compute_recall_at_k([result], k=10) == 1.0


def test_recall_at_k_empty_results() -> None:
    """Empty result set → recall = 0.0."""
    assert compute_recall_at_k([], k=5) == 0.0


# --- MRR tests -------------------------------------------------------------


def test_mrr_perfect_rank() -> None:
    """Expected chunk at rank 1 → MRR = 1.0."""
    question = _make_question()
    result = EvalResult(
        question=question,
        retrieved_chunks=[_make_hit(content="the expected chunk content")],
        grounded=True,
    )
    assert compute_mrr([result]) == pytest.approx(1.0)


def test_mrr_rank_2() -> None:
    """Expected chunk at rank 2 → MRR = 0.5."""
    question = _make_question()
    result = EvalResult(
        question=question,
        retrieved_chunks=[
            _make_hit(content="noise"),
            _make_hit(content="the expected chunk content"),
        ],
        grounded=True,
    )
    assert compute_mrr([result]) == pytest.approx(0.5)


def test_mrr_no_relevant_chunk() -> None:
    """No expected chunk in results → MRR = 0.0."""
    question = _make_question()
    result = EvalResult(
        question=question,
        retrieved_chunks=[_make_hit(content="no match at all")],
        grounded=True,
    )
    assert compute_mrr([result]) == pytest.approx(0.0)


def test_mrr_out_of_scope_correct_refusal() -> None:
    """Out-of-scope with no retrieval → MRR = 1.0 (correct refusal)."""
    question = _make_question(
        category=EvalCategory.OUT_OF_SCOPE,
        expected_chunks=(),
        should_be_grounded=False,
    )
    result = EvalResult(
        question=question,
        retrieved_chunks=[],
        grounded=False,
    )
    assert compute_mrr([result]) == pytest.approx(1.0)


def test_mrr_average_across_questions() -> None:
    """MRR averages reciprocal ranks across questions."""
    q1 = _make_question()
    q2 = _make_question()
    results = [
        EvalResult(
            question=q1,
            retrieved_chunks=[_make_hit(content="the expected chunk content")],
            grounded=True,
        ),
        EvalResult(
            question=q2,
            retrieved_chunks=[
                _make_hit(content="noise only"),
                _make_hit(content="the expected chunk content"),
            ],
            grounded=True,
        ),
    ]
    assert compute_mrr(results) == pytest.approx(0.75)  # (1.0 + 0.5) / 2


def test_mrr_empty_results() -> None:
    """Empty result set → MRR = 0.0."""
    assert compute_mrr([]) == 0.0


# --- Citation correctness tests -------------------------------------------


def test_citation_correctness_grounded_with_valid_sources() -> None:
    """Grounded answer with sources referencing retrieved chunks → correct."""
    hit = _make_hit()
    result = EvalResult(
        question=_make_question(),
        retrieved_chunks=[hit],
        grounded=True,
        sources=[{
            "chunk_id": str(hit.chunk_id),
            "document_id": str(hit.document_id),
            "filename": hit.filename,
            "rerank_score": hit.rerank_score,
        }],
    )
    assert compute_citation_correctness([result]) == 1.0


def test_citation_correctness_grounded_without_sources() -> None:
    """Grounded answer with no sources → incorrect."""
    result = EvalResult(
        question=_make_question(),
        retrieved_chunks=[_make_hit()],
        grounded=True,
        sources=None,
    )
    assert compute_citation_correctness([result]) == 0.0


def test_citation_correctness_refused_with_no_sources() -> None:
    """Refused answer with no sources → correct."""
    result = EvalResult(
        question=_make_question(should_be_grounded=False),
        retrieved_chunks=[],
        grounded=False,
        sources=None,
    )
    assert compute_citation_correctness([result]) == 1.0


def test_citation_correctness_refused_with_sources() -> None:
    """Refused answer but sources present → incorrect."""
    result = EvalResult(
        question=_make_question(should_be_grounded=False),
        retrieved_chunks=[],
        grounded=False,
        sources=[{
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "filename": "f.pdf",
        }],
    )
    assert compute_citation_correctness([result]) == 0.0


def test_citation_correctness_source_references_unknown_chunk() -> None:
    """Source references a chunk not in the retrieved set → incorrect."""
    hit = _make_hit()
    result = EvalResult(
        question=_make_question(),
        retrieved_chunks=[hit],
        grounded=True,
        sources=[{
            "chunk_id": str(uuid.uuid4()),  # Different from hit.chunk_id
            "document_id": str(hit.document_id),
            "filename": hit.filename,
        }],
    )
    assert compute_citation_correctness([result]) == 0.0


def test_citation_correctness_source_missing_required_field() -> None:
    """Source missing 'filename' field → incorrect."""
    hit = _make_hit()
    result = EvalResult(
        question=_make_question(),
        retrieved_chunks=[hit],
        grounded=True,
        sources=[{
            "chunk_id": str(hit.chunk_id),
            "document_id": str(hit.document_id),
            # Missing 'filename'
        }],
    )
    assert compute_citation_correctness([result]) == 0.0


# --- Grounded answer correctness tests -------------------------------------


def test_grounded_answer_correctness_all_correct() -> None:
    """All grounding decisions match expectations → 1.0."""
    results = [
        EvalResult(
            question=_make_question(should_be_grounded=True),
            retrieved_chunks=[_make_hit()],
            grounded=True,
        ),
        EvalResult(
            question=_make_question(should_be_grounded=False),
            retrieved_chunks=[],
            grounded=False,
        ),
    ]
    assert compute_grounded_answer_correctness(results) == 1.0


def test_grounded_answer_correctness_false_positive() -> None:
    """Out-of-scope question incorrectly grounded → partial failure."""
    results = [
        EvalResult(
            question=_make_question(should_be_grounded=False),
            retrieved_chunks=[_make_hit()],
            grounded=True,  # Wrong: should be refused
        ),
    ]
    assert compute_grounded_answer_correctness(results) == 0.0


def test_grounded_answer_correctness_false_negative() -> None:
    """In-scope question incorrectly refused → partial failure."""
    results = [
        EvalResult(
            question=_make_question(should_be_grounded=True),
            retrieved_chunks=[],
            grounded=False,  # Wrong: should be grounded
        ),
    ]
    assert compute_grounded_answer_correctness(results) == 0.0


def test_grounded_answer_correctness_mixed() -> None:
    """Mix of correct and incorrect → fraction correct."""
    results = [
        EvalResult(
            question=_make_question(should_be_grounded=True),
            retrieved_chunks=[_make_hit()],
            grounded=True,
        ),
        EvalResult(
            question=_make_question(should_be_grounded=False),
            retrieved_chunks=[_make_hit()],
            grounded=True,  # Wrong
        ),
        EvalResult(
            question=_make_question(should_be_grounded=True),
            retrieved_chunks=[_make_hit()],
            grounded=True,
        ),
    ]
    assert compute_grounded_answer_correctness(results) == pytest.approx(2.0 / 3.0)


def test_grounded_answer_correctness_empty() -> None:
    assert compute_grounded_answer_correctness([]) == 0.0


# --- Refusal correctness tests ---------------------------------------------


def test_refusal_correctness_all_correct() -> None:
    """Both in-scope grounded and out-of-scope refused → 1.0."""
    results = [
        EvalResult(
            question=_make_question(should_be_grounded=True),
            retrieved_chunks=[_make_hit()],
            grounded=True,
        ),
        EvalResult(
            question=_make_question(
                category=EvalCategory.OUT_OF_SCOPE,
                expected_chunks=(),
                should_be_grounded=False,
            ),
            retrieved_chunks=[],
            grounded=False,
        ),
    ]
    assert compute_refusal_correctness(results) == 1.0


def test_refusal_correctness_false_positive() -> None:
    """Out-of-scope question answered instead of refused → failure."""
    results = [
        EvalResult(
            question=_make_question(
                category=EvalCategory.OUT_OF_SCOPE,
                expected_chunks=(),
                should_be_grounded=False,
            ),
            retrieved_chunks=[_make_hit()],
            grounded=True,  # Should be refused
        ),
    ]
    assert compute_refusal_correctness(results) == 0.0


def test_refusal_correctness_false_negative() -> None:
    """In-scope question refused instead of answered → failure."""
    results = [
        EvalResult(
            question=_make_question(should_be_grounded=True),
            retrieved_chunks=[],
            grounded=False,  # Should be grounded
        ),
    ]
    assert compute_refusal_correctness(results) == 0.0


def test_refusal_correctness_empty() -> None:
    assert compute_refusal_correctness([]) == 0.0


# --- Workspace isolation tests ---------------------------------------------


def test_workspace_isolation_no_violations() -> None:
    """All chunks belong to the query workspace → 0 violations."""
    ws = uuid.uuid4()
    result = EvalResult(
        question=_make_question(),
        retrieved_chunks=[_make_hit(workspace_id=ws), _make_hit(workspace_id=ws)],
        grounded=True,
    )
    assert check_workspace_isolation([result], ws) == 0


def test_workspace_isolation_violations() -> None:
    """Chunks from a different workspace appear → violation count."""
    ws_a = uuid.uuid4()
    ws_b = uuid.uuid4()
    result = EvalResult(
        question=_make_question(),
        retrieved_chunks=[
            _make_hit(workspace_id=ws_a),
            _make_hit(workspace_id=ws_b),  # Violation
            _make_hit(workspace_id=ws_b),  # Violation
        ],
        grounded=True,
    )
    assert check_workspace_isolation([result], ws_a) == 2


def test_workspace_isolation_empty_results() -> None:
    """No chunks → no violations."""
    result = EvalResult(
        question=_make_question(),
        retrieved_chunks=[],
        grounded=False,
    )
    assert check_workspace_isolation([result], uuid.uuid4()) == 0


# --- generate_report tests -------------------------------------------------


def test_generate_report_basic() -> None:
    """Report aggregation works on a simple result set."""
    ws = uuid.uuid4()
    results = [
        EvalResult(
            question=_make_question(should_be_grounded=True),
            retrieved_chunks=[_make_hit(workspace_id=ws, content="the expected chunk")],
            grounded=True,
            sources=[{
                "chunk_id": str(uuid.uuid4()),
                "document_id": str(uuid.uuid4()),
                "filename": "handbook.pdf",
            }],
        ),
    ]
    report = generate_report(results, ws)
    assert isinstance(report, MetricReport)
    assert report.total_questions == 1
    assert report.recall_at_k[5] == 1.0
    assert report.mrr == pytest.approx(1.0)
    assert report.workspace_isolation_violations == 0


def test_generate_report_empty() -> None:
    """Empty results → zeroed report."""
    report = generate_report([], uuid.uuid4())
    assert report.total_questions == 0
    assert report.mrr == 0.0
