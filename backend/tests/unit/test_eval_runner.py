"""Phase 9 evaluation runner tests: end-to-end stub-based verification.

The runner is tested with a stub retrieval function that returns deterministic
results, verifying the full pipeline from dataset → retrieval → metrics → report.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from eval.dataset import EvalCategory, get_dataset, get_dataset_by_category
from eval.metrics import RetrievalHit
from eval.runner import format_report, run_evaluation

pytestmark = pytest.mark.usefixtures("valid_env")


def _stub_retrieve_factory(
    behavior: str = "perfect",
) -> Callable[[Any, str, uuid.UUID], Coroutine[Any, Any, list[RetrievalHit]]]:
    """Create a stub retrieval function with configurable behavior.

    - 'perfect': returns a matching hit for every question
    - 'refuse': returns empty for all questions
    - 'biased': returns matching hits for semantic_paraphrase, empty for others
    """

    async def _stub(
        session: Any, query: str, workspace_id: uuid.UUID
    ) -> list[RetrievalHit]:
        if behavior == "perfect":
            return [
                RetrievalHit(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    workspace_id=workspace_id,
                    filename="handbook.pdf",
                    content=f"chunk content about {query}",
                    page_number=1,
                    rerank_score=0.9,
                )
            ]
        elif behavior == "refuse":
            return []
        elif behavior == "biased":
            # Only return results for semantic_paraphrase questions
            if any(
                word in query.lower()
                for word in ["time off", "security", "work from home", "rules"]
            ):
                return [
                    RetrievalHit(
                        chunk_id=uuid.uuid4(),
                        document_id=uuid.uuid4(),
                        workspace_id=workspace_id,
                        filename="handbook.pdf",
                        content=f"chunk content about {query}",
                        page_number=1,
                        rerank_score=0.9,
                    )
                ]
            return []
        return []

    return _stub


async def test_run_evaluation_perfect_retrieval() -> None:
    """With a perfect retrieval stub, every question gets results."""
    ws = uuid.uuid4()
    stub = _stub_retrieve_factory("perfect")
    dataset = get_dataset()

    report = await run_evaluation(stub, None, ws, dataset)

    assert report.total_questions == len(dataset)
    # The stub always returns one hit, so questions with content containing
    # the expected keywords get recall; others don't — this is expected.
    assert 0.0 <= report.recall_at_k[5] <= 1.0
    assert report.mrr > 0.0
    # The stub always returns hits, so grounding is determined by rerank_score
    assert 0.0 <= report.grounded_answer_correctness <= 1.0


async def test_run_evaluation_refuse_all() -> None:
    """With a refusing stub, no chunks are retrieved for any question."""
    ws = uuid.uuid4()
    stub = _stub_retrieve_factory("refuse")
    dataset = get_dataset()

    report = await run_evaluation(stub, None, ws, dataset)

    assert report.total_questions == len(dataset)
    # No chunks retrieved: out-of-scope questions count as correct refusals
    # (recall@5 counts them as hits), but answerable questions are misses.
    refusal_count = sum(1 for q in dataset if not q.should_be_grounded)
    expected_recall = refusal_count / len(dataset)
    assert report.recall_at_k[5] == pytest.approx(expected_recall)
    assert report.mrr == pytest.approx(expected_recall)  # same logic for MRR
    # Refusal correctness: out-of-scope correctly refused, in-scope incorrectly refused
    assert report.refusal_correctness == pytest.approx(refusal_count / len(dataset))


async def test_run_evaluation_workspace_isolation() -> None:
    """All hits should belong to the query workspace."""
    ws = uuid.uuid4()
    stub = _stub_retrieve_factory("perfect")
    dataset = get_dataset()

    report = await run_evaluation(stub, None, ws, dataset)

    # The stub returns hits with the correct workspace_id
    assert report.workspace_isolation_violations == 0


async def test_run_evaluation_returns_per_category() -> None:
    """Report should have per-category breakdown."""
    ws = uuid.uuid4()
    stub = _stub_retrieve_factory("perfect")

    report = await run_evaluation(stub, None, ws)

    assert len(report.per_category) > 0
    for cat in EvalCategory:
        assert cat.value in report.per_category


async def test_run_evaluation_single_category() -> None:
    """Runner works with a subset of the dataset."""
    ws = uuid.uuid4()
    stub = _stub_retrieve_factory("perfect")
    dataset = get_dataset_by_category(EvalCategory.SEMANTIC_PARAPHRASE)

    report = await run_evaluation(stub, None, ws, dataset)

    assert report.total_questions == len(dataset)
    assert report.per_category["semantic_paraphrase"]["count"] == len(dataset)


def test_format_report_produces_string() -> None:
    """format_report produces a non-empty string."""
    from eval.metrics import MetricReport

    report = MetricReport(
        total_questions=40,
        recall_at_k={5: 0.85, 10: 0.92, 15: 0.95},
        mrr=0.78,
        citation_correctness=0.95,
        grounded_answer_correctness=0.90,
        refusal_correctness=0.88,
        workspace_isolation_violations=0,
    )
    formatted = format_report(report)
    assert isinstance(formatted, str)
    assert len(formatted) > 100
    assert "0.8500" in formatted  # Recall@5
    assert "0.7800" in formatted  # MRR
    assert "Recall@5" in formatted
