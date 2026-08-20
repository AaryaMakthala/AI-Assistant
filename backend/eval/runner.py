"""Evaluation runner for the RAG retrieval pipeline (CLAUDE.md Section 15).

Runs the evaluation dataset against a retrieval function (injected for testability)
and produces a MetricReport with actual measured values.

The runner is dependency-injected: it accepts a retrieval function that matches the
signature of ``app.retrieval.pipeline.retrieve``. This makes it testable with stubs
and also usable against the real pipeline in integration tests.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from loguru import logger

from eval.dataset import EvalQuestion, get_dataset
from eval.metrics import (
    EvalResult,
    MetricReport,
    RetrievalHit,
    generate_report,
)


async def _default_retrieve(
    session: Any,
    *,
    query: str,
    workspace_id: uuid.UUID,
) -> Any:
    """Placeholder retrieve function. Must be replaced in real usage."""
    raise NotImplementedError("Provide a real retrieve function")


async def run_evaluation(
    retrieve_fn: Callable[
        [Any, str, uuid.UUID],
        Coroutine[Any, Any, list[RetrievalHit]],
    ],
    session: Any,
    workspace_id: uuid.UUID,
    dataset: list[EvalQuestion] | None = None,
) -> MetricReport:
    """Run the full evaluation against the provided retrieval function.

    Parameters
    ----------
    retrieve_fn :
        An async callable ``(session, query, workspace_id) -> list[RetrievalHit]``.
        This is a thin wrapper around the real ``retrieve`` function that converts
        ``RetrievalResult.chunks`` into ``list[RetrievalHit]``. The wrapper is
        provided by the caller so the runner itself never imports the pipeline.
    session :
        The database session (or stub) to pass through.
    workspace_id :
        The workspace to evaluate against.
    dataset :
        Optional override; defaults to the full dataset.

    Returns
    -------
    MetricReport
        Aggregated metrics across all questions.
    """
    questions = dataset or get_dataset()
    results: list[EvalResult] = []

    for question in questions:
        logger.info(
            "Evaluating [{category}] {id}: {question}",
            category=question.category.value,
            id=question.id,
            question=question.question[:60],
        )
        try:
            chunks = await retrieve_fn(session, question.question, workspace_id)
        except Exception as exc:
            logger.error(
                "Retrieval failed for question {id}: {error}",
                id=question.id,
                error=exc,
            )
            chunks = []

        # Determine grounding from the top score if available
        grounded = len(chunks) > 0 and any(
            c.rerank_score >= 0.3 for c in chunks
        )

        # Build sources in the same format the chat endpoint returns
        sources = []
        if grounded:
            for chunk in chunks:
                sources.append({
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "rerank_score": chunk.rerank_score,
                })

        result = EvalResult(
            question=question,
            retrieved_chunks=chunks,
            grounded=grounded,
            sources=sources if sources else None,
        )
        results.append(result)

    report = generate_report(results, workspace_id)
    return report


def format_report(report: MetricReport) -> str:
    """Format a MetricReport as a human-readable string."""
    lines = [
        "=" * 72,
        "PHASE 9 — EVALUATION REPORT",
        "=" * 72,
        f"Total questions: {report.total_questions}",
        "",
        "Retrieval Quality:",
        f"  Recall@5:  {report.recall_at_k.get(5, 0.0):.4f}",
        f"  Recall@10: {report.recall_at_k.get(10, 0.0):.4f}",
        f"  Recall@15: {report.recall_at_k.get(15, 0.0):.4f}",
        f"  MRR:       {report.mrr:.4f}",
        "",
        "Citation & Grounding:",
        f"  Citation correctness:           {report.citation_correctness:.4f}",
        f"  Grounded-answer correctness:     {report.grounded_answer_correctness:.4f}",
        f"  Refusal correctness:             {report.refusal_correctness:.4f}",
        "",
        "Workspace Isolation:",
        f"  Violations: {report.workspace_isolation_violations}",
        "",
        "Per-Category Breakdown:",
    ]

    for cat, metrics in report.per_category.items():
        lines.append(f"  [{cat}] (n={int(metrics['count'])})")
        lines.append(f"    Recall@5:  {metrics['recall_at_5']:.4f}")
        lines.append(f"    MRR:       {metrics['mrr']:.4f}")
        lines.append(f"    Grounded:  {metrics['grounded_answer_correctness']:.4f}")
        lines.append(f"    Refusal:   {metrics['refusal_correctness']:.4f}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def save_report(report: MetricReport, path: Path) -> None:
    """Save the report as JSON for reproducibility."""
    data = {
        "total_questions": report.total_questions,
        "recall_at_k": {str(k): v for k, v in report.recall_at_k.items()},
        "mrr": report.mrr,
        "citation_correctness": report.citation_correctness,
        "grounded_answer_correctness": report.grounded_answer_correctness,
        "refusal_correctness": report.refusal_correctness,
        "workspace_isolation_violations": report.workspace_isolation_violations,
        "per_category": report.per_category,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = [
    "format_report",
    "run_evaluation",
    "save_report",
]
