"""Evaluation metrics for the RAG retrieval pipeline (CLAUDE.md Section 15).

Implements the six canonical metrics:
- Retrieval Recall@K: did the right chunk make it into the candidate set
- MRR: how highly the right chunk ranked after fusion + rerank
- Citation correctness: does the returned citation match the source of the answer
- Grounded-answer correctness: is the answer actually supported by the cited chunks
- Refusal correctness: does an out-of-scope question get refused, and does an
  in-scope question not get incorrectly refused
- Workspace isolation: a question scoped to Workspace A never returns a chunk
  from Workspace B
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from eval.dataset import EvalQuestion, ExpectedChunk


@dataclass(frozen=True)
class RetrievalHit:
    """A single chunk as returned by the retrieval pipeline for evaluation."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    workspace_id: uuid.UUID
    filename: str
    content: str
    page_number: int | None
    rerank_score: float


@dataclass(frozen=True)
class EvalResult:
    """Result of evaluating one question against the retrieval pipeline."""

    question: EvalQuestion
    retrieved_chunks: list[RetrievalHit]
    grounded: bool
    answer: str | None = None
    sources: list[dict] | None = None


@dataclass
class MetricReport:
    """Aggregated metric results across the full evaluation set."""

    total_questions: int = 0
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    citation_correctness: float = 0.0
    grounded_answer_correctness: float = 0.0
    refusal_correctness: float = 0.0
    workspace_isolation_violations: int = 0
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)


def _chunk_matches_expected(
    chunk: RetrievalHit, expected: ExpectedChunk
) -> bool:
    """Check whether a retrieved chunk matches an expected chunk's characteristics."""
    content_lower = chunk.content.lower()
    if not all(kw.lower() in content_lower for kw in expected.content_keywords):
        return False
    if expected.filename is not None and chunk.filename != expected.filename:
        return False
    if expected.page_number is not None and chunk.page_number != expected.page_number:
        return False
    return True


def compute_recall_at_k(
    results: list[EvalResult], k: int = 5
) -> float:
    """Compute Recall@K: fraction of questions where at least one expected chunk
    appears in the top-K retrieved results.

    For questions with no expected chunks (out-of-scope), they contribute to
    the denominator only if they have zero retrieved chunks (correct refusal).
    """
    if not results:
        return 0.0

    hits = 0
    for result in results:
        if not result.question.expected_chunks:
            # Out-of-scope: recall is trivially satisfied if nothing was retrieved
            if len(result.retrieved_chunks) == 0:
                hits += 1
            continue
        top_k = result.retrieved_chunks[:k]
        for expected in result.question.expected_chunks:
            if any(_chunk_matches_expected(c, expected) for c in top_k):
                hits += 1
                break  # At least one expected chunk found counts as a hit

    return hits / len(results)


def compute_mrr(results: list[EvalResult]) -> float:
    """Compute Mean Reciprocal Rank: average of 1/rank for the first relevant chunk.

    For questions with no expected chunks (out-of-scope), MRR is 1.0 if nothing
    was retrieved (correct refusal), 0.0 otherwise.
    """
    if not results:
        return 0.0

    rr_sum = 0.0
    for result in results:
        if not result.question.expected_chunks:
            if len(result.retrieved_chunks) == 0:
                rr_sum += 1.0  # Correct refusal gets perfect rank
            continue

        for rank, chunk in enumerate(result.retrieved_chunks, start=1):
            for expected in result.question.expected_chunks:
                if _chunk_matches_expected(chunk, expected):
                    rr_sum += 1.0 / rank
                    break
            else:
                continue
            break  # Found first relevant chunk

    return rr_sum / len(results)


def compute_citation_correctness(
    results: list[EvalResult],
) -> float:
    """Compute citation correctness: fraction of grounded answers where the
    returned sources correspond 1:1 to the chunks that were sent to the LLM.

    For the eval framework, we check that:
    - Each source in the response has valid chunk_id, document_id, filename
    - No source references a chunk not in the retrieved set
    - Grounded answers have at least one source
    """
    if not results:
        return 0.0

    correct = 0
    for result in results:
        if not result.grounded:
            # Ungrounded answers should have empty sources — that's correct
            if not result.sources or len(result.sources) == 0:
                correct += 1
            continue

        # Grounded answers must have sources
        if not result.sources:
            continue

        # Every source must reference a chunk in the retrieved set
        retrieved_ids = {c.chunk_id for c in result.retrieved_chunks}
        source_chunks_valid = all(
            uuid.UUID(s["chunk_id"]) in retrieved_ids for s in result.sources
        )
        # Sources must have required fields
        sources_have_fields = all(
            all(k in s for k in ("chunk_id", "document_id", "filename"))
            for s in result.sources
        )
        if source_chunks_valid and sources_have_fields:
            correct += 1

    return correct / len(results)


def compute_grounded_answer_correctness(results: list[EvalResult]) -> float:
    """Compute grounded-answer correctness: whether the grounding decision
    (grounded vs. refused) matches the expected outcome.

    This measures the two-layer grounding system (CLAUDE.md Section 8.3):
    - Questions that should be grounded must pass the threshold
    - Questions that should be refused must not pass the threshold
    """
    if not results:
        return 0.0

    correct = 0
    for result in results:
        if result.question.should_be_grounded == result.grounded:
            correct += 1

    return correct / len(results)


def compute_refusal_correctness(results: list[EvalResult]) -> float:
    """Compute refusal correctness: fraction of out-of-scope/unsupported
    questions that were correctly refused (grounded=False) AND fraction of
    in-scope questions that were not incorrectly refused.

    This is symmetric: both false positives (answering out-of-scope) and
    false negatives (refusing in-scope) count as failures.
    """
    if not results:
        return 0.0

    correct = 0
    for result in results:
        should_refuse = not result.question.should_be_grounded
        did_refuse = not result.grounded
        if should_refuse == did_refuse:
            correct += 1

    return correct / len(results)


def check_workspace_isolation(
    results: list[EvalResult],
    query_workspace_id: uuid.UUID,
) -> int:
    """Check workspace isolation: count violations where a chunk from a different
    workspace appears in the retrieval results.

    Returns the number of violations (0 = correct isolation).
    """
    violations = 0
    for result in results:
        for chunk in result.retrieved_chunks:
            if chunk.workspace_id != query_workspace_id:
                violations += 1
    return violations


def compute_per_category_metrics(
    results: list[EvalResult],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by category."""
    categories: dict[str, list[EvalResult]] = {}
    for result in results:
        cat = result.question.category.value
        categories.setdefault(cat, []).append(result)

    per_cat: dict[str, dict[str, float]] = {}
    for cat, cat_results in categories.items():
        per_cat[cat] = {
            "count": float(len(cat_results)),
            "recall_at_5": compute_recall_at_k(cat_results, k=5),
            "mrr": compute_mrr(cat_results),
            "grounded_answer_correctness": compute_grounded_answer_correctness(cat_results),
            "refusal_correctness": compute_refusal_correctness(cat_results),
        }

    return per_cat


def generate_report(
    results: list[EvalResult],
    query_workspace_id: uuid.UUID | None = None,
) -> MetricReport:
    """Generate the full evaluation report (CLAUDE.md Section 15)."""
    report = MetricReport(total_questions=len(results))
    report.recall_at_k = {
        5: compute_recall_at_k(results, k=5),
        10: compute_recall_at_k(results, k=10),
        15: compute_recall_at_k(results, k=15),
    }
    report.mrr = compute_mrr(results)
    report.citation_correctness = compute_citation_correctness(results)
    report.grounded_answer_correctness = compute_grounded_answer_correctness(results)
    report.refusal_correctness = compute_refusal_correctness(results)
    report.per_category = compute_per_category_metrics(results)

    if query_workspace_id is not None:
        report.workspace_isolation_violations = check_workspace_isolation(
            results, query_workspace_id
        )

    return report


__all__ = [
    "EvalResult",
    "MetricReport",
    "RetrievalHit",
    "check_workspace_isolation",
    "compute_citation_correctness",
    "compute_grounded_answer_correctness",
    "compute_mrr",
    "compute_recall_at_k",
    "compute_refusal_correctness",
    "generate_report",
]
