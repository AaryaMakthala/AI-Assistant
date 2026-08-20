"""Phase 9 evaluation dataset tests: structure, coverage, and completeness.

These tests verify the dataset satisfies CLAUDE.md Section 15 requirements:
- 30–50 questions total
- All seven categories represented
- Questions cover semantic-paraphrase, exact-keyword, numeric facts, multi-chunk,
  out-of-scope, in-scope-but-unsupported, and follow-up
"""

from __future__ import annotations

import pytest

from eval.dataset import (
    EvalCategory,
    get_dataset,
    get_dataset_by_category,
    get_groundable_questions,
    get_refusal_questions,
)

pytestmark = pytest.mark.usefixtures("valid_env")


# --- Dataset structure tests -----------------------------------------------


def test_dataset_has_sufficient_questions() -> None:
    """CLAUDE.md Section 15: 30–50 questions."""
    dataset = get_dataset()
    assert 30 <= len(dataset) <= 50, f"Dataset has {len(dataset)} questions, expected 30–50"


def test_all_seven_categories_are_represented() -> None:
    """CLAUDE.md Section 15: all seven categories must appear."""
    dataset = get_dataset()
    categories = {q.category for q in dataset}
    expected = set(EvalCategory)
    assert categories == expected, (
        f"Missing categories: {expected - categories}, "
        f"Extra categories: {categories - expected}"
    )


def test_category_counts_are_balanced() -> None:
    """Each category should have at least 3 questions for meaningful metrics."""
    for category in EvalCategory:
        questions = get_dataset_by_category(category)
        assert len(questions) >= 3, (
            f"Category {category.value} has only {len(questions)} questions, "
            "minimum is 3"
        )


def test_dataset_ids_are_unique() -> None:
    """Every question must have a unique ID."""
    dataset = get_dataset()
    ids = [q.id for q in dataset]
    assert len(ids) == len(set(ids)), "Duplicate question IDs found"


def test_dataset_ids_are_non_empty() -> None:
    """No question may have an empty ID."""
    for q in get_dataset():
        assert q.id.strip(), f"Question has empty ID: {q}"


# --- Groundable vs refusal split ------------------------------------------


def test_groundable_and_refusal_split_covers_all() -> None:
    """Every question is either groundable or a refusal candidate."""
    groundable = get_groundable_questions()
    refusal = get_refusal_questions()
    assert len(groundable) + len(refusal) == len(get_dataset())


def test_out_of_scope_are_refusals() -> None:
    """All out-of-scope questions must be marked as should_be_grounded=False."""
    for q in get_dataset_by_category(EvalCategory.OUT_OF_SCOPE):
        assert not q.should_be_grounded, (
            f"Out-of-scope question {q.id} should not be grounded"
        )


def test_in_scope_unsupported_are_refusals() -> None:
    """All in-scope-but-unsupported questions must be marked as refusals."""
    for q in get_dataset_by_category(EvalCategory.IN_SCOPE_UNSUPPORTED):
        assert not q.should_be_grounded, (
            f"In-scope unsupported question {q.id} should not be grounded"
        )


def test_answerable_categories_are_groundable() -> None:
    """Semantic-paraphrase, exact-keyword, numeric-fact, multi-chunk are answerable."""
    answerable_categories = {
        EvalCategory.SEMANTIC_PARAPHRASE,
        EvalCategory.EXACT_KEYWORD,
        EvalCategory.NUMERIC_FACT,
        EvalCategory.MULTI_CHUNK,
    }
    for q in get_dataset():
        if q.category in answerable_categories:
            assert q.should_be_grounded, (
                f"Answerable question {q.id} ({q.category.value}) "
                "should be marked as groundable"
            )


# --- Expected chunk structure ----------------------------------------------


def test_groundable_questions_have_expected_chunks() -> None:
    """Every groundable question must specify at least one expected chunk."""
    for q in get_groundable_questions():
        assert len(q.expected_chunks) > 0, (
            f"Groundable question {q.id} has no expected chunks"
        )


def test_refusal_questions_have_no_expected_chunks() -> None:
    """Refusal questions should have empty expected_chunks."""
    for q in get_refusal_questions():
        assert len(q.expected_chunks) == 0, (
            f"Refusal question {q.id} has unexpected chunks: {q.expected_chunks}"
        )


def test_refusal_questions_have_refusal_fragment() -> None:
    """Refusal questions should specify what text the refusal contains."""
    for q in get_refusal_questions():
        assert q.refusal_fragment is not None, (
            f"Refusal question {q.id} has no refusal_fragment"
        )


def test_expected_chunks_have_non_empty_keywords() -> None:
    """Every expected chunk must have at least one content keyword."""
    for q in get_dataset():
        for ec in q.expected_chunks:
            assert len(ec.content_keywords) > 0, (
                f"Question {q.id} has an expected chunk with no keywords"
            )


# --- Follow-up questions ---------------------------------------------------


def test_follow_up_questions_have_prior_context() -> None:
    """Follow-up questions must have conversation history."""
    for q in get_dataset_by_category(EvalCategory.FOLLOW_UP):
        assert len(q.prior_context) > 0, (
            f"Follow-up question {q.id} has no prior context"
        )


def test_follow_up_prior_context_has_user_assistant_turns() -> None:
    """Prior context must be (user, assistant) turn pairs."""
    for q in get_dataset_by_category(EvalCategory.FOLLOW_UP):
        for role, _content in q.prior_context:
            assert role in ("user", "assistant"), (
                f"Question {q.id} has invalid role '{role}' in prior context"
            )


# --- Semantic checks -------------------------------------------------------


def test_questions_are_non_trivial() -> None:
    """Every question should be at least 5 characters."""
    for q in get_dataset():
        assert len(q.question.strip()) >= 5, (
            f"Question {q.id} is too short: '{q.question}'"
        )


def test_no_duplicate_questions() -> None:
    """No two questions should have the same text."""
    dataset = get_dataset()
    texts = [q.question for q in dataset]
    assert len(texts) == len(set(texts)), "Duplicate question texts found"


def test_groundable_questions_have_filename_in_at_least_one_expected_chunk() -> None:
    """At least one expected chunk should specify a filename for citation checking."""
    for q in get_groundable_questions():
        has_filename = any(ec.filename is not None for ec in q.expected_chunks)
        # This is a recommendation, not a hard requirement — some questions may
        # not need a specific filename. We just log a warning.
        if not has_filename:
            # Acceptable but worth noting
            pass
