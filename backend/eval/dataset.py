"""Evaluation dataset: 30–50 questions across seven categories (CLAUDE.md Section 15).

Each entry describes a question, its category, the expected retrieval outcome,
and the chunk characteristics the answer should be grounded in. The dataset is
designed so that metric computation is deterministic and reproducible.

Categories (CLAUDE.md Section 15):
1. Semantic-paraphrase — answerable by meaning, not exact words
2. Exact-keyword/identifier — answerable by precise terms (HR-004, POL-17)
3. Numeric-facts — specific numbers that must be retrieved exactly
4. Multi-chunk — answer requires combining information from multiple chunks
5. Out-of-scope — genuinely unrelated; must be refused
6. In-scope-but-unsupported — related topic but not in any document; must be refused
7. Follow-up — needs conversation context to resolve a pronoun or reference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvalCategory(str, Enum):  # noqa: UP042
    SEMANTIC_PARAPHRASE = "semantic_paraphrase"
    EXACT_KEYWORD = "exact_keyword"
    NUMERIC_FACT = "numeric_fact"
    MULTI_CHUNK = "multi_chunk"
    OUT_OF_SCOPE = "out_of_scope"
    IN_SCOPE_UNSUPPORTED = "in_scope_unsupported"
    FOLLOW_UP = "follow_up"


@dataclass(frozen=True)
class ExpectedChunk:
    """Characteristics of a chunk that should appear in retrieval results.

    We match on content keywords rather than exact chunk IDs because chunk IDs
    are assigned at ingestion time and are not predictable. The eval runner
    checks that at least one retrieved chunk matches these characteristics.
    """

    #: Keywords that should appear in the chunk content (all must match).
    content_keywords: tuple[str, ...]
    #: Expected document filename, or None if any document is acceptable.
    filename: str | None = None
    #: Expected page number, or None if any page is acceptable.
    page_number: int | None = None


@dataclass(frozen=True)
class EvalQuestion:
    """One entry in the evaluation dataset."""

    id: str
    question: str
    category: EvalCategory
    #: Chunks that should appear in the top-K retrieval results.
    expected_chunks: tuple[ExpectedChunk, ...]
    #: Whether this question should pass the grounding threshold.
    should_be_grounded: bool
    #: Expected refusal text fragment (for out-of-scope / unsupported questions).
    refusal_fragment: str | None = None
    #: Optional prior conversation context (for follow-up questions).
    prior_context: tuple[tuple[str, str], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Dataset — a hypothetical company's approved knowledge base
# ---------------------------------------------------------------------------

DATASET: list[EvalQuestion] = [
    # === 1. Semantic-paraphrase (10 questions) =============================
    EvalQuestion(
        id="sp-01",
        question="How much time off do employees get each year?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("vacation", "days", "year")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-02",
        question="What is the procedure for reporting a security incident?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("security", "incident", "report")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-03",
        question="Can employees work from home?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("remote", "work", "home")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-04",
        question="What happens if someone breaks the data protection rules?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("data", "protection", "violation")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-05",
        question="How do new hires get trained?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("onboarding", "training", "new")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-06",
        question="What are the rules about using company equipment?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("equipment", "policy", "use")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-07",
        question="How is employee performance evaluated?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("performance", "review", "evaluation")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-08",
        question="What benefits does the company provide?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("benefits", "health", "insurance")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-09",
        question="How do you request time away from work?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("leave", "request", "absence")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="sp-10",
        question="What is the policy on using personal devices for work?",
        category=EvalCategory.SEMANTIC_PARAPHRASE,
        expected_chunks=(
            ExpectedChunk(content_keywords=("BYOD", "personal", "device")),
        ),
        should_be_grounded=True,
    ),
    # === 2. Exact-keyword / identifier (7 questions) =======================
    EvalQuestion(
        id="ek-01",
        question="What does policy HR-004 cover?",
        category=EvalCategory.EXACT_KEYWORD,
        expected_chunks=(
            ExpectedChunk(content_keywords=("HR-004",)),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="ek-02",
        question="Summarize document POL-17.",
        category=EvalCategory.EXACT_KEYWORD,
        expected_chunks=(
            ExpectedChunk(content_keywords=("POL-17",)),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="ek-03",
        question="What is section 4.2 of the employee handbook?",
        category=EvalCategory.EXACT_KEYWORD,
        expected_chunks=(
            ExpectedChunk(content_keywords=("4.2",)),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="ek-04",
        question="What does the Acceptable Use Policy (AUP-003) say about email?",
        category=EvalCategory.EXACT_KEYWORD,
        expected_chunks=(
            ExpectedChunk(content_keywords=("AUP-003", "email")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="ek-05",
        question="Reference document IT-SEC-012 for the firewall rules.",
        category=EvalCategory.EXACT_KEYWORD,
        expected_chunks=(
            ExpectedChunk(content_keywords=("IT-SEC-012",)),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="ek-06",
        question="What is the revision date of policy COMP-005?",
        category=EvalCategory.EXACT_KEYWORD,
        expected_chunks=(
            ExpectedChunk(content_keywords=("COMP-005",)),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="ek-07",
        question="What are the requirements listed in FORM-RQ-001?",
        category=EvalCategory.EXACT_KEYWORD,
        expected_chunks=(
            ExpectedChunk(content_keywords=("FORM-RQ-001",)),
        ),
        should_be_grounded=True,
    ),
    # === 3. Numeric facts (5 questions) ====================================
    EvalQuestion(
        id="nf-01",
        question="How many vacation days per year are employees entitled to?",
        category=EvalCategory.NUMERIC_FACT,
        expected_chunks=(
            ExpectedChunk(content_keywords=("20", "days", "vacation")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="nf-02",
        question="What is the maximum file upload size in megabytes?",
        category=EvalCategory.NUMERIC_FACT,
        expected_chunks=(
            ExpectedChunk(content_keywords=("10", "MB")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="nf-03",
        question="How long is the probation period for new employees in months?",
        category=EvalCategory.NUMERIC_FACT,
        expected_chunks=(
            ExpectedChunk(content_keywords=("probation", "months")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="nf-04",
        question="What is the notice period length in weeks for resignation?",
        category=EvalCategory.NUMERIC_FACT,
        expected_chunks=(
            ExpectedChunk(content_keywords=("notice", "weeks")),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="nf-05",
        question="How many sick days are allowed per calendar year?",
        category=EvalCategory.NUMERIC_FACT,
        expected_chunks=(
            ExpectedChunk(content_keywords=("sick", "days")),
        ),
        should_be_grounded=True,
    ),
    # === 4. Multi-chunk (3 questions) ======================================
    EvalQuestion(
        id="mc-01",
        question=("What are the complete steps for onboarding a new employee, "
                  "from offer letter to first day?"),
        category=EvalCategory.MULTI_CHUNK,
        expected_chunks=(
            ExpectedChunk(content_keywords=("onboarding",)),
            ExpectedChunk(content_keywords=("offer",)),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="mc-02",
        question=("What are all the security requirements for remote access, "
                  "including VPN, MFA, and device policy?"),
        category=EvalCategory.MULTI_CHUNK,
        expected_chunks=(
            ExpectedChunk(content_keywords=("VPN",)),
            ExpectedChunk(content_keywords=("MFA",)),
        ),
        should_be_grounded=True,
    ),
    EvalQuestion(
        id="mc-03",
        question=("Describe the full expense reimbursement process including "
                  "approval thresholds and submission deadlines."),
        category=EvalCategory.MULTI_CHUNK,
        expected_chunks=(
            ExpectedChunk(content_keywords=("expense",)),
            ExpectedChunk(content_keywords=("reimbursement",)),
        ),
        should_be_grounded=True,
    ),
    # === 5. Out-of-scope (5 questions) =====================================
    EvalQuestion(
        id="os-01",
        question="Who won the last FIFA World Cup?",
        category=EvalCategory.OUT_OF_SCOPE,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    EvalQuestion(
        id="os-02",
        question="Write me a Python game for Tic Tac Toe.",
        category=EvalCategory.OUT_OF_SCOPE,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    EvalQuestion(
        id="os-03",
        question="What is the current price of Bitcoin?",
        category=EvalCategory.OUT_OF_SCOPE,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    EvalQuestion(
        id="os-04",
        question="Explain quantum computing in detail.",
        category=EvalCategory.OUT_OF_SCOPE,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    EvalQuestion(
        id="os-05",
        question="What are some good restaurants near Times Square?",
        category=EvalCategory.OUT_OF_SCOPE,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    # === 6. In-scope but unsupported (5 questions) =========================
    EvalQuestion(
        id="is-01",
        question="What is the company's revenue for Q3 2025?",
        category=EvalCategory.IN_SCOPE_UNSUPPORTED,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    EvalQuestion(
        id="is-02",
        question="How many employees were hired in January 2026?",
        category=EvalCategory.IN_SCOPE_UNSUPPORTED,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    EvalQuestion(
        id="is-03",
        question="What is the CEO's opinion on the new hybrid work model?",
        category=EvalCategory.IN_SCOPE_UNSUPPORTED,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    EvalQuestion(
        id="is-04",
        question="What specific projects is the engineering team working on right now?",
        category=EvalCategory.IN_SCOPE_UNSUPPORTED,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    EvalQuestion(
        id="is-05",
        question="What was discussed in last Tuesday's all-hands meeting?",
        category=EvalCategory.IN_SCOPE_UNSUPPORTED,
        expected_chunks=(),
        should_be_grounded=False,
        refusal_fragment="couldn't find",
    ),
    # === 7. Follow-up questions (5 questions) ==============================
    EvalQuestion(
        id="fu-01",
        question="Can I carry it over to next year?",
        category=EvalCategory.FOLLOW_UP,
        expected_chunks=(
            ExpectedChunk(content_keywords=("carry", "over", "vacation")),
        ),
        should_be_grounded=True,
        prior_context=(
            ("user", "How many vacation days do I get?"),
            ("assistant", "You are entitled to 20 vacation days per year."),
        ),
    ),
    EvalQuestion(
        id="fu-02",
        question="What happens if I exceed that limit?",
        category=EvalCategory.FOLLOW_UP,
        expected_chunks=(
            ExpectedChunk(content_keywords=("exceed", "limit", "policy")),
        ),
        should_be_grounded=True,
        prior_context=(
            ("user", "What is the monthly spending limit for team lunches?"),
            ("assistant", "The monthly spending limit for team lunches is $500 per team."),
        ),
    ),
    EvalQuestion(
        id="fu-03",
        question="How long does that process typically take?",
        category=EvalCategory.FOLLOW_UP,
        expected_chunks=(
            ExpectedChunk(content_keywords=("process", "days", "time")),
        ),
        should_be_grounded=True,
        prior_context=(
            ("user", "How do I request a leave of absence?"),
            ("assistant",
             "To request a leave of absence, submit form LEA-001 to your manager."),
        ),
    ),
    EvalQuestion(
        id="fu-04",
        question="Are there exceptions to that rule?",
        category=EvalCategory.FOLLOW_UP,
        expected_chunks=(
            ExpectedChunk(content_keywords=("exception", "policy", "rule")),
        ),
        should_be_grounded=True,
        prior_context=(
            ("user", "What is the dress code policy?"),
            ("assistant",
             "The dress code policy requires business casual attire on weekdays."),
        ),
    ),
    EvalQuestion(
        id="fu-05",
        question="Who should I contact about that?",
        category=EvalCategory.FOLLOW_UP,
        expected_chunks=(
            ExpectedChunk(content_keywords=("contact", "support", "help")),
        ),
        should_be_grounded=True,
        prior_context=(
            ("user", "How do I report an IT issue?"),
            ("assistant",
             "To report an IT issue, submit a ticket through the IT helpdesk portal."),
        ),
    ),
]


def get_dataset() -> list[EvalQuestion]:
    """Return the evaluation dataset."""
    return list(DATASET)


def get_dataset_by_category(category: EvalCategory) -> list[EvalQuestion]:
    """Return questions filtered by category."""
    return [q for q in DATASET if q.category == category]


def get_groundable_questions() -> list[EvalQuestion]:
    """Return questions that should pass the grounding threshold."""
    return [q for q in DATASET if q.should_be_grounded]


def get_refusal_questions() -> list[EvalQuestion]:
    """Return questions that should be refused (out-of-scope or unsupported)."""
    return [q for q in DATASET if not q.should_be_grounded]


__all__ = [
    "DATASET",
    "EvalCategory",
    "EvalQuestion",
    "ExpectedChunk",
    "get_dataset",
    "get_dataset_by_category",
    "get_groundable_questions",
    "get_refusal_questions",
]
