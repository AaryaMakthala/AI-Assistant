"""Layered relevance gate for chat questions (CLAUDE.md section 8).

Three separate decisions in the RAG pipeline, implemented as independent modules:

1. **Query/company relevance** (this module): Is this question about this workspace's
   documents at all?  Runs BEFORE retrieval to avoid wasting a retrieval call on
   obviously unrelated questions.  Uses two layers:
   - Layer 1 (deterministic): metadata routing, greetings, obviously unrelated.
   - Layer 2 (semantic): LLM-based relevance classifier using workspace document
     titles/topics as context — never chunk text, never cross-workspace.

2. **Retrieval relevance** (``app/retrieval/grounding.py``): After retrieval, did we
   get any reasonably on-topic chunks?  The top rerank score is compared against
   ``RETRIEVAL_RELEVANCE_THRESHOLD``.

3. **Answer grounding** (system prompt in ``app/rag/prompts.py``): Do the top chunks
   actually support an answer?  The LLM is instructed to refuse if the context
   doesn't cover the question.

The reranker's raw score is NOT a relevance signal by itself — cross-encoder scores
are unbounded logits, not probabilities.  A score of -11.0 is a clear non-match,
while 0.70 is a strong match, but the boundary between "ambiguous" and "relevant"
depends on the query and workspace, not a fixed threshold.  This is why Layer 2
exists: to make the relevance decision with full context rather than a raw number.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Literal

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Document


# ---------------------------------------------------------------------------
# Layer 1 — deterministic heuristics
# ---------------------------------------------------------------------------

# Greeting / chitchat patterns — these get a direct friendly response, no RAG.
# Expanded to handle elongated chars and common non-English greetings,
# matching the intent classifier's expanded greeting pattern.
_GREETING_PATTERN = re.compile(
    r"^(?:hi+|hello+|hey+|good\s+(?:morning|afternoon|evening)|thanks|thank you|"
    r"what'?s*\s+up|how\s+are\s+you|help|"
    r"hola|bonjour|salut|guten\s+(?:tag|morgen)|namaste|salaam|shalom|ciao)"
    r"\s*[!.]?\s*$",
    re.IGNORECASE,
)

# Obviously unrelated patterns — well-known general-knowledge questions that
# have nothing to do with any workspace.  Conservative: only reject the truly
# obvious cases, push anything uncertain to Layer 2.
_UNRELATED_PATTERNS = [
    re.compile(
        r"(?:capital\s+of\s+(?:france|germany|italy|spain|japan|india|china|"
        r"brazil|australia|canada|mexico|uk|united\s+kingdom|usa|united\s+states))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:who\s+(?:won|scored|played)\s+.*(?:world\s+cup|super\s+bowl|olympics|championship))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:what\s+is\s+(?:the\s+)?(?:speed\s+of\s+light|boiling\s+point\s+of\s+water|"
        r"sqrt\s+of\s+\d+|pi\s+equal))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:write\s+(?:me\s+)?(?:a\s+)?(?:python|javascript|java|c\+\+|rust|go)\s+"
        r"(?:program|script|function|game|app))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:tell\s+me\s+a\s+joke|why\s+(?:did|does|is|are|was|were)\s+the\s+chicken)",
        re.IGNORECASE,
    ),
]


@dataclass(frozen=True)
class RelevanceDecision:
    """The result of the relevance gate."""

    relevant: bool
    reason: str
    confidence: float = 1.0
    layer: Literal["deterministic", "llm_classifier", "fallback"] = "deterministic"


async def check_relevance(
    session: AsyncSession,
    *,
    question: str,
    workspace_id: uuid.UUID,
) -> RelevanceDecision:
    """Two-layer relevance gate: deterministic heuristics, then LLM classifier.

    Layer 1 (deterministic) is conservative — only rejects truly obvious non-matches.
    Anything uncertain passes to Layer 2 (LLM-based) which uses workspace document
    titles/topics as context to make the final call.

    If the LLM relevance check itself fails (network/503), the question falls
    through to attempting retrieval + the retrieval-level threshold instead.
    """
    q = question.strip()

    # --- Layer 1: deterministic ---

    # Greetings get a direct response.
    if _GREETING_PATTERN.match(q):
        return RelevanceDecision(
            relevant=True,
            reason="greeting_or_chitchat",
            confidence=1.0,
            layer="deterministic",
        )

    # Obviously unrelated questions are rejected without retrieval.
    for pattern in _UNRELATED_PATTERNS:
        if pattern.search(q):
            return RelevanceDecision(
                relevant=False,
                reason="obviously_unrelated",
                confidence=1.0,
                layer="deterministic",
            )

    # --- Layer 2: LLM-based relevance classifier ---
    # Use workspace document titles/topics as context — never chunk text,
    # never cross-workspace data.
    return await _llm_relevance_check(
        session=session,
        question=q,
        workspace_id=workspace_id,
    )


async def _llm_relevance_check(
    *,
    session: AsyncSession,
    question: str,
    workspace_id: uuid.UUID,
) -> RelevanceDecision:
    """LLM-based relevance classifier using workspace document titles.

    Fetches READY document filenames for this workspace and asks the LLM whether
    the question is plausibly about any of them.  Scoped to workspace_id — no
    document text or cross-workspace data is included.

    If the LLM call fails, returns ``relevant=True`` (optimistic) so the normal
    retrieval + grounding pipeline handles it.  The retrieval-level threshold
    (Layer 1 grounding) is the backstop.
    """
    try:
        # Fetch workspace document titles/topics (workspace-scoped).
        rows = (
            await session.execute(
                select(Document.filename).where(
                    Document.workspace_id == workspace_id,
                    Document.status == "READY",
                )
            )
        ).all()

        doc_titles = [row.filename for row in rows]
        if not doc_titles:
            # Empty workspace — question is "relevant" in the sense that retrieval
            # will return zero chunks and the normal grounding check handles it.
            return RelevanceDecision(
                relevant=True,
                reason="empty_workspace_fall_through",
                confidence=0.5,
                layer="llm_classifier",
            )

        # Build a lightweight relevance-classification prompt.
        titles_context = "\n".join(f"- {title}" for title in doc_titles)
        classification_prompt = (
            f"Workspace document titles:\n{titles_context}\n\n"
            f"User question: {question}\n\n"
            "Is this question plausibly about content that might be found in one of "
            "these documents? Answer with ONLY a JSON object: "
            '{"relevant": true/false, "confidence": 0.0-1.0, "reason": "..."}'
        )

        # Import the LLM provider from the same dependency the chat endpoint uses.
        from app.llm.base import Completion, Message
        from app.config import get_settings

        settings = get_settings()
        chain_count = sum(
            1
            for key in (
                settings.gemini_api_key,
                settings.groq_api_key,
                settings.openrouter_api_key,
            )
            if key is not None
        )
        if chain_count > 1:
            from app.llm.fallback import FallbackChainProvider
            provider = FallbackChainProvider()
        else:
            from app.llm.generic import GenericProvider
            provider = GenericProvider()
        messages = [Message(role="user", content=classification_prompt)]
        completion = Completion()

        # Consume the full stream (this is a classification call, not user-facing).
        async for _token in provider.stream(messages, completion=completion):
            pass

        # Parse the structured output.
        response_text = completion.text.strip()
        return _parse_relevance_response(response_text)

    except Exception as exc:
        # LLM relevance check failed — fall through to retrieval + grounding threshold.
        # This is the correct behavior per Part 2: "If the relevance-gate LLM call
        # itself fails, do not silently treat that as 'not relevant'".
        logger.warning(
            "LLM relevance check failed for workspace {ws}, falling through to "
            "retrieval threshold: {error}",
            ws=workspace_id,
            error=str(exc)[:200],
        )
        return RelevanceDecision(
            relevant=True,
            reason="relevance_gate_degraded",
            confidence=0.0,
            layer="fallback",
        )


def _parse_relevance_response(response_text: str) -> RelevanceDecision:
    """Parse the LLM's JSON relevance classification response."""
    import json

    try:
        # Try to extract JSON from the response (LLM may wrap it in markdown).
        json_match = re.search(r"\{[^}]+\}", response_text)
        if json_match:
            data = json.loads(json_match.group())
            relevant = bool(data.get("relevant", True))
            confidence = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", "llm_classification"))
            return RelevanceDecision(
                relevant=relevant,
                reason=reason,
                confidence=confidence,
                layer="llm_classifier",
            )
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Parsing failed — be optimistic and let retrieval handle it.
    logger.warning(
        "Could not parse LLM relevance response, assuming relevant: {text}",
        text=response_text[:200],
    )
    return RelevanceDecision(
        relevant=True,
        reason="parse_failure_optimistic",
        confidence=0.0,
        layer="fallback",
    )


__all__ = ["RelevanceDecision", "check_relevance"]
