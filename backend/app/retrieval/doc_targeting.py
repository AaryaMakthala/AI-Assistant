"""Document reference detection and fuzzy matching for document-specific questions.

When a user explicitly names a document in their question (e.g. "What does the DevOps
document say about Kanban?"), retrieval should be biased toward that document.  This
module provides:

1. **Reference detection**: regex patterns that extract the document name fragment from
   natural-language phrasing ("in the DevOps document", "from the DevOps doc", etc.).
2. **Fuzzy matching**: normalize and compare the extracted fragment against READY
   documents in the current workspace.
3. **Ambiguity detection**: if multiple documents are similarly plausible, refuse to
   silently select one — fall back to normal workspace-wide retrieval.

The module does NOT perform retrieval itself.  It returns a
:class:`DocumentTargetingResult` that the pipeline uses to optionally filter chunks.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document


# ---------------------------------------------------------------------------
# Reference detection patterns
# ---------------------------------------------------------------------------

# Patterns that capture a document name fragment from natural-language phrasing.
# Group 1 is the extracted document name.
_DOC_REFERENCE_PATTERNS = [
    # "in the DevOps document" / "from the DevOps doc" / "according to the DevOps document"
    re.compile(
        r"(?:in|from|according\s+to|per|within)\s+"
        r"(?:the|this|that|my|our)\s+"
        r"(?:"
        r"(\S+(?:\s+\S+){0,5})\s+"  # capture up to 6 words as doc name
        r"(?:document|doc|file|pdf|docx|upload|handbook|guide|manual|bank|question\s*bank)"
        r")",
        re.IGNORECASE,
    ),
    # "what does the DevOps document say about X"
    re.compile(
        r"(?:what|how)\s+(?:does|do|did|is|are|was|were)\s+"
        r"(?:the|this|that|my|our)\s+"
        r"(\S+(?:\s+\S+){0,5})\s+"
        r"(?:document|doc|file|pdf|docx|upload|handbook|guide|manual|bank|question\s*bank)"
        r"\s+(?:say|state|describe|mention|cover|discuss|contain|include|tell)",
        re.IGNORECASE,
    ),
    # "the DevOps document says ..." / "the DevOps doc states ..."
    re.compile(
        r"(?:the|this|that|my|our)\s+"
        r"(\S+(?:\s+\S+){0,5})\s+"
        r"(?:document|doc|file|pdf|docx|upload|handbook|guide|manual|bank|question\s*bank)"
        r"\s+(?:says?|states?|describes?|mentions?|covers?|discusses?|contains?|includes?|tells?)",
        re.IGNORECASE,
    ),
    # "about Kanban are present in the DevOps document"
    re.compile(
        r"in\s+(?:the|this|that|my|our)\s+"
        r"(\S+(?:\s+\S+){0,5})\s+"
        r"(?:document|doc|file|pdf|docx|upload|handbook|guide|manual|bank|question\s*bank)",
        re.IGNORECASE,
    ),
    # Fallback: just "the DevOps document" / "the DevOps doc"
    re.compile(
        r"(?:the|this|that|my|our)\s+"
        r"(\S+(?:\s+\S+){0,4})\s+"
        r"(?:document|doc|file|pdf|docx)",
        re.IGNORECASE,
    ),
]

# Words to strip from the extracted fragment during normalization.
# Only generic document-type words that add no discriminative value.
# Domain words (devops, software, engineering, etc.) are NEVER stripped.
_STRIP_WORDS = frozenset({
    "document", "doc", "file", "pdf", "docx", "upload", "uploads",
    "handbook", "guide", "manual", "the", "a", "an",
})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentTargetingResult:
    """Outcome of document reference detection."""

    #: The detected document name fragment (raw, before normalization).
    detected_name: str | None
    #: The matched document's UUID, if a confident match was found.
    matched_document_id: uuid.UUID | None
    #: The matched document's filename, for logging/citations.
    matched_filename: str | None
    #: Confidence score: 0.0 = no match, 1.0 = exact match.
    confidence: float
    #: Why this result was produced.
    reason: str  # "exact_match" | "fuzzy_match" | "ambiguous" | "no_match" | "no_reference"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_for_match(text: str) -> str:
    """Normalize a string for fuzzy comparison.

    Handles case, punctuation, underscores, hyphens, file extensions, and
    repeated whitespace.  Does NOT strip meaningful domain words — only
    generic document-type words that add no discriminative value.
    """
    import unicodedata

    t = text.lower().strip()

    # Remove file extensions.
    t = re.sub(r"\.(pdf|docx?|xlsx?|csv|txt|pptx?)$", "", t)

    # Replace punctuation and special characters with spaces.
    t = re.sub(r"[_\-/\\.,;:!?\"'()\[\]{}]", " ", t)

    # Normalize unicode.
    t = unicodedata.normalize("NFKD", t)

    # Collapse whitespace.
    t = re.sub(r"\s+", " ", t).strip()

    return t


def _token_set(text: str) -> set[str]:
    """Extract the meaningful token set from normalized text."""
    normalized = _normalize_for_match(text)
    tokens = set(normalized.split())
    # Remove generic document-type words that don't help distinguish.
    tokens -= _STRIP_WORDS
    return tokens


def _fuzzy_score(query_fragment: str, doc_filename: str) -> float:
    """Compute a similarity score between a query fragment and a document filename.

    Returns a float in [0.0, 1.0] where 1.0 is an exact match.
    Uses token-set overlap (Jaccard-like) as the primary signal.
    """
    query_tokens = _token_set(query_fragment)
    doc_tokens = _token_set(doc_filename)

    if not query_tokens or not doc_tokens:
        return 0.0

    intersection = query_tokens & doc_tokens
    union = query_tokens | doc_tokens

    if not union:
        return 0.0

    jaccard = len(intersection) / len(union)

    # Bonus: if all query tokens appear in the doc tokens, it's a strong match.
    if query_tokens <= doc_tokens:
        jaccard = max(jaccard, 0.9)

    # Bonus: partial containment — if most query tokens are in doc tokens.
    if query_tokens:
        containment = len(intersection) / len(query_tokens)
        if containment >= 0.7:
            jaccard = max(jaccard, containment * 0.95)

    return min(jaccard, 1.0)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_document_reference(question: str) -> str | None:
    """Extract a document name fragment from the question.

    Returns the raw extracted fragment, or None if no document reference is detected.
    """
    for pattern in _DOC_REFERENCE_PATTERNS:
        match = pattern.search(question)
        if match:
            fragment = match.group(1).strip()
            # Reject very short fragments (likely false positives).
            if len(fragment) >= 3:
                return fragment
    return None


async def resolve_document_target(
    session: AsyncSession,
    *,
    question: str,
    workspace_id: uuid.UUID,
) -> DocumentTargetingResult:
    """Detect and resolve a document reference in the question.

    Steps:
    1. Extract the document name fragment from the question.
    2. Fetch READY document filenames for this workspace.
    3. Fuzzy-match the fragment against filenames.
    4. Return a targeting result with the resolved document ID (if confident).

    If the match is ambiguous (multiple documents score similarly) or if no
    reference is detected, the result tells the caller to fall back to normal
    workspace-wide retrieval.
    """
    fragment = detect_document_reference(question)

    if fragment is None:
        return DocumentTargetingResult(
            detected_name=None,
            matched_document_id=None,
            matched_filename=None,
            confidence=0.0,
            reason="no_reference",
        )

    # Fetch READY documents for this workspace.
    rows = (
        await session.execute(
            select(Document.id, Document.filename).where(
                Document.workspace_id == workspace_id,
                Document.status == "READY",
            )
        )
    ).all()

    if not rows:
        logger.info(
            "Document reference detected ('{fragment}') but workspace {ws} has no READY docs",
            fragment=fragment,
            ws=workspace_id,
        )
        return DocumentTargetingResult(
            detected_name=fragment,
            matched_document_id=None,
            matched_filename=None,
            confidence=0.0,
            reason="no_match",
        )

    # Score each document against the fragment.
    scores: list[tuple[uuid.UUID, str, float]] = []
    for doc_id, filename in rows:
        score = _fuzzy_score(fragment, filename)
        if score > 0.0:
            scores.append((doc_id, filename, score))

    if not scores:
        logger.info(
            "Document reference detected ('{fragment}') but no READY docs matched in workspace {ws}",
            fragment=fragment,
            ws=workspace_id,
        )
        return DocumentTargetingResult(
            detected_name=fragment,
            matched_document_id=None,
            matched_filename=None,
            confidence=0.0,
            reason="no_match",
        )

    # Sort by score descending.
    scores.sort(key=lambda x: x[2], reverse=True)

    best_id, best_filename, best_score = scores[0]

    # Check for ambiguity: if the second-best score is within 0.15 of the best,
    # we have an ambiguous match — refuse to silently select one.
    if len(scores) > 1:
        second_score = scores[1][2]
        if best_score - second_score < 0.15 and best_score < 0.9:
            logger.info(
                "Ambiguous document reference '{fragment}': top matches are {matches}",
                fragment=fragment,
                matches=[(f, round(s, 3)) for _, f, s in scores[:3]],
            )
            return DocumentTargetingResult(
                detected_name=fragment,
                matched_document_id=None,
                matched_filename=None,
                confidence=best_score,
                reason="ambiguous",
            )

    # Confidence thresholds:
    # - >= 0.5: confident match (fuzzy match with good token overlap)
    # - < 0.5: weak match, don't target
    if best_score < 0.5:
        logger.info(
            "Document reference '{fragment}' weak match (score={score:.3f}) — "
            "falling back to workspace-wide retrieval",
            fragment=fragment,
            score=best_score,
        )
        return DocumentTargetingResult(
            detected_name=fragment,
            matched_document_id=None,
            matched_filename=None,
            confidence=best_score,
            reason="no_match",
        )

    match_type = "exact_match" if best_score >= 0.9 else "fuzzy_match"
    logger.info(
        "Document reference '{fragment}' resolved to '{filename}' "
        "(id={doc_id}, score={score:.3f}, type={type})",
        fragment=fragment,
        filename=best_filename,
        doc_id=best_id,
        score=best_score,
        type=match_type,
    )
    return DocumentTargetingResult(
        detected_name=fragment,
        matched_document_id=best_id,
        matched_filename=best_filename,
        confidence=best_score,
        reason=match_type,
    )


__all__ = [
    "DocumentTargetingResult",
    "detect_document_reference",
    "resolve_document_target",
]
