"""Deterministic intent classification for chat questions.

Phase A: replaces the ad-hoc ``_is_metadata_question`` check with a proper
multi-intent router.  Deterministic regex/heuristics handle obvious cases;
an LLM fallback is available for genuinely ambiguous queries (but NOT used
during routing — only when the caller needs disambiguation).

The classifier returns an :class:`Intent` that tells the chat handler which
lane to route the question into.  Every lane (including identity,
conversation_history, app_help, and out_of_scope) runs through
``assert_workspace_role`` first — authorization is non-negotiable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class IntentCategory(str, Enum):
    """High-level routing lane."""

    METADATA = "metadata"
    CONVERSATION_HISTORY = "conversation_history"
    IDENTITY = "identity"
    APP_HELP = "app_help"
    DOCUMENT_CONTENT = "document_content"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


class MetadataSubIntent(str, Enum):
    """What kind of metadata the question asks for."""

    DOC_LIST = "doc_list"
    DOC_COUNT = "doc_count"
    DOC_PAGE_COUNT = "doc_page_count"
    MEMBER_COUNT = "member_count"
    MEMBER_LIST = "member_list"
    ROLE = "role"


class ConversationHistorySubIntent(str, Enum):
    """What aspect of conversation history is requested."""

    PREVIOUS_QUESTIONS = "previous_questions"
    PREVIOUS_ANSWER = "previous_answer"
    RECENT_CONVERSATION = "recent_conversation"


class QueryShape(str, Enum):
    """The shape of a document-content query (Phase B, B2)."""

    FACT_LOOKUP = "fact_lookup"
    TARGETED_QUERY = "targeted_query"
    OVERVIEW = "overview"
    LIST_EXTRACTION = "list_extraction"
    COMPARISON = "comparison"


@dataclass(frozen=True)
class Intent:
    """A classified intent with all metadata the handler needs.

    Attributes
    ----------
    category:
        The high-level routing lane.
    metadata_sub:
        ``None`` unless ``category == METADATA``.
    member_status:
        For ``member_count`` / ``member_list``, the status filter
        (e.g. ``"ACTIVE"`` or ``"INVITED"``).  ``None`` means no filter.
    conversation_history_sub:
        ``None`` unless ``category == CONVERSATION_HISTORY``.
    needs_clarification:
        ``True`` when the query is genuinely ambiguous and cannot be resolved
        deterministically.
    skip_rewrite:
        ``True`` when the query is already a clear, structured metadata or
        out-of-scope question and rewriting would only risk losing
        constraints.
    reason:
        Human-readable classification reason (for logging).
    """

    category: IntentCategory
    metadata_sub: MetadataSubIntent | None = None
    member_status: str | None = None
    conversation_history_sub: ConversationHistorySubIntent | None = None
    query_shape: QueryShape | None = None
    needs_clarification: bool = False
    skip_rewrite: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Query shape classification (Phase B, B2)
# ---------------------------------------------------------------------------

# Overview patterns: "what is X", "tell me about X", "summarize X"
# "what is" must NOT be followed by "the" (that's a fact lookup: "What is the rate?")
_OVERVIEW_PATTERNS = re.compile(
    r"(?:what\s+is\s+(?!the\s)|what(?:'?s)\s+(?!the\s)|tell\s+me\s+about|describe|summarize|"
    r"overview\s+of|explain|give\s+me\s+(?:an\s+)?overview)\b",
    re.IGNORECASE,
)

# Comparison patterns: "how does X compare to Y", "X vs Y", "difference between X and Y"
_COMPARISON_PATTERNS = re.compile(
    r"(?:how\s+(?:does|do|did|is|are)\s+.+?\s+(?:compare|differ|stack))\b"
    r"|\bvs\.?\b"
    r"|(?:difference\s+between)\b"
    r"|(?:compare|contrast)\s+",
    re.IGNORECASE,
)

# List extraction patterns: "list", "what are all", "enumerate"
_LIST_EXTRACTION_PATTERNS = re.compile(
    r"(?:list|enumerate|what\s+are\s+all|name\s+all|give\s+me\s+all)\b",
    re.IGNORECASE,
)

# Targeted query: has a document reference (already resolved by doc_targeting)
# or asks about a specific entity/identifier.
_TARGETED_PATTERNS = re.compile(
    r"(?:in\s+(?:the|this|that|my|our)\s+\w+\s+"
    r"(?:document|doc|file|pdf|handbook|guide|manual))\b"
    r"|(?:what\s+(?:does|do|did)\s+(?:the|this|that)\s+\w+\s+"
    r"(?:document|doc|file)\s+say)\b",
    re.IGNORECASE,
)


def classify_query_shape(query: str, *, has_doc_target: bool = False) -> QueryShape:
    """Classify the shape of a document-content query.

    Uses deterministic heuristics first.  Falls back to FACT_LOOKUP for
    queries that don't match any specific pattern.

    Parameters
    ----------
    query:
        The user's question (already rewritten if applicable).
    has_doc_target:
        Whether document targeting resolved a specific document.
    """
    q = query.strip()

    # Targeted query: named a specific document.
    if has_doc_target or _TARGETED_PATTERNS.search(q):
        # But if it also looks like an overview, prefer OVERVIEW.
        if _OVERVIEW_PATTERNS.search(q):
            return QueryShape.OVERVIEW
        return QueryShape.TARGETED_QUERY

    # Comparison.
    if _COMPARISON_PATTERNS.search(q):
        return QueryShape.COMPARISON

    # Overview: "what is X", "tell me about X", etc.
    if _OVERVIEW_PATTERNS.search(q):
        return QueryShape.OVERVIEW

    # List extraction.
    if _LIST_EXTRACTION_PATTERNS.search(q):
        return QueryShape.LIST_EXTRACTION

    # Default: fact lookup.
    return QueryShape.FACT_LOOKUP


# ---------------------------------------------------------------------------
# Deterministic patterns — order matters (first match wins)
# ---------------------------------------------------------------------------

# --- Out-of-scope patterns ---
# General-knowledge questions that have nothing to do with workspace documents.

_MATH_PATTERN = re.compile(
    r"^\s*(?:what\s+is\s+)?(?:\d+\s*[\+\-\*\/\=]\s*\d+|sqrt\s+of\s+\d+|"
    r"what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+)\s*[?.]?\s*$",
    re.IGNORECASE,
)

_CAPITAL_OF_PATTERN = re.compile(
    r"\b(?:capital\s+of)\s+\w+",
    re.IGNORECASE,
)

_GENERAL_KNOW_PATTERN = re.compile(
    r"(?:who\s+(?:won|scored|played)\s+.*(?:world\s+cup|super\s+bowl|olympics))"
    r"|(?:write\s+(?:me\s+)?(?:a\s+)?(?:python|javascript|java|c\+\+|rust|go)\s+"
    r"(?:program|script|function|game|app))"
    r"|(?:tell\s+me\s+a\s+joke)",
    re.IGNORECASE,
)

# --- Identity patterns ---
_IDENTITY_PATTERN = re.compile(
    r"(?:what(?:'?s|\s+is)\s+my\s+(?:name|email))\b"
    r"|who\s+am\s+i\b"
    r"|my\s+name\b",
    re.IGNORECASE,
)

# --- App-help patterns ---
_APP_HELP_PATTERN = re.compile(
    r"(?:who\s+can\s+(?:upload|add|submit)\s+(?:document|file)s?)\b"
    r"|(?:how\s+(?:can|do|should)\s+(?:i|we)\s+"
    r"(?:upload|add|submit)\s+(?:.*?\s+)?(?:document|file)s?)\b"
    r"|(?:how\s+(?:can|do|should)\s+(?:i|we)\s+"
    r"(?:upload|add|submit).*?\b(?:and\s+)?(?:ask|question))\b"
    r"|(?:what\s+can\s+(?:i|we)\s+do)\b"
    r"|(?:am\s+i\s+being\s+(?:monitored|tracked|watched))\b"
    r"|(?:do\s+you\s+(?:track|monitor|log)\s+(?:my|us|activity))\b"
    r"|(?:who\s+has\s+(?:access|permission))\b"
    r"|(?:what\s+(?:are\s+)?(?:my|the)\s+(?:permission|role|access))\b"
    r"|(?:how\s+(?:does\s+)?(?:this|it)\s+work)\b",
    re.IGNORECASE,
)

# --- Conversation history patterns ---
_CONVERSATION_HISTORY_PATTERN = re.compile(
    r"(?:what\s+(?:are|were)\s+(?:the\s+)?(?:questions?|things?)\s+"
    r"(?:i|we)\s+(?:ask|asked|said|mentioned))\b"
    r"|(?:what\s+(?:was|did)\s+(?:my|the)\s+(?:previous|last|earlier)\s+"
    r"(?:question|query|thing|message))\b"
    r"|(?:what\s+did\s+(?:i|we)\s+(?:ask|say))\b"
    r"|(?:what\s+did\s+you\s+(?:just\s+)?(?:say|tell\s+me|answer))\b"
    r"|(?:what\s+(?:questions?|messages?)\s+(?:have\s+)?i\s+"
    r"(?:ask|asked|sent|made))\b"
    r"|(?:show\s+(?:me\s+)?(?:my\s+)?(?:previous|recent|last)\s+"
    r"(?:questions?|messages?|conversation))\b"
    r"|(?:what\s+have\s+(?:i|we)\s+(?:been\s+)?(?:asking|discussing|talking))\b",
    re.IGNORECASE,
)

# --- Metadata: member patterns ---
_MEMBER_COUNT_PATTERN = re.compile(
    r"(?:how\s+many|number\s+of|count\s+of|total\s+(?:number\s+of)?)\s+"
    r"(?:\w+\s+)?"  # optional adjective before the noun (e.g. "active members")
    r"(?:people|members?|users?|employees?|team\s*members?|contributors?)",
    re.IGNORECASE,
)

_MEMBER_LIST_PATTERN = re.compile(
    r"(?:list|show|what|which|name)s?\s+"
    r"(?:are\s+the\s+)?(?:me\s+)?(?:all\s+)?"
    r"(?:the\s+|this\s+|our\s+)?(?:workspace\s+)?"
    r"(?:people|members?|users?|employees?|team\s*members?|contributors?)"
    r"|(?:who(?:'?s|\s+is|\s+are))\s+"
    r"(?:in|of|on|at)\s+"
    r"(?:the\s+|this\s+|our\s+)?(?:workspace|company|team)?"
    r"|(?:who(?:'?s|\s+is|\s+are))\s+"
    r"(?:invited|pending|active|removed|waiting|confirmed|accepted)",
    re.IGNORECASE,
)

# Status qualifiers for member queries
_INVITED_PATTERN = re.compile(
    r"\b(?:invited|pending|waiting|not\s+(?:yet\s+)?active|awaiting)\b",
    re.IGNORECASE,
)

_ACTIVE_PATTERN = re.compile(
    r"\b(?:active|confirmed|current|accepted|joined)\b",
    re.IGNORECASE,
)

_REMOVED_PATTERN = re.compile(
    r"\b(?:removed|deleted|inactive|deactivated)\b",
    re.IGNORECASE,
)

# --- Metadata: document patterns ---
_TOPIC_QUALIFIERS = re.compile(
    r"\b(?:about|discuss|cover|mention|regarding|on\s+the\s+topic\s+of|concerning)\b",
    re.IGNORECASE,
)

_DOC_COUNT_PATTERN = re.compile(
    r"(?:how\s+many|number\s+of|count\s+of|total\s+(?:number\s+of)?)\s+"
    r"(?:uploaded\s+)?(?:my\s+|the\s+|this\s+)?(?:own\s+)?"
    r"(?:files|documents?)",
    re.IGNORECASE,
)

_DOC_LIST_PATTERN = re.compile(
    r"(?:list|show|what|which|names?)\s+"
    r"(?:are\s+the\s+)?(?:me\s+)?(?:all\s+)?"
    r"(?:my\s+|the\s+|this\s+)?(?:uploaded\s+)?(?:own\s+)?"
    r"(?:of\s+(?:the\s+|my\s+|this\s+)?)?"  # optional "of the" before noun
    r"(?:files|documents?)",
    re.IGNORECASE,
)

_DOC_PAGE_COUNT_PATTERN = re.compile(
    r"(?:how\s+many|number\s+of|total)\s+"
    r"(?:pages?|sheets?)\s+"
    r"(?:in\s+|of\s+|are\s+(?:there\s+)?in\s+)?"
    r"(?:each|every|the|this|all)?\s*"
    r"(?:document|file|upload)?s?",
    re.IGNORECASE,
)

_ROLE_PATTERN = re.compile(
    r"(?:what(?:'?s|\s+is)\s+my|my\s+current|what\s+role\s+(?:do\s+i|am\s+i))\s+"
    r"(?:role|access|permission|level)"
    r"|what(?:'?s|\s+is)\s+my\s+(?:role|access|permission|level)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> Intent:
    """Classify a user query into a routing intent.

    Deterministic regex patterns handle obvious cases first.  This function
    is synchronous and does NOT call any LLM — it is pure pattern matching.
    """
    q = query.strip()
    if not q:
        return Intent(
            category=IntentCategory.AMBIGUOUS,
            needs_clarification=True,
            reason="empty_query",
        )

    # --- 1. Out-of-scope (obvious general knowledge) ---
    if _is_out_of_scope(q):
        return Intent(
            category=IntentCategory.OUT_OF_SCOPE,
            skip_rewrite=True,
            reason="general_knowledge",
        )

    # --- 2. Identity ---
    if _IDENTITY_PATTERN.search(q):
        return Intent(
            category=IntentCategory.IDENTITY,
            skip_rewrite=True,
            reason="identity_query",
        )

    # --- 3. Conversation history ---
    if _CONVERSATION_HISTORY_PATTERN.search(q):
        sub = _classify_conversation_history(q)
        return Intent(
            category=IntentCategory.CONVERSATION_HISTORY,
            conversation_history_sub=sub,
            skip_rewrite=True,
            reason="conversation_history_query",
        )

    # --- 4. App help ---
    if _APP_HELP_PATTERN.search(q):
        return Intent(
            category=IntentCategory.APP_HELP,
            skip_rewrite=True,
            reason="app_help_query",
        )

    # --- 5. Metadata: member queries ---
    member_intent = _classify_member_metadata(q)
    if member_intent is not None:
        return member_intent

    # --- 6. Metadata: document queries ---
    doc_intent = _classify_document_metadata(q)
    if doc_intent is not None:
        return doc_intent

    # --- 7. Document content (everything else goes to RAG) ---
    return Intent(
        category=IntentCategory.DOCUMENT_CONTENT,
        reason="default_document_content",
    )


def _is_out_of_scope(q: str) -> bool:
    """Detect obvious general-knowledge questions."""
    if _MATH_PATTERN.search(q):
        return True
    if _CAPITAL_OF_PATTERN.search(q):
        return True
    if _GENERAL_KNOW_PATTERN.search(q):
        return True
    return False


def _classify_conversation_history(q: str) -> ConversationHistorySubIntent:
    """Determine which aspect of conversation history is requested."""
    if re.search(r"(?:previous|last|earlier)\s+question", q, re.IGNORECASE):
        return ConversationHistorySubIntent.PREVIOUS_QUESTIONS
    if re.search(r"(?:what\s+did\s+you\s+)?(?:say|answer|tell)", q, re.IGNORECASE):
        return ConversationHistorySubIntent.PREVIOUS_ANSWER
    return ConversationHistorySubIntent.PREVIOUS_QUESTIONS


# Pattern for follow-up style member count: "how many are invited?"
# without the explicit word "members" — relies on status keywords.
_MEMBER_COUNT_IMPLICIT_PATTERN = re.compile(
    r"(?:how\s+many|number\s+of|count\s+of)\s+"
    r"(?:are|is|were|was)\s+"
    r"(\w+)",
    re.IGNORECASE,
)

# Status keywords that can follow "how many are" to imply member count.
_IMPLICIT_MEMBER_STATUSES = frozenset({
    "invited", "pending", "waiting", "active", "confirmed",
    "accepted", "removed", "inactive", "deleted",
})


def _classify_member_metadata(q: str) -> Intent | None:
    """Detect member count/list queries with status filters."""
    # Don't match if this is a topic-qualified content question.
    if _TOPIC_QUALIFIERS.search(q):
        return None

    # Must match a member-related pattern.
    is_member_count = bool(_MEMBER_COUNT_PATTERN.search(q))
    is_member_list = bool(_MEMBER_LIST_PATTERN.search(q))

    # Also detect follow-up style: "how many are invited?" (no "members" word)
    if not is_member_count and not is_member_list:
        implicit = _MEMBER_COUNT_IMPLICIT_PATTERN.search(q)
        if implicit:
            candidate = implicit.group(1).lower().rstrip("?!.;")
            if candidate in _IMPLICIT_MEMBER_STATUSES:
                # Determine status filter.
                status: str | None = None
                if _INVITED_PATTERN.search(q):
                    status = "INVITED"
                elif _ACTIVE_PATTERN.search(q):
                    status = "ACTIVE"
                elif _REMOVED_PATTERN.search(q):
                    status = "REMOVED"
                elif candidate in ("invited", "pending", "waiting"):
                    status = "INVITED"
                elif candidate in ("active", "confirmed", "accepted"):
                    status = "ACTIVE"
                elif candidate in ("removed", "inactive", "deleted"):
                    status = "REMOVED"
                return Intent(
                    category=IntentCategory.METADATA,
                    metadata_sub=MetadataSubIntent.MEMBER_COUNT,
                    member_status=status,
                    skip_rewrite=True,
                    reason=f"member_count_implicit status={status or 'all'}",
                )
        return None

    # Determine status filter.
    status = None
    if _INVITED_PATTERN.search(q):
        status = "INVITED"
    elif _ACTIVE_PATTERN.search(q):
        status = "ACTIVE"
    elif _REMOVED_PATTERN.search(q):
        status = "REMOVED"

    if is_member_count:
        return Intent(
            category=IntentCategory.METADATA,
            metadata_sub=MetadataSubIntent.MEMBER_COUNT,
            member_status=status,
            skip_rewrite=True,
            reason=f"member_count status={status or 'all'}",
        )
    else:
        return Intent(
            category=IntentCategory.METADATA,
            metadata_sub=MetadataSubIntent.MEMBER_LIST,
            member_status=status,
            skip_rewrite=True,
            reason=f"member_list status={status or 'all'}",
        )


def _classify_document_metadata(q: str) -> Intent | None:
    """Detect document count/list/page-count queries."""
    # Don't match topic-qualified content questions.
    if _TOPIC_QUALIFIERS.search(q):
        return None

    # Page count must be checked before general count to avoid false matches.
    if _DOC_PAGE_COUNT_PATTERN.search(q):
        return Intent(
            category=IntentCategory.METADATA,
            metadata_sub=MetadataSubIntent.DOC_PAGE_COUNT,
            skip_rewrite=True,
            reason="doc_page_count",
        )

    if _DOC_COUNT_PATTERN.search(q):
        return Intent(
            category=IntentCategory.METADATA,
            metadata_sub=MetadataSubIntent.DOC_COUNT,
            skip_rewrite=True,
            reason="doc_count",
        )

    if _DOC_LIST_PATTERN.search(q):
        return Intent(
            category=IntentCategory.METADATA,
            metadata_sub=MetadataSubIntent.DOC_LIST,
            skip_rewrite=True,
            reason="doc_list",
        )

    # Role query.
    if _ROLE_PATTERN.search(q):
        return Intent(
            category=IntentCategory.METADATA,
            metadata_sub=MetadataSubIntent.ROLE,
            skip_rewrite=True,
            reason="role_query",
        )

    return None


__all__ = [
    "ConversationHistorySubIntent",
    "Intent",
    "IntentCategory",
    "MetadataSubIntent",
    "QueryShape",
    "classify_intent",
    "classify_query_shape",
]
