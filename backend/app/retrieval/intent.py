"""Deterministic intent classification for chat questions.

Expanded routing layer: determines which lane a query enters.

  GREETING             → conversational greeting, no retrieval
  APP_HELP             → questions about the application itself
  IDENTITY             → who-am-I questions from session data
  WORKSPACE_METADATA   → workspace/member/document count queries
  WORKSPACE_PERMISSION → who-can-do-what questions
  DOCUMENT_LIST        → list/show documents (metadata, not content)
  DOCUMENT_CONTENT     → questions answered from document content (Phase B-2 RAG)
  DOCUMENT_COMPARISON  → compare two or more documents
  CONVERSATION_HISTORY → questions about prior conversation
  OUT_OF_SCOPE         → general knowledge, no retrieval
  GENERAL_CONVERSATION → general chat not matching any specific lane
  AMBIGUOUS            → genuinely unclear, needs clarification

Deterministic regex/heuristics handle obvious cases; an LLM fallback is
available for genuinely ambiguous queries (but NOT used during routing —
only when the caller needs disambiguation).

The classifier returns an :class:`Intent` that tells the chat handler which
lane to route the question into.  Every lane runs through
``assert_workspace_role`` first — authorization is non-negotiable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from loguru import logger


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class IntentCategory(str, Enum):
    """High-level routing lane."""

    GREETING = "greeting"
    APP_HELP = "app_help"
    IDENTITY = "identity"
    IDENTITY_ASSISTANT = "identity_assistant"
    IDENTITY_USER = "identity_user"
    PERMISSIONS = "permissions"
    WORKSPACE_METADATA = "workspace_metadata"
    WORKSPACE_PERMISSION = "workspace_permission"
    DOCUMENT_LIST = "document_list"
    DOCUMENT_CONTENT = "document_content"
    DOCUMENT_COMPARISON = "document_comparison"
    CONVERSATION_HISTORY = "conversation_history"
    OUT_OF_SCOPE = "out_of_scope"
    GENERAL_CONVERSATION = "general_conversation"
    AMBIGUOUS = "ambiguous"


class MetadataSubIntent(str, Enum):
    """What kind of metadata the question asks for."""

    DOC_LIST = "doc_list"
    DOC_COUNT = "doc_count"
    DOC_PAGE_COUNT = "doc_page_count"
    DOC_DESCRIPTION = "doc_description"
    MEMBER_COUNT = "member_count"
    MEMBER_LIST = "member_list"
    ROLE = "role"
    COMPANY_NAME = "company_name"


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
        ``None`` unless ``category`` is a metadata/document lane.
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
    member_role: str | None = None
    conversation_history_sub: ConversationHistorySubIntent | None = None
    query_shape: QueryShape | None = None
    needs_clarification: bool = False
    skip_rewrite: bool = False
    rewritten_query: str | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Query shape classification (Phase B, B2)
# ---------------------------------------------------------------------------

# Overview patterns: "what is X", "tell me about X", "summarize X"
# "what is" must NOT be followed by "the" (that's a fact lookup: "What is the rate?").
# Coverage/scope questions ("what does X cover", "what topics does X cover") are
# treated as overviews too: they ask for the full scope of a document, so they need
# the broad-retrieval + aggregate-grounding treatment rather than a narrow fact lookup.
_OVERVIEW_PATTERNS = re.compile(
    r"(?:what\s+is\s+(?!the\s)|what(?:'?s)\s+(?!the\s)|tell\s+me\s+about|describe|summarize|"
    r"overview\s+of|explain|give\s+me\s+(?:an\s+)?overview|"
    r"what\s+is\s+covered(?:\s+in|\s+by)|"
    r"what\s+(?:topics|areas|subjects)\s+(?:does|do)\s+.{1,40}\s+cover|"
    r"what\s+does\s+.{1,40}\s+cover)"
    r"\b",
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

# --- Greeting patterns ---
# Expanded to handle: elongated chars (heyyyy→hey), non-English greetings,
# casual phrasing, and common variations.
_GREETING_PATTERN = re.compile(
    r"^\s*(?:hi+|hello+|hey+|howdy|good\s+(?:morning|afternoon|evening|day)|"
    r"what'?s*\s+up|sup|yo+|greetings|how\s+are\s+(?:you|things|it\s+going)|"
    r"how\s+do\s+you\s+do|nice\s+to\s+meet\s+you|"
    r"thank(?:s|\s+you)|thanks\s+a\s+lot|cheers|"
    r"bye+|goodbye|see\s+you|take\s+care|good\s+night|"
    r"ok+|okay|sure|yes|no+|nah|yep|nope|"
    r"help|/help|/start|"
    r"hola|bonjour|salut|guten\s+(?:tag|morgen)|namaste|salaam|shalom|ciao)\s*$",
    re.IGNORECASE,
)

# --- Out-of-scope patterns ---
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
    r"|(?:tell\s+me\s+a\s+joke)"
    r"|(?:what(?:'?s|\s+is)\s+(?:the\s+)?weather\s+(?:today|tomorrow|like|outside|forecast))"
    r"|(?:how(?:'?s|\s+is)\s+(?:the\s+)?weather)"
    r"|(?:who\s+(?:is|are)\s+the\s+(?:president|ceo|pm|prime\s+minister|king|queen))"
    r"|(?:what\s+time\s+(?:is\s+it|do\s+(?:we|you|they)\s+(?:start|finish|close|open)))"
    r"|(?:translate\s+.*\s+to\s+(?:french|spanish|german|chinese|japanese|hindi|arabic))",
    re.IGNORECASE,
)

# --- Identity patterns ---
# Covers: "who am I", "what is my name", "who are you" (bot identity).
# NOTE: "what is my role" is handled by _ROLE_PATTERN in metadata, not here.
# "who is admin" is handled by _IDENTITY_PATTERN as a bot/knowledge question.
_IDENTITY_PATTERN = re.compile(
    r"(?:what(?:'?s|\s+is)\s+my\s+(?:name|email|username|display\s+name))\b"
    r"|who\s+am\s+i\b"
    r"|my\s+name\b"
    r"|what\s+(?:is|are)\s+my\s+(?:credentials|profile|account\s+details?)\b"
    r"|who\s+(?:are|is)\s+you\b"
    r"|what\s+(?:is|are)\s+your\s+(?:name|purpose|function|role)\b"
    r"|who\s+is\s+(?:the\s+)?(?:admin|owner|manager|lead|administrator)\b",
    re.IGNORECASE,
)

# Identity: greeting + name statement ("hi my name is X", "hey my name is Y")
# Must be checked BEFORE the greeting pattern to avoid "hi" matching as greeting.
_GREETING_NAME_PATTERN = re.compile(
    r"^(?:hi+|hey+|hello+)\s+.*?\bmy\s+name\b",
    re.IGNORECASE,
)

# General conversation: casual statements/questions not matching any specific lane.
# "i have a doubt", "i need help", "can you assist me", etc.
_GENERAL_CONVERSATION_PATTERNS = [
    re.compile(r"^\s*(?:i\s+have\s+(?:an?\s+)?(?:doubt|question|query|issue|problem|concern))\b", re.IGNORECASE),
    re.compile(r"\b(?:can\s+you\s+(?:help|assist|guide)\s+me)\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+need\s+(?:some\s+)?(?:help|assistance|guidance))\b", re.IGNORECASE),
    re.compile(r"^\s*(?:hey\s+(?:there|buddy|bot|assistant))\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:what(?:'?s|\s+is)\s+up)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:sup)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:yo+)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:um+|umm+|hmm+|uh+)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:ok(?:ay)?|sure|yep|nope|nah|cool|nice|great|awesome|alright)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:test(?:ing)?|ping)\s*[!.?]*\s*$", re.IGNORECASE),
]

# Out-of-scope: code/programming requests (catches typos like "pyathon")
_CODE_REQUEST_PATTERN = re.compile(
    r"\bwrite\s+(?:me\s+)?(?:a\s+)?\w*\s*(?:code|program|script|function|game|app)\b"
    r"|(?:create|make|build|generate)\s+(?:me\s+)?(?:a\s+)?\w*\s*(?:code|program|script|function|game|app)\b",
    re.IGNORECASE,
)

# Metadata: document count with common typos ("how manu", "how manyy", etc.)
_DOC_COUNT_TYPO_PATTERN = re.compile(
    r"(?:how\s+man+[uy]+|how\s+many|number\s+of|count\s+of|total\s+(?:number\s+of)?)\s+"
    r"(?:uploaded\s+)?(?:my\s+|the\s+|this\s+)?(?:own\s+)?"
    r"(?:files|docs?|documents?)\s*(?:there|are\s+there|do\s+i\s+have)?\s*$",
    re.IGNORECASE,
)

# --- App-help patterns (questions about the application itself) ---
_APP_HELP_PATTERN = re.compile(
    r"(?:what\s+(?:does|do)\s+(?:this|the)\s+(?:chatbot|assistant|app(?:lication)?|bot)\s+"
    r"(?:do|does|offer|provide|support|know|answer))\b"
    r"|(?:what\s+can\s+(?:i|we|you)\s+do)\b"
    r"|(?:what\s+can\s+(?:i|we)\s+(?:ask|use\s+this\s+for))\b"
    r"|(?:how\s+(?:does\s+)?(?:this|it|the\s+app)\s+work)\b"
    r"|(?:how\s+(?:do|can|should)\s+(?:i|we)\s+(?:use|start|begin)\s+this)\b"
    r"|(?:how\s+do\s+i\s+(?:use|access|open|get\s+started))\b"
    r"|(?:what\s+(?:are\s+)?(?:the\s+)?(?:features?|capabilities|functions?))\b"
    r"|(?:what\s+(?:kind|type)\s+of\s+(?:questions?|things?)\s+(?:can|do)\s+"
    r"(?:i|we|you)\s+(?:ask|answer|handle))\b"
    r"|(?:how\s+do\s+i\s+(?:upload|add|submit)\s+(?:.*?\s+)?(?:document|file)s?)\b"
    r"|(?:who\s+can\s+(?:upload|add|submit)\s+(?:document|file)s?)\b"
    r"|(?:can\s+(?:i|we|members?)\s+upload)\b"
    r"|(?:am\s+i\s+being\s+(?:monitored|tracked|watched))\b"
    r"|(?:do\s+you\s+(?:track|monitor|log)\s+(?:my|us|activity))\b"
    r"|(?:how\s+(?:can|do|should)\s+(?:i|we|you)\s+"
    r"(?:invite|add|send)\s+(?:.*?\s+)?(?:member|user|person|colleague|someone)?)\b"
    r"|(?:how\s+(?:do\s+)?i\s+"
    r"(?:invite|add|onboard)\s+(?:a\s+)?(?:member|user|person|colleague|someone)?)\b"
    r"|(?:tell\s+me\s+about\s+(?:this\s+)?(?:chatbot|assistant|app(?:lication)?|system|bot))\b"
    r"|(?:what\s+(?:is|are)\s+this\s+(?:chatbot|assistant|app(?:lication)?|system|bot))\b",
    re.IGNORECASE,
)

# --- Workspace permission patterns ---
# NOTE: "What is my role?" is handled by _ROLE_PATTERN in metadata, not here.
# This pattern covers action-level permissions (who can upload, can I invite, etc.).
# "How can I upload and ask" is APP_HELP (application usage), not permission.
# "who can" always matches permission; "can I/members" matches permission when
# not preceded by "how" (which makes it a how-to/app-help question).
_WORKSPACE_PERMISSION_PATTERN = re.compile(
    r"(?:who\s+can\s+(?:upload|add|submit|delete|remove|approve|reject|invite|manage|edit|view|read|access|see))\b"
    r"|(?:can\s+(?:i|we|members?|users?|everyone|anyone)\s+(?:only\s+)?"
    r"(?:upload|add|submit|delete|remove|approve|reject|invite|manage|edit|view|read|access|see))\b"
    r"|(?:what\s+(?:can|am\s+i\s+allowed\s+to)\s+(?:i|we)\s+"
    r"(?:do|upload|add|delete|remove|approve|reject|invite|manage|edit|view|read|access))\b"
    r"|(?:who\s+(?:has|have)\s+(?:access|permission|rights?))\b"
    r"|(?:who\s+(?:is|are)\s+(?:allowed|permitted|authorized)\s+to)\b",
    re.IGNORECASE,
)

# "how can I upload" is a how-to question, not a permission question.
_HOW_TO_UPLOAD_PATTERN = re.compile(
    r"(?:how\s+(?:can|do|should)\s+(?:i|we)\s+"
    r"(?:upload|add|submit|invite|add|send))\b",
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
    r"|(?:what\s+have\s+(?:i|we)\s+(?:been\s+)?(?:asking|discussing|talking))\b"
    r"|(?:what\s+have\s+(?:i|we)\s+(?:been\s+)?discussing\s*(?:recently)?\b)"
    r"|(?:what\s+(?:have|did)\s+(?:i|we)\s+"
    r"(?:discussed|covered|talked\s+about|spoken\s+about))\b",
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
    r"(?:\w+\s+)?"  # optional adjective before noun (e.g. "active members")
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

# Role qualifiers for member count/list queries: "how many owners", "list all admins"
_ROLE_OWNER_PATTERN = re.compile(
    r"\b(?:owners?|founders?|workspace\s+owners?)\b",
    re.IGNORECASE,
)

_ROLE_MEMBER_PATTERN = re.compile(
    r"\b(?:members?|team\s*members?|contributors?|employees?)\b",
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

# Typo-tolerant variant: catches "what are doucuments names"
_DOC_LIST_TYPO_PATTERN = re.compile(
    r"what\s+(?:are\s+(?:the\s+)?)?"
    r"(?:my\s+|the\s+|this\s+)?"
    r"(?:files|docs?|doucuments?|documents?)"
    r"\s+names?\s*$",
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

# Description/summary patterns: "description", "give me a description of each document",
# "summary of each document", "summary of each file", "summary of [specific doc name]".
_DOC_DESCRIPTION_PATTERN = re.compile(
    r"(?:can\s+you\s+)?(?:give\s+(?:me\s+)?(?:a\s+)?)?"
    r"(?:description|summary|summery|descrption|descriction)"
    r"\s*(?:of|for|about|on)?\s*"
    r"(?:\d+\s+(?:lines?|sentences?|paragraphs?)\s+(?:of\s+)?)?"
    r"(?:each|every|all|the|this|my)?\s*"
    r"(?:uploaded\s+)?(?:own\s+)?(?:document|file|doc|files|documents)?s?\s*$",
    re.IGNORECASE,
)

# Company name pattern: "what is the name of this company", "what's our company name".
# Includes common typo variants for "company" (comapny, compnay, etc.).
_COMPANY_WORDS = r"(?:comapny|compnay|company|workspace|organization|org|team)"
_COMPANY_NAME_PATTERN = re.compile(
    r"what(?:'?s|\s+is)\s+(?:the\s+)?(?:name\s+(?:of\s+(?:the\s+|this\s+|our\s+)?)?"
    + _COMPANY_WORDS
    + r"|(?:the\s+|this\s+|our\s+)?" + _COMPANY_WORDS + r"\s+name)"
    r"|(?:what\s+(?:is|are)\s+(?:the\s+)?" + _COMPANY_WORDS + r"\s+name)"
    r"|" + _COMPANY_WORDS + r"\s+(?:is\s+)?(?:called|named)",
    re.IGNORECASE,
)

# Specific document summary: "summary of [docname]", "description of [docname]".
_DOC_SPECIFIC_DESCRIPTION_PATTERN = re.compile(
    r"(?:give\s+(?:me\s+)?(?:a\s+)?)?"
    r"(?:description|summary|summery|descrption|descriction)"
    r"\s+(?:of|for|about|on)\s+"
    r"(.+?)\s*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Text normalization for classification
# ---------------------------------------------------------------------------

# Common non-English greetings mapped to English equivalents.
_GREETING_EXPANSIONS: dict[str, str] = {
    "hola": "hello",
    "bonjour": "hello",
    "salut": "hello",
    "guten tag": "hello",
    "guten morgen": "good morning",
    "namaste": "hello",
    "salaam": "hello",
    "shalom": "hello",
    "ciao": "hello",
    "yo yo": "hey",
}


def normalize_for_classification(query: str) -> str:
    """Normalize a query for intent classification.

    Handles: repeated/elongated characters (e.g. "heyyyy" → "hey"),
    extra whitespace, common non-English greetings, and punctuation.
    Applied *before* regex pattern matching so rigid patterns can match
    casual, noisy input.
    """
    text = query.strip()
    if not text:
        return text

    # 1. Unicode normalize (NFKD) to decompose accented characters.
    text = unicodedata.normalize("NFKD", text)

    # 2. Strip combining marks (accents produced by NFKD decomposition).
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    # 3. Lowercase for matching.
    text = text.lower()

    # 4. Collapse runs of 3+ identical letters to 2 (e.g. "heyyy" → "hey").
    # Preserves natural doubles like "ll" in "hello" or "oo" in "book".
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 5. Strip all punctuation (handles "hola!", "hey?", etc.).
    text = re.sub(r"[\W_]+", " ", text)

    # 6. Collapse whitespace.
    text = " ".join(text.split())

    # 7. Expand common non-English greetings.
    for foreign, english in _GREETING_EXPANSIONS.items():
        if text == foreign or text.startswith(foreign + " "):
            text = text.replace(foreign, english, 1)
            break

    return text


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Map LLM router routes to IntentCategory values.
# New 5-route taxonomy: direct, metadata, retrieval, clarification, out_of_scope.
_LLM_ROUTE_TO_CATEGORY = {
    # New routes
    "direct": IntentCategory.GENERAL_CONVERSATION,
    "metadata": IntentCategory.WORKSPACE_METADATA,
    "retrieval": IntentCategory.DOCUMENT_CONTENT,
    "clarification": IntentCategory.AMBIGUOUS,
    "out_of_scope": IntentCategory.OUT_OF_SCOPE,
    # Legacy route names (for backward compat with old mocks/tests)
    "GREETING": IntentCategory.GREETING,
    "IDENTITY_ASSISTANT": IntentCategory.IDENTITY_ASSISTANT,
    "IDENTITY_USER": IntentCategory.IDENTITY_USER,
    "METADATA": IntentCategory.WORKSPACE_METADATA,
    "PERMISSIONS": IntentCategory.PERMISSIONS,
    "DOCUMENT_CONTENT": IntentCategory.DOCUMENT_CONTENT,
    "CONVERSATION_HISTORY": IntentCategory.CONVERSATION_HISTORY,
    "APP_HELP": IntentCategory.APP_HELP,
    "OUT_OF_SCOPE": IntentCategory.OUT_OF_SCOPE,
    "GENERAL_CONVERSATION": IntentCategory.GENERAL_CONVERSATION,
    "NEEDS_CLARIFICATION": IntentCategory.AMBIGUOUS,
}


def _llm_route_to_intent(
    route_result: "RouteResult",
    original_query: str = "",
) -> Intent:
    """Map an LLM router RouteResult to an Intent.

    For METADATA routes, uses regex sub-classification to determine the
    specific metadata sub-intent (member count, doc list, role, etc.).
    """
    from app.retrieval.llm_router import CONFIDENCE_THRESHOLD

    category = _LLM_ROUTE_TO_CATEGORY.get(route_result.route)
    if category is None:
        return Intent(
            category=IntentCategory.AMBIGUOUS,
            needs_clarification=True,
            reason=f"unknown_route:{route_result.route}",
        )

    # Low confidence → force clarification.
    if route_result.confidence < CONFIDENCE_THRESHOLD:
        return Intent(
            category=IntentCategory.AMBIGUOUS,
            needs_clarification=True,
            reason=f"low_confidence:{route_result.route}={route_result.confidence:.2f}",
        )

    needs_clarification = category == IntentCategory.AMBIGUOUS

    # Determine the query to use for downstream processing.
    # If the LLM flagged needs_rewrite and provided a rewritten query, use it.
    effective_query = original_query
    if route_result.needs_rewrite and route_result.query:
        effective_query = route_result.query

    # For METADATA routes, determine the specific sub-intent via regex.
    metadata_sub = None
    member_status = None
    if category == IntentCategory.WORKSPACE_METADATA and original_query:
        member_intent = _classify_member_metadata(effective_query)
        if member_intent is not None:
            return Intent(
                category=category,
                metadata_sub=member_intent.metadata_sub,
                member_status=member_intent.member_status,
                member_role=member_intent.member_role,
                skip_rewrite=True,
                reason=f"llm_router:{route_result.route} sub={member_intent.metadata_sub.value} conf={route_result.confidence:.2f}",
            )
        doc_intent = _classify_document_metadata(effective_query)
        if doc_intent is not None:
            return Intent(
                category=doc_intent.category,
                metadata_sub=doc_intent.metadata_sub,
                skip_rewrite=True,
                reason=f"llm_router:{route_result.route} sub={doc_intent.metadata_sub.value} conf={route_result.confidence:.2f}",
            )

    return Intent(
        category=category,
        needs_clarification=needs_clarification,
        skip_rewrite=category not in (
            IntentCategory.DOCUMENT_CONTENT,
            IntentCategory.DOCUMENT_LIST,
        ),
        rewritten_query=route_result.query if route_result.needs_rewrite and route_result.query else None,
        reason=f"llm_router:{route_result.route} conf={route_result.confidence:.2f}",
    )


def classify_intent_regex(query: str) -> Intent:
    """Regex-only fast-path classification (synchronous, no LLM).

    Returns an Intent when a high-confidence regex match is found, or
    ``None`` (as Intent with category=DOCUMENT_CONTENT as fallback) when
    no pattern matches — signaling the caller should fall through to the
    LLM router.

    Fast-path cases (no LLM needed):
    - Greetings (exact or elongated matches)
    - Out-of-scope (math, geography, obvious general knowledge)
    - Conversation history (specific question phrasings)
    - App help (specific application question phrasings)
    - Workspace permissions (specific action-level questions)
    - Metadata queries (member count/list, document count/list, role)
    - Document comparison
    """
    q = query.strip()
    if not q:
        return Intent(
            category=IntentCategory.AMBIGUOUS,
            needs_clarification=True,
            reason="empty_query",
        )

    q_normalized = normalize_for_classification(q)

    # --- 0a. Greeting + name statement ("hi my name is X") — before greeting ---
    # Must be checked BEFORE the greeting pattern because "hi my name is aarya"
    # starts with "hi" which would match as a greeting.
    if _GREETING_NAME_PATTERN.search(q) or _GREETING_NAME_PATTERN.search(q_normalized):
        return Intent(
            category=IntentCategory.IDENTITY_USER,
            skip_rewrite=True,
            reason="greeting_name_statement",
        )

    # --- 0. Greeting (highest priority — obvious single words/phrases) ---
    if _GREETING_PATTERN.search(q) or _GREETING_PATTERN.search(q_normalized):
        return Intent(
            category=IntentCategory.GREETING,
            skip_rewrite=True,
            reason="greeting",
        )

    # --- 0b. Ambiguous / anaphoric references ---
    _ANAPHORIC_PATTERN = re.compile(
        r"(?:tell\s+me\s+about\s+(?:that|it|them|this))\b"
        r"|(?:what\s+about\s+(?:that|it|them|this))\b"
        r"|(?:how\s+about\s+(?:that|it|them|this))\b"
        r"|(?:explain\s+(?:that|it|them|this))\b"
        r"|(?:describe\s+(?:that|it|them|this))\b",
        re.IGNORECASE,
    )
    if _ANAPHORIC_PATTERN.search(q) or _ANAPHORIC_PATTERN.search(q_normalized):
        return Intent(
            category=IntentCategory.AMBIGUOUS,
            needs_clarification=True,
            reason="anaphoric_reference",
        )

    # --- 1. Out-of-scope (obvious general knowledge) ---
    if _is_out_of_scope(q) or _is_out_of_scope(q_normalized):
        return Intent(
            category=IntentCategory.OUT_OF_SCOPE,
            skip_rewrite=True,
            reason="general_knowledge",
        )

    # --- 1b. Bot identity fast-path ("who are you", "what are you") ---
    # Unambiguous — no LLM call needed.  Must be the full question,
    # not a prefix like "what is you doing" which is a different intent.
    if re.search(r"(?:who|what)\s+(?:are|is)\s+you\s*[?!.,]*\s*$", q, re.IGNORECASE) or \
       re.search(r"(?:who|what)\s+(?:are|is)\s+you\s*[?!.,]*\s*$", q_normalized, re.IGNORECASE):
        return Intent(
            category=IntentCategory.IDENTITY_ASSISTANT,
            skip_rewrite=True,
            reason="identity_bot_fast_path",
        )

    # --- 1c. Code/programming requests (catches typos like "pyathon") ---
    if _CODE_REQUEST_PATTERN.search(q) or _CODE_REQUEST_PATTERN.search(q_normalized):
        return Intent(
            category=IntentCategory.OUT_OF_SCOPE,
            skip_rewrite=True,
            reason="code_request",
        )

    # --- 1d. General conversation (casual statements/questions) ---
    for _gcp in _GENERAL_CONVERSATION_PATTERNS:
        if _gcp.search(q) or _gcp.search(q_normalized):
            return Intent(
                category=IntentCategory.GENERAL_CONVERSATION,
                skip_rewrite=True,
                reason="general_conversation",
            )

    # NOTE: Identity queries deliberately NOT in the fast-path.
    # "who are you" (IDENTITY_ASSISTANT) and "my name is X" (IDENTITY_USER)
    # need different handling that only the LLM router can provide.
    # The regex _IDENTITY_PATTERN still exists for backward compat in tests
    # but is not used in the fast-path — all identity queries go to the LLM.

    # --- 2. Workspace permission (before app_help to avoid overlap) ---
    if ((_WORKSPACE_PERMISSION_PATTERN.search(q) or _WORKSPACE_PERMISSION_PATTERN.search(q_normalized))
            and not (_HOW_TO_UPLOAD_PATTERN.search(q) or _HOW_TO_UPLOAD_PATTERN.search(q_normalized))):
        return Intent(
            category=IntentCategory.WORKSPACE_PERMISSION,
            skip_rewrite=True,
            reason="workspace_permission_query",
        )

    # --- 4. Conversation history ---
    if _CONVERSATION_HISTORY_PATTERN.search(q) or _CONVERSATION_HISTORY_PATTERN.search(q_normalized):
        sub = _classify_conversation_history(q)
        return Intent(
            category=IntentCategory.CONVERSATION_HISTORY,
            conversation_history_sub=sub,
            skip_rewrite=True,
            reason="conversation_history_query",
        )

    # --- 3. App help ---
    if _APP_HELP_PATTERN.search(q) or _APP_HELP_PATTERN.search(q_normalized):
        return Intent(
            category=IntentCategory.APP_HELP,
            skip_rewrite=True,
            reason="app_help_query",
        )

    # --- 7. Document comparison (before content — comparison is a distinct lane) ---
    if _is_document_comparison(q) or _is_document_comparison(q_normalized):
        return Intent(
            category=IntentCategory.DOCUMENT_CONTENT,
            query_shape=QueryShape.COMPARISON,
            skip_rewrite=False,
            reason="document_comparison",
        )

    # --- 8. Metadata: member queries (exact pattern match) ---
    member_intent = _classify_member_metadata(q) or _classify_member_metadata(q_normalized)
    if member_intent is not None:
        return member_intent

    # --- 9. Metadata: document list/count queries (exact pattern match) ---
    doc_intent = _classify_document_metadata(q) or _classify_document_metadata(q_normalized)
    if doc_intent is not None:
        return doc_intent

    # --- 9b. Metadata: document count with common typos ("how manu docs") ---
    if _DOC_COUNT_TYPO_PATTERN.search(q) or _DOC_COUNT_TYPO_PATTERN.search(q_normalized):
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_COUNT,
            skip_rewrite=True,
            reason="doc_count_typo",
        )

    # --- 10. No regex match — signal to fall through to LLM router ---
    # Return DOCUMENT_CONTENT as default; the caller replaces this with
    # the LLM router's result.
    return Intent(
        category=IntentCategory.DOCUMENT_CONTENT,
        reason="regex_fallback_to_llm",
    )


async def _llm_classify_metadata_subintent(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> MetadataSubIntent | None:
    """LLM fallback for metadata sub-classification when regex doesn't match.

    When the outer LLM router identifies a question as METADATA but the
    regex sub-classifier can't determine the specific operation (doc_list,
    doc_count, member_count, etc.), this function asks the LLM — including
    recent conversation history for pronoun/reference resolution.

    Returns the specific MetadataSubIntent, or None if the LLM call fails.
    """
    import json as _json

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

    system_prompt = (
        "You are a metadata sub-classifier for a company knowledge assistant.\n"
        "A question has been identified as a metadata/workspace question.\n"
        "Determine which specific metadata operation is being requested.\n\n"
        "Possible operations (use the EXACT key):\n"
        "- doc_list: List, show, or name document filenames/names.\n"
        "  Includes: 'what are the documents', 'what documents are there',\n"
        "  'name any documents', 'documents details', 'what are they' (when\n"
        "  history shows a document-related question), etc.\n"
        "- doc_count: Count how many documents exist.\n"
        "- doc_page_count: Count pages in documents.\n"
        "- doc_description: Get description or summary of documents.\n"
        "- member_count: Count workspace members.\n"
        "- member_list: List workspace members.\n"
        "- role: Ask about the user's own role/permissions.\n"
        "- company_name: Ask about the workspace or company name.\n\n"
        "Rules:\n"
        "1. If the question uses pronouns (they, those, these, it), use the\n"
        "   conversation history to resolve what they refer to.\n"
        "2. 'what are they' after a document count question → doc_list\n"
        "3. 'what are those documents' / 'what are documents present' → doc_list\n"
        "4. 'name any two documents' / 'name these 7 documents' → doc_list\n"
        "5. 'documents details' / 'give me 3 document uploaded' → doc_list\n"
        "6. Return ONLY a JSON object, no markdown fences.\n\n"
        'Return: {"sub_intent": "<key>", "confidence": <0.0-1.0>}'
    )

    # Build context with history for pronoun resolution.
    context_lines: list[str] = []
    if history:
        for turn in history[-4:]:  # last 2 pairs max
            label = "User" if turn.get("role") == "user" else "Assistant"
            context_lines.append(f"{label}: {turn.get('content', '')}")

    user_msg = f"User question: {query}"
    if context_lines:
        user_msg = (
            "Recent conversation:\n"
            + "\n".join(context_lines)
            + f"\n\n{user_msg}"
        )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_msg),
    ]
    completion = Completion()

    try:
        async for _token in provider.stream(messages, completion=completion):
            pass

        response_text = completion.text.strip()
        if not response_text:
            return None

        # Parse structured output.
        import re as _re
        json_match = _re.search(r'\{[^}]+\}', response_text)
        if json_match:
            data = _json.loads(json_match.group())
            sub = str(data.get("sub_intent", "")).strip()
            try:
                return MetadataSubIntent(sub)
            except ValueError:
                logger.warning(
                    "LLM metadata sub-classifier returned unknown sub_intent '{sub}'",
                    sub=sub,
                )
                return None

    except Exception as exc:
        logger.warning(
            "LLM metadata sub-classifier failed: {error}",
            error=str(exc)[:200],
        )

    return None


async def classify_intent(
    query: str,
    *,
    history: list[dict[str, str]] | None = None,
    workspace_id: "uuid.UUID | None" = None,
) -> Intent:
    """Classify a user query into a routing intent.

    LLM-first routing:
    1. Check the routing cache for a previously computed route.
    2. On cache miss, call the LLM router with workspace knowledge.
    3. If the LLM call fails, fall back to the regex classifier.

    The regex layer is no longer the default path — it exists only as a
    failure-mode fallback so the system degrades gracefully when the LLM
    is unavailable.

    Parameters
    ----------
    query:
        The raw user query.
    history:
        Optional recent conversation turns for context.
    workspace_id:
        The workspace UUID.  Used for cache keying and to inject
        workspace-specific knowledge into the LLM router prompt.
    """
    import uuid as _uuid

    q_normalized = normalize_for_classification(query)

    # --- Stage 0: Check routing cache ---
    if workspace_id is not None:
        from app.retrieval.routing_cache import get_cached_route, set_cached_route

        cached = get_cached_route(workspace_id, q_normalized)
        if cached is not None:
            route, reasoning, confidence = cached
            # Build a synthetic RouteResult and map it.
            from app.retrieval.llm_router import RouteResult
            cached_result = RouteResult(
                route=route,
                reasoning=f"cache_hit:{reasoning}",
                confidence=confidence,
                status="success",
            )
            return _llm_route_to_intent(cached_result, original_query=query)

    # --- Stage 0.5: Deterministic fast-path for obvious cases ---
    # These patterns are unambiguous and should never need an LLM call.
    # Checking them here avoids unnecessary LLM latency and prevents
    # misclassification for common, obvious inputs.
    fast_path = classify_intent_regex(query)
    if fast_path.category not in (
        IntentCategory.DOCUMENT_CONTENT,  # regex fallback, not a real match
        IntentCategory.AMBIGUOUS,           # needs LLM disambiguation
    ):
        return fast_path

    # --- Stage 1: LLM router (primary path) ---
    from app.retrieval.llm_router import route_with_llm

    # Build workspace knowledge context for the LLM prompt.
    ws_knowledge_context: str | None = None
    if workspace_id is not None:
        try:
            from app.retrieval.workspace_knowledge import get_workspace_knowledge
            from app.security.rls import tenant_session

            async with tenant_session(
                workspace_id=workspace_id,
                user_id=_uuid.UUID(int=0),  # system-level read, no user scope needed
            ) as db:
                knowledge = await get_workspace_knowledge(db, workspace_id)
                ws_knowledge_context = knowledge.to_prompt_context()
        except Exception:
            # If knowledge loading fails, proceed without workspace context.
            pass

    llm_result = await route_with_llm(
        query=q_normalized,
        history=history,
        workspace_knowledge_context=ws_knowledge_context,
    )

    # Cache the LLM result (unless it's degraded).
    if (
        workspace_id is not None
        and llm_result.status == "success"
        and llm_result.confidence >= 0.3
    ):
        from app.retrieval.routing_cache import set_cached_route
        set_cached_route(
            workspace_id=workspace_id,
            normalized_message=q_normalized,
            route=llm_result.route,
            reasoning=llm_result.reasoning,
            confidence=llm_result.confidence,
        )

    # If the LLM call succeeded, use its result.
    if llm_result.status == "success":
        intent = _llm_route_to_intent(llm_result, original_query=query)

        # If METADATA but no sub-intent resolved via regex, try LLM sub-classification.
        # This handles pronouns ("what are they") and unusual phrasings
        # ("documents details", "name any two documents") that regex misses.
        if (
            intent.category == IntentCategory.WORKSPACE_METADATA
            and intent.metadata_sub is None
        ):
            sub = await _llm_classify_metadata_subintent(query, history)
            if sub is not None:
                return Intent(
                    category=intent.category,
                    metadata_sub=sub,
                    skip_rewrite=True,
                    reason=f"llm_metadata_sub:{sub.value}",
                )

        return intent

    # --- Stage 2: Regex fallback (LLM failed) ---
    logger.warning(
        "LLM router degraded, falling back to regex for query: {query}",
        query=q_normalized[:100],
    )
    return classify_intent_regex(query)


def _is_out_of_scope(q: str) -> bool:
    """Detect obvious general-knowledge questions."""
    if _MATH_PATTERN.search(q):
        return True
    if _CAPITAL_OF_PATTERN.search(q):
        return True
    if _GENERAL_KNOW_PATTERN.search(q):
        return True
    return False


def _is_document_comparison(q: str) -> bool:
    """Detect document comparison queries."""
    return bool(_COMPARISON_PATTERNS.search(q))


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
    """Detect member count/list queries with status and role filters."""
    # Don't match if this is a topic-qualified content question.
    if _TOPIC_QUALIFIERS.search(q):
        return None

    # Must match a member-related pattern OR a role-specific count pattern.
    # The role pattern catches "how many owners", "how many admins", etc.
    # which use a role keyword instead of a generic member noun.
    is_member_count = bool(_MEMBER_COUNT_PATTERN.search(q))
    is_member_list = bool(_MEMBER_LIST_PATTERN.search(q))
    is_role_count = bool(
        re.search(
            r"(?:how\s+many|number\s+of|count\s+of|total\s+(?:number\s+of)?)\s+"
            r"(?:\w+\s+)?"
            r"(?:owners?|admins?|administrator)",
            q, re.IGNORECASE,
        )
    )

    # Also detect follow-up style: "how many are invited?" (no "members" word)
    if not is_member_count and not is_member_list and not is_role_count:
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
                    category=IntentCategory.WORKSPACE_METADATA,
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

    # Determine role filter.
    # "members" is the generic term for all workspace membership — do NOT
    # restrict it to the MEMBER role.  Only an explicit, non-generic role
    # keyword (owner, admin, contributor, etc.) narrows the count to one role.
    role = None
    if _ROLE_OWNER_PATTERN.search(q):
        role = "OWNER"
    elif re.search(
        r"\b(?:admins?|administrators?)\b",
        q, re.IGNORECASE,
    ):
        # No ADMIN role exists in this system — handled downstream, but mark
        # the intent so the handler can respond appropriately.
        role = "ADMIN"
    elif _ROLE_MEMBER_PATTERN.search(q) and not (
        # Generic member phrasing that should mean "all members", not role=MEMBER:
        # "how many members", "number of members", "count of members", "team members"
        re.search(
            r"(?:how\s+many|number\s+of|count\s+of|total\s+(?:number\s+of)?)\s+"
            r"(?:\w+\s+)?(?:members?|team\s*members?)\b",
            q, re.IGNORECASE,
        )
    ):
        role = "MEMBER"

    if is_member_count or is_role_count:
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.MEMBER_COUNT,
            member_status=status,
            member_role=role,
            skip_rewrite=True,
            reason=f"member_count status={status or 'all'} role={role or 'all'}",
        )
    else:
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.MEMBER_LIST,
            member_status=status,
            member_role=role,
            skip_rewrite=True,
            reason=f"member_list status={status or 'all'} role={role or 'all'}",
        )


def _classify_document_metadata(q: str) -> Intent | None:
    """Detect document count/list/page-count queries."""
    # Don't match topic-qualified content questions.
    if _TOPIC_QUALIFIERS.search(q):
        return None

    # Page count must be checked before general count to avoid false matches.
    if _DOC_PAGE_COUNT_PATTERN.search(q):
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_PAGE_COUNT,
            skip_rewrite=True,
            reason="doc_page_count",
        )

    if _DOC_COUNT_PATTERN.search(q):
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_COUNT,
            skip_rewrite=True,
            reason="doc_count",
        )

    if _DOC_LIST_PATTERN.search(q):
        return Intent(
            category=IntentCategory.DOCUMENT_LIST,
            metadata_sub=MetadataSubIntent.DOC_LIST,
            skip_rewrite=True,
            reason="doc_list",
        )

    # Typo-tolerant: "what are doucuments names"
    if _DOC_LIST_TYPO_PATTERN.search(q):
        return Intent(
            category=IntentCategory.DOCUMENT_LIST,
            metadata_sub=MetadataSubIntent.DOC_LIST,
            skip_rewrite=True,
            reason="doc_list_typo",
        )

    # Role query.
    if _ROLE_PATTERN.search(q):
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.ROLE,
            skip_rewrite=True,
            reason="role_query",
        )

    # Company/workspace name query.
    if _COMPANY_NAME_PATTERN.search(q):
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.COMPANY_NAME,
            skip_rewrite=True,
            reason="company_name",
        )

    # Document description/summary (generic: "description", "summary of each document").
    if _DOC_DESCRIPTION_PATTERN.search(q):
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_DESCRIPTION,
            skip_rewrite=True,
            reason="doc_description",
        )

    # Specific document description: "summary of [docname]".
    desc_match = _DOC_SPECIFIC_DESCRIPTION_PATTERN.search(q)
    if desc_match:
        return Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_DESCRIPTION,
            skip_rewrite=True,
            reason="doc_description_specific",
        )

    return None


__all__ = [
    "ConversationHistorySubIntent",
    "Intent",
    "IntentCategory",
    "MetadataSubIntent",
    "QueryShape",
    "classify_intent",
    "classify_intent_regex",
    "classify_query_shape",
    "normalize_for_classification",
]
