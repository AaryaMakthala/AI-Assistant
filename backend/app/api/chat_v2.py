"""Chat endpoints — grounded retrieval + streaming and session management.

Two interfaces to the same retrieval pipeline:

1. ``POST /chat`` — SSE streaming endpoint the frontend calls.  Returns a typed
   event stream (``session``, ``sources``, ``token``, ``citations``, ``done``,
   ``error``) matching the contract in ``frontend/src/lib/api/types.ts``.  Creates
   or reuses a chat session and persists every turn.

2. ``POST /chat/grounded`` — synchronous JSON endpoint used by the test suite and
   for programmatic access.  Returns the complete answer in one response.

Both endpoints share the same pipeline:

    authenticate → workspace membership → Phase 5 retrieval (hybrid → RRF → rerank)
        → Layer-1 grounding check
            ├── grounded    → build prompt from the final chunks → LLM → cited answer
            └── ungrounded  → honest refusal, NO LLM call (CLAUDE.md 8.3)

Session management:
  - ``GET  /chat/sessions``                    — list the user's sessions
  - ``GET  /chat/sessions/{id}/messages``       — load a session's transcript
  - ``DELETE /chat/sessions/{id}``              — delete a session and its messages
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

UTC = timezone.utc
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, insert, select

from app.llm.utils import strip_think_tags as _strip_think_tags
from app.llm.utils import stream_think_filtered as _stream_think_filtered

from app.api.dependencies import get_generic_llm
from app.api.workspace_deps import assert_workspace_role
from app.db.models import ChatMessage, ChatSession, Document
from app.llm.base import Completion, LLMError, LLMProvider
from app.rag.prompts import build_messages
from app.retrieval.intent import (
    _DOC_SPECIFIC_DESCRIPTION_PATTERN,
    ConversationHistorySubIntent,
    Intent,
    IntentCategory,
    MetadataSubIntent,
    classify_query_shape,
)
from app.retrieval.pipeline import RetrievedChunk, retrieve
from app.retrieval.query_rewrite import ChatTurn
from app.retrieval.query_understanding import understand_query
from app.retrieval.refusals import ResponseReason, refusal_message
from app.security.auth import CurrentPrincipal
from app.security.rate_limit import CHAT_RATE_LIMIT, limiter
from app.security.rls import tenant_session

router = APIRouter(prefix="/chat", tags=["chat"])

# Kept for backward compatibility.
REFUSAL_NO_EVIDENCE = refusal_message(ResponseReason.NO_EVIDENCE)
REFUSAL_NOT_RELEVANT = refusal_message(ResponseReason.NOT_RELEVANT)
REFUSAL_ANSWER = REFUSAL_NO_EVIDENCE

#: Matches "[1]" / "[2, 3]" / "[1][4]" in the model's answer.
_CITATION_REF = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(answer: str) -> list[int]:
    """The bracketed source numbers `answer` refers to, in order of first appearance.

    Raw and unvalidated — the caller decides what range is legitimate.
    """
    numbers: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_REF.finditer(answer):
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number not in seen:
                seen.add(number)
                numbers.append(number)
    return numbers


def _log_chat_diag(
    *,
    workspace_id,
    intent: str,
    reason: str = "",
    grounded: bool = True,
    provider: str = "",
    candidates: int = 0,
    final: int = 0,
    tokens: int = 0,
    retrieval_called: bool = False,
    diag: dict[str, float] | None = None,
) -> None:
    """Emit a single structured per-request diagnostic line.

    Called once per chat request at every terminal exit of the streaming
    endpoint so operators can split a given request's cost and latency by
    stage without correlating several ad-hoc logs.  Stage durations are
    milliseconds; keys that a fast path never touched default to 0.0.
    """
    d = diag or {}
    request_started = d.get("request_started", time.perf_counter())
    total_ms = (time.perf_counter() - request_started) * 1000.0
    logger.info(
        "chat_request intent={intent} reason={reason} retrieval_called={rc} "
        "grounded={grounded} provider={provider} candidates={candidates} "
        "final={final} tokens={tokens} "
        "qu_ms={qu_ms:.2f} retrieval_ms={retrieval_ms:.2f} total_ms={total_ms:.2f} "
        "workspace={ws}",
        intent=intent,
        reason=reason,
        rc=retrieval_called,
        grounded=grounded,
        provider=provider,
        candidates=candidates,
        final=final,
        tokens=tokens,
        qu_ms=d.get("qu_ms", 0.0),
        retrieval_ms=d.get("retrieval_ms", 0.0),
        total_ms=total_ms,
        ws=workspace_id,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _refine_intent_from_qu(
    *,
    qu_result: object,
    original_query: str,
) -> Intent:
    """Refine a QU classification with regex-based deterministic checks.

    QU uses a 5-category model (greeting, workspace_metadata, document_content,
    off_topic, needs_clarification) that may misclassify certain query types:
    - Identity questions ("who are you", "what is your name") may be classified
      as greeting or off_topic instead of identity_assistant.
    - Personal-name queries ("my name is X", "what is my name") may be
      classified as document_content or off_topic; the regex fast-path routes
      them to the personal-name boundary before any retrieval.
    - Capability requests ("write code", "create a file") may be classified as
      off_topic, which is correct, but we need to distinguish them from general
      knowledge questions for a more specific refusal.

    This function applies regex-based deterministic checks on top of QU's
    classification to refine the intent where the regex patterns are confident.
    """
    from app.retrieval.intent import classify_intent_regex
    from app.retrieval.query_understanding import QueryUnderstanding
    from app.retrieval.intent import IntentCategory as _IC

    # Type the QU result properly.
    if not isinstance(qu_result, QueryUnderstanding):
        raise TypeError(f"Expected QueryUnderstanding, got {type(qu_result)}")

    # Start with QU's classification.
    intent = Intent(
        category=qu_result.intent,
        needs_clarification=(qu_result.intent == _IC.AMBIGUOUS),
        reason=f"query_understanding conf={qu_result.confidence:.2f}",
    )

    # Use regex to refine specific cases.
    regex_intent = classify_intent_regex(original_query)

    # If regex is confident about identity, greeting, OR metadata sub-intent,
    # use it to refine QU's classification.  QU's 5-category model correctly
    # identifies workspace_metadata questions but does not populate the specific
    # sub-intent (doc_count, doc_list, member_count, etc.) that the metadata
    # handler needs — the regex classifier does.
    #
    # Priority order: identity > greeting > metadata-refined > QU's original.
    #
    # This handles:
    #   "who are you" → identity_assistant (not greeting/off_topic)
    #   "hi" → greeting (not document_content)
    #   "how many files are there" → workspace_metadata + metadata_sub=doc_count
    #   (even when QU classifies it as workspace_metadata, the regex populates
    #    the sub-intent the handler needs)
    if (
        regex_intent.category
        in (_IC.IDENTITY_ASSISTANT, _IC.IDENTITY_USER, _IC.GREETING)
    ) or regex_intent.reason == "personal_name_boundary":
        return regex_intent

    # Metadata refinement: when QU classified the query as workspace_metadata or
    # document_list and the regex classifier can resolve the specific sub-intent,
    # use the regex's richer Intent (with metadata_sub populated).
    if regex_intent.category in (
        _IC.WORKSPACE_METADATA,
        _IC.DOCUMENT_LIST,
    ) and regex_intent.metadata_sub is not None:
        # Use the regex's Intent which has the specific sub-intent set.
        # Build a fresh Intent that merges the regex's sub-intent with QU's
        # confidence for traceability (the regex Intent is frozen).
        return Intent(
            category=regex_intent.category,
            metadata_sub=regex_intent.metadata_sub,
            member_status=regex_intent.member_status,
            member_role=regex_intent.member_role,
            skip_rewrite=regex_intent.skip_rewrite,
            reason=f"query_understanding conf={qu_result.confidence:.2f} + regex:{regex_intent.reason}",
        )

    # --- ISSUE 2 fallback: typo-tolerant metadata sub-intent inference ---
    # When QU classified the query as workspace_metadata or DOCUMENT_LIST
    # but the regex couldn't resolve the specific sub-intent (due to typos,
    # unusual phrasing, etc.), try to infer the sub-intent from the corrected
    # query.  This avoids sending clearly-metadata queries through the
    # retrieval + LLM pipeline when we have enough signal to answer from DB.
    if qu_result.intent in (_IC.WORKSPACE_METADATA, _IC.DOCUMENT_LIST):
        inferred_sub = _infer_metadata_sub_from_query(qu_result.corrected_query)
        if inferred_sub is not None:
            from app.retrieval.intent import MetadataSubIntent as _MSI
            return Intent(
                category=qu_result.intent,
                metadata_sub=inferred_sub,
                skip_rewrite=True,
                reason=(
                    f"query_understanding conf={qu_result.confidence:.2f}"
                    f" + inferred_sub:{inferred_sub.value}"
                ),
            )

    # --- Degraded-QU fallback: honor a confident regex classification ---
    # When QU itself degraded (LLM failure / parse failure / truncation) it
    # returns DOCUMENT_CONTENT with confidence 0.0 for *everything*, so a query
    # like "i have an doubt" or "write a pyathon code" would otherwise fall
    # through to retrieval with no evidence it is really general conversation
    # or out-of-scope.  When QU is clearly degraded, trust a confident regex
    # classification.  When QU ran normally (confidence >= 0.5), trust QU even
    # if the regex disagrees — a healthy QU resolves "can you help me with the
    # vacation policy" better than a substring regex ("can you help me").
    if (
        qu_result.intent == _IC.DOCUMENT_CONTENT
        and qu_result.confidence < 0.5
        and regex_intent.category not in (
            _IC.DOCUMENT_CONTENT,
            _IC.AMBIGUOUS,
        )
    ):
        return regex_intent

    return intent


def _infer_metadata_sub_from_query(query: str) -> "MetadataSubIntent | None":
    """Infer a metadata sub-intent from the corrected query string.

    Used as a fallback when regex classification doesn't match (e.g. due to
    typos) but QU correctly identified the query as workspace_metadata.
    Returns a MetadataSubIntent if a confident match is found, None otherwise.
    """
    from app.retrieval.intent import MetadataSubIntent as _MSI
    q = query.strip().lower()

    # Company/workspace name
    if re.search(r"\b(?:name|title)\b.*\b(?:workspace|company|org)\b", q) or \
       re.search(r"\b(?:workspace|company|org)\b.*\b(?:name|title)\b", q):
        return _MSI.COMPANY_NAME

    # Member count/list — accept "member" or "members"
    if re.search(r"\bmembers?\b", q):
        if re.search(r"\b(?:how\s+\w*(?:many|manu|much)|count|number)\b", q):
            return _MSI.MEMBER_COUNT
        if re.search(r"\b(?:list|show|name|who)\b", q):
            return _MSI.MEMBER_LIST
        return _MSI.MEMBER_COUNT

    # Role/permission
    if re.search(r"\b(?:role|permission|access)\b", q):
        return _MSI.ROLE

    # Document page count — accept "page" or "pages"
    if re.search(r"\bpages?\b", q):
        return _MSI.DOC_PAGE_COUNT

    # Document description/summary
    if re.search(r"\b(?:description|summary|summarize|describe)\b", q):
        return _MSI.DOC_DESCRIPTION

    # Document list: questions about what docs exist, their names, etc.
    # Accept singular and plural forms: document/documents, file/files, doc/docs
    if re.search(r"\b(?:what|which|list|show|name)\b.*\b(?:documents?|files?|docs?)\b", q) or \
       re.search(r"\b(?:documents?|files?|docs?)\b.*\b(?:present|presnt|exist|are|have|there)\b", q):
        # Differentiate: "how many" → count, "what are" → list
        # Typo-tolerant: "how manu" or "how meny" via how\s+\w*
        if re.search(r"\b(?:how\s+\w*(?:many|manu|much)|count|number)\b", q):
            return _MSI.DOC_COUNT
        return _MSI.DOC_LIST

    # Document count: "how many" (typo-tolerant) + docs/files context
    if re.search(r"\b(?:how\s+\w*(?:many|manu|much)|count|number)\b", q) and \
       re.search(r"\b(?:documents?|files?|docs?|uploaded|there)\b", q):
        return _MSI.DOC_COUNT

    # Generic "what are the docs" / "docs present" → doc_list
    if re.search(r"\b(?:what|which)\b.*\b(?:documents?|files?|docs?)\b", q):
        return _MSI.DOC_LIST

    return None


def _is_capability_request(question: str) -> bool:
    """Detect requests for capabilities the assistant doesn't provide.

    Catches: write code, create files, design posters, generate images,
    build websites, make apps, etc. — things a document Q&A assistant
    should refuse with a capability-specific message rather than the
    generic out-of-scope response.
    """
    q = question.strip().lower()

    # Code/programming requests
    if re.search(
        r"\b(?:write|create|make|build|generate|develop|code|program)\s+(?:me\s+)?"
        r"(?:a\s+)?\w*\s*(?:code|program|script|function|game|app|software|application)\b",
        q,
    ):
        return True

    # File creation requests
    if re.search(
        r"\b(?:create|make|generate|produce|build|write)\s+(?:me\s+)?"
        r"(?:a\s+)?\w*\s*(?:file|document|pdf|image|picture|graphic|poster|drawing|logo)\b",
        q,
    ):
        return True

    # Design/creative requests
    if re.search(
        r"\b(?:design|create|make|generate|create|draw|paint|illustrate|compose)\s+"
        r"(?:me\s+)?(?:a\s+)?\w*\s*(?:poster|logo|image|graphic|design|banner|illustration|art|drawing)|"
        r"\b(?:can|could|would|will)\s+you\s+(?:design|create|make|generate|build|write|develop)\b",
        q,
    ):
        return True

    # Website/app building requests
    if re.search(
        r"\b(?:build|create|make|generate|develop|design|code)\s+(?:me\s+)?"
        r"(?:a\s+)?\w*\s*(?:website|web\s+site|page|landing|portfolio|blog|store)\b",
        q,
    ):
        return True

    # General "can you make/create/do X" where X is a creative deliverable
    if re.search(
        r"\b(?:can|could|would|will)\s+you\s+(?:make|create|generate|build|design|write|draw|compose|produce)\b",
        q,
    ):
        return True

    return False


# ---------------------------------------------------------------------------
# Person-info and injection gates
# ---------------------------------------------------------------------------

# Questions that ask for one specific person's identity, role, or contact
# details (who is the owner/CEO, who uploaded this document, who handles X).
# These are plausibly answered from org charts / team directories in approved
# documents, so they must go through the evidence path (RAG) — never be
# answered from the LLM's prior knowledge, and never fabricated when no
# verified source exists.  Matches the intent of relevance.py's
# _CONTACT_PERSONNEL_PATTERN, which already passes personnel questions
# through the relevance gate (Layer 1 gives relevant=True).
_PERSON_INFO_PATTERN = re.compile(
    r"\bwho\s+(?:is|are|was|were)\s+(?:the\s+)?"
    r"(?:owner|co-?owner|ceo|founder|co-?founder|admin|administrator|"
    r"manager|lead|leader|head|director|supervisor|recruiter|hr)\b"
    r"|(?:who\s+(?:uploaded|added|created|wrote|authored|submitted|published)"
    r"\s(?:the|this|that|our))"
    r"|(?:who\s+is\s+the\s+(?:author|creator|uploader)\s+of)"
    r"|(?:who\s+(?:handles?|manages?|deals?\s+with|is\s+(?:responsible|in\s+charge)"
    r"\s+(?:for|of)))"
    r"|(?:who\s+(?:should\s+)?(?:i\s+)?(?:contact|email|call|reach\s+out\s+to|"
    r"talk\s+to|get\s+in\s+touch\s+with))"
    r"|(?:what(?:'?s|\s+is|\s+are)\s+"
    r"(?:my|mine|your|yours|youe|you|yor|yr|ur|ours?|his|hers?|theirs?)\s+names?)",
    re.IGNORECASE,
)


def _is_person_info_request(question: str) -> bool:
    """Detect questions asking about a specific person's identity or contact."""
    q = question.strip()
    if not q:
        return False
    return bool(_PERSON_INFO_PATTERN.search(q))


_INJECTION_PATTERN = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions"
    r"|ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+prompts?"
    r"|ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+context"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above|earlier)"
    r"|forget\s+(?:all\s+)?(?:your\s+)?(?:previous|prior|above|earlier)"
    r"|reveal\s+(?:your|the)\s+(?:system|initial)\s+prompts?"
    r"|show\s+me\s+(?:your|the)\s+(?:system|initial)\s+prompts?"
    r"|output\s+(?:your|the)\s+(?:system|initial)\s+prompts?"
    r"|you\s+are\s+now\s+.*\b(?:no\s+longer|acting\s+as)\b"
    r"|do\s+not\s+follow\s+(?:your\s+)?(?:previous|initial)\s+instructions",
    re.IGNORECASE,
)


def _is_injection_attempt(question: str) -> bool:
    """Detect prompt-injection attempts in the user's own message."""
    q = question.strip()
    if not q:
        return False
    return bool(_INJECTION_PATTERN.search(q))


class GroundedChatRequest(BaseModel):
    """A question for the workspace's knowledge base (synchronous endpoint)."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)


class ChatStreamRequest(BaseModel):
    """The SSE streaming chat request body (what the frontend sends)."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    session_id: uuid.UUID | None = None


class Source(BaseModel):
    """One backend-constructed citation (CLAUDE.md 8.4).

    Built from the chunk rows actually sent to the LLM — the LLM contributes
    nothing to this list.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None = None
    section_title: str | None = None
    content: str
    #: Cross-encoder score that earned this chunk a place in the context.
    rerank_score: float


class GroundedChatResponse(BaseModel):
    """The answer plus enough metadata to audit how it was grounded."""

    answer: str
    grounded: bool
    insufficient_evidence: bool
    sources: list[Source]
    provider: str = ""
    model: str = ""


class ChatSessionResponse(BaseModel):
    """Session as the frontend renders it in the sidebar."""

    id: str
    title: str | None = None
    created_at: str
    updated_at: str


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionResponse]


class ChatMessageResponse(BaseModel):
    """One persisted turn, matching the frontend ``ChatMessage`` type."""

    id: str
    role: str
    content: str
    citations: list[dict] = Field(default_factory=list)
    created_at: str
    incomplete: bool = False
    routes: list[str] = Field(default_factory=list)
    sql_query: str = ""


class ChatMessageListResponse(BaseModel):
    messages: list[ChatMessageResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source(chunk: RetrievedChunk) -> Source:
    return Source(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        content=chunk.content,
        rerank_score=chunk.rerank_score,
    )


def _source_dict(chunk: RetrievedChunk, *, number: int) -> dict:
    """Flat citation dict matching the frontend ``Source`` / ``Citation`` type."""
    return {
        "number": number,
        "document_id": str(chunk.document_id),
        "chunk_id": str(chunk.chunk_id),
        "filename": chunk.filename,
        "page": chunk.page_number,
        "label": chunk.citation_label,
        "excerpt": chunk.content[:240] if chunk.content else "",
        "score": round(chunk.rerank_score, 4),
    }


# Think-tag stripping: definitions moved to app.llm.utils to avoid duplication.
# _strip_think_tags and _stream_think_filtered are imported from there.


def _display_provider_name(internal_name: str) -> str:
    """Map internal provider name to a generic user-facing display name.

    Internal names like "gemini", "groq", "openrouter" are mapped to
    "primary", "fallback", "secondary_fallback" respectively. Unknown
    names pass through unchanged.
    """
    from app.config import _PROVIDER_DISPLAY_NAMES
    return _PROVIDER_DISPLAY_NAMES.get(internal_name, internal_name)


async def _sse_event(name: str, data: object) -> str:
    """Format one SSE ``event:`` / ``data:`` frame."""
    payload = json.dumps(data, default=str)
    return f"event: {name}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Metadata questions  (bypass retrieval for count/list/role/member queries)
# ---------------------------------------------------------------------------

# Topic qualifiers that make a question about document *content*, not metadata.
_TOPIC_QUALIFIERS = re.compile(
    r"\b(?:about|discuss|cover|mention|regarding|on the topic of|concerning)\b",
    re.IGNORECASE,
)

# Matches questions that ask about documents/files themselves (count or list).
# A topic qualifier appearing AFTER the document phrase makes it a content
# question -- "How many documents discuss X?" must go through retrieval, not
# this path.
_COUNT_PATTERN = re.compile(
    r"(?:how\s+many|number\s+of|count\s+of|total\s+(?:number\s+of)?)"
    r"\s+"
    r"(?:uploaded\s+)?(?:my\s+|the\s+|this\s+)?(?:own\s+)?"
    r"(?:files|documents?)",
    re.IGNORECASE,
)

_LIST_PATTERN = re.compile(
    r"(?:list|show|what|which|name)\s+"
    r"(?:are\s+the\s+)?(?:me\s+)?(?:all\s+)?"
    r"(?:my\s+|the\s+|this\s+)?(?:uploaded\s+)?(?:own\s+)?"
    r"(?:files|documents?)",
    re.IGNORECASE,
)

# --- Member/workspace metadata patterns ---

_MEMBER_COUNT_PATTERN = re.compile(
    r"(?:how\s+many|number\s+of|count\s+of|total\s+(?:number\s+of)?)"
    r"\s+"
    r"(?:people|members?|users?|employees?|team\s*members?|contributors?)",
    re.IGNORECASE,
)

_MEMBER_LIST_PATTERN = re.compile(
    r"(?:list|show|what|which|name)\s+"
    r"(?:are\s+the\s+)?(?:me\s+)?(?:all\s+)?"
    r"(?:the\s+|this\s+|our\s+)?(?:workspace\s+)?"
    r"(?:people|members?|users?|employees?|team\s*members?|contributors?)"
    r"|"
    r"(?:who(?:'?s|\s+is|\s+are))\s+"
    r"(?:in|of|on|at)\s+"
    r"(?:the\s+|this\s+|our\s+)?(?:workspace|company|team)?",
    re.IGNORECASE,
)

_ROLE_PATTERN = re.compile(
    r"(?:what(?:'?s|\s+is)\s+my|my\s+current|what\s+role\s+(?:do\s+i|am\s+i))"
    r"\s+"
    r"(?:role|access|permission|level)"
    r"|"
    r"what(?:'?s|\s+is)\s+my\s+(?:role|access|permission|level)",
    re.IGNORECASE,
)

# "This month" date filter pattern.
_THIS_MONTH_PATTERN = re.compile(
    r"\b(?:this\s+month|current\s+month|in\s+the\s+current\s+month|"
    r"uploaded\s+(?:this|in\s+this)\s+month)\b",
    re.IGNORECASE,
)


def _is_metadata_question(question: str) -> str | None:
    """Detect document/member-metadata questions that bypass retrieval.

    Returns "count", "list", "member_count", "member_list",
    "role", or None.
    """
    normalised = question.strip().rstrip("?").rstrip(".").strip()

    if _TOPIC_QUALIFIERS.search(normalised):
        return None

    # --- Document metadata ---
    if _COUNT_PATTERN.search(normalised):
        return "count"
    if _LIST_PATTERN.search(normalised):
        return "list"

    # --- Member/workspace metadata ---
    if _ROLE_PATTERN.search(normalised):
        return "role"
    if _MEMBER_COUNT_PATTERN.search(normalised):
        return "member_count"
    if _MEMBER_LIST_PATTERN.search(normalised):
        return "member_list"

    return None


async def _answer_metadata_question(
    *,
    intent: Intent,
    question: str,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
) -> tuple[str, ResponseReason | None]:
    """Answer a metadata question directly from the database.

    No retrieval, no reranking, no LLM call.
    Returns (answer_text, refusal_reason_or_None).
    """
    from app.db.models import Member

    sub = intent.metadata_sub
    normalised = question.strip().rstrip("?").rstrip(".").strip()
    this_month = bool(_THIS_MONTH_PATTERN.search(normalised))

    async with tenant_session(workspace_id=workspace_id, user_id=user_id) as db:
        if sub == MetadataSubIntent.DOC_COUNT:
            stmt = select(func.count()).select_from(Document).where(
                Document.workspace_id == workspace_id,
                Document.status == "READY",
            )
            if this_month:
                from datetime import datetime
                now = datetime.now(UTC)
                month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                stmt = stmt.where(Document.created_at >= month_start)
            count = (await db.execute(stmt)).scalar_one()
            if count == 0:
                if this_month:
                    return "You have no uploaded documents in this workspace this month.", ResponseReason.METADATA_EMPTY
                return "You have no uploaded documents in this workspace.", ResponseReason.METADATA_EMPTY
            word = "document" if count == 1 else "documents"
            if this_month:
                return f"You have {count} uploaded {word} in this workspace this month.", None
            return f"You have {count} uploaded {word} in this workspace.", None

        if sub == MetadataSubIntent.DOC_LIST:
            rows = (
                await db.execute(
                    select(Document.filename, Document.status, Document.created_at)
                    .where(
                        Document.workspace_id == workspace_id,
                        Document.status == "READY",
                    )
                    .order_by(Document.created_at.desc())
                )
            ).all()
            if not rows:
                return "You have no uploaded documents in this workspace.", ResponseReason.METADATA_EMPTY
            items = [f"- {row.filename}" for row in rows]
            count = len(rows)
            word = "document" if count == 1 else "documents"
            header = f"You have {count} uploaded {word} in this workspace:"
            return header + "\n" + "\n".join(items), None

        if sub == MetadataSubIntent.DOC_PAGE_COUNT:
            # Check if authoritative page-count metadata exists.
            # The documents table does NOT store page counts, so we cannot
            # answer this honestly.  Returning a clear "not available" rather
            # than computing pages from chunks (which are not pages).
            return (
                "Page count information is not available for documents in this workspace."
                " Document pages are not tracked as a metadata field.",
                None,
            )

        if sub == MetadataSubIntent.MEMBER_COUNT:
            # ADMIN is not a role in this system — clarify rather than guess.
            if intent.member_role == "ADMIN":
                return (
                    "There is no 'admin' role in this workspace. Roles are "
                    "OWNER and MEMBER only.",
                    None,
                )
            # Build the query — optionally filter by status and/or role.
            stmt = select(func.count()).select_from(Member).where(
                Member.workspace_id == workspace_id,
            )
            if intent.member_status:
                stmt = stmt.where(Member.status == intent.member_status)
            if intent.member_role:
                stmt = stmt.where(Member.role == intent.member_role)
            count = (await db.execute(stmt)).scalar_one()
            status_label = (intent.member_status or "ACTIVE").lower()
            role_label = (intent.member_role or "").lower()
            if count == 0:
                descriptor = " ".join(filter(None, [role_label, status_label, "members"]))
                if not descriptor.strip():
                    descriptor = "members"
                return (
                    f"There are no {descriptor} in this workspace.",
                    ResponseReason.METADATA_EMPTY,
                )
            word = "member" if count == 1 else "members"
            verb = "is" if count == 1 else "are"
            descriptor = " ".join(filter(None, [role_label, status_label, word]))
            if not descriptor.strip():
                descriptor = word
            if intent.member_status or intent.member_role:
                return f"There {verb} {count} {descriptor} in this workspace.", None
            return f"There {verb} {count} {word} in this workspace.", None

        if sub == MetadataSubIntent.MEMBER_LIST:
            stmt = (
                select(Member.user_id, Member.role, Member.status)
                .where(Member.workspace_id == workspace_id)
            )
            if intent.member_status:
                stmt = stmt.where(Member.status == intent.member_status)
            if intent.member_role:
                stmt = stmt.where(Member.role == intent.member_role)
            rows = (
                await db.execute(stmt.order_by(Member.created_at.asc()))
            ).all()
            if not rows:
                role_label = (intent.member_role or "").lower()
                status_label = (intent.member_status or "").lower()
                descriptor = " ".join(filter(None, [role_label, status_label, "members"]))
                if not descriptor.strip():
                    descriptor = "members"
                return f"There are no {descriptor} in this workspace.", ResponseReason.METADATA_EMPTY
            items = [
                f"- User {str(row.user_id)[:8]}... (role: {row.role}, status: {row.status})"
                for row in rows
            ]
            count = len(rows)
            word = "member" if count == 1 else "members"
            verb = "is" if count == 1 else "are"
            role_label = (intent.member_role or "").lower()
            status_label = (intent.member_status or "all").lower()
            descriptor = " ".join(filter(None, [role_label, status_label, word]))
            header = f"There {verb} {count} {descriptor} in this workspace:"
            return header + "\n" + "\n".join(items), None

        if sub == MetadataSubIntent.ROLE:
            rows = (
                await db.execute(
                    select(Member.role).where(
                        Member.workspace_id == workspace_id,
                        Member.user_id == user_id,
                        Member.status == "ACTIVE",
                    )
                )
            ).all()
            if not rows:
                return "You are not an active member of this workspace.", None
            return f"Your role in this workspace is {rows[0].role}.", None

        if sub == MetadataSubIntent.COMPANY_NAME:
            from app.db.models import Workspace as WorkspaceModel
            ws_row = (
                await db.execute(
                    select(WorkspaceModel.name).where(
                        WorkspaceModel.id == workspace_id,
                    )
                )
            ).scalar_one_or_none()
            if ws_row:
                return f"Your workspace is named \"{ws_row}\".", None
            return "This workspace does not have a name configured.", None

        if sub == MetadataSubIntent.DOC_DESCRIPTION:
            # Check if a specific document name was mentioned.
            desc_match = _DOC_SPECIFIC_DESCRIPTION_PATTERN.search(question)
            target_name = desc_match.group(1).strip() if desc_match else None

            # Reject quantifier phrases as "specific" document names.
            # "summary of each file" → "each file" is not a real document name.
            _QUANTIFIER_ONLY_RE = re.compile(
                r"^(?:(?:each|every|all|the|this|my|some|any)\s+)*"
                r"(?:uploaded\s+)?(?:own\s+)?"
                r"(?:document|file|doc|files|documents)s?$",
                re.IGNORECASE,
            )
            if target_name and _QUANTIFIER_ONLY_RE.match(target_name):
                target_name = None

            if target_name:
                # Typo-tolerant filename match for a specific document.
                from app.retrieval.hybrid import _normalize_filename_for_match
                norm_target = _normalize_filename_for_match(target_name)
                doc_rows = (
                    await db.execute(
                        select(Document.filename, Document.description)
                        .where(
                            Document.workspace_id == workspace_id,
                            Document.status == "READY",
                        )
                    )
                ).all()
                best_match = None
                best_score = 0.0
                for row in doc_rows:
                    norm_doc = _normalize_filename_for_match(row.filename)
                    # Simple containment check: if normalized target is contained in
                    # doc name or vice versa.
                    if norm_target in norm_doc or norm_doc in norm_target:
                        score = len(norm_target) / max(len(norm_doc), 1)
                        if score > best_score:
                            best_score = score
                            best_match = row
                if best_match:
                    if best_match.description:
                        return best_match.description, None
                    return (
                        f"The document \"{best_match.filename}\" does not have a generated description yet.",
                        ResponseReason.METADATA_EMPTY,
                    )
                return (
                    f"I could not find a document matching \"{target_name}\" in this workspace.",
                    ResponseReason.METADATA_EMPTY,
                )

            # No specific document — return descriptions for all.
            doc_rows = (
                await db.execute(
                    select(Document.filename, Document.description)
                    .where(
                        Document.workspace_id == workspace_id,
                        Document.status == "READY",
                    )
                    .order_by(Document.created_at.desc())
                )
            ).all()
            if not doc_rows:
                return "You have no uploaded documents in this workspace.", ResponseReason.METADATA_EMPTY

            items = []
            for row in doc_rows:
                if row.description:
                    items.append(f"**{row.filename}**: {row.description}")
                else:
                    items.append(f"**{row.filename}**: No description available.")
            header = f"Here are the descriptions of your {len(doc_rows)} document(s):"
            return header + "\n\n" + "\n\n".join(items), None

    return "I could not determine what metadata you are asking about.", ResponseReason.METADATA_EMPTY


def _pick_refusal_reason(had_candidates: bool) -> ResponseReason:
    """Choose the right refusal reason based on retrieval output."""
    return ResponseReason.NOT_RELEVANT if had_candidates else ResponseReason.NO_EVIDENCE


def _pick_refusal(had_candidates: bool) -> str:
    """Choose the right refusal message based on retrieval output."""
    return refusal_message(_pick_refusal_reason(had_candidates))


# ---------------------------------------------------------------------------
# Conversation history handler (Phase A, step 5)
# ---------------------------------------------------------------------------


async def _answer_conversation_history(
    *,
    intent: Intent,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID | None = None,
) -> str:
    """Answer a conversation-history question from the current user's session.

    Scoped to workspace_id, user_id, and session_id — never exposes other
    users' conversations.
    """
    async with tenant_session(workspace_id=workspace_id, user_id=user_id) as db:
        # Resolve the session to query.
        target_session_id = session_id
        if target_session_id is None:
            row = (
                await db.execute(
                    select(ChatSession.id).where(
                        ChatSession.workspace_id == workspace_id,
                        ChatSession.user_id == user_id,
                    ).order_by(ChatSession.created_at.desc()).limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return "You haven't asked any questions in this session yet."
            target_session_id = row
        else:
            # Verify the session belongs to this user in this workspace.
            exists = (
                await db.execute(
                    select(ChatSession.id).where(
                        ChatSession.id == target_session_id,
                        ChatSession.workspace_id == workspace_id,
                        ChatSession.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if exists is None:
                return "Session not found."

        # Load messages from the session.
        rows = (
            await db.execute(
                select(ChatMessage.role, ChatMessage.content)
                .where(ChatMessage.session_id == target_session_id)
                .order_by(ChatMessage.created_at.asc())
            )
        ).all()

    user_messages = [r.content for r in rows if r.role == "user"]
    assistant_messages = [r.content for r in rows if r.role == "assistant"]

    sub = intent.conversation_history_sub

    if sub == ConversationHistorySubIntent.PREVIOUS_QUESTIONS:
        if not user_messages:
            return "You haven't asked any questions in this session yet."
        items = [f"- {msg}" for msg in user_messages]
        count = len(user_messages)
        word = "question" if count == 1 else "questions"
        header = f"You have asked {count} {word} in this session:"
        return header + "\n" + "\n".join(items)

    if sub == ConversationHistorySubIntent.PREVIOUS_ANSWER:
        if not assistant_messages:
            return "I haven't given any answers in this session yet."
        last_answer = assistant_messages[-1]
        # Truncate long answers for readability.
        if len(last_answer) > 500:
            last_answer = last_answer[:500] + "..."
        return f"My most recent answer was:\n\n{last_answer}"

    # Default: show recent conversation.
    if not rows:
        return "There's no conversation history in this session yet."
    items = []
    for r in rows:
        label = "You" if r.role == "user" else "Assistant"
        content = r.content[:200] + "..." if len(r.content) > 200 else r.content
        items.append(f"{label}: {content}")
    return "Recent conversation:\n" + "\n".join(items)


# ---------------------------------------------------------------------------
# Greeting handler
# ---------------------------------------------------------------------------


def _answer_greeting(*, question: str) -> str:
    """Return a simple conversational greeting response.

    No database query, no retrieval, no LLM call.
    """
    q = question.strip().lower()

    # Farewells
    if re.match(r"^(?:bye|goodbye|see\s+you|take\s+care|good\s+night)\s*[!.?]*$", q):
        return "Goodbye! Feel free to come back anytime if you have questions about your documents."

    # Thanks
    if re.match(r"^(?:thank(?:s|\s+you)|thanks\s+a\s+lot|cheers)\s*[!.?]*$", q):
        return "You're welcome! Let me know if you need anything else."

    # Help command
    if re.match(r"^(?:help|/help|/start)\s*$", q):
        return (
            "I'm your company knowledge assistant. I can help you:\n"
            "- Answer questions about your workspace documents\n"
            "- List and find documents\n"
            "- Compare information across documents\n"
            "Just ask a question about your documents to get started!"
        )

    # Simple social acknowledgements — short and natural, never via RAG.
    if re.match(r"^(?:ok+|okay|sure|yep|yes|alright)\s*[!.?]*$", q):
        return "Sounds good! 👍"
    if re.match(r"^(?:no+|nah|nope)\s*[!.?]*$", q):
        return "Alright! Let me know if you need anything else. 👍"
    if re.match(r"^(?:good\s+(?:boy|girl|job|work)|well\s+done|nice\s+one)\s*[!.?]*$", q):
        return "Haha, thank you! 😄"
    if re.match(r"^(?:nice|cool|great|awesome|perfect|sweet)\s*[!.?]*$", q):
        return "Thank you! 😊"

    # Time-of-day greetings.
    m = re.match(r"^(good\s+(?:morning|afternoon|evening))\b", q)
    if m:
        return f"{m.group(1).capitalize()}! How can I help you today? ☀️"

    # How-are-you greetings.
    if re.match(r"^how(?:'?s|\s+are)\s+(?:you|things|it\s+going)", q) or \
       re.match(r"^how\s+do\s+you\s+do\b", q):
        return "I'm doing great, thanks for asking! How can I help you today?"

    # Echo the greeting back naturally ("hi/hey/heyyy/hello/yo/howdy",
    # including typos like "nameste" and regional "vanakam").
    if re.match(r"^(?:hi+|hello+|hey+|yo+|howdy|greetings|namaste|nameste|vanakam)\b", q):
        word = q.split()[0]
        if word.startswith("nam"):
            return "Namaste! 🙏 How can I help you today?"
        return "Hi! How can I help you today?"

    # Default greeting
    return "Hello! I'm your company knowledge assistant. How can I help you today?"


# ---------------------------------------------------------------------------
# General conversation handler
# ---------------------------------------------------------------------------


def _answer_general_conversation(*, question: str) -> str:
    """Return a friendly response for casual/general conversation.

    No database query, no retrieval, no LLM call.
    Handles statements like "I have a doubt", "can you help me", casual
    chat that doesn't fit any specific lane.
    """
    q = question.strip().lower()

    # Specific patterns
    if re.search(r"(?:i\s+have\s+(?:an?\s+)?(?:doubt|question|query|issue|problem|concern))", q):
        return (
            "Of course! I'm here to help. Could you tell me more about what"
            " you'd like to know? I can answer questions about your workspace's"
            " approved documents, help with member or document counts, and more."
        )

    if re.search(r"(?:can\s+you\s+(?:help|assist|guide)\s+me)", q):
        return (
            "Absolutely! I can help you find information from your workspace's"
            " approved documents. Just ask a question about your documents,"
            " policies, or workspace — I'll search and answer with citations."
        )

    if re.search(r"(?:i\s+need\s+(?:some\s+)?(?:help|assistance|guidance))", q):
        return (
            "I'm here to help! I can answer questions about your workspace's"
            " approved documents, list documents, show member counts, and more."
            " What would you like to know?"
        )

    # Default general conversation
    return (
        "I'm a company knowledge assistant. I help you find information from"
        " your workspace's approved documents. Try asking a question about"
        " your documents, policies, or workspace members!"
    )


# ---------------------------------------------------------------------------
# Identity handler (Phase A, step 6)
# ---------------------------------------------------------------------------


async def _answer_identity(
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
) -> tuple[str, ResponseReason]:
    """Answer an identity question from the authenticated user's context.

    Returns (answer, refusal_reason).  If the user's email is available from
    the principal, we use it.  Otherwise we return IDENTITY_UNAVAILABLE.
    """
    # The principal's email is available from the JWT, but we don't have it
    # in this function's signature yet.  We query the members table to see
    # if we have any identifying info.
    from app.db.models import Member

    async with tenant_session(workspace_id=workspace_id, user_id=user_id) as db:
        rows = (
            await db.execute(
                select(Member.user_id).where(
                    Member.workspace_id == workspace_id,
                    Member.user_id == user_id,
                    Member.status == "ACTIVE",
                )
            )
        ).all()

    if not rows:
        return (
            refusal_message(ResponseReason.IDENTITY_UNAVAILABLE),
            ResponseReason.IDENTITY_UNAVAILABLE,
        )

    # We have a membership but no display name stored — Supabase Auth profiles
    # are not queried here.  Return the honest answer.
    return (
        "I can see you're a member of this workspace, but I don't have your name"
        " available from the current session.",
        ResponseReason.IDENTITY_UNAVAILABLE,
    )


# ---------------------------------------------------------------------------
# Identity sub-type handlers (LLM router)
# ---------------------------------------------------------------------------


def _answer_identity_assistant(*, question: str = "") -> str:
    """Answer a question about what/who the assistant is.

    No database query, no retrieval, no LLM call.  When the user's question
    has an obvious spelling mistake (e.g. "what is youe name"), mention the
    correction before answering — but only in that case.
    """
    prefix = ""
    if re.search(r"\b(?:youe|yur|yor|ur)\b", question.strip().lower()):
        prefix = "I think you meant \u201cyour name.\u201d "
    return prefix + (
        "I'm Office Brain, your company knowledge assistant. I help you find "
        "information from your workspace's approved documents. You can ask me "
        "questions about uploaded documents, and I'll answer with citations "
        "from the relevant sources. I can also help with workspace metadata "
        "like member counts, document counts, and roles."
    )


async def _answer_identity_user(
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    question: str = "",
) -> tuple[str, ResponseReason]:
    """Answer a question about the user's own workspace info.

    Handles questions like "what is my info".  Personal-name statements
    ("my name is X") are routed to the personal-name boundary before this
    handler, so nothing here ever accepts, extracts, or stores a user name.
    Returns (answer, refusal_reason).
    """
    from app.db.models import Member

    q_lower = question.lower().strip()

    # Personal-name statement reaching this handler (defensive; the boundary
    # intercept in classify_intent_regex should have caught it first).  Any
    # user-name intent is answered with the single fixed boundary message —
    # no name is stored, extracted, learned, or echoed back.
    is_statement = bool(
        re.match(r"^(?:my\s+name\s+is|my\s+name's|i\s+am|i'?m|iam|call\s+me)\b", q_lower)
    )

    if is_statement:
        return (
            refusal_message(ResponseReason.PERSONAL_NAME),
            ResponseReason.PERSONAL_NAME,
        )

    # It's a question about user info.
    async with tenant_session(workspace_id=workspace_id, user_id=user_id) as db:
        rows = (
            await db.execute(
                select(Member.user_id, Member.role, Member.status).where(
                    Member.workspace_id == workspace_id,
                    Member.user_id == user_id,
                    Member.status == "ACTIVE",
                )
            )
        ).all()

    if not rows:
        return (
            "I can see you're using this workspace, but I don't have detailed "
            "profile information available. I can help you find information from "
            "your workspace's documents instead.",
            ResponseReason.IDENTITY_UNAVAILABLE,
        )

    role = rows[0].role
    return (
        f"You're a member of this workspace with the role '{role}'. "
        "I don't store personal details like your name or email — those come "
        "from your authentication session. If you need information about "
        "workspace policies, documents, or members, I can help with that.",
        None,
    )


# ---------------------------------------------------------------------------
# App-help handler (Phase A, step 7)
# ---------------------------------------------------------------------------

_APP_HELP_RESPONSES: dict[str, str] = {
    # Permission-related: derived from the actual authorization model.
    "who_can_upload": (
        "Any workspace member can upload documents. Uploaded documents are"
        " immediately available to the uploader. The workspace owner can"
        " approve or reject member uploads to make them searchable."
    ),
    "how_to_upload_and_ask": (
        "To use this assistant:\n"
        "1. Upload documents through the Documents page."
        " Owner uploads are immediately searchable."
        " Member uploads need owner approval.\n"
        "2. Ask questions in this chat."
        " I'll search your workspace's approved documents and answer"
        " with citations."
    ),
    "what_can_i_do": (
        "You can:\n"
        "- Ask questions about your workspace documents\n"
        "- Upload documents (owner uploads are immediate; member uploads need approval)\n"
        "- View document status and manage uploads\n"
        "- Chat with the assistant using your workspace's knowledge base"
    ),
    "how_does_it_work": (
        "This assistant searches your workspace's approved documents to"
        " answer questions. It uses hybrid retrieval (semantic + keyword"
        " search) with reranking to find the most relevant passages,"
        " then generates an answer grounded in those sources."
    ),
    "what_is_this": (
        "I'm a company knowledge assistant. I help you find information"
        " from your workspace's approved documents. You can ask me questions"
        " about uploaded documents, and I'll answer with citations from"
        " the relevant sources."
    ),
    "what_can_i_ask": (
        "You can ask me about any information in your workspace's approved"
        " documents. For example:\n"
        "- Questions about policies, procedures, or guidelines\n"
        "- Summaries of specific documents\n"
        "- Comparisons between documents\n"
        "I'll search the documents and provide answers with citations."
    ),
    "how_do_i_use": (
        "To get started:\n"
        "1. Upload documents through the Documents page\n"
        "2. Ask questions in this chat about your documents\n"
        "3. I'll search and answer with citations from the sources"
    ),
    "monitored": (
        "I don't have authoritative information about monitoring or"
        " tracking policies. Please check your company's privacy policy"
        " or IT department for details."
    ),
}


def _answer_workspace_permission(
    *,
    question: str,
    principal_role: str | None = None,
) -> tuple[str, ResponseReason | None]:
    """Answer a workspace permission question from the authorization model."""
    q = question.lower()

    # Permission-specific answers derived from the actual authorization model.
    if re.search(r"who\s+can\s+(?:upload|add|submit)", q):
        return _APP_HELP_RESPONSES["who_can_upload"], None
    if re.search(r"can\s+(?:i|we|members?)\s+upload", q):
        return (
            "Yes, any workspace member can upload documents."
            " Owner uploads are immediately searchable."
            " Member uploads need owner approval before becoming searchable.",
            None,
        )
    if re.search(r"(?:who\s+has\s+(?:access|permission))", q):
        role_info = f"Your role is {principal_role}." if principal_role else ""
        return (
            f"{role_info} All active workspace members can read documents"
            " and chat. Only the workspace owner can approve documents"
            " and manage members.",
            None,
        )
    if re.search(r"(?:what\s+(?:are|is)\s+(?:my|the|our)\s+(?:permission|role|access))", q):
        if principal_role:
            return (
                f"Your role in this workspace is {principal_role}."
                " Members can upload documents (pending owner approval) and"
                " ask questions. Owners can also approve/reject documents"
                " and manage members.",
                None,
            )
        return (
            "I don't have your role information available.",
            ResponseReason.IDENTITY_UNAVAILABLE,
        )

    return (
        refusal_message(ResponseReason.APP_HELP_UNAVAILABLE),
        ResponseReason.APP_HELP_UNAVAILABLE,
    )


def _answer_app_help(
    *,
    question: str,
    intent: Intent,
    principal_role: str | None = None,
) -> tuple[str, ResponseReason | None]:
    """Answer an app-help question.

    Permission-related answers are derived from the actual authorization
    model.  Privacy/monitoring questions only get answered if an authoritative
    source exists.
    """
    q = question.lower()

    # Detect specific help sub-intents.
    if re.search(r"who\s+can\s+(?:upload|add|submit)", q):
        return _APP_HELP_RESPONSES["who_can_upload"], None
    if re.search(r"how\s+(?:can|do|should)\s+(?:i|we)\s+(?:upload|ask)", q):
        return _APP_HELP_RESPONSES["how_to_upload_and_ask"], None
    if re.search(r"what\s+can\s+(?:i|we)\s+do", q):
        return _APP_HELP_RESPONSES["what_can_i_do"], None
    if re.search(r"how\s+(?:does|do)\s+(?:this|it)\s+work", q):
        return _APP_HELP_RESPONSES["how_does_it_work"], None
    if re.search(r"(?:what\s+(?:does|do)\s+(?:this|the)\s+(?:chatbot|assistant|app|bot)\s+(?:do|does|offer|provide))", q):
        return _APP_HELP_RESPONSES["what_is_this"], None
    if re.search(r"(?:tell\s+me\s+about\s+(?:this\s+)?(?:chatbot|assistant|app|system|bot))", q):
        return _APP_HELP_RESPONSES["what_is_this"], None
    if re.search(r"(?:what\s+(?:is|are)\s+this\s+(?:chatbot|assistant|app|system|bot))", q):
        return _APP_HELP_RESPONSES["what_is_this"], None
    if re.search(r"(?:what\s+(?:can|kind|type)\s+(?:i|we)\s+(?:ask|use))", q):
        return _APP_HELP_RESPONSES["what_can_i_ask"], None
    if re.search(r"(?:how\s+(?:do|can)\s+i\s+(?:use|start|get\s+started))", q):
        return _APP_HELP_RESPONSES["how_do_i_use"], None
    if re.search(r"(?:am\s+i\s+being|do\s+you\s+(?:track|monitor))", q):
        return _APP_HELP_RESPONSES["monitored"], ResponseReason.APP_HELP_UNAVAILABLE
    if re.search(r"(?:who\s+has\s+(?:access|permission))", q):
        role_info = f"Your role is {principal_role}." if principal_role else ""
        return (
            f"{role_info} All active workspace members can read documents"
            " and chat. Only the workspace owner can approve documents"
            " and manage members.",
            None,
        )

    return (
        refusal_message(ResponseReason.APP_HELP_UNAVAILABLE),
        ResponseReason.APP_HELP_UNAVAILABLE,
    )


async def _load_recent_history(
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID | None = None,
    max_turns: int = 6,
) -> list[ChatTurn]:
    """Load the most recent conversation turns for query rewriting.

    If ``session_id`` is provided, loads messages from that session.
    Otherwise, loads from the user's most recent session in the workspace.
    Returns at most ``max_turns`` messages (3 user/assistant pairs).
    """
    try:
        async with tenant_session(workspace_id=workspace_id, user_id=user_id) as db:
            # Find the session to load from.
            target_session_id = session_id
            if target_session_id is None:
                # Find the user's most recent session in this workspace.
                row = (
                    await db.execute(
                        select(ChatSession.id).where(
                            ChatSession.workspace_id == workspace_id,
                            ChatSession.user_id == user_id,
                        ).order_by(ChatSession.created_at.desc()).limit(1)
                    )
                ).scalar_one_or_none()
                if row is None:
                    return []
                target_session_id = row

            # Load recent messages from that session.
            rows = (
                await db.execute(
                    select(ChatMessage.role, ChatMessage.content)
                    .where(ChatMessage.session_id == target_session_id)
                    .order_by(ChatMessage.created_at.desc())
                    .limit(max_turns)
                )
            ).all()

            # Reverse to chronological order (oldest first).
            rows = list(reversed(rows))

            return [ChatTurn(role=r.role, content=r.content) for r in rows]
    except Exception as exc:
        # If history loading fails, proceed without context.
        logger.debug(
            "Failed to load conversation history for rewrite: {error}",
            error=str(exc)[:200],
        )
        return []


# SSE streaming chat endpoint  (POST /chat)
# ---------------------------------------------------------------------------


async def _stream_chat(
    principal: CurrentPrincipal,
    payload: ChatStreamRequest,
    llm: LLMProvider,
    member_role: str,
) -> AsyncIterator[str]:
    """SSE generator: intent → (metadata | history | identity | help | RAG) → stream.

    ``member_role`` must be resolved *before* this generator is passed to
    ``StreamingResponse`` — raising ``HTTPException`` inside an already-
    streaming response triggers a ``RuntimeError``.
    """

    workspace_id = principal.workspace_id
    question = payload.message.strip()

    # 0. Resolve the session to append to — ONCE, before anything else.
    # Reuse payload.session_id when it names a session this user owns in this
    # workspace (a continuation); otherwise create a new session (first message
    # of a conversation, or an unknown/stale id).  Every persistence branch
    # below — RAG, direct answers, clarification — writes under this one id,
    # so a conversation can never fragment across ChatSession rows.
    session_id: uuid.UUID | None = None
    session_created = False
    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        if payload.session_id is not None:
            existing = (
                await db.execute(
                    select(ChatSession.id).where(
                        ChatSession.id == payload.session_id,
                        ChatSession.workspace_id == workspace_id,
                        ChatSession.user_id == principal.user_id,
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                session_id = existing
        if session_id is None:
            row = (
                await db.execute(
                    insert(ChatSession)
                    .values(
                        workspace_id=workspace_id,
                        user_id=principal.user_id,
                    )
                    .returning(ChatSession.id)
                )
            ).scalar_one()
            session_id = row
            session_created = True
    assert session_id is not None

    # Announce a genuinely new session to the client (sidebar entry; the
    # frontend adopts this id for subsequent turns).  Turns that continue an
    # existing conversation reuse the same id and must NOT re-emit it — the
    # frontend re-points at whatever session_id the last event delivered, so a
    # per-message event would re-aim every later answer at the latest fragment.
    if session_created:
        yield await _sse_event("session", {"session_id": str(session_id)})

    # 1a. Query Understanding: single LLM call for intent, typo correction,
    # and search-query optimization.  Replaces the fragmented regex + LLM
    # router approach.
    effective_query = question  # may be overridden by QU below
    search_query_for_retrieval: str | None = None
    qu_confidence: float | None = None
    needs_clarification = False
    refusal_reason: ResponseReason | None = None
    history_turns: list[ChatTurn] = []
    #: Per-request stage timings, consumed by _log_chat_diag at the terminal
    #: exits below.  Keys are set as each phase completes; fast paths leave
    #: them unset and the helper defaults them to 0.0.
    diag: dict[str, float] = {"request_started": time.perf_counter()}

    # 1b. Prompt-injection attempts are refused immediately — never routed,
    # never retrieved, never sent to the LLM.  Checked BEFORE any QU call so a
    # payload aimed at the understanding stage never reaches a model.
    if _is_injection_attempt(question):
        injection_text = refusal_message(ResponseReason.INJECTION_ATTEMPT)
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=session_id, role="user", content=question,
                )
            )
        yield await _sse_event("status", {"stage": "generating"})
        yield await _sse_event("sources", {"sources": []})
        yield await _sse_event("token", {"text": injection_text})
        yield await _sse_event("citations", {"citations": []})
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=session_id, role="assistant",
                    content=injection_text, sources=[],
                )
            )
        yield await _sse_event("done", {
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "provider": "", "model": "", "grounded": True, "routes": [], "sql_query": "",
        })
        logger.info(
            "intent=injection_attempt workspace={ws} retrieval_called=False",
            ws=workspace_id,
        )
        _log_chat_diag(
            workspace_id=workspace_id,
            intent="injection_attempt",
            reason="injection_attempt",
            grounded=True,
            provider="",
            candidates=0,
            final=0,
            tokens=0,
            retrieval_called=False,
            diag=diag,
        )
        return

    # Load history for context.
    history_turns = await _load_recent_history(
        workspace_id=workspace_id,
        user_id=principal.user_id,
        session_id=session_id,
    )
    history_dicts = [{"role": t.role, "content": t.content} for t in history_turns]

    # 1a'. Deterministic zero-cost fast path.  These intents are resolved by
    # regex alone and their answers never need retrieval: running the QU LLM
    # call on them wastes latency and token budget, and a rewrite can only eat
    # a constraint a regex match already respected.  Person-info questions are
    # deliberately excluded (they must go through the evidence path), and the
    # generic GENERAL_CONVERSATION lane is excluded — its regex is too eager
    # ("can you help me with X" would skip retrieval on a vacation-policy
    # question).  The personal_name_boundary sub-lane is included: it is a
    # fixed refusal and needs no LLM.
    from app.retrieval.intent import classify_intent_regex as _classify_regex

    regex_intent = _classify_regex(question)
    fast_intent = (
        regex_intent.category
        in (
            IntentCategory.GREETING,
            IntentCategory.APP_HELP,
            IntentCategory.OUT_OF_SCOPE,
            IntentCategory.IDENTITY_ASSISTANT,
            IntentCategory.IDENTITY_USER,
        )
        or (
            regex_intent.category == IntentCategory.GENERAL_CONVERSATION
            and regex_intent.reason == "personal_name_boundary"
        )
    )
    if fast_intent and not _is_person_info_request(question):
        intent = regex_intent
        effective_query = question
        diag["qu_ms"] = 0.0
        logger.info(
            "intent={category} reason={reason} workspace={ws} QU_skipped=True",
            category=intent.category.value,
            reason=intent.reason,
            ws=workspace_id,
        )
    else:
        # Query Understanding stage — single LLM call for everything else.
        _qu_started = time.perf_counter()
        qu_result = await understand_query(
            query=question,
            workspace_id=workspace_id,
            history=history_dicts,
        )
        diag["qu_ms"] = (time.perf_counter() - _qu_started) * 1000.0

        # Build intent from QU result, but refine with regex for specific cases
        # that QU's 5-category model may misclassify (identity questions, name
        # statements) — these are deterministic and should never need an LLM.
        intent = _refine_intent_from_qu(
            qu_result=qu_result,
            original_query=question,
        )
        effective_query = qu_result.corrected_query
        search_query_for_retrieval = qu_result.search_query
        qu_confidence = qu_result.confidence

        # --- Progress status ---
        # Query understanding is complete; the next major phase is search or a
        # direct answer.  Emit a status so the frontend can show "Searching
        # your documents…" (the regex fast path above goes straight to its
        # answer and skips this).
        yield await _sse_event("status", {"stage": "searching"})

    # Person-info questions must go through the evidence path (RAG) — they are
    # plausibly answered from org charts / team directories in approved
    # documents and must never be handled as general knowledge.  When QU call
    # tends to classify these as off_topic / general_conversation, route them
    # to document-content retrieval instead.
    if (
        _is_person_info_request(question)
        and intent.reason != "personal_name_boundary"
        and intent.category in (
            IntentCategory.OUT_OF_SCOPE,
            IntentCategory.GENERAL_CONVERSATION,
        )
    ):
        intent = Intent(
            category=IntentCategory.DOCUMENT_CONTENT,
            reason="person_info_override",
        )

    # 1c. Handle ambiguity: ask for clarification instead of refusing.
    if needs_clarification or intent.needs_clarification:
        # Persist the exchange under the session resolved in step 0.
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=session_id, role="user", content=question,
                )
            )
        clarification_text = refusal_message(ResponseReason.NEEDS_CLARIFICATION)
        yield await _sse_event("status", {"stage": "generating"})
        yield await _sse_event("sources", {"sources": []})
        yield await _sse_event("token", {"text": clarification_text})
        yield await _sse_event("citations", {"citations": []})
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=session_id, role="assistant",
                    content=clarification_text, sources=[],
                )
            )
        yield await _sse_event("done", {
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "provider": "", "model": "", "grounded": True, "routes": [], "sql_query": "",
        })
        logger.info(
            "intent=ambiguous reason=needs_clarification workspace={ws}",
            ws=workspace_id,
        )
        _log_chat_diag(
            workspace_id=workspace_id,
            intent=intent.category.value,
            reason="needs_clarification",
            grounded=True,
            provider="",
            candidates=0,
            final=0,
            tokens=0,
            retrieval_called=False,
            diag=diag,
        )
        return

    # 1d. Route by intent category.
    answer: str | None = None

    if intent.category == IntentCategory.GREETING:
        answer = _answer_greeting(question=effective_query)
        logger.info(
            "intent=greeting workspace={ws} retrieval_called=False",
            ws=workspace_id,
        )

    elif intent.category == IntentCategory.GENERAL_CONVERSATION:
        if intent.reason == "personal_name_boundary":
            answer = refusal_message(ResponseReason.PERSONAL_NAME)
            refusal_reason = ResponseReason.PERSONAL_NAME
            logger.info(
                "intent=general_conversation reason=personal_name_boundary "
                "workspace={ws} retrieval_called=False",
                ws=workspace_id,
            )
        else:
            answer = _answer_general_conversation(question=effective_query)
            logger.info(
                "intent=general_conversation workspace={ws} retrieval_called=False",
                ws=workspace_id,
            )

    elif intent.category == IntentCategory.OUT_OF_SCOPE:
        # Distinguish capability requests (write code, create files, design)
        # from general knowledge questions for a more specific response.
        if _is_capability_request(effective_query):
            answer = refusal_message(ResponseReason.OUT_OF_SCOPE_CAPABILITY)
        else:
            answer = refusal_message(ResponseReason.OUT_OF_SCOPE)
        refusal_reason = ResponseReason.OUT_OF_SCOPE
        logger.info(
            "intent=out_of_scope workspace={ws} retrieval_called=False "
            "capability_request={cap}",
            ws=workspace_id,
            cap=_is_capability_request(effective_query),
        )

    elif intent.category == IntentCategory.IDENTITY_ASSISTANT:
        answer = _answer_identity_assistant(question=question)
        logger.info(
            "intent=identity_assistant workspace={ws} retrieval_called=False",
            ws=workspace_id,
        )

    elif intent.category == IntentCategory.IDENTITY_USER:
        answer, refusal_reason = await _answer_identity_user(
            workspace_id=workspace_id,
            user_id=principal.user_id,
            question=effective_query,
        )
        logger.info(
            "intent=identity_user workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )

    elif intent.category == IntentCategory.IDENTITY:
        # Legacy identity route (regex fast-path) — treat as user identity.
        answer, refusal_reason = await _answer_identity_user(
            workspace_id=workspace_id,
            user_id=principal.user_id,
            question=effective_query,
        )
        logger.info(
            "intent=identity workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )

    elif intent.category in (
        IntentCategory.PERMISSIONS,
        IntentCategory.WORKSPACE_PERMISSION,
    ):
        # PERMISSIONS (legacy LLM-router route name) and WORKSPACE_PERMISSION
        # (regex fast-path category) are the same lane — one answer function.
        answer, refusal_reason = _answer_workspace_permission(
            question=effective_query,
            principal_role=member_role,
        )
        logger.info(
            "intent=permissions workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )

    elif intent.category == IntentCategory.APP_HELP:
        answer, refusal_reason = _answer_app_help(
            question=effective_query,
            intent=intent,
            principal_role=member_role,
        )
        logger.info(
            "intent=app_help workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )

    elif intent.category == IntentCategory.CONVERSATION_HISTORY:
        answer = await _answer_conversation_history(
            intent=intent,
            workspace_id=workspace_id,
            user_id=principal.user_id,
            session_id=session_id,
        )
        logger.info(
            "intent=conversation_history sub={sub} workspace={ws} retrieval_called=False",
            sub=intent.conversation_history_sub.value if intent.conversation_history_sub else None,
            ws=workspace_id,
        )

    elif intent.category in (IntentCategory.WORKSPACE_METADATA, IntentCategory.DOCUMENT_LIST):
        answer, refusal_reason = await _answer_metadata_question(
            intent=intent,
            question=effective_query,
            workspace_id=workspace_id,
            user_id=principal.user_id,
        )
        logger.info(
            "intent={intent_type} sub={sub} workspace={ws} retrieval_called=False refusal={refusal}",
            intent_type=intent.category.value,
            sub=intent.metadata_sub.value if intent.metadata_sub else None,
            ws=workspace_id,
            refusal=refusal_reason.value if refusal_reason else None,
        )
        # Unify with the sync endpoint: when the metadata handler can't resolve
        # the specific field (sub is None) and returns a refusal, fall through
        # to document-content search instead of dead-ending with a canned
        # "I could not determine" refusal.
        if refusal_reason == ResponseReason.METADATA_EMPTY and intent.metadata_sub is None:
            logger.info(
                "Metadata sub-intent unresolved for workspace={ws}, "
                "falling through to document content search",
                ws=workspace_id,
            )
            answer = None

    # For non-document intents, emit the answer and persist under the session
    # resolved in step 0 — no new ChatSession per turn.
    if answer is not None:
        # Persist user message.
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=session_id,
                    role="user",
                    content=question,
                )
            )
        # Emit the direct answer — no LLM, no retrieval.
        yield await _sse_event("status", {"stage": "generating"})
        yield await _sse_event("sources", {"sources": []})
        yield await _sse_event("token", {"text": answer})
        yield await _sse_event("citations", {"citations": []})
        # Persist assistant message.
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=session_id,
                    role="assistant",
                    content=answer,
                    sources=[],
                )
            )
        yield await _sse_event(
            "done",
            {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "provider": "",
                "model": "",
                "grounded": True,
                "routes": [],
                "sql_query": "",
            },
        )
        _log_chat_diag(
            workspace_id=workspace_id,
            intent=intent.category.value,
            reason=refusal_reason.value if refusal_reason else "direct_answer",
            grounded=True,
            provider="",
            candidates=0,
            final=0,
            tokens=0,
            retrieval_called=False,
            diag=diag,
        )
        return

    # --- Document content path (RAG) ---
    # 2. The session was resolved up front (step 0) — just append to it.

    # 3. Persist user message.
    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        await db.execute(
            insert(ChatMessage).values(
                session_id=session_id,
                role="user",
                content=question,
            )
        )

    # 4. Retrieve evidence (session closes before the LLM call).
    # Use the rewritten query for all downstream operations.
    # Phase B-2: classify query shape + resolve doc target for filename-aware retrieval.
    doc_target_result = None
    if intent.category == IntentCategory.DOCUMENT_CONTENT:
        from app.retrieval.doc_targeting import resolve_document_target
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as dt_db:
            doc_target_result = await resolve_document_target(
                session=dt_db,
                question=effective_query,
                workspace_id=workspace_id,
            )
    has_doc_target = doc_target_result is not None and doc_target_result.matched_document_id is not None
    query_shape = classify_query_shape(
        effective_query,
        has_doc_target=has_doc_target,
    )
    logger.info(
        "intent=document_content query_shape={shape} workspace={ws} "
        "retrieval_called=True filename_match={fm} matched_filename={mf} "
        "doc_target_confidence={dtc}",
        shape=query_shape.value, ws=workspace_id,
        fm=doc_target_result is not None and doc_target_result.matched_filename is not None,
        mf=doc_target_result.matched_filename if doc_target_result else None,
        dtc=doc_target_result.confidence if doc_target_result else 0.0,
    )

    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        _retrieval_started = time.perf_counter()
        result = await retrieve(
            db, query=effective_query, workspace_id=workspace_id,
            query_shape=query_shape,
            doc_target_result=doc_target_result,
            search_query=search_query_for_retrieval,
            qu_confidence=qu_confidence,
        )
        diag["retrieval_ms"] = (time.perf_counter() - _retrieval_started) * 1000.0

    if not result.grounded:
        # Person-info questions that found no evidence get the unsupported-
        # information refusal — a specific person's identity/contact must never
        # be fabricated, even when retrieval came up empty.
        if _is_person_info_request(question) or _is_person_info_request(effective_query):
            refusal_reason = ResponseReason.UNSUPPORTED_INFORMATION
        else:
            # Choose the right refusal: documents exist but irrelevant, or nothing found.
            refusal_reason = _pick_refusal_reason(had_candidates=bool(result.chunks))
        refusal = refusal_message(refusal_reason)
        logger.info(
            "intent=document_content refusal={reason} top_score={score} "
            "candidates={n} retrieval_called=True workspace={ws}",
            ws=workspace_id,
            score=result.top_score,
            n=len(result.chunks),
            reason=refusal_reason.value,
        )
        _log_chat_diag(
            workspace_id=workspace_id,
            intent=intent.category.value,
            reason=refusal_reason.value,
            grounded=False,
            provider="",
            candidates=len(result.chunks),
            final=0,
            tokens=0,
            retrieval_called=True,
            diag=diag,
        )
        # Emit empty sources, the refusal text as a token, and done.
        yield await _sse_event("status", {"stage": "generating"})
        yield await _sse_event("sources", {"sources": []})
        yield await _sse_event("token", {"text": refusal})
        yield await _sse_event("citations", {"citations": []})

        # Persist refusal.
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=session_id,
                    role="assistant",
                    content=refusal,
                    sources=[],
                )
            )

        yield await _sse_event(
            "done",
            {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "provider": "",
                "model": "",
                "grounded": False,
                "routes": [],
                "sql_query": "",
            },
        )
        return

    # 5. Emit sources.
    sources_list = [
        _source_dict(chunk, number=i + 1)
        for i, chunk in enumerate(result.chunks)
    ]
    yield await _sse_event("sources", {"sources": sources_list})

    # 6. Stream LLM tokens.
    yield await _sse_event("status", {"stage": "generating"})
    messages = build_messages(question=effective_query, chunks=result.chunks)
    completion = Completion()
    full_text = ""
    _gen_started = time.perf_counter()
    try:
        # Suppress model-injected think/reasoning blocks as they stream, so the
        # client never sees the raw tags or reasoning text (Qwen3 on Groq,
        # Gemini-2.5 on OpenRouter).  Any tokens that pass through are also
        # stripped post-hoc below, so every path (streaming + sync) is covered.
        async for token in _stream_think_filtered(
            llm.stream(messages, completion=completion)
        ):
            full_text += token
            yield await _sse_event("token", {"text": token})
        diag["gen_ms"] = (time.perf_counter() - _gen_started) * 1000.0
    except LLMError as exc:
        diag["gen_ms"] = (time.perf_counter() - _gen_started) * 1000.0
        logger.error(
            "Generation failed for user {user} in workspace {ws}: {error}",
            user=principal.user_id,
            ws=workspace_id,
            error=exc,
        )
        _log_chat_diag(
            workspace_id=workspace_id,
            intent=intent.category.value,
            reason="llm_error",
            grounded=result.grounded,
            provider=completion.provider or llm.name,
            candidates=len(result.chunks),
            final=0,
            tokens=completion.usage.prompt_tokens,
            retrieval_called=True,
            diag=diag,
        )
        yield await _sse_event(
            "error",
            {
                "detail": "The language model is currently unavailable. Please try again.",
                "partial": bool(full_text),
            },
        )
        return

    # Strip model-injected thinking/reasoning blocks (e.g. Qwen3 `` tags).
    full_text = _strip_think_tags(full_text)

    if not full_text or not full_text.strip():
        logger.warning(
            "LLM returned empty answer after stripping think tags (streaming) "
            "for user {user} in workspace {ws}: grounded={grounded} sources={n} "
            "provider={provider}",
            user=principal.user_id,
            ws=workspace_id,
            grounded=result.grounded,
            n=len(result.chunks),
            provider=completion.provider or llm.name,
        )
        _log_chat_diag(
            workspace_id=workspace_id,
            intent=intent.category.value,
            reason="empty_answer",
            grounded=True,
            provider=completion.provider or llm.name,
            candidates=len(result.chunks),
            final=0,
            tokens=completion.usage.completion_tokens,
            retrieval_called=True,
            diag=diag,
        )
        yield await _sse_event(
            "error",
            {
                "detail": (
                    "I found relevant documents, but I couldn't generate an answer "
                    "from them. Please try asking the question again."
                ),
                "partial": False,
            },
        )
        return

    # 6b. Resolve which retrieved sources the LLM actually cited, so the
    # user-facing source/citation lists contain only chunks that grounded the
    # answer (CLAUDE.md 8.4 / ISSUE E).  Chunks that survived retrieval but
    # were never referenced by the answer are dropped, not passed through.
    # If the answer contains no bracketed citations at all, we cannot determine
    # what was referenced — keep the full retrieved set rather than dropping
    # everything.
    cited_nums = cited_numbers(full_text)
    if cited_nums:
        cited_set = set(cited_nums)
        final_sources = [
            src for src in sources_list if src["number"] in cited_set
        ]
    else:
        final_sources = sources_list

    # 7. Emit citations (only what the answer actually referenced).
    yield await _sse_event("citations", {"citations": final_sources})

    # 8. Persist assistant message.
    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        await db.execute(
            insert(ChatMessage).values(
                session_id=session_id,
                role="assistant",
                content=full_text,
                sources=final_sources,
            )
        )

    # 9. Emit done.
    yield await _sse_event(
        "done",
        {
            "usage": completion.usage.as_dict(),
            "provider": _display_provider_name(completion.provider or llm.name),
            "model": "",
            "grounded": True,
            "routes": [],
            "sql_query": "",
        },
    )

    logger.info(
        "Streamed answer for workspace {ws}: {n} sources, {tokens} tokens",
        ws=workspace_id,
        n=len(sources_list),
        tokens=completion.usage.completion_tokens,
    )
    _log_chat_diag(
        workspace_id=workspace_id,
        intent=intent.category.value,
        reason="",
        grounded=True,
        provider=_display_provider_name(completion.provider or llm.name),
        candidates=len(sources_list),
        final=len(final_sources),
        tokens=completion.usage.completion_tokens,
        retrieval_called=True,
        diag=diag,
    )


@router.post(
    "",
    response_class=StreamingResponse,
    summary="SSE streaming chat — what the frontend calls",
)
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_stream(
    request: Request,  # noqa: ARG001 — required by slowapi's decorator
    principal: CurrentPrincipal,
    payload: ChatStreamRequest,
    llm: Annotated[LLMProvider, Depends(get_generic_llm)],
) -> StreamingResponse:
    """SSE streaming chat: retrieval → grounding → LLM stream → persist.

    The frontend calls ``POST /chat`` with ``Accept: text/event-stream`` and
    receives a typed event stream matching the protocol in
    ``frontend/src/lib/api/types.ts``.

    Session management is transparent: if ``session_id`` is omitted, a new
    session is created.  If it refers to a session the user owns, the message
    is appended.  If it refers to a session that no longer exists, a new one
    is created.
    """
    # Workspace membership check must happen BEFORE streaming starts.
    # StreamingResponse sends HTTP headers as soon as iteration begins;
    # raising HTTPException after that point triggers a RuntimeError.
    member_role = await assert_workspace_role(principal.workspace_id, principal)

    return StreamingResponse(
        _stream_chat(principal, payload, llm, member_role),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Synchronous JSON chat endpoint  (POST /chat/grounded)  — test suite & API
# ---------------------------------------------------------------------------


@router.post(
    "/grounded",
    response_model=GroundedChatResponse,
    summary="Ask a question grounded in the workspace's approved documents (sync JSON)",
)
@limiter.limit(CHAT_RATE_LIMIT)
async def grounded_chat(
    request: Request,  # noqa: ARG001 — required by slowapi's decorator
    principal: CurrentPrincipal,
    payload: GroundedChatRequest,
    llm: Annotated[LLMProvider, Depends(get_generic_llm)],
) -> GroundedChatResponse:
    """Answer ``payload.message`` from the caller's workspace's approved documents.

    Fail-closed contract: if retrieval finds no acceptable evidence the request
    returns a refusal *without calling the LLM*, and no answer is ever fabricated
    (CLAUDE.md 8.3).
    """
    question = payload.message.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be empty.",
        )

    workspace_id = principal.workspace_id
    member_role = await assert_workspace_role(workspace_id, principal)

    # Query Understanding stage — single LLM call for intent, typo correction,
    # and search-query optimization.
    effective_query = question
    search_query_for_retrieval: str | None = None
    qu_confidence: float | None = None

    # Load history for context.
    history_turns = await _load_recent_history(
        workspace_id=workspace_id,
        user_id=principal.user_id,
    )
    history_dicts = [{"role": t.role, "content": t.content} for t in history_turns]

    qu_result = await understand_query(
        query=question,
        workspace_id=workspace_id,
        history=history_dicts,
    )

    # QU degraded (LLM failure, empty response, or truncation): record the
    # routing attribution that the normal intent= log lines would have produced,
    # so every request has a traceable routing decision even when QU itself
    # failed.  Only fire this for genuine QU failures — a successful
    # workspace_metadata classification must not be mislabeled as degraded.
    # Failure reasoning values: parse_failure, empty_response,
    # unclosed_think_block, empty_query, llm_error:*
    _QU_FAILURE_REASONS = {"parse_failure", "empty_response", "unclosed_think_block", "empty_query"}
    _qu_failed = bool(
        qu_result.reasoning
        and qu_result.reasoning not in ("", "parse_success")
        and (
            qu_result.reasoning in _QU_FAILURE_REASONS
            or qu_result.reasoning.startswith("llm_error:")
        ),
    )
    if _qu_failed:
        logger.info(
            "intent=document_content qu_degraded=true reason={reason} "
            "retrieval_called=True workspace={ws}",
            reason=qu_result.reasoning[:80],
            ws=workspace_id,
        )

    # Build intent from QU result, but refine with regex for specific cases
    # that QU's 5-category model may misclassify (identity questions, name
    # statements).
    intent = _refine_intent_from_qu(
        qu_result=qu_result,
        original_query=question,
    )
    effective_query = qu_result.corrected_query
    search_query_for_retrieval = qu_result.search_query
    qu_confidence = qu_result.confidence

    # Prompt-injection attempts are refused immediately — never routed, never
    # retrieved, never sent to the LLM.
    if _is_injection_attempt(question):
        logger.info(
            "intent=injection_attempt workspace={ws} retrieval_called=False",
            ws=workspace_id,
        )
        return GroundedChatResponse(
            answer=refusal_message(ResponseReason.INJECTION_ATTEMPT),
            grounded=True,
            insufficient_evidence=False,
            sources=[],
        )

    # Person-info questions must go through the evidence path (RAG) — they are
    # plausibly answered from org charts / team directories in approved
    # documents and must never be handled as general knowledge.  When QU tends
    # to classify these as off_topic / general_conversation, route them to
    # document-content retrieval instead.
    if (
        _is_person_info_request(question)
        and intent.reason != "personal_name_boundary"
        and intent.category in (
            IntentCategory.OUT_OF_SCOPE,
            IntentCategory.GENERAL_CONVERSATION,
        )
    ):
        intent = Intent(
            category=IntentCategory.DOCUMENT_CONTENT,
            reason="person_info_override",
        )

    if intent.category == IntentCategory.AMBIGUOUS:
        return GroundedChatResponse(
            answer=refusal_message(ResponseReason.NEEDS_CLARIFICATION),
            grounded=True,
            insufficient_evidence=False,
            sources=[],
        )

    # Route by intent category.
    if intent.category == IntentCategory.GREETING:
        answer = _answer_greeting(question=effective_query)
        logger.info(
            "intent=greeting workspace={ws} retrieval_called=False", ws=workspace_id,
        )
        return GroundedChatResponse(
            answer=answer, grounded=True, insufficient_evidence=False, sources=[],
        )

    if intent.category == IntentCategory.GENERAL_CONVERSATION:
        if intent.reason == "personal_name_boundary":
            answer = refusal_message(ResponseReason.PERSONAL_NAME)
            logger.info(
                "intent=general_conversation reason=personal_name_boundary "
                "workspace={ws} retrieval_called=False", ws=workspace_id,
            )
        else:
            answer = _answer_general_conversation(question=effective_query)
            logger.info(
                "intent=general_conversation workspace={ws} retrieval_called=False", ws=workspace_id,
            )
        return GroundedChatResponse(
            answer=answer, grounded=True, insufficient_evidence=False, sources=[],
        )

    if intent.category == IntentCategory.OUT_OF_SCOPE:
        if _is_capability_request(effective_query):
            answer = refusal_message(ResponseReason.OUT_OF_SCOPE_CAPABILITY)
        else:
            answer = refusal_message(ResponseReason.OUT_OF_SCOPE)
        logger.info(
            "intent=out_of_scope workspace={ws} retrieval_called=False "
            "capability_request={cap}",
            ws=workspace_id,
            cap=_is_capability_request(effective_query),
        )
        return GroundedChatResponse(
            answer=answer,
            grounded=True, insufficient_evidence=False, sources=[],
        )

    if intent.category == IntentCategory.IDENTITY_ASSISTANT:
        answer = _answer_identity_assistant(question=question)
        logger.info(
            "intent=identity_assistant workspace={ws} retrieval_called=False", ws=workspace_id,
        )
        return GroundedChatResponse(
            answer=answer, grounded=True, insufficient_evidence=False, sources=[],
        )

    if intent.category == IntentCategory.IDENTITY_USER:
        answer, refusal_reason = await _answer_identity_user(
            workspace_id=workspace_id, user_id=principal.user_id,
            question=effective_query,
        )
        logger.info(
            "intent=identity_user workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )
        return GroundedChatResponse(
            answer=answer, grounded=True,
            insufficient_evidence=refusal_reason is not None, sources=[],
        )

    if intent.category == IntentCategory.IDENTITY:
        # Legacy identity route (regex fast-path) — treat as user identity.
        answer, refusal_reason = await _answer_identity_user(
            workspace_id=workspace_id, user_id=principal.user_id,
            question=effective_query,
        )
        logger.info(
            "intent=identity workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )
        return GroundedChatResponse(
            answer=answer, grounded=True,
            insufficient_evidence=refusal_reason is not None, sources=[],
        )

    if intent.category in (
        IntentCategory.PERMISSIONS,
        IntentCategory.WORKSPACE_PERMISSION,
    ):
        # PERMISSIONS (legacy LLM-router route name) and WORKSPACE_PERMISSION
        # (regex fast-path category) are the same lane — one answer function.
        answer, refusal_reason = _answer_workspace_permission(
            question=effective_query, principal_role=member_role,
        )
        logger.info(
            "intent=permissions workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )
        return GroundedChatResponse(
            answer=answer, grounded=True,
            insufficient_evidence=refusal_reason is not None, sources=[],
        )

    if intent.category == IntentCategory.APP_HELP:
        answer, refusal_reason = _answer_app_help(
            question=effective_query, intent=intent, principal_role=member_role,
        )
        logger.info(
            "intent=app_help workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )
        return GroundedChatResponse(
            answer=answer, grounded=True,
            insufficient_evidence=refusal_reason is not None, sources=[],
        )

    if intent.category == IntentCategory.CONVERSATION_HISTORY:
        answer = await _answer_conversation_history(
            intent=intent, workspace_id=workspace_id, user_id=principal.user_id,
        )
        logger.info(
            "intent=conversation_history workspace={ws} retrieval_called=False", ws=workspace_id,
        )
        return GroundedChatResponse(
            answer=answer, grounded=True, insufficient_evidence=False, sources=[],
        )

    if intent.category in (IntentCategory.WORKSPACE_METADATA, IntentCategory.DOCUMENT_LIST):
        answer, refusal_reason = await _answer_metadata_question(
            intent=intent, question=effective_query,
            workspace_id=workspace_id, user_id=principal.user_id,
        )
        logger.info(
            "intent={intent_type} sub={sub} workspace={ws} retrieval_called=False refusal={refusal}",
            intent_type=intent.category.value,
            sub=intent.metadata_sub.value if intent.metadata_sub else None,
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )
        # When the metadata handler can't resolve the specific field (sub is
        # None) and returns a refusal, fall through to document-content search
        # instead of dead-ending.  A well-spelled metadata question with a
        # recognized sub-intent (DOC_COUNT, DOC_LIST, etc.) is answered above;
        # an ambiguous or unresolvable one deserves a retrieval attempt, not a
        # canned "I could not determine" refusal.
        if refusal_reason == ResponseReason.METADATA_EMPTY and intent.metadata_sub is None:
            logger.info(
                "Metadata sub-intent unresolved for workspace={ws}, "
                "falling through to document content search",
                ws=workspace_id,
            )
        else:
            return GroundedChatResponse(
                answer=answer, grounded=True,
                insufficient_evidence=refusal_reason is not None, sources=[],
            )

    # --- Document content path (RAG) ---
    # Phase B-2: classify query shape + resolve doc target for filename-aware retrieval.
    from app.retrieval.doc_targeting import resolve_document_target
    async with tenant_session(
        workspace_id=workspace_id, user_id=principal.user_id
    ) as dt_db:
        doc_target_result = await resolve_document_target(
            session=dt_db,
            question=effective_query,
            workspace_id=workspace_id,
        )
    has_doc_target = doc_target_result is not None and doc_target_result.matched_document_id is not None
    query_shape = classify_query_shape(
        effective_query,
        has_doc_target=has_doc_target,
    )
    logger.info(
        "intent=document_content query_shape={shape} workspace={ws} "
        "retrieval_called=True filename_match={fm} matched_filename={mf} "
        "doc_target_confidence={dtc}",
        shape=query_shape.value, ws=workspace_id,
        fm=doc_target_result.matched_filename is not None,
        mf=doc_target_result.matched_filename,
        dtc=doc_target_result.confidence,
    )
    async with tenant_session(
        workspace_id=workspace_id, user_id=principal.user_id
    ) as session:
        result = await retrieve(
            session, query=effective_query, workspace_id=workspace_id,
            query_shape=query_shape,
            doc_target_result=doc_target_result,
            search_query=search_query_for_retrieval,
            qu_confidence=qu_confidence,
        )

    if not result.grounded:
        # Person-info questions that found no evidence get the unsupported-
        # information refusal — a specific person's identity/contact must never
        # be fabricated, even when retrieval came up empty.
        if _is_person_info_request(question) or _is_person_info_request(effective_query):
            refusal_reason = ResponseReason.UNSUPPORTED_INFORMATION
        else:
            refusal_reason = _pick_refusal_reason(had_candidates=bool(result.chunks))
        refusal = refusal_message(refusal_reason)
        logger.info(
            "intent=document_content refusal={reason} top_score={score} "
            "candidates={n} retrieval_called=True workspace={ws}",
            ws=workspace_id,
            score=result.top_score,
            n=len(result.chunks),
            reason=refusal_reason.value,
        )
        return GroundedChatResponse(
            answer=refusal,
            grounded=False,
            insufficient_evidence=True,
            sources=[],
        )

    messages = build_messages(question=effective_query, chunks=result.chunks)
    completion = Completion()
    try:
        async for _token in llm.stream(messages, completion=completion):
            pass
    except LLMError as exc:
        logger.error(
            "Generation failed for user {user} in workspace {ws}: {error}",
            user=principal.user_id,
            ws=workspace_id,
            error=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is currently unavailable. Please try again.",
        ) from exc

    # Strip model-injected thinking/reasoning blocks (e.g. Qwen3 `` tags)
    # BEFORE the empty-response check.  The raw completion may be non-empty
    # (it contains a thinking block) but the stripped answer is empty — that
    # must be treated as a generation failure, not passed to the frontend.
    answer_text = _strip_think_tags(completion.text)

    if not answer_text or not answer_text.strip():
        logger.warning(
            "LLM returned empty answer after stripping think tags "
            "for user {user} in workspace {ws}: grounded={grounded} "
            "insufficient_evidence={ie} sources={n} provider={provider}",
            user=principal.user_id,
            ws=workspace_id,
            grounded=result.grounded,
            ie=not result.grounded,
            n=len(result.chunks),
            provider=completion.provider or llm.name,
        )
        answer_text = (
            "I found relevant documents, but I couldn't generate an answer "
            "from them. Please try asking the question again."
        )
        sources = [_source(chunk) for chunk in result.chunks]
        return GroundedChatResponse(
            answer=answer_text,
            grounded=True,
            insufficient_evidence=False,
            sources=sources,
        )

    sources = [_source(chunk) for chunk in result.chunks]

    # ISSUE E: filter the sources returned to only those the answer actually cited.
    # The LLM cites [1], [2], ... matching the prompt position (index+1) of
    # `result.chunks`.  Any chunk that survived retrieval but was never referenced
    # by the answer is dropped, so the returned list has no noise citations.
    # If the answer contains no bracketed citations at all, we cannot determine
    # what was referenced — fall back to the full retrieved set rather than
    # dropping everything.
    cited_nums = cited_numbers(answer_text)
    if cited_nums:
        cited_set = set(cited_nums)
        cited_chunk_ids = {
            chunk.chunk_id
            for i, chunk in enumerate(result.chunks)
            if (i + 1) in cited_set
        }
        sources = [src for src in sources if src.chunk_id in cited_chunk_ids]
    logger.info(
        "Grounded answer for workspace {ws}: {n} sources, {tokens} completion tokens",
        ws=workspace_id,
        n=len(sources),
        tokens=completion.usage.completion_tokens,
    )
    return GroundedChatResponse(
        answer=answer_text,
        grounded=True,
        insufficient_evidence=False,
        sources=sources,
        provider=_display_provider_name(completion.provider or llm.name),
        model="",
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@router.get(
    "/sessions",
    response_model=ChatSessionListResponse,
    summary="List the user's chat sessions",
)
async def list_chat_sessions(
    principal: CurrentPrincipal,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChatSessionListResponse:
    """Sessions the caller owns in their workspace, most recently active first.

    ``title`` is derived from the first user message; ``updated_at`` from the
    most recent message (or the session's ``created_at`` when no messages exist).
    """
    workspace_id = principal.workspace_id

    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        sessions = (
            await db.execute(
                select(ChatSession)
                .where(
                    ChatSession.workspace_id == workspace_id,
                    ChatSession.user_id == principal.user_id,
                )
                .order_by(ChatSession.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).scalars().all()

        if not sessions:
            return ChatSessionListResponse(sessions=[])

        session_ids = [s.id for s in sessions]

        # First user message per session → title.
        title_sq = (
            select(
                ChatMessage.session_id,
                ChatMessage.content.label("title"),
                func.row_number()
                .over(
                    partition_by=ChatMessage.session_id,
                    order_by=ChatMessage.created_at,
                )
                .label("rn"),
            )
            .where(
                ChatMessage.session_id.in_(session_ids),
                ChatMessage.role == "user",
            )
            .subquery()
        )
        title_rows = await db.execute(
            select(title_sq.c.session_id, title_sq.c.title).where(
                title_sq.c.rn == 1
            )
        )
        titles: dict[uuid.UUID, str] = {
            row.session_id: row.title for row in title_rows
        }

        # Most recent message timestamp per session → updated_at.
        updated_rows = await db.execute(
            select(
                ChatMessage.session_id,
                func.max(ChatMessage.created_at).label("updated_at"),
            )
            .where(ChatMessage.session_id.in_(session_ids))
            .group_by(ChatMessage.session_id)
        )
        updated_at: dict[uuid.UUID, object] = {
            row.session_id: row.updated_at for row in updated_rows
        }

    result = [
        ChatSessionResponse(
            id=str(s.id),
            title=(titles[s.id][:100] if s.id in titles else None),
            created_at=s.created_at.isoformat(),
            updated_at=(
                updated_at[s.id].isoformat()  # type: ignore[union-attr]
                if s.id in updated_at
                else s.created_at.isoformat()
            ),
        )
        for s in sessions
    ]
    return ChatSessionListResponse(sessions=result)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ChatMessageListResponse,
    summary="Load a session's transcript",
)
async def get_chat_messages(
    principal: CurrentPrincipal,
    session_id: uuid.UUID,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> ChatMessageListResponse:
    """Messages in a session, oldest first.  Scoped to the caller's workspace."""
    workspace_id = principal.workspace_id

    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        # Verify the session belongs to this user in this workspace.
        session_exists = (
            await db.execute(
                select(ChatSession.id).where(
                    ChatSession.id == session_id,
                    ChatSession.workspace_id == workspace_id,
                    ChatSession.user_id == principal.user_id,
                )
            )
        ).scalar_one_or_none()
        if session_exists is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )

        rows = (
            await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
            )
        ).scalars().all()

    messages = [
        ChatMessageResponse(
            id=str(m.id),
            role=m.role,
            content=m.content,
            citations=m.sources if isinstance(m.sources, list) else [],
            created_at=m.created_at.isoformat(),
        )
        for m in rows
    ]
    return ChatMessageListResponse(messages=messages)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session and its messages",
)
async def delete_chat_session(
    principal: CurrentPrincipal,
    session_id: uuid.UUID,
) -> None:
    """Delete a session and all its messages.  Messages cascade-delete via FK."""
    workspace_id = principal.workspace_id

    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        existing = (
            await db.execute(
                select(ChatSession.id).where(
                    ChatSession.id == session_id,
                    ChatSession.workspace_id == workspace_id,
                    ChatSession.user_id == principal.user_id,
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        await db.execute(
            select(ChatSession)
            .where(
                ChatSession.id == session_id,
                ChatSession.workspace_id == workspace_id,
            )
        )
        # Explicit delete rather than relying solely on cascade — makes the
        # intent visible and survives a future FK change.
        from sqlalchemy import delete

        await db.execute(
            delete(ChatMessage).where(ChatMessage.session_id == session_id)
        )
        await db.execute(
            delete(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.workspace_id == workspace_id,
            )
        )

    logger.info(
        "Deleted chat session {sid} for workspace {ws}",
        sid=session_id,
        ws=workspace_id,
    )


__all__ = [
    "GroundedChatRequest",
    "GroundedChatResponse",
    "REFUSAL_ANSWER",
    "REFUSAL_NO_EVIDENCE",
    "REFUSAL_NOT_RELEVANT",
    "Source",
    "_is_metadata_question",
    "router",
]
