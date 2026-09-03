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
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, insert, select

from app.api.dependencies import get_generic_llm
from app.api.workspace_deps import assert_workspace_role
from app.db.models import ChatMessage, ChatSession, Document
from app.llm.base import Completion, LLMError, LLMProvider
from app.rag.prompts import build_messages
from app.retrieval.intent import (
    ConversationHistorySubIntent,
    Intent,
    IntentCategory,
    MetadataSubIntent,
    QueryShape,
    _DOC_SPECIFIC_DESCRIPTION_PATTERN,
    classify_intent,
    classify_intent_regex,
    classify_query_shape,
)
from app.retrieval.pipeline import RetrievedChunk, retrieve
from app.retrieval.query_rewrite import ChatTurn, RewriteResult, rewrite_query
from app.retrieval.refusals import ResponseReason, refusal_message
from app.security.auth import CurrentPrincipal
from app.security.rls import tenant_session

router = APIRouter(prefix="/chat", tags=["chat"])

# Kept for backward compatibility.
REFUSAL_NO_EVIDENCE = refusal_message(ResponseReason.NO_EVIDENCE)
REFUSAL_NOT_RELEVANT = refusal_message(ResponseReason.NOT_RELEVANT)
REFUSAL_ANSWER = REFUSAL_NO_EVIDENCE




# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


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


# Openers and closers for model-injected thinking/reasoning blocks.  Groq runs
# qwen/qwen3.x and OpenRouter runs gemini-2.5-flash, both of which can emit a
# reasoning preamble wrapped in control tags before the final answer.  The
# most common is Qwen3's ``<|start_of_thought|>...</<|end_of_thought|>``.
_THINK_OPENERS = re.compile(
    r"<\|?(?:start_of_thought|thinking_start|thinking|think)\|?>",
    re.IGNORECASE,
)
_THINK_CLOSERS = re.compile(
    r"<\|?(?:end_of_thought|thinking_end|/think|/thinking)\|?>",
    re.IGNORECASE,
)


def _strip_think_tags(text: str) -> str:
    """Remove model-injected thinking/reasoning blocks from LLM output.

    Some models (e.g. Qwen3 on Groq, Gemini-2.5 on OpenRouter) emit reasoning
    wrapped in tags like ``<|start_of_thought|>...</<|end_of_thought|>``.
    The user must see only the final answer.  This removes the entire block
    (tags and the reasoning text between them), and also drops anything after
    an opener that was never closed (a max-tokens cutoff mid-reasoning).
    """
    parts = _THINK_OPENERS.split(text)
    if len(parts) == 1:
        # No opener — nothing to strip.
        return text.strip()

    # Reassemble: keep the text before the first opener (parts[0]) and any
    # segment that comes after a closing tag.  Content inside an unclosed block
    # (no closer before the next opener or the end) is dropped.
    kept = [parts[0]]
    for segment in parts[1:]:
        closed = _THINK_CLOSERS.split(segment, maxsplit=1)
        if len(closed) == 2:
            # Opener ... closer: drop the reasoning, keep what follows.
            kept.append(closed[1])
        # else: unclosed block — drop it entirely.
    return "".join(kept).strip()


#: Length of the rolling lookahead buffer used by the streaming think filter.
#: Markers are short (``<|start_of_thought|>`` is 20 chars), so keeping the tail
#: well past that means any marker is captured whole inside the window before its
#: text could be emitted.  This bounds the added streaming latency to ~one short
#: phrase while still never leaking reasoning.
_THINK_TAIL = 40


async def _stream_think_filtered(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Wrap an LLM token stream, suppressing think-block reasoning text.

    The streaming path must not send reasoning to the client, so this filters
    tokens as they arrive rather than relying on a post-hoc strip.  It keeps a
    rolling buffer and only yields text once it is known to be outside a
    reasoning block (``<|start_of_thought|>...<|end_of_thought|>`` or
    ``<thinking>...</thinking>``).  Markers that arrive split across stream
    tokens are still caught because we scan the whole buffered window.  An
    unclosed block (max-tokens cutoff mid-reasoning) is discarded entirely.
    """
    buf = ""
    in_think = False
    async for token in source:
        buf += token
        while True:
            if in_think:
                close = _THINK_CLOSERS.search(buf)
                if close is not None:
                    buf = buf[close.end():]
                    in_think = False
                else:
                    # Still inside reasoning — hold.  Keep only the tail so an
                    # arbitrarily long reasoning block cannot grow memory.
                    if len(buf) > _THINK_TAIL:
                        buf = buf[-_THINK_TAIL:]
                    break
            else:
                open_m = _THINK_OPENERS.search(buf)
                close_m = _THINK_CLOSERS.search(buf)
                if open_m is not None and (close_m is None or open_m.start() < close_m.start()):
                    # Emit everything before the opener, drop the opener, hide.
                    if open_m.start():
                        yield buf[:open_m.start()]
                    buf = buf[open_m.end():]
                    in_think = True
                elif close_m is not None:
                    # Stray closer with no preceding opener — drop just the closer.
                    if close_m.start():
                        yield buf[:close_m.start()]
                    buf = buf[close_m.end():]
                else:
                    # No markers in the buffered window: emit all but the tail,
                    # which we hold back in case a marker starts at the boundary.
                    if len(buf) > _THINK_TAIL:
                        yield buf[:-_THINK_TAIL]
                        buf = buf[-_THINK_TAIL:]
                    break
    # End of stream.
    if in_think:
        return  # unclosed thinking block — discard the remainder (caught above too)
    if buf:
        yield buf


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
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
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
            # Build the query — optionally filter by status.
            stmt = select(func.count()).select_from(Member).where(
                Member.workspace_id == workspace_id,
            )
            if intent.member_status:
                stmt = stmt.where(Member.status == intent.member_status)
            count = (await db.execute(stmt)).scalar_one()
            status_label = (intent.member_status or "ACTIVE").lower()
            if count == 0:
                if intent.member_status:
                    return (
                        f"There are no {status_label} members in this workspace.",
                        ResponseReason.METADATA_EMPTY,
                    )
                return "There are no members in this workspace yet.", ResponseReason.METADATA_EMPTY
            word = "member" if count == 1 else "members"
            verb = "is" if count == 1 else "are"
            if intent.member_status:
                return f"There {verb} {count} {status_label} {word} in this workspace.", None
            return f"There {verb} {count} {word} in this workspace.", None

        if sub == MetadataSubIntent.MEMBER_LIST:
            stmt = (
                select(Member.user_id, Member.role, Member.status)
                .where(Member.workspace_id == workspace_id)
            )
            if intent.member_status:
                stmt = stmt.where(Member.status == intent.member_status)
            rows = (
                await db.execute(stmt.order_by(Member.created_at.asc()))
            ).all()
            if not rows:
                if intent.member_status:
                    status_label = intent.member_status.lower()
                    return f"There are no {status_label} members in this workspace.", ResponseReason.METADATA_EMPTY
                return "There are no members in this workspace yet.", ResponseReason.METADATA_EMPTY
            items = [
                f"- User {str(row.user_id)[:8]}... (role: {row.role}, status: {row.status})"
                for row in rows
            ]
            count = len(rows)
            word = "member" if count == 1 else "members"
            verb = "is" if count == 1 else "are"
            status_label = (intent.member_status or "all").lower()
            header = f"There {verb} {count} {status_label} {word} in this workspace:"
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


def _answer_identity_assistant() -> str:
    """Answer a question about what/who the assistant is.

    No database query, no retrieval, no LLM call.
    """
    return (
        "I'm a company knowledge assistant. I help you find information from "
        "your workspace's approved documents. You can ask me questions about "
        "uploaded documents, and I'll answer with citations from the relevant "
        "sources. I can also help with workspace metadata like member counts, "
        "document counts, and roles."
    )


async def _answer_identity_user(
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    question: str = "",
) -> tuple[str, ResponseReason]:
    """Answer a question about the user's own info, or acknowledge a statement.

    Handles both questions ("what is my info") and statements ("my name is X").
    Returns (answer, refusal_reason).
    """
    from app.db.models import Member

    q_lower = question.lower().strip()

    # Detect if this is a statement ("my name is X") vs a question.
    is_statement = bool(
        re.match(r"^(?:my\s+name\s+is|i\s+am|i'm|call\s+me)\b", q_lower)
    )

    if is_statement:
        # Acknowledge the statement — we can't store it, but we should not
        # pretend we didn't hear it.
        return (
            "Thanks for letting me know! I don't store personal information "
            "like names, but I can help you find information from your "
            "workspace's approved documents. What would you like to know?",
            None,
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

    # 1a. Classify intent: LLM-first with cache, regex as failure fallback.
    effective_query = question  # may be overridden by rewrite below
    needs_clarification = False
    refusal_reason: ResponseReason | None = None
    history_turns: list[ChatTurn] = []

    # Load history for context (needed by LLM router and query rewrite).
    history_turns = await _load_recent_history(
        workspace_id=workspace_id,
        user_id=principal.user_id,
        session_id=payload.session_id,
    )
    history_dicts = [{"role": t.role, "content": t.content} for t in history_turns]

    # LLM-first classification with workspace knowledge.
    intent = await classify_intent(
        question,
        history=history_dicts,
        workspace_id=workspace_id,
    )

    # 1b. Rewrite for document_content intents.
    if intent.category == IntentCategory.DOCUMENT_CONTENT:
        rewrite_result = await rewrite_query(query=question, history=history_turns)
        effective_query = rewrite_result.rewritten_query
        needs_clarification = rewrite_result.needs_clarification
        # Re-classify after rewrite in case the rewritten query changed the intent.
        if rewrite_result.status == "success" and effective_query != question:
            intent = await classify_intent(
                effective_query,
                history=history_dicts,
                workspace_id=workspace_id,
            )
    elif intent.category == IntentCategory.AMBIGUOUS:
        needs_clarification = True

    # 1c. Handle ambiguity: ask for clarification instead of refusing.
    if needs_clarification or intent.needs_clarification:
        # Create session and persist.
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            row = (
                await db.execute(
                    insert(ChatSession)
                    .values(workspace_id=workspace_id, user_id=principal.user_id)
                    .returning(ChatSession.id)
                )
            ).scalar_one()
            clarify_session_id = row
        yield await _sse_event("session", {"session_id": str(clarify_session_id)})
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=clarify_session_id, role="user", content=question,
                )
            )
        clarification_text = refusal_message(ResponseReason.NEEDS_CLARIFICATION)
        yield await _sse_event("sources", {"sources": []})
        yield await _sse_event("token", {"text": clarification_text})
        yield await _sse_event("citations", {"citations": []})
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
            await db.execute(
                insert(ChatMessage).values(
                    session_id=clarify_session_id, role="assistant",
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
        answer = _answer_general_conversation(question=effective_query)
        logger.info(
            "intent=general_conversation workspace={ws} retrieval_called=False",
            ws=workspace_id,
        )

    elif intent.category == IntentCategory.OUT_OF_SCOPE:
        answer = refusal_message(ResponseReason.OUT_OF_SCOPE)
        refusal_reason = ResponseReason.OUT_OF_SCOPE
        logger.info(
            "intent=out_of_scope workspace={ws} retrieval_called=False",
            ws=workspace_id,
        )

    elif intent.category == IntentCategory.IDENTITY_ASSISTANT:
        answer = _answer_identity_assistant()
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

    elif intent.category == IntentCategory.PERMISSIONS:
        answer, refusal_reason = _answer_workspace_permission(
            question=effective_query,
            principal_role=member_role,
        )
        logger.info(
            "intent=permissions workspace={ws} retrieval_called=False refusal={refusal}",
            ws=workspace_id, refusal=refusal_reason.value if refusal_reason else None,
        )

    elif intent.category == IntentCategory.WORKSPACE_PERMISSION:
        answer, refusal_reason = _answer_workspace_permission(
            question=effective_query,
            principal_role=member_role,
        )
        logger.info(
            "intent=workspace_permission workspace={ws} retrieval_called=False refusal={refusal}",
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
            session_id=payload.session_id,
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

    # For non-document intents, emit the answer and persist.
    if answer is not None:
        # We still need a session to persist the turn.
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
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
        assert session_id is not None
        yield await _sse_event("session", {"session_id": str(session_id)})
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
        return

    # --- Document content path (RAG) ---
    # 2. Create or verify session.
    session_id: uuid.UUID | None = None
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
            else:
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
        else:
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

    assert session_id is not None
    yield await _sse_event("session", {"session_id": str(session_id)})

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
        result = await retrieve(
            db, query=effective_query, workspace_id=workspace_id,
            query_shape=query_shape,
            doc_target_result=doc_target_result,
        )

    if not result.grounded:
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
        # Emit empty sources, the refusal text as a token, and done.
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
    messages = build_messages(question=effective_query, chunks=result.chunks)
    completion = Completion()
    full_text = ""
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
    except LLMError as exc:
        logger.error(
            "Generation failed for user {user} in workspace {ws}: {error}",
            user=principal.user_id,
            ws=workspace_id,
            error=exc,
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

    if not full_text.strip():
        yield await _sse_event(
            "error",
            {
                "detail": "The language model returned an empty response. Please try again.",
                "partial": False,
            },
        )
        return

    # 7. Emit citations.
    yield await _sse_event("citations", {"citations": sources_list})

    # 8. Persist assistant message.
    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        await db.execute(
            insert(ChatMessage).values(
                session_id=session_id,
                role="assistant",
                content=full_text,
                sources=sources_list,
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


@router.post(
    "",
    response_class=StreamingResponse,
    summary="SSE streaming chat — what the frontend calls",
)
async def chat_stream(
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
async def grounded_chat(
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

    # LLM-first classification with workspace knowledge.
    effective_query = question
    history_turns: list[ChatTurn] = []

    # Load history for context.
    history_turns = await _load_recent_history(
        workspace_id=workspace_id,
        user_id=principal.user_id,
    )
    history_dicts = [{"role": t.role, "content": t.content} for t in history_turns]

    intent = await classify_intent(
        question,
        history=history_dicts,
        workspace_id=workspace_id,
    )

    # Only rewrite for document_content intents.
    if intent.category == IntentCategory.DOCUMENT_CONTENT:
        rewrite_result = await rewrite_query(query=question, history=history_turns)
        effective_query = rewrite_result.rewritten_query
        if rewrite_result.needs_clarification:
            return GroundedChatResponse(
                answer=refusal_message(ResponseReason.NEEDS_CLARIFICATION),
                grounded=True,
                insufficient_evidence=False,
                sources=[],
            )
        if rewrite_result.status == "success" and effective_query != question:
            intent = await classify_intent(
                effective_query,
                history=history_dicts,
                workspace_id=workspace_id,
            )
    elif intent.category == IntentCategory.AMBIGUOUS:
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
        answer = _answer_general_conversation(question=effective_query)
        logger.info(
            "intent=general_conversation workspace={ws} retrieval_called=False", ws=workspace_id,
        )
        return GroundedChatResponse(
            answer=answer, grounded=True, insufficient_evidence=False, sources=[],
        )

    if intent.category == IntentCategory.OUT_OF_SCOPE:
        logger.info("intent=out_of_scope workspace={ws} retrieval_called=False", ws=workspace_id)
        return GroundedChatResponse(
            answer=refusal_message(ResponseReason.OUT_OF_SCOPE),
            grounded=True, insufficient_evidence=False, sources=[],
        )

    if intent.category == IntentCategory.IDENTITY_ASSISTANT:
        answer = _answer_identity_assistant()
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

    if intent.category == IntentCategory.PERMISSIONS:
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

    if intent.category == IntentCategory.WORKSPACE_PERMISSION:
        answer, refusal_reason = _answer_workspace_permission(
            question=effective_query, principal_role=member_role,
        )
        logger.info(
            "intent=workspace_permission workspace={ws} retrieval_called=False refusal={refusal}",
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
        )

    if not result.grounded:
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

    if not completion.text.strip():
        logger.error(
            "Provider returned an empty completion for user {user} in workspace {ws}",
            user=principal.user_id,
            ws=workspace_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model returned an empty response. Please try again.",
        )

    sources = [_source(chunk) for chunk in result.chunks]
    # Strip model-injected thinking/reasoning blocks (e.g. Qwen3 `` tags).
    answer_text = _strip_think_tags(completion.text)
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
