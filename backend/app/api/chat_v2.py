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
from app.retrieval.pipeline import RetrievedChunk, retrieve
from app.security.auth import CurrentPrincipal
from app.security.rls import tenant_session

router = APIRouter(prefix="/chat", tags=["chat"])#: Returned when retrieval finds zero candidate chunks.
REFUSAL_NO_EVIDENCE = (
    "I couldn't find any relevant information about that topic in your uploaded documents."
)

#: Returned when retrieval finds chunks but none pass the grounding threshold.
REFUSAL_NOT_RELEVANT = (
    "Your workspace contains documents, but none of them contain information"
    " about that specific topic. Try rephrasing your question or check that"
    " the relevant document has been uploaded."
)

# Kept for backward compatibility; prefer the specific variants above.
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
    r"(?:list|show|what|which|name|who)\s+"
    r"(?:are\s+the\s+)?(?:me\s+)?(?:all\s+)?"
    r"(?:the\s+|this\s+|our\s+)?(?:workspace\s+)?"
    r"(?:people|members?|users?|employees?|team\s*members?|contributors?)",
    re.IGNORECASE,
)

_ROLE_PATTERN = re.compile(
    r"(?:what\s+is\s+my|my\s+current|what\s+role\s+(?:do\s+i|am\s+i))"
    r"\s+"
    r"(?:role|access|permission|level)",
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
    question: str,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
) -> str:
    """Answer a metadata question directly from the database.

    No retrieval, no reranking, no LLM call.
    """
    from app.db.models import Member

    intent = _is_metadata_question(question)
    normalised = question.strip().rstrip("?").rstrip(".").strip()
    this_month = bool(_THIS_MONTH_PATTERN.search(normalised))

    async with tenant_session(workspace_id=workspace_id, user_id=user_id) as db:
        if intent == "count":
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
                    return "You have no uploaded documents in this workspace this month."
                return "You have no uploaded documents in this workspace."
            word = "document" if count == 1 else "documents"
            if this_month:
                return f"You have {count} uploaded {word} in this workspace this month."
            return f"You have {count} uploaded {word} in this workspace."

        if intent == "list":
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
                return "You have no uploaded documents in this workspace."
            items = [f"- {row.filename}" for row in rows]
            count = len(rows)
            word = "document" if count == 1 else "documents"
            header = f"You have {count} uploaded {word} in this workspace:"
            return header + "\n" + "\n".join(items)

        if intent == "member_count":
            count = (
                await db.execute(
                    select(func.count()).select_from(Member).where(
                        Member.workspace_id == workspace_id,
                        Member.status == "ACTIVE",
                    )
                )
            ).scalar_one()
            if count == 0:
                return "There are no members in this workspace yet."
            word = "member" if count == 1 else "members"
            verb = "is" if count == 1 else "are"
            return f"There {verb} {count} {word} in this workspace."

        if intent == "member_list":
            rows = (
                await db.execute(
                    select(Member.user_id, Member.role, Member.status)
                    .where(
                        Member.workspace_id == workspace_id,
                        Member.status == "ACTIVE",
                    )
                    .order_by(Member.created_at.asc())
                )
            ).all()
            if not rows:
                return "There are no members in this workspace yet."
            items = [f"- User {str(row.user_id)[:8]}... (role: {row.role})" for row in rows]
            count = len(rows)
            word = "member" if count == 1 else "members"
            verb = "is" if count == 1 else "are"
            header = f"There {verb} {count} {word} in this workspace:"
            return header + "\n" + "\n".join(items)

        if intent == "role":
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
                return "You are not an active member of this workspace."
            return f"Your role in this workspace is {rows[0].role}."

    return "I could not determine what metadata you are asking about."


def _pick_refusal(had_candidates: bool) -> str:
    """Choose the right refusal message based on retrieval output."""
    return REFUSAL_NOT_RELEVANT if had_candidates else REFUSAL_NO_EVIDENCE


# SSE streaming chat endpoint  (POST /chat)
# ---------------------------------------------------------------------------


async def _stream_chat(
    principal: CurrentPrincipal,
    payload: ChatStreamRequest,
    llm: LLMProvider,
) -> AsyncIterator[str]:
    """SSE generator: retrieval → grounding → LLM stream → persist."""

    workspace_id = principal.workspace_id
    question = payload.message.strip()

    # 1. Workspace membership check (before any DB write).
    await assert_workspace_role(workspace_id, principal)

    # 1a. Metadata questions bypass the retrieval pipeline entirely.
    metadata_intent = _is_metadata_question(question)
    if metadata_intent is not None:
        answer = await _answer_metadata_question(
            question=question,
            workspace_id=workspace_id,
            user_id=principal.user_id,
        )
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
        logger.info(
            "Metadata route: intent={intent} source=database workspace={ws}",
            ws=workspace_id,
            intent=metadata_intent,
        )
        return

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
    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as db:
        result = await retrieve(db, query=question, workspace_id=workspace_id)

    if not result.grounded:
        # Choose the right refusal: documents exist but irrelevant, or nothing found.
        refusal = _pick_refusal(had_candidates=bool(result.chunks))
        logger.info(
            "Refused ungrounded question for workspace {ws} (top_score={score}, "
            "candidates={n}): {reason}",
            ws=workspace_id,
            score=result.top_score,
            n=len(result.chunks),
            reason="not_relevant" if result.chunks else "no_evidence",
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
    messages = build_messages(question=question, chunks=result.chunks)
    completion = Completion()
    full_text = ""
    try:
        async for token in llm.stream(messages, completion=completion):
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
            "provider": completion.provider or llm.name,
            "model": completion.model or llm.model,
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
    return StreamingResponse(
        _stream_chat(principal, payload, llm),
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
    await assert_workspace_role(workspace_id, principal)

    # Metadata questions bypass the retrieval pipeline entirely.
    metadata_intent = _is_metadata_question(question)
    if metadata_intent is not None:
        answer = await _answer_metadata_question(
            question=question,
            workspace_id=workspace_id,
            user_id=principal.user_id,
        )
        logger.info(
            "Metadata route: intent={intent} source=database workspace={ws}",
            ws=workspace_id,
            intent=metadata_intent,
        )
        return GroundedChatResponse(
            answer=answer,
            grounded=True,
            insufficient_evidence=False,
            sources=[],
        )

    async with tenant_session(
        workspace_id=workspace_id, user_id=principal.user_id
    ) as session:
        result = await retrieve(session, query=question, workspace_id=workspace_id)

    if not result.grounded:
        refusal = _pick_refusal(had_candidates=bool(result.chunks))
        logger.info(
            "Refused ungrounded question for workspace {ws} (top_score={score}, "
            "candidates={n}): {reason}",
            ws=workspace_id,
            score=result.top_score,
            n=len(result.chunks),
            reason="not_relevant" if result.chunks else "no_evidence",
        )
        return GroundedChatResponse(
            answer=refusal,
            grounded=False,
            insufficient_evidence=True,
            sources=[],
        )

    messages = build_messages(question=question, chunks=result.chunks)
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
    logger.info(
        "Grounded answer for workspace {ws}: {n} sources, {tokens} completion tokens",
        ws=workspace_id,
        n=len(sources),
        tokens=completion.usage.completion_tokens,
    )
    return GroundedChatResponse(
        answer=completion.text,
        grounded=True,
        insufficient_evidence=False,
        sources=sources,
        provider=completion.provider or llm.name,
        model=completion.model or llm.model,
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
