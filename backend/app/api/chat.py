"""Streaming RAG chat (Phase 4).

The response is Server-Sent Events. Each event is a typed JSON envelope rather than bare
text, so the client can distinguish tokens from citations from errors without parsing
prose — and so an error that happens mid-stream is still reportable after a 200 header has
already gone out.

Persistence deliberately happens after the stream completes: writing the assistant's turn
requires knowing the whole answer and its citations, and holding a transaction open across
a multi-second generation would pin a pooled connection for its duration.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import case, desc, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_llm_router
from app.config import get_settings
from app.db.models import ChatMessage, ChatSession
from app.llm.base import LLMError, LLMRouterProtocol, Message
from app.rag.pipeline import AnswerResult, retrieve_for_question, stream_answer
from app.security.auth import CurrentPrincipal, Principal
from app.security.rate_limit import CHAT_RATE_LIMIT, limiter
from app.security.rls import tenant_session

router = APIRouter(prefix="/chat", tags=["chat"])

#: Roles replayed into the prompt. Tool turns are internal bookkeeping, not conversation.
_HISTORY_ROLES = ("user", "assistant")

#: Both rows of a turn are written in one transaction, and Postgres `now()` is
#: transaction-start time — so they share an identical `created_at`. Ordering by timestamp
#: alone would leave the question and its answer in arbitrary order; this breaks the tie.
_ROLE_ORDER = case((ChatMessage.role == "user", 0), else_=1)


class ChatRequest(BaseModel):
    """A question, optionally continuing an existing session."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    session_id: uuid.UUID | None = None


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionResponse]


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: datetime


class ChatMessageListResponse(BaseModel):
    messages: list[ChatMessageResponse]


def _sse(event: str, data: dict[str, Any]) -> str:
    """Encode one SSE frame.

    json.dumps cannot emit a raw newline inside a string, so the payload can never break
    out of the single `data:` line the SSE framing depends on.
    """
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def _load_history(
    session: AsyncSession, *, session_id: uuid.UUID, limit: int
) -> list[Message]:
    """Recent turns of an existing conversation, oldest first."""
    rows = (
        await session.execute(
            select(ChatMessage.role, ChatMessage.content)
            .where(
                ChatMessage.session_id == session_id,
                ChatMessage.role.in_(_HISTORY_ROLES),
            )
            .order_by(desc(ChatMessage.created_at), desc(_ROLE_ORDER))
            .limit(limit)
        )
    ).all()
    return [Message(role=row.role, content=row.content) for row in reversed(rows)]


async def _ensure_session(
    session: AsyncSession, *, principal: Principal, session_id: uuid.UUID | None, question: str
) -> uuid.UUID:
    """Return the id of the session to append to, creating one if needed.

    A client-supplied session_id is verified under RLS before use: it arrived over the
    wire and may name another user's conversation.
    """
    if session_id is not None:
        exists = (
            await session.execute(select(ChatSession.id).where(ChatSession.id == session_id))
        ).scalar_one_or_none()
        if exists is None:
            # RLS already scoped this to the caller, so invisible and absent are the same
            # case — 404 for both avoids confirming that another user's session exists.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found."
            )
        return session_id

    title = question.strip()[:80] or None
    return (
        await session.execute(
            insert(ChatSession)
            .values(org_id=principal.org_id, user_id=principal.user_id, title=title)
            .returning(ChatSession.id)
        )
    ).scalar_one()


async def _persist_turn(
    *,
    principal: Principal,
    session_id: uuid.UUID,
    question: str,
    result: AnswerResult,
    incomplete: bool = False,
) -> None:
    """Record the user's question and the assistant's answer.

    `incomplete` marks a turn whose generation stopped early. The partial text is still
    stored: the user saw it, so a transcript that omits it would not match the
    conversation they remember. The flag lets the UI say so rather than presenting a
    truncated answer as finished.
    """
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        rows: list[dict[str, Any]] = [
            {
                "org_id": principal.org_id,
                "user_id": principal.user_id,
                "session_id": session_id,
                "role": "user",
                "content": question,
                "citations": [],
                "token_usage": None,
            }
        ]
        # An assistant row is only written if there is something to show. A failure before
        # the first token leaves the question recorded and nothing else, which is the
        # honest record of what happened.
        if result.text:
            rows.append(
                {
                    "org_id": principal.org_id,
                    "user_id": principal.user_id,
                    "session_id": session_id,
                    "role": "assistant",
                    "content": result.text,
                    "citations": [citation.as_dict() for citation in result.citations],
                    "token_usage": {
                        **result.completion.usage.as_dict(),
                        "provider": result.completion.provider,
                        "model": result.completion.model,
                        "incomplete": incomplete,
                    },
                }
            )
        await session.execute(insert(ChatMessage), rows)
        await session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(updated_at=datetime.now().astimezone())
        )


async def _event_stream(
    *,
    principal: Principal,
    llm: LLMRouterProtocol,
    question: str,
    session_id: uuid.UUID,
    history: list[Message],
    chunks: list[Any],
) -> AsyncIterator[str]:
    result = AnswerResult()

    yield _sse("session", {"session_id": str(session_id)})
    # Sent before generation so the UI can show what is being consulted while it waits,
    # and so a user can see an answer rests on nothing at all.
    yield _sse(
        "sources",
        {
            "sources": [
                {
                    "number": index,
                    "document_id": str(chunk.document_id),
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "label": chunk.citation_label,
                }
                for index, chunk in enumerate(chunks, start=1)
            ]
        },
    )

    try:
        async for token in stream_answer(
            llm, question=question, chunks=chunks, history=history, result=result
        ):
            yield _sse("token", {"text": token})
    except LLMError as exc:
        # The 200 header is long gone, so the failure has to travel as an event.
        logger.opt(exception=exc).error(
            "Generation failed for user {user}", user=principal.user_id
        )
        yield _sse(
            "error",
            {"detail": "The language model is currently unavailable. Please try again."},
        )
        return
    except Exception as exc:
        logger.opt(exception=exc).error(
            "Unexpected error streaming answer for user {user}", user=principal.user_id
        )
        yield _sse("error", {"detail": "Something went wrong generating the answer."})
        return

    yield _sse(
        "citations", {"citations": [citation.as_dict() for citation in result.citations]}
    )
    yield _sse(
        "done",
        {
            "usage": result.completion.usage.as_dict(),
            "provider": result.completion.provider,
            "model": result.completion.model,
            "grounded": result.had_context,
        },
    )

    logger.info(
        "Chat answered for user {user}: {tokens} completion tokens via {provider}, "
        "{sources} sources",
        user=principal.user_id,
        tokens=result.completion.usage.completion_tokens,
        provider=result.completion.provider,
        sources=len(result.citations),
    )

    try:
        await _persist_turn(
            principal=principal, session_id=session_id, question=question, result=result
        )
    except (IntegrityError, OSError) as exc:
        # The user already has their answer; losing the transcript must not retroactively
        # turn a successful response into an error.
        logger.opt(exception=exc).error(
            "Could not persist chat turn for session {session}", session=session_id
        )


@router.post("", summary="Ask a question over the organization's documents")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(
    request: Request,  # noqa: ARG001 — required by slowapi's decorator
    principal: CurrentPrincipal,
    payload: ChatRequest,
    llm: Annotated[LLMRouterProtocol, Depends(get_llm_router)],
) -> StreamingResponse:
    """Stream a cited answer as Server-Sent Events."""
    question = payload.message.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Message cannot be empty."
        )

    settings = get_settings()

    # Retrieval and history run in their own short transaction, which closes before the
    # LLM call begins.
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        session_id = await _ensure_session(
            session, principal=principal, session_id=payload.session_id, question=question
        )
        history = (
            await _load_history(
                session, session_id=session_id, limit=settings.chat_history_limit
            )
            if payload.session_id is not None
            else []
        )
        chunks = await retrieve_for_question(
            session, question=question, org_id=principal.org_id
        )

    return StreamingResponse(
        _event_stream(
            principal=principal,
            llm=llm,
            question=question,
            session_id=session_id,
            history=history,
            chunks=chunks,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Nginx and similar proxies buffer by default, which would defeat streaming.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=ChatSessionListResponse, summary="List chat sessions")
async def list_sessions(
    principal: CurrentPrincipal,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChatSessionListResponse:
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        rows = (
            await session.execute(
                select(ChatSession)
                .order_by(desc(ChatSession.updated_at))
                .limit(limit)
                .offset(offset)
            )
        ).scalars()
        sessions = [ChatSessionResponse.model_validate(row) for row in rows]
    return ChatSessionListResponse(sessions=sessions)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ChatMessageListResponse,
    summary="Read a session's transcript",
)
async def list_messages(
    principal: CurrentPrincipal,
    session_id: uuid.UUID,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> ChatMessageListResponse:
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        owns = (
            await session.execute(select(ChatSession.id).where(ChatSession.id == session_id))
        ).scalar_one_or_none()
        if owns is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found."
            )

        rows = (
            await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at, _ROLE_ORDER)
                .limit(limit)
            )
        ).scalars()
        messages = [ChatMessageResponse.model_validate(row) for row in rows]
    return ChatMessageListResponse(messages=messages)


__all__ = ["router"]
