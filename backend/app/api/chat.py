"""Streaming multi-agent chat (Phase 7).

A question is answered by the LangGraph supervisor rather than by the RAG pipeline
directly: the supervisor routes it to the document, business-data and code agents — one,
several or none of them — and synthesizes a single answer over whatever they returned.

The response is Server-Sent Events. Each event is a typed JSON envelope rather than bare
text, so the client can distinguish tokens from citations from errors without parsing
prose — and so an error that happens mid-stream is still reportable after a 200 header has
already gone out.

Persistence deliberately happens after the stream completes: writing the assistant's turn
requires knowing the whole answer and its citations, and holding a transaction open across
a multi-second generation would pin a pooled connection for its duration.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import case, delete, desc, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import stream_supervisor
from app.agents.state import outcomes
from app.api.dependencies import get_llm_router
from app.config import get_settings
from app.db.models import ChatMessage, ChatSession
from app.llm.base import Completion, LLMError, LLMRouterProtocol, Message
from app.observability import capture_exception, record_token_usage
from app.rag.pipeline import cited_numbers
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
    #: True when generation stopped early. Reloading a conversation must not present a
    #: truncated answer as a finished one just because the stream that produced it is over.
    incomplete: bool = False
    #: Which sources the supervisor consulted for this turn.
    routes: list[str] = []
    #: The validated SELECT behind a business-data answer, if one ran.
    sql_query: str = ""

    @model_validator(mode="before")
    @classmethod
    def _lift_generation_metadata(cls, value: Any) -> Any:
        """Surface generation metadata out of the stored token_usage blob.

        It lives there rather than in columns of its own because it is all metadata about
        one generation, and separate columns would mean a migration per field the UI reads.
        """
        usage = getattr(value, "token_usage", None)
        if not isinstance(usage, dict):
            return value
        return {
            "id": value.id,
            "role": value.role,
            "content": value.content,
            "citations": value.citations,
            "created_at": value.created_at,
            "incomplete": bool(usage.get("incomplete")),
            "routes": usage.get("routes") or [],
            "sql_query": usage.get("sql_query") or "",
        }


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


@dataclass
class _TurnResult:
    """What the supervisor produced for one turn, as the persistence layer needs it.

    Filled in as the stream proceeds and stays consistent with the text the user has
    actually seen, so a turn cut short partway still records what happened.
    """

    text: str = ""
    #: Already in wire shape — the supervisor resolved these against the retrieved chunks.
    citations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    completion: Completion = field(default_factory=Completion)
    routes: list[str] = field(default_factory=list)
    #: The validated SELECT the SQL agent ran, if any. Stored so reloading a conversation
    #: can still show how a figure was obtained.
    sql_query: str = ""
    #: Whether any agent actually supplied evidence. Not derivable from `sources` alone: a
    #: SQL- or code-only answer is grounded in material that never appears as a citation.
    grounded: bool = False


async def _persist_turn(
    *,
    principal: Principal,
    session_id: uuid.UUID,
    question: str,
    result: _TurnResult,
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
                    "citations": result.citations,
                    # Generation metadata shares the one JSON column rather than each field
                    # earning a migration for something only the UI reads.
                    "token_usage": {
                        **result.completion.usage.as_dict(),
                        "provider": result.completion.provider,
                        "model": result.completion.model,
                        "incomplete": incomplete,
                        "routes": result.routes,
                        "sql_query": result.sql_query,
                    },
                }
            )
        await session.execute(insert(ChatMessage), rows)
        # Bumps the session to the top of the sidebar. `func.now()` is transaction time,
        # so it matches the created_at the messages in this same transaction receive.
        await session.execute(
            update(ChatSession).where(ChatSession.id == session_id).values(updated_at=func.now())
        )


async def _save_turn(
    *,
    principal: Principal,
    session_id: uuid.UUID,
    question: str,
    result: _TurnResult,
    incomplete: bool,
) -> None:
    """Persist one turn, absorbing storage failures."""
    try:
        await _persist_turn(
            principal=principal,
            session_id=session_id,
            question=question,
            result=result,
            incomplete=incomplete,
        )
    except Exception as exc:
        # The user already has their answer; losing the transcript must not retroactively
        # turn a successful response into an error. Reported, because a chat that answers
        # correctly and saves nothing looks entirely healthy from outside.
        logger.opt(exception=exc).error(
            "Could not persist chat turn for session {session}", session=session_id
        )
        capture_exception(exc, subsystem="chat_persistence")


async def _shielded_save(
    *,
    principal: Principal,
    session_id: uuid.UUID,
    question: str,
    result: _TurnResult,
    incomplete: bool,
) -> None:
    """Run the write to completion even if the surrounding task is being cancelled.

    When a client disconnects mid-answer, Starlette cancels the response task. An
    unshielded write would then be interrupted at whichever statement it had reached,
    committing the question with no answer — or nothing at all. The task must be created
    before shielding: `shield` only protects an already-scheduled task, not a coroutine
    it is still awaiting inline.
    """
    task = asyncio.create_task(
        _save_turn(
            principal=principal,
            session_id=session_id,
            question=question,
            result=result,
            incomplete=incomplete,
        )
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # The shield was cancelled from outside, not the task. Give the write the moment
        # it needs to finish rather than abandoning a half-open transaction.
        await task
        raise


def _absorb_final(result: _TurnResult, state: dict[str, Any]) -> None:
    """Copy the supervisor's final state into the turn record.

    Read defensively with `.get`: a run that ended at an early node never wrote the later
    keys, and a missing key there is an ordinary outcome rather than a fault.
    """
    result.text = state.get("answer") or result.text
    result.citations = state.get("citations") or []
    result.sources = state.get("sources") or result.sources
    result.sql_query = state.get("sql_query") or ""
    result.routes = list(state.get("routes") or [])
    # An agent that succeeded and returned content is evidence, whether or not it produced
    # anything citable — reporting a SQL-only answer as ungrounded would misread it as a
    # guess.
    result.grounded = any(
        outcome.ok and outcome.has_content for outcome in outcomes(state)  # type: ignore[arg-type]
    )
    completion = state.get("completion")
    if isinstance(completion, Completion):
        result.completion = completion


def _partial_citations(result: _TurnResult) -> list[dict[str, Any]]:
    """Citations for text that was streamed before the run failed.

    Synthesis resolves citations only after it finishes, so a failure partway through
    leaves the partial answer on screen with none. The partial text is still cited against
    `sources` — what was actually retrieved and sent — so a number the model invented
    resolves to nothing, exactly as on the success path.
    """
    if result.citations or not result.text:
        return result.citations
    by_number = {source["number"]: source for source in result.sources}
    return [by_number[n] for n in cited_numbers(result.text) if n in by_number]


async def _event_stream(
    *,
    principal: Principal,
    llm: LLMRouterProtocol,
    question: str,
    session_id: uuid.UUID,
    history: list[Message],
) -> AsyncIterator[str]:
    result = _TurnResult()
    # Assume the worst until the `done` frame is reached. Every early exit — provider
    # failure, client disconnect, cancelled task — leaves this true, so the flag records
    # what happened without each exit path having to remember to set it.
    incomplete = True

    try:
        yield _sse("session", {"session_id": str(session_id)})

        try:
            async for kind, payload in stream_supervisor(
                llm, question=question, principal=principal, history=history
            ):
                if kind == "token":
                    result.text += payload
                    yield _sse("token", {"text": payload})
                elif kind == "step":
                    # Drives the UI's "thinking/routing" indicator (CLAUDE.md section 6).
                    yield _sse("step", {"text": payload})
                elif kind == "routes":
                    yield _sse("routing", payload)
                elif kind == "sources":
                    # Before the first token where retrieval ran, so the UI can show what is
                    # being consulted while it waits — and so a user can see when an answer
                    # rests on nothing at all.
                    result.sources = payload
                    yield _sse("sources", {"sources": payload})
                elif kind == "final":
                    _absorb_final(result, payload)
        except LLMError as exc:
            # The 200 header is long gone, so the failure has to travel as an event.
            logger.opt(exception=exc).error(
                "Generation failed for user {user}", user=principal.user_id
            )
            # Whatever partial text arrived is still on screen, so its citations are
            # still worth resolving.
            result.citations = _partial_citations(result)
            yield _sse("citations", {"citations": result.citations})
            yield _sse(
                "error",
                {
                    "detail": ("The language model is currently unavailable. Please try again."),
                    "partial": bool(result.text),
                },
            )
            return
        except Exception as exc:
            logger.opt(exception=exc).error(
                "Unexpected error streaming answer for user {user}", user=principal.user_id
            )
            result.citations = _partial_citations(result)
            yield _sse("citations", {"citations": result.citations})
            yield _sse(
                "error",
                {
                    "detail": "Something went wrong generating the answer.",
                    "partial": bool(result.text),
                },
            )
            return

        yield _sse("citations", {"citations": result.citations})
        yield _sse(
            "done",
            {
                "usage": result.completion.usage.as_dict(),
                "provider": result.completion.provider,
                "model": result.completion.model,
                "grounded": result.grounded,
                "routes": result.routes,
                # Empty unless the SQL agent ran. The UI shows it collapsed, so an answer
                # over business data stays inspectable without cluttering the reply.
                "sql_query": result.sql_query,
            },
        )
        incomplete = False

        logger.info(
            "Chat answered for user {user} via {routes}: {tokens} completion tokens "
            "from {provider}, {sources} sources",
            user=principal.user_id,
            routes=",".join(result.routes) or "none",
            tokens=result.completion.usage.completion_tokens,
            provider=result.completion.provider,
            sources=len(result.citations),
        )
        # Cost telemetry, and the alert when one turn is unexpectedly expensive
        # (CLAUDE.md section 7, cost/quota overrun). After the `done` frame, so the
        # accounting can never delay the answer.
        record_token_usage(result.completion, route="chat")
    finally:
        # In `finally` so a disconnect partway through still records what the user saw.
        # Starlette throws GeneratorExit/CancelledError into this generator when the
        # client goes away, and neither is caught above — but both run this block.
        if incomplete:
            logger.warning(
                "Chat turn ended early for user {user} after {n} chars",
                user=principal.user_id,
                n=len(result.text),
            )
        await _shielded_save(
            principal=principal,
            session_id=session_id,
            question=question,
            result=result,
            incomplete=incomplete,
        )


@router.post("", summary="Ask a question over documents, business data and connected systems")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(
    request: Request,  # noqa: ARG001 — required by slowapi's decorator
    principal: CurrentPrincipal,
    payload: ChatRequest,
    llm: Annotated[LLMRouterProtocol, Depends(get_llm_router)],
) -> StreamingResponse:
    """Route the question through the supervisor and stream the answer as SSE."""
    question = payload.message.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Message cannot be empty."
        )

    settings = get_settings()

    # Session bookkeeping and history run in their own short transaction, which closes
    # before the graph starts. Retrieval now happens inside the RAG sub-agent — under its
    # own org-scoped session — so no connection is held across the multi-second run.
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        session_id = await _ensure_session(
            session, principal=principal, session_id=payload.session_id, question=question
        )
        history = (
            await _load_history(session, session_id=session_id, limit=settings.chat_history_limit)
            if payload.session_id is not None
            else []
        )

    return StreamingResponse(
        _event_stream(
            principal=principal,
            llm=llm,
            question=question,
            session_id=session_id,
            history=history,
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

        # Newest-first under the limit, then reversed for display. Ordering ascending and
        # limiting would return the *oldest* N — the start of a long conversation rather
        # than the part the user was just reading.
        rows = (
            await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(desc(ChatMessage.created_at), desc(_ROLE_ORDER))
                .limit(limit)
            )
        ).scalars()
        messages = [ChatMessageResponse.model_validate(row) for row in reversed(list(rows))]
    return ChatMessageListResponse(messages=messages)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session",
)
async def delete_session(principal: CurrentPrincipal, session_id: uuid.UUID) -> None:
    """Remove a conversation and, by cascade, its messages."""
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        # RLS confines this to the caller's own sessions, so the rowcount doubles as the
        # existence check — no separate SELECT that another request could race against.
        deleted = (
            await session.execute(delete(ChatSession).where(ChatSession.id == session_id))
        ).rowcount
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")


__all__ = ["router"]
