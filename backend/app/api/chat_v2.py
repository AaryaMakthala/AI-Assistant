"""Grounded chat endpoint (Phase 6) — CLAUDE.md section 8.

One question, answered from the workspace's approved documents and nothing else.
The endpoint is a thin, deterministic wrapper around the Phase 5 retrieval
pipeline (``app/retrieval/pipeline.py``):

    authenticate → workspace membership → Phase 5 retrieval (hybrid → RRF → rerank)
        → Layer-1 grounding check
            ├── grounded    → build prompt from the final chunks → LLM → cited answer
            └── ungrounded  → honest refusal, NO LLM call (CLAUDE.md 8.3)

It never bypasses the pipeline, never duplicates its logic, and never lets the
client choose the workspace: the tenant is the caller's default workspace from the
verified JWT claim, and membership is resolved from the ``members`` table at
request time (CLAUDE.md section 4) — the same contract the Phase 3 document
endpoints use. A client-supplied ``workspace_id`` is rejected outright
(``extra="forbid"``), so a tenant can never be asserted from the wire.

The response is synchronous JSON rather than SSE. The provider's token stream is
consumed internally; the returned contract is the complete answer plus the
backend-constructed citations, which is what the frontend needs to render the
turn (CLAUDE.md section 10: synchronous responses are fine when the provider
supports streaming only as a transport detail).

Citations are built by the backend from the chunks that were actually sent to the
LLM (CLAUDE.md 8.4) — the model never invents citation metadata, because it never
receives any.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import get_generic_llm
from app.api.workspace_deps import assert_workspace_role
from app.llm.base import Completion, LLMError, LLMProvider
from app.rag.prompts import build_messages
from app.retrieval.pipeline import RetrievedChunk, retrieve
from app.security.auth import CurrentPrincipal
from app.security.rls import tenant_session

router = APIRouter(prefix="/chat", tags=["chat"])

#: Returned verbatim when Layer-1 grounding fails (CLAUDE.md 8.3). This is the
#: honest refusal that replaces any LLM call on an ungrounded question — the LLM
#: never sees the question, so it can never be tempted to fill the gap.
REFUSAL_ANSWER = "I couldn't find that information in the approved company knowledge base."


class GroundedChatRequest(BaseModel):
    """A question for the workspace's knowledge base.

    Only the message travels from the client. The workspace is the caller's
    authenticated default workspace — never a field on this request.
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)


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
    #: Whether the answer is grounded in retrieved evidence (CLAUDE.md 8.3).
    grounded: bool
    #: True when retrieval produced no acceptable evidence and the LLM was never
    #: called — the machine-readable refusal the frontend can branch on.
    insufficient_evidence: bool
    sources: list[Source]
    provider: str = ""
    model: str = ""


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


@router.post(
    "/grounded",
    response_model=GroundedChatResponse,
    summary="Ask a question grounded in the workspace's approved documents",
)
async def grounded_chat(
    principal: CurrentPrincipal,
    payload: GroundedChatRequest,
    llm: Annotated[LLMProvider, Depends(get_generic_llm)],
) -> GroundedChatResponse:
    """Answer `payload.message` from the caller's workspace's approved documents.

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

    # The workspace is the caller's default workspace from the verified JWT claim;
    # membership is resolved from the canonical members table at request time and
    # never from the token (CLAUDE.md section 4). Both OWNER and MEMBER may chat.
    workspace_id = principal.workspace_id
    await assert_workspace_role(workspace_id, principal)

    # Phase 5 pipeline: hybrid search → RRF fusion → rerank → Layer-1 grounding.
    # The retrieval session closes before the LLM call, so no pooled connection is
    # pinned across a multi-second generation.
    async with tenant_session(
        workspace_id=workspace_id, user_id=principal.user_id
    ) as session:
        result = await retrieve(session, query=question, workspace_id=workspace_id)

    if not result.grounded:
        logger.info(
            "Refused ungrounded question for workspace {ws} (top_score={score})",
            ws=workspace_id,
            score=result.top_score,
        )
        return GroundedChatResponse(
            answer=REFUSAL_ANSWER,
            grounded=False,
            insufficient_evidence=True,
            sources=[],
        )

    # Layer-1 passed: build the prompt from the final reranked chunks only. The
    # prompt treats retrieved text as untrusted quoted data (CLAUDE.md 4.4).
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


__all__ = [
    "GroundedChatRequest",
    "GroundedChatResponse",
    "REFUSAL_ANSWER",
    "Source",
    "router",
]
