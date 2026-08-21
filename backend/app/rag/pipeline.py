"""The RAG answer pipeline: retrieve → prompt → stream → cite.

Kept separate from the API layer so the same pipeline is testable independently
of HTTP routing.

Citations are derived from what was *retrieved and sent*, not from what the model claims.
A model can emit "[7]" when six sources were supplied; resolving citations against the
real chunk list means such a reference is dropped rather than rendered as a broken chip.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.llm.base import Completion, LLMRouterProtocol, Message
from app.rag.prompts import build_messages
from app.rag.retrieval import RetrievedChunk, build_context, retrieve_chunks

#: Matches "[1]" / "[2, 3]" / "[1][4]" in the model's answer.
_CITATION_REF = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Citation:
    """A source the answer actually drew on, safe to render in the UI."""

    number: int
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    filename: str
    page: int | None
    label: str
    #: Short preview for a hover card — never the whole chunk, which may be long.
    excerpt: str
    #: Retrieval similarity, 0–1. How close the passage was to the question, not how
    #: strongly it supports the sentence citing it.
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "document_id": str(self.document_id),
            "chunk_id": str(self.chunk_id),
            "filename": self.filename,
            "page": self.page,
            "label": self.label,
            "excerpt": self.excerpt,
            "score": round(self.score, 4),
        }


@dataclass
class AnswerResult:
    """Everything the caller needs once the stream is done."""

    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    completion: Completion = field(default_factory=Completion)
    retrieved: list[RetrievedChunk] = field(default_factory=list)

    @property
    def had_context(self) -> bool:
        return bool(self.retrieved)


def _excerpt(text: str, *, limit: int = 240) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def cited_numbers(answer: str) -> list[int]:
    """The bracketed source numbers `answer` refers to, in order of first appearance.

    Raw and unvalidated — the caller decides what range is legitimate. Split out so a
    caller holding sources in wire shape rather than as `RetrievedChunk` objects (the
    supervisor's partial-failure path) can still resolve citations against what was
    actually sent.
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


def resolve_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Map the bracketed numbers in `answer` onto the chunks that were supplied.

    Out-of-range numbers are ignored: the model referring to a source that was never sent
    is a hallucinated citation, and rendering it would lend it credibility.
    """
    cited: list[Citation] = []

    for number in cited_numbers(answer):
        if not 1 <= number <= len(chunks):
            continue
        chunk = chunks[number - 1]
        cited.append(
            Citation(
                number=number,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page=chunk.page,
                label=chunk.citation_label,
                excerpt=_excerpt(chunk.content),
                score=chunk.similarity,
            )
        )

    return sorted(cited, key=lambda citation: citation.number)


def source_payload(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    """The retrieved set as the UI sees it, numbered exactly as the prompt numbers it.

    Deliberately the same shape as :meth:`Citation.as_dict`, so "what was consulted" and
    "what was cited" render through one component: a citation is the subset of this list
    the answer actually leaned on, not a different kind of thing.
    """
    return [
        {
            "number": index,
            "document_id": str(chunk.document_id),
            "chunk_id": str(chunk.chunk_id),
            "filename": chunk.filename,
            "page": chunk.page,
            "label": chunk.citation_label,
            "excerpt": _excerpt(chunk.content),
            "score": round(chunk.similarity, 4),
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


async def retrieve_for_question(
    session: AsyncSession, *, question: str, org_id: uuid.UUID, user_id: uuid.UUID
) -> list[RetrievedChunk]:
    """Fetch and trim the evidence for one question.

    Split from generation so the caller can close its transaction before the LLM call:
    a pooled connection must not be pinned for the many seconds a stream can take.

    `user_id` scopes retrieval to documents the asker may read — org-wide plus their own.
    """
    settings = get_settings()
    retrieved = await retrieve_chunks(session, query=question, org_id=org_id, user_id=user_id)
    chunks = build_context(retrieved, max_chars=settings.retrieval_max_context_chars)
    if not chunks:
        logger.info("No chunks passed the relevance threshold for org {org}", org=org_id)
    return chunks


async def stream_answer(
    router: LLMRouterProtocol,
    *,
    question: str,
    chunks: list[RetrievedChunk],
    history: list[Message] | None = None,
    result: AnswerResult,
) -> AsyncIterator[str]:
    """Generate an answer over already-retrieved evidence, yielding text as it arrives.

    `result` is filled in as the stream proceeds and stays consistent with the text
    produced so far even if the stream fails or the client disconnects partway — the
    caller may still want to persist what the user actually saw.
    """
    result.retrieved = chunks
    messages = build_messages(question=question, chunks=chunks, history=history)

    try:
        async for token in router.stream(messages, completion=result.completion):
            result.text += token
            yield token
    finally:
        result.citations = resolve_citations(result.text, chunks)


__all__ = [
    "AnswerResult",
    "Citation",
    "cited_numbers",
    "resolve_citations",
    "retrieve_for_question",
    "source_payload",
    "stream_answer",
]
