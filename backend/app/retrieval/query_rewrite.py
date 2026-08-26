"""Conversational query rewriting for follow-up handling.

When a user asks a follow-up question like "What about last month?" or
"Why is that required?", the question contains references that can only
be resolved using recent conversation context.  This module rewrites such
queries into standalone form *before* the relevance gate and retrieval
pipeline, so downstream components see a self-contained question.

Flow:
    original query + recent conversation history
        → rewrite LLM call (lightweight, bounded timeout)
        → standalone query  →  metadata routing / relevance / retrieval

Failure handling: if the rewrite LLM fails, the original query is used as-is.
The request is never rejected because of a rewrite failure.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Literal

from loguru import logger


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewriteResult:
    """Outcome of the query rewrite step."""

    #: The rewritten standalone query, or the original if no rewrite occurred.
    rewritten_query: str
    #: Whether the system genuinely cannot resolve the reference.
    needs_clarification: bool = False
    #: Confidence that the rewrite is correct (0.0–1.0).
    confidence: float = 1.0
    #: Why this result was produced.
    status: Literal["success", "skipped", "degraded", "ambiguous"] = "success"
    #: The original user query (for logging/debugging).
    original_query: str = ""


# ---------------------------------------------------------------------------
# Conversation history type (lightweight — no DB dependency)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatTurn:
    """One turn in recent conversation history."""

    role: str  # "user" or "assistant"
    content: str


# ---------------------------------------------------------------------------
# Rewrite prompt
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """\
You are a query rewriting component for a company knowledge assistant.

Rewrite the user's newest message into a standalone search/query form.
Use the recent conversation only to resolve references, pronouns, ellipsis, and omitted subjects.

Rules:
1. Preserve the user's original intent.
2. Resolve references such as "this", "that", "it", "they", "them", "there", and document names using conversation context.
3. Do not answer the question.
4. Do not invent facts.
5. Do not add information that is not supported by the conversation.
6. If the message is already standalone, return it unchanged.
7. If the message is ambiguous even after using conversation context, set needs_clarification to true rather than inventing a referent.

Return ONLY a JSON object, no markdown fences:
{{
"rewritten_query": "...",
"needs_clarification": false,
"confidence": 0.0
}}"""


# ---------------------------------------------------------------------------
# Heuristic: does this query need rewriting?
# ---------------------------------------------------------------------------

# Pronouns and demonstratives that typically need context to resolve.
_FOLLOWUP_SIGNALS = re.compile(
    r"\b(?:this|that|it|they|them|there|these|those|"
    r"his|her|its|their|"
    r"what about|how about|why is that|why are they|"
    r"how many are there|how many are|"
    r"what does it|what does that|what do they|"
    r"tell me more|more about|continue|"
    r"any more|anything else|"
    r"how many\b)",
    re.IGNORECASE,
)


def _needs_rewrite(
    query: str,
    history: list[ChatTurn],
) -> bool:
    """Decide whether the query would benefit from conversational rewriting.

    Conservative strategy: skip rewrite when there is no history, or when
    the query is clearly self-contained.
    """
    if not history:
        return False

    # If the query already contains a full subject and is long enough,
    # it is likely standalone.
    q = query.strip()
    if len(q) > 30 and not _FOLLOWUP_SIGNALS.search(q):
        return False

    # Short queries (< 30 chars) with follow-up signals likely need context.
    if _FOLLOWUP_SIGNALS.search(q):
        return True

    # Very short queries (< 15 chars) in the presence of history likely need context.
    if len(q) < 15 and history:
        return True

    return False


# ---------------------------------------------------------------------------
# Parse structured output
# ---------------------------------------------------------------------------

def _parse_rewrite_response(response_text: str, original: str) -> RewriteResult:
    """Parse the LLM's JSON rewrite response."""
    try:
        # Strip markdown fences if present.
        cleaned = re.sub(r"```(?:json)?\s*", "", response_text)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        json_match = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            rewritten = str(data.get("rewritten_query", original)).strip()
            needs_clarification = bool(data.get("needs_clarification", False))
            confidence = float(data.get("confidence", 0.5))

            if not rewritten:
                rewritten = original

            if needs_clarification:
                return RewriteResult(
                    rewritten_query=original,
                    needs_clarification=True,
                    confidence=confidence,
                    status="ambiguous",
                    original_query=original,
                )

            return RewriteResult(
                rewritten_query=rewritten,
                needs_clarification=False,
                confidence=confidence,
                status="success",
                original_query=original,
            )
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Parsing failed — fall back to the original query.
    logger.warning(
        "Could not parse query rewrite response, using original: {text}",
        text=response_text[:200],
    )
    return RewriteResult(
        rewritten_query=original,
        needs_clarification=False,
        confidence=0.0,
        status="degraded",
        original_query=original,
    )


# ---------------------------------------------------------------------------
# Main rewrite function
# ---------------------------------------------------------------------------

async def rewrite_query(
    *,
    query: str,
    history: list[ChatTurn],
) -> RewriteResult:
    """Rewrite a follow-up query into standalone form using conversation context.

    Uses the configured LLM provider via the same fallback chain the chat
    endpoint uses.  The rewrite call is lightweight (bounded timeout, no
    retrieved chunks).  On failure, returns the original query unchanged.

    Parameters
    ----------
    query:
        The raw user query.
    history:
        Recent conversation turns (last 2–3 user/assistant pairs).

    Returns
    -------
    RewriteResult with the standalone query, or the original on failure.
    """
    original = query.strip()
    if not original:
        return RewriteResult(
            rewritten_query=original,
            status="skipped",
            original_query=original,
        )

    # Check if rewriting is needed.
    if not _needs_rewrite(original, history):
        logger.debug(
            "query_rewrite status=skipped reason=no_rewrite_needed query_len={len}",
            len=len(original),
        )
        return RewriteResult(
            rewritten_query=original,
            status="skipped",
            original_query=original,
        )

    # Build the conversation context block.
    context_lines: list[str] = []
    for turn in history[-6:]:  # at most 3 pairs = 6 messages
        label = "User" if turn.role == "user" else "Assistant"
        context_lines.append(f"{label}: {turn.content}")

    context_block = "\n".join(context_lines)

    prompt = (
        f"{_REWRITE_PROMPT}\n\n"
        f"Recent conversation:\n{context_block}\n\n"
        f"New user message:\n{original}"
    )

    # Call the LLM provider.
    try:
        from app.llm.base import Completion, Message
        from app.llm.generic import GenericProvider

        provider = GenericProvider()
        messages = [Message(role="user", content=prompt)]
        completion = Completion()

        async for _token in provider.stream(messages, completion=completion):
            pass

        response_text = completion.text.strip()
        if not response_text:
            logger.warning(
                "query_rewrite status=degraded reason=empty_response",
            )
            return RewriteResult(
                rewritten_query=original,
                status="degraded",
                original_query=original,
            )

        result = _parse_rewrite_response(response_text, original)
        logger.info(
            "query_rewrite status={status} confidence={confidence:.2f}",
            status=result.status,
            confidence=result.confidence,
        )
        return RewriteResult(
            rewritten_query=result.rewritten_query,
            needs_clarification=result.needs_clarification,
            confidence=result.confidence,
            status=result.status,
            original_query=original,
        )

    except Exception as exc:
        # Rewrite LLM failed — fall through to the original query.
        # Do NOT reject the request; do NOT expose provider errors.
        logger.warning(
            "query_rewrite status=degraded reason={error}",
            error=str(exc)[:200],
        )
        return RewriteResult(
            rewritten_query=original,
            status="degraded",
            original_query=original,
        )


__all__ = [
    "ChatTurn",
    "RewriteResult",
    "rewrite_query",
]
