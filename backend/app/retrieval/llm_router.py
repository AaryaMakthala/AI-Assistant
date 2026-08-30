"""LLM-based intent router for ambiguous queries.

When regex/heuristic classification is uncertain, this module sends the
(normalized) user message plus recent conversation context to the configured
LLM fallback chain and asks it to classify the intent as structured JSON.

The router ONLY decides which path a message takes — it never generates
the final user-facing answer for DOCUMENT_CONTENT or any other route.
Grounding guarantees from the retrieval pipeline stay intact.

Latency note: this adds one LLM call (~200-800ms on Groq) for messages
that don't match the regex fast-path.  The routing prompt and expected
output are kept small (single JSON object, no chain-of-thought) to
minimize cost and latency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from loguru import logger


# ---------------------------------------------------------------------------
# Route taxonomy
# ---------------------------------------------------------------------------

# The LLM router returns one of these route strings.  The caller maps it
# to an IntentCategory.
ROUTE_GREETING = "GREETING"
ROUTE_IDENTITY_ASSISTANT = "IDENTITY_ASSISTANT"
ROUTE_IDENTITY_USER = "IDENTITY_USER"
ROUTE_METADATA = "METADATA"
ROUTE_PERMISSIONS = "PERMISSIONS"
ROUTE_DOCUMENT_CONTENT = "DOCUMENT_CONTENT"
ROUTE_OUT_OF_SCOPE = "OUT_OF_SCOPE"
ROUTE_NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

# Conversation history route (questions about prior conversations).
ROUTE_CONVERSATION_HISTORY = "CONVERSATION_HISTORY"
# App-help route (questions about the application itself).
ROUTE_APP_HELP = "APP_HELP"
# General conversation route (casual chat, greetings, etc.).
ROUTE_GENERAL_CONVERSATION = "GENERAL_CONVERSATION"

# All valid route strings.
_VALID_ROUTES = frozenset({
    ROUTE_GREETING,
    ROUTE_IDENTITY_ASSISTANT,
    ROUTE_IDENTITY_USER,
    ROUTE_METADATA,
    ROUTE_PERMISSIONS,
    ROUTE_DOCUMENT_CONTENT,
    ROUTE_OUT_OF_SCOPE,
    ROUTE_NEEDS_CLARIFICATION,
    ROUTE_CONVERSATION_HISTORY,
    ROUTE_APP_HELP,
    ROUTE_GENERAL_CONVERSATION,
})

# Below this confidence threshold, force NEEDS_CLARIFICATION.
CONFIDENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Route result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteResult:
    """Outcome of the LLM router call."""

    #: The route the LLM assigned.
    route: str
    #: Free-text reasoning (for logging/debugging, never user-facing).
    reasoning: str = ""
    #: Confidence that the route is correct (0.0–1.0).
    confidence: float = 1.0
    #: Whether the LLM call succeeded or we fell back.
    status: Literal["success", "degraded"] = "success"


# ---------------------------------------------------------------------------
# Routing prompt
# ---------------------------------------------------------------------------

_ROUTING_SYSTEM_PROMPT_TEMPLATE = """\
You are a fast intent classifier for a company knowledge assistant chatbot.

{workspace_context}

Given the user's message (and optional recent conversation for context),
classify the message into exactly ONE of these routes:

- GREETING: Hello, goodbye, thanks, or any casual social message.
- IDENTITY_ASSISTANT: User asking what/who the assistant is, what it can do,
  or making a statement about the assistant (e.g. "who are you", "what are you").
- IDENTITY_USER: User asking about or stating their own info
  (e.g. "my name is X", "what is my info", "who am I").
- METADATA: Questions about workspace data — member count, document count,
  who is admin/owner, what is my role, document listing, etc.
- PERMISSIONS: "Can I do X?" questions — upload, invite, approve, delete, etc.
- DOCUMENT_CONTENT: Questions that should be answered from uploaded documents
  (policies, procedures, facts, summaries, comparisons).
- CONVERSATION_HISTORY: Questions about the user's prior conversation — what
  they asked before, what the assistant answered, recent chat history.
- APP_HELP: Questions about the application itself — what it does, how to use it,
  whether it tracks/monitors activity, features and capabilities.
- OUT_OF_SCOPE: General knowledge unrelated to the workspace
  (math, geography, weather, programming, jokes, etc.).
- GENERAL_CONVERSATION: Casual chat, statements like "I have a doubt",
  "can you help me", testing messages, or any non-specific interaction that
  doesn't fit another category.
- NEEDS_CLARIFICATION: Genuinely ambiguous — cannot determine the route.

Rules:
1. Return ONLY a JSON object, no markdown fences, no explanation outside JSON.
2. If the message is a statement about the user (e.g. "my name is X"), route
   to IDENTITY_USER — do NOT treat it as a question about the assistant.
3. "who are you" / "what are you" → IDENTITY_ASSISTANT (the user asks about
   the assistant, not themselves).
4. "who is admin" / "who is the owner" → METADATA (a workspace data question).
5. "can I add someone" / "can I upload" → PERMISSIONS.
6. "what are the file names" / "what files are present" / "what are the
   document names" → METADATA (document listing).
7. When unsure, prefer NEEDS_CLARIFICATION over guessing.
8. Document-name-aware routing: If the user's message mentions or approximates
   a document title listed in the workspace context above (even with typos),
   route to DOCUMENT_CONTENT — the retrieval pipeline will find the matching
   document. For example, "what does devops contain" targets the DevOps
   document, "tell me about the resume" targets the resume, etc.
9. Vague content questions ("what does it say", "what about that", "tell me
   about it") with no document name or topic should be NEEDS_CLARIFICATION
   if the context doesn't resolve the reference, or DOCUMENT_CONTENT if the
   conversation history makes the target clear.

Return ONLY:
{{"route": "<route>", "reasoning": "<brief>", "confidence": <0.0-1.0>}}
"""


# ---------------------------------------------------------------------------
# Parse structured output
# ---------------------------------------------------------------------------

def _parse_route_response(response_text: str) -> RouteResult:
    """Parse the LLM's JSON route response."""
    try:
        # Strip markdown fences if present.
        cleaned = re.sub(r"```(?:json)?\s*", "", response_text)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        json_match = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            route = str(data.get("route", "")).strip().upper()
            reasoning = str(data.get("reasoning", ""))
            confidence = float(data.get("confidence", 0.5))

            # Validate the route.
            if route not in _VALID_ROUTES:
                logger.warning(
                    "LLM router returned invalid route '{route}', "
                    "falling back to NEEDS_CLARIFICATION",
                    route=route,
                )
                return RouteResult(
                    route=ROUTE_NEEDS_CLARIFICATION,
                    reasoning=f"invalid_route: {route}",
                    confidence=0.0,
                    status="degraded",
                )

            return RouteResult(
                route=route,
                reasoning=reasoning,
                confidence=max(0.0, min(1.0, confidence)),
                status="success",
            )
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(
            "Could not parse LLM router response: {error}",
            error=str(exc)[:200],
        )

    # Parsing failed.
    return RouteResult(
        route=ROUTE_NEEDS_CLARIFICATION,
        reasoning="parse_failure",
        confidence=0.0,
        status="degraded",
    )


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------

async def route_with_llm(
    *,
    query: str,
    history: list[dict[str, str]] | None = None,
    workspace_knowledge_context: str | None = None,
) -> RouteResult:
    """Classify a user message using the LLM fallback chain.

    Sends the normalized query + optional recent conversation turns to the
    configured LLM provider (via the existing fallback chain) and asks it
    to return structured JSON with a route classification.

    Parameters
    ----------
    query:
        The user's message (already normalized if applicable).
    history:
        Optional recent conversation turns as [{"role": "user"/"assistant",
        "content": "..."}].  Used to resolve pronouns and follow-ups.
    workspace_knowledge_context:
        Optional workspace-specific context (document titles, capabilities)
        to inject into the system prompt.  If None, a generic prompt is used.

    Returns
    -------
    RouteResult with the classified route, reasoning, and confidence.
    On LLM failure, returns NEEDS_CLARIFICATION with status=degraded.
    """
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

    # Build the system prompt with workspace context.
    ws_context = (
        f"This workspace contains:\n{workspace_knowledge_context}"
        if workspace_knowledge_context
        else "No workspace-specific information available."
    )
    system_prompt = _ROUTING_SYSTEM_PROMPT_TEMPLATE.format(
        workspace_context=ws_context,
    )

    # Build the user message with conversation context.
    context_lines: list[str] = []
    if history:
        for turn in history[-4:]:  # last 2 pairs max
            label = "User" if turn.get("role") == "user" else "Assistant"
            context_lines.append(f"{label}: {turn.get('content', '')}")

    user_msg = f"User message: {query}"
    if context_lines:
        user_msg = f"Recent conversation:\n" + "\n".join(context_lines) + f"\n\n{user_msg}"

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
            logger.warning("LLM router returned empty response")
            return RouteResult(
                route=ROUTE_NEEDS_CLARIFICATION,
                reasoning="empty_response",
                confidence=0.0,
                status="degraded",
            )

        result = _parse_route_response(response_text)
        logger.info(
            "LLM router route={route} confidence={confidence:.2f} "
            "reasoning={reasoning}",
            route=result.route,
            confidence=result.confidence,
            reasoning=result.reasoning[:100],
        )
        return result

    except Exception as exc:
        logger.warning(
            "LLM router failed: {error}, falling back to NEEDS_CLARIFICATION",
            error=str(exc)[:200],
        )
        return RouteResult(
            route=ROUTE_NEEDS_CLARIFICATION,
            reasoning=f"llm_error: {str(exc)[:100]}",
            confidence=0.0,
            status="degraded",
        )


__all__ = [
    "CONFIDENCE_THRESHOLD",
    "RouteResult",
    "route_with_llm",
]
