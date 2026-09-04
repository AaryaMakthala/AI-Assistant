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
ROUTE_DIRECT = "direct"
ROUTE_METADATA = "metadata"
ROUTE_RETRIEVAL = "retrieval"
ROUTE_CLARIFICATION = "clarification"
ROUTE_OUT_OF_SCOPE = "out_of_scope"

# Legacy aliases kept for backward compat in tests that reference old names.
ROUTE_GREETING = ROUTE_DIRECT
ROUTE_IDENTITY_ASSISTANT = ROUTE_DIRECT
ROUTE_IDENTITY_USER = ROUTE_DIRECT
ROUTE_PERMISSIONS = ROUTE_DIRECT
ROUTE_DOCUMENT_CONTENT = ROUTE_RETRIEVAL
ROUTE_NEEDS_CLARIFICATION = ROUTE_CLARIFICATION
ROUTE_CONVERSATION_HISTORY = ROUTE_DIRECT
ROUTE_APP_HELP = ROUTE_DIRECT
ROUTE_GENERAL_CONVERSATION = ROUTE_DIRECT

# All valid route strings.
_VALID_ROUTES = frozenset({
    ROUTE_DIRECT,
    ROUTE_METADATA,
    ROUTE_RETRIEVAL,
    ROUTE_CLARIFICATION,
    ROUTE_OUT_OF_SCOPE,
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
    #: Rewritten standalone query (only set when needs_rewrite is True).
    query: str | None = None
    #: Whether the query needed rewriting (pronouns, garbled, context-dependent).
    needs_rewrite: bool = False
    #: Whether the question is plausibly answerable from workspace documents.
    relevant: bool = False


# ---------------------------------------------------------------------------
# Routing prompt
# ---------------------------------------------------------------------------

_ROUTING_SYSTEM_PROMPT_TEMPLATE = """\
You are the routing and relevance layer for Office Brain, an internal document
Q&A assistant.

Given the user's message and recent conversation context, return ONLY a JSON
object (no markdown, no preamble) with these fields:

{{
  "route": "direct" | "metadata" | "retrieval" | "clarification" | "out_of_scope",
  "query": "<rewritten standalone query, only if route is 'retrieval' and rewrite
is needed, else null>",
  "needs_rewrite": true | false,
  "relevant": true | false,
  "reasoning": "<one short phrase, not a sentence>"
}}

Routing rules:
- "direct": greetings, identity questions, app help, general conversation not
  requiring documents.
- "metadata": questions about the workspace itself (doc count, doc names, who
  uploaded what).
- "retrieval": questions that require searching document content.
- "clarification": message is ambiguous and cannot be routed without asking the
  user.
- "out_of_scope": unrelated to this workspace's documents or purpose.

Rewrite rules:
- Set "needs_rewrite": true only if the query contains unresolved pronouns
  ("they", "it", "that"), is garbled/malformed, or depends on prior turns to
  make sense standalone.
- If needs_rewrite is true, "query" must be a fully standalone version using
  conversation context.
- If needs_rewrite is false, set "query" to the original message unchanged.

Relevance rules:
- "relevant": true only if route is "retrieval" AND the question is plausibly
  answerable from workspace documents (not a request for external/general
  knowledge).
- If route is not "retrieval", set "relevant": false.

{workspace_context}

Conversation context:
{conversation_context}

User message:
{user_message}

Return ONLY:
{{"route": "<route>", "query": "<rewritten or original>", "needs_rewrite": <bool>, "relevant": <bool>, "reasoning": "<brief>"}}
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

        json_match = re.search(r"\{.+\}", cleaned, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            route = str(data.get("route", "")).strip().lower()
            reasoning = str(data.get("reasoning", ""))
            confidence = float(data.get("confidence", 0.5))
            query = data.get("query")
            needs_rewrite = bool(data.get("needs_rewrite", False))
            relevant = bool(data.get("relevant", False))

            # Validate the route.
            if route not in _VALID_ROUTES:
                logger.warning(
                    "LLM router returned invalid route '{route}', "
                    "falling back to clarification",
                    route=route,
                )
                return RouteResult(
                    route=ROUTE_CLARIFICATION,
                    reasoning=f"invalid_route: {route}",
                    confidence=0.0,
                    status="degraded",
                )

            return RouteResult(
                route=route,
                reasoning=reasoning,
                confidence=max(0.0, min(1.0, confidence)),
                status="success",
                query=str(query) if query else None,
                needs_rewrite=needs_rewrite,
                relevant=relevant,
            )
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(
            "Could not parse LLM router response: {error}",
            error=str(exc)[:200],
        )

    # Parsing failed.
    return RouteResult(
        route=ROUTE_CLARIFICATION,
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

    # Build workspace context section.
    ws_context = (
        f"Workspace knowledge:\n{workspace_knowledge_context}"
        if workspace_knowledge_context
        else "No workspace-specific information available."
    )

    # Build conversation context section.
    context_lines: list[str] = []
    if history:
        for turn in history[-4:]:  # last 2 pairs max
            label = "User" if turn.get("role") == "user" else "Assistant"
            context_lines.append(f"{label}: {turn.get('content', '')}")
    conv_context = "\n".join(context_lines) if context_lines else "(none)"

    # Build the full prompt using the template.
    system_prompt = (
        _ROUTING_SYSTEM_PROMPT_TEMPLATE
        .replace("{workspace_context}", ws_context, 1)
        .replace("{conversation_context}", conv_context, 1)
        .replace("{user_message}", query, 1)
    )

    messages = [
        Message(role="system", content=system_prompt),
    ]

    completion = Completion()

    try:
        async for _token in provider.stream(messages, completion=completion):
            pass

        response_text = completion.text.strip()
        if not response_text:
            logger.warning("LLM router returned empty response")
            return RouteResult(
                route=ROUTE_CLARIFICATION,
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
            route=ROUTE_CLARIFICATION,
            reasoning=f"llm_error: {str(exc)[:100]}",
            confidence=0.0,
            status="degraded",
        )


__all__ = [
    "CONFIDENCE_THRESHOLD",
    "RouteResult",
    "route_with_llm",
    "ROUTE_DIRECT",
    "ROUTE_METADATA",
    "ROUTE_RETRIEVAL",
    "ROUTE_CLARIFICATION",
    "ROUTE_OUT_OF_SCOPE",
]
