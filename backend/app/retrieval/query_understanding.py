"""Query Understanding stage — single LLM call before any retrieval.

Replaces the fragmented regex + LLM router + filename-token-matching approach
with one structured LLM call that produces:

- ``corrected_query``: typo-fixed version of the user's message (for the
  answer-generation prompt).
- ``intent``: routing classification (greeting, workspace_metadata,
  document_content, off_topic, needs_clarification).
- ``search_query``: retrieval-optimized version (pronouns resolved, abbreviations
  expanded, filler dropped — what gets embedded and reranked).
- ``confidence``: 0–1, how confident the model is in this classification.
- ``reasoning``: short string, for logging/debugging only.

The message list is built as ``system`` + ``user`` (never a single ``system``
message with no ``user`` turn), fixing the HTTP 400 errors some providers
threw on the old ``route_with_llm`` implementation.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.utils import detect_unclosed_think_block, strip_think_tags
from app.retrieval.intent import IntentCategory


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryUnderstanding:
    """Structured output from the Query Understanding stage."""

    corrected_query: str
    search_query: str
    intent: IntentCategory
    confidence: float
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the query understanding layer for Office Brain, an internal document
Q&A assistant.  You receive the user's message plus recent conversation context,
and return a JSON object that corrects typos, classifies intent, and produces
a search-optimized query.

Return ONLY a JSON object (no markdown fences, no preamble) with these fields:

{
  "corrected_query": "<user message with spelling/typo errors fixed, e.g. 'conpamy name' → 'company name'>",
  "intent": "greeting" | "workspace_metadata" | "document_content" | "off_topic" | "needs_clarification",
  "search_query": "<retrieval-optimized version: expand abbreviations, resolve pronouns from conversation context, drop filler words. e.g. 'describe them' → 'describe the uploaded documents'>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one short phrase, never shown to the user>"
}

Rules:

corrected_query:
- Fix ONLY spelling/typo errors.  Do NOT rephrase intent.
- If the message is already correct, return it unchanged.

intent:
- "greeting": greetings, thanks, farewells, casual chitchat not requiring documents.
- "workspace_metadata": questions about the workspace itself (doc count, doc names,
  member count, workspace name, who uploaded what).
- "document_content": questions that require searching document content to answer.
- "off_topic": general knowledge, trivia, code requests — unrelated to this
  workspace's documents.
- "needs_clarification": message is genuinely ambiguous and cannot be resolved.

search_query:
- A retrieval-optimized version of the corrected query.
- Resolve pronouns/references from conversation context ("describe them" →
  "describe the uploaded documents").
- Expand abbreviations ("hr policy" → "human resources policy").
- Drop filler words ("um", "so", "like", "can you").
- This is what gets embedded and reranked — it should be clear and specific.

confidence:
- How confident you are in the intent classification.
- High (0.9–1.0): obvious match (clear greeting, exact keyword match to a document).
- Medium (0.7–0.9): likely match but some ambiguity.
- Low (0.3–0.7): genuinely uncertain.

Workspace documents:
{workspace_context}

Conversation context:
{conversation_context}
"""


# ---------------------------------------------------------------------------
# Parse structured output
# ---------------------------------------------------------------------------

_INTENT_MAP = {
    "greeting": IntentCategory.GREETING,
    "workspace_metadata": IntentCategory.WORKSPACE_METADATA,
    "document_content": IntentCategory.DOCUMENT_CONTENT,
    "off_topic": IntentCategory.OUT_OF_SCOPE,
    "needs_clarification": IntentCategory.AMBIGUOUS,
}


def _extract_first_json(text: str) -> str | None:
    """Extract the first complete JSON object from text.

    Handles trailing garbage, nested braces, and escaped quotes — unlike a
    greedy ``{.+}`` regex which captures everything from the first ``{`` to
    the *last* ``}`` in the string (breaking when the model appends text).

    Also handles the case where text before the JSON is not a markdown fence
    (e.g. model returns "Here is the JSON: {...}" or just appends text after).
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # If we reach here, we found an opening brace but never closed it.
    # This can happen if the model's output was truncated (max_tokens cutoff)
    # mid-JSON. Return None so the caller can log and retry.
    return None


def _parse_query_understanding(
    response_text: str,
    original_query: str,
) -> QueryUnderstanding:
    """Parse the LLM's JSON response into a QueryUnderstanding."""
    try:
        # Strip markdown fences if present.
        cleaned = re.sub(r"```(?:json)?\s*", "", response_text)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        json_str = _extract_first_json(cleaned)
        if json_str:
            data = json.loads(json_str)

            corrected_query = str(data.get("corrected_query", original_query)).strip()
            search_query = str(data.get("search_query", corrected_query)).strip()
            intent_str = str(data.get("intent", "document_content")).strip().lower()
            confidence = float(data.get("confidence", 0.5))
            reasoning = str(data.get("reasoning", ""))

            if not corrected_query:
                corrected_query = original_query
            if not search_query:
                search_query = corrected_query

            intent = _INTENT_MAP.get(intent_str, IntentCategory.DOCUMENT_CONTENT)

            return QueryUnderstanding(
                corrected_query=corrected_query,
                search_query=search_query,
                intent=intent,
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=reasoning,
            )

        # _extract_first_json returned None — log what we actually got.
        logger.warning(
            "Query understanding: _extract_first_json returned None. "
            "Raw cleaned content (first 500 chars): {raw}",
            raw=cleaned[:500],
        )
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(
            "Query understanding: JSON parse failed. "
            "Raw content (first 500 chars): {raw} error={error}",
            raw=response_text[:500],
            error=str(exc)[:200],
        )

    # Parsing failed — default to document_content with the original query.
    return QueryUnderstanding(
        corrected_query=original_query,
        search_query=original_query,
        intent=IntentCategory.DOCUMENT_CONTENT,
        confidence=0.0,
        reasoning="parse_failure",
    )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def understand_query(
    *,
    query: str,
    workspace_id: uuid.UUID,
    history: list[dict[str, str]] | None = None,
) -> QueryUnderstanding:
    """Run the Query Understanding stage on a user message.

    One LLM call that corrects typos, classifies intent, and produces a
    search-optimized query.  On LLM failure, returns a degraded result
    (DOCUMENT_CONTENT with the original query) so the pipeline can still
    attempt retrieval.

    Parameters
    ----------
    query:
        The raw user message.
    workspace_id:
        The workspace UUID.
    history:
        Optional recent conversation turns as [{"role": "user"/"assistant",
        "content": "..."}].
    """
    text = query.strip()
    if not text:
        return QueryUnderstanding(
            corrected_query="",
            search_query="",
            intent=IntentCategory.AMBIGUOUS,
            confidence=1.0,
            reasoning="empty_query",
        )

    # Fetch workspace document titles + descriptions for context.
    from app.db.models import Document
    from sqlalchemy import select
    from app.security.rls import tenant_session

    ws_context = "No workspace-specific information available."
    try:
        async with tenant_session(workspace_id=workspace_id, user_id=uuid.UUID(int=0)) as db:
            rows = (
                await db.execute(
                    select(Document.filename, Document.description).where(
                        Document.workspace_id == workspace_id,
                        Document.status == "READY",
                    )
                )
            ).all()
            if rows:
                lines = []
                for row in rows:
                    if row.description:
                        lines.append(f"- {row.filename}: {row.description}")
                    else:
                        lines.append(f"- {row.filename}")
                ws_context = "Workspace documents:\n" + "\n".join(lines)
    except Exception as exc:
        logger.debug(
            "Failed to load workspace docs for query understanding: {error}",
            error=str(exc)[:200],
        )

    # Build conversation context section.
    context_lines: list[str] = []
    if history:
        for turn in history[-4:]:  # last 2 pairs max
            label = "User" if turn.get("role") == "user" else "Assistant"
            context_lines.append(f"{label}: {turn.get('content', '')}")
    conv_context = "\n".join(context_lines) if context_lines else "(none)"

    # Build the prompt.
    system_prompt = (
        _SYSTEM_PROMPT
        .replace("{workspace_context}", ws_context, 1)
        .replace("{conversation_context}", conv_context, 1)
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"User message:\n{text}",
        },
    ]

    # Call the LLM provider (shared factory — same chain for QU and answer gen).
    try:
        from app.llm.base import Completion, Message
        from app.config import get_settings
        from app.api.dependencies import get_llm_provider

        settings = get_settings()
        provider = get_llm_provider()

        llm_messages = [
            Message(role=m["role"], content=m["content"]) for m in messages
        ]
        completion = Completion()

        async for _token in provider.stream(
            llm_messages, completion=completion,
            max_tokens=settings.qu_max_output_tokens,
            disable_thinking=True,
        ):
            pass

        # Strip thinking tags before JSON parsing — Qwen3-Thinking and
        # similar models wrap their output in <think> blocks.
        response_text = strip_think_tags(completion.text)
        if not response_text:
            # Log the raw completion text BEFORE stripping for debugging.
            raw_preview = completion.text[:500] if completion.text else "(empty)"
            if detect_unclosed_think_block(completion.text or ""):
                # Distinct failure mode: the model was still reasoning when the
                # token budget ran out — its JSON answer was never emitted. This
                # is a truncation, not an empty response, and must not be
                # counted the same way in metrics.  Fall through to the retry
                # path below: the second attempt may land on a different
                # provider and produce a complete response.
                logger.warning(
                    "Query understanding truncated mid-thinking (unclosed think "
                    "block, max_tokens={max_tokens}). Raw completion text "
                    "(first 500 chars): {raw}",
                    max_tokens=settings.qu_max_output_tokens,
                    raw=raw_preview,
                )
                result = QueryUnderstanding(
                    corrected_query=text,
                    search_query=text,
                    intent=IntentCategory.DOCUMENT_CONTENT,
                    confidence=0.0,
                    reasoning="unclosed_think_block",
                )
            else:
                logger.warning(
                    "Query understanding returned empty after strip_think_tags. "
                    "Raw completion text (first 500 chars): {raw}",
                    raw=raw_preview,
                )
                return QueryUnderstanding(
                    corrected_query=text,
                    search_query=text,
                    intent=IntentCategory.DOCUMENT_CONTENT,
                    confidence=0.0,
                    reasoning="empty_response",
                )
        else:
            result = _parse_query_understanding(response_text, text)

        # If parsing failed (model returned unparseable text) or the response
        # was truncated mid-thinking, retry once.  A second attempt often
        # succeeds because the failure is non-deterministic, and the rotating
        # provider may pick a different provider on the retry.
        if result.reasoning in ("parse_failure", "unclosed_think_block"):
            logger.info(
                "Query understanding parse failed, retrying (attempt 2/2)",
            )
            completion2 = Completion()
            async for _token in provider.stream(
                llm_messages, completion=completion2,
                max_tokens=settings.qu_max_output_tokens,
                disable_thinking=True,
            ):
                pass
            response_text2 = strip_think_tags(completion2.text)
            if response_text2:
                result2 = _parse_query_understanding(response_text2, text)
                if result2.reasoning != "parse_failure":
                    logger.info(
                        "Query understanding retry succeeded: intent={intent} "
                        "confidence={confidence:.2f} corrected='{corrected}'",
                        intent=result2.intent.value,
                        confidence=result2.confidence,
                        corrected=result2.corrected_query[:60],
                    )
                    return result2
                logger.warning(
                    "Query understanding retry also failed to parse. "
                    "Raw content (first 500 chars): {raw}",
                    raw=response_text2[:500],
                )
            else:
                logger.warning(
                    "Query understanding retry returned empty after strip_think_tags. "
                    "Raw completion (first 500 chars): {raw}",
                    raw=completion2.text[:500] if completion2.text else "(empty)",
                )

        logger.info(
            "Query understanding: intent={intent} confidence={confidence:.2f} "
            "corrected='{corrected}' search='{search}' reasoning={reasoning}",
            intent=result.intent.value,
            confidence=result.confidence,
            corrected=result.corrected_query[:60],
            search=result.search_query[:60],
            reasoning=result.reasoning[:80],
        )
        return result

    except Exception as exc:
        # LLM call failed — degrade gracefully to document_content.
        logger.warning(
            "Query understanding LLM call failed, degrading to document_content: "
            "{error}",
            error=str(exc)[:200],
        )
        return QueryUnderstanding(
            corrected_query=text,
            search_query=text,
            intent=IntentCategory.DOCUMENT_CONTENT,
            confidence=0.0,
            reasoning=f"llm_error:{str(exc)[:80]}",
        )


__all__ = ["QueryUnderstanding", "understand_query"]
