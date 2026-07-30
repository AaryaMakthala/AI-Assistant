"""The guarded SQL agent: question → validated SELECT → rows → answer.

Kept free of HTTP so the Phase 7 LangGraph SQL sub-agent can drive it directly, matching
how `app/rag/pipeline.py` is structured.

The retry loop is deliberately shallow. A rejected query is fed back once with the
validator's own reason, because the common failure is a model reaching for a column that
does not exist — which one corrective message fixes. Beyond that the model is usually
restating the same idea, and each further attempt is another chance for it to stumble into
something that passes validation but answers a different question.

Generation is non-streaming even though the provider supports streaming: a partial SELECT
cannot be validated, so there is nothing safe to emit until the whole statement exists.
Only the final answer streams, and that happens in the caller.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from loguru import logger

from app.config import get_settings
from app.llm.base import Completion, LLMError, LLMRouterProtocol
from app.sql_agent.audit import record_query
from app.sql_agent.execution import QueryResult
from app.sql_agent.prompts import CANNOT_ANSWER, build_answer_messages, build_sql_messages
from app.sql_agent.tools import ExecuteQueryArgs, ToolError, execute_query

#: Markdown fencing the model adds despite being told not to. Stripped rather than rejected:
#: it is a formatting habit, not an attempt at anything, and the SQL inside is still fully
#: validated afterwards.
_FENCE_MARKERS = ("```sql", "```postgresql", "```")


@dataclass
class SQLAnswer:
    """The outcome of one question put to the SQL agent."""

    #: The validated SQL that ran, for the UI's collapsed "show query" panel (section 6).
    sql: str | None = None
    result: QueryResult | None = None
    #: Set when no query could be produced or run. Safe to show the user.
    refusal: str | None = None
    attempts: int = 0
    rejections: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.result is not None


def _strip_fences(text: str) -> str:
    """Pull the SQL out of whatever wrapping the model put around it."""
    cleaned = text.strip()
    for marker in _FENCE_MARKERS:
        if cleaned.lower().startswith(marker):
            cleaned = cleaned[len(marker) :]
            break
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


async def _generate(llm: LLMRouterProtocol, *, question: str, retry_reason: str | None) -> str:
    """Ask the model for one SELECT statement, returning it unvalidated."""
    messages = build_sql_messages(question=question, retry_reason=retry_reason)
    completion = Completion()
    text = ""
    async for token in llm.stream(messages, completion=completion):
        text += token
    return _strip_fences(text)


async def answer_question(
    llm: LLMRouterProtocol, *, question: str, org_id: uuid.UUID, user_id: uuid.UUID
) -> SQLAnswer:
    """Turn a question into rows, or into an honest refusal.

    Never raises for a rejected or unanswerable query — those are the normal outcomes and
    are reported on :class:`SQLAnswer`. Only a provider failure propagates.
    """
    settings = get_settings()
    answer = SQLAnswer()
    retry_reason: str | None = None

    for attempt in range(1, settings.sql_max_generation_attempts + 1):
        answer.attempts = attempt

        try:
            raw = await _generate(llm, question=question, retry_reason=retry_reason)
        except LLMError:
            # Nothing was generated, so there is no query to audit — but the attempt is
            # still worth recording as a failure to answer.
            await record_query(
                org_id=org_id,
                user_id=user_id,
                question=question,
                status="failed",
                rejection_reason="The language model was unavailable.",
            )
            raise

        if not raw or raw.upper().startswith(CANNOT_ANSWER):
            # The model declined. Audited because a refusal is a real outcome, and a run of
            # them is a signal about either the schema or the questions being asked.
            await record_query(
                org_id=org_id,
                user_id=user_id,
                question=question,
                status="rejected",
                generated_sql=raw or None,
                rejection_reason="Model declined to answer from the available schema.",
            )
            answer.refusal = (
                "I can't answer that from the business data available to me. I can "
                "report on the organization's uploaded documents — how many there are, "
                "their sizes, types, and ingestion status."
            )
            return answer

        # Validation, execution and auditing all happen inside execute_query.
        outcome = await execute_query(
            ExecuteQueryArgs(sql=raw),
            org_id=org_id,
            user_id=user_id,
            question=question,
        )

        if isinstance(outcome, ToolError):
            answer.rejections.append(outcome.message)
            retry_reason = outcome.message
            logger.warning(
                "SQL agent attempt {n} rejected for user {user}: {reason}",
                n=attempt,
                user=user_id,
                reason=outcome.message,
            )
            continue

        answer.sql = raw
        answer.result = outcome
        return answer

    answer.refusal = (
        "I couldn't build a safe query for that question. Try asking it more simply, or "
        "in terms of the documents that have been uploaded."
    )
    return answer


async def synthesize_answer(llm: LLMRouterProtocol, *, question: str, answer: SQLAnswer) -> str:
    """Phrase a completed result in plain language.

    Separate from :func:`answer_question` so a caller can stream this half while the
    query itself, which cannot be streamed safely, has already finished.
    """
    if answer.result is None or answer.sql is None:
        return answer.refusal or "I don't have an answer for that."

    messages = build_answer_messages(question=question, sql=answer.sql, result=answer.result)
    completion = Completion()
    text = ""
    async for token in llm.stream(messages, completion=completion):
        text += token
    return text.strip()


__all__ = ["SQLAnswer", "answer_question", "synthesize_answer"]
