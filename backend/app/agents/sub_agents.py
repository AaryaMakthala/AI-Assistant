"""The sub-agent nodes: RAG, SQL and MCP wrappers (Phase 7).

Each function here is a LangGraph node that adapts one existing capability to the supervisor's
state. They are **wrappers, not reimplementations** — that is the whole design constraint:

- `rag_agent`  calls `app.rag.pipeline` / `retrieval`, so chunks stay org-scoped and citations
               still resolve against the real retrieved set.
- `sql_agent`  calls `app.sql_agent.answer_question`, so sqlglot validation, the allowlist, the
               read-only role, the row cap, the timeout and the audit log all still apply
               (CLAUDE.md 4.3). This module adds no SQL path of its own.
- `mcp_agent`  calls the Phase 6 MCP tool functions inside `acting_as(principal)`, so identity
               comes from the verified JWT and never from a tool argument (CLAUDE.md 4.6).

Two rules hold across all three:

**A sub-agent failure is data, not an exception.** Each returns an `AgentOutcome` with `ok`
false and a `detail`, so one dead branch of a fan-out cannot take down a two-source answer.
The only thing that propagates is cancellation.

**No sub-agent result may trigger another tool call.** Every node runs exactly the lookups the
*router* selected from the user's question, then stops. There is no loop in which retrieved
text can request more retrieval (CLAUDE.md 4.4).
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from functools import wraps

from loguru import logger

from app.agents.state import AgentOutcome, SupervisorState
from app.llm.base import LLMError, LLMRouterProtocol
from app.mcp_servers.github_server import MissingToken, SearchCodeArgs, search_code
from app.mcp_servers.identity import acting_as
from app.rag.pipeline import retrieve_for_question
from app.rag.prompts import format_context
from app.security.rls import tenant_session
from app.sql_agent.agent import answer_question

#: How many rows of a SQL result are rendered into the synthesis prompt. The query itself is
#: already capped at `sql_max_rows` (500); 50 is what fits usefully in a prompt without
#: crowding out document passages, and aggregates — the common case — return far fewer.
_MAX_SQL_ROWS_IN_PROMPT = 50

NodeFn = Callable[[SupervisorState, LLMRouterProtocol], Awaitable[dict[str, object]]]


def isolated(state_key: str, source: str) -> Callable[[NodeFn], NodeFn]:
    """Contain any unhandled exception in a sub-agent node.

    LangGraph does **not** isolate node failures: one branch of a fan-out raising takes the
    whole graph run down with it, so a broken GitHub call would lose an otherwise good answer
    from documents and SQL. Each node already handles the failures it anticipates; this
    decorator is what makes "a sub-agent failure is data, not an exception" true in general
    rather than only for the paths someone thought of.

    `CancelledError` is deliberately re-raised — a cancelled request is not a degraded answer,
    and swallowing it would keep work running after the client has gone.
    """

    def decorate(func: NodeFn) -> NodeFn:
        @wraps(func)
        async def wrapper(state: SupervisorState, llm: LLMRouterProtocol) -> dict[str, object]:
            try:
                return await func(state, llm)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.opt(exception=exc).error(
                    "Sub-agent {source} raised unexpectedly for user {user}",
                    source=source,
                    user=state["principal"].user_id,
                )
                return {
                    state_key: AgentOutcome(
                        source=source, ok=False, detail="this lookup did not complete"
                    ),
                    "steps": [f"{source}: failed unexpectedly"],
                }

        return wrapper

    return decorate


@isolated("rag", "documents")
async def rag_agent(state: SupervisorState, llm: LLMRouterProtocol) -> dict[str, object]:
    """Retrieve document passages relevant to the question.

    Retrieval only — generation happens once, in the synthesis node. Generating here and again
    at synthesis would summarize a summary, which is how citations drift away from the text
    they supposedly support.
    """
    principal = state["principal"]
    question = state["question"]

    try:
        async with tenant_session(
            org_id=principal.org_id, user_id=principal.user_id, role=principal.role
        ) as session:
            chunks = await retrieve_for_question(
                session, question=question, org_id=principal.org_id
            )
    except Exception as exc:
        logger.opt(exception=exc).error("RAG agent failed for user {user}", user=principal.user_id)
        return {
            "rag": AgentOutcome(
                source="documents", ok=False, detail="the document search did not complete"
            ),
            "steps": ["documents: search failed"],
            "chunks": [],
        }

    if not chunks:
        return {
            "rag": AgentOutcome(source="documents", ok=True, detail="no relevant passages"),
            "steps": ["documents: no relevant passages found"],
            "chunks": [],
        }

    # Reuses the Phase 4 formatter, so passages are numbered exactly as the citation
    # resolver expects and are already neutralized against fence forgery.
    content = format_context(chunks, nonce=_context_nonce())
    return {
        "rag": AgentOutcome(source="documents", ok=True, content=content),
        "steps": [f"documents: retrieved {len(chunks)} passage(s)"],
        "chunks": chunks,
    }


def _context_nonce() -> str:
    """A nonce for the inner context fence.

    The synthesis prompt applies its own outer fence with its own nonce; this inner one keeps
    `format_context`'s guarantees intact rather than passing a predictable constant.
    """
    return secrets.token_hex(8)


@isolated("sql", "business_data")
async def sql_agent(state: SupervisorState, llm: LLMRouterProtocol) -> dict[str, object]:
    """Answer the question from structured business data, via the guarded SQL agent."""
    principal = state["principal"]
    question = state["question"]

    try:
        answer = await answer_question(
            llm, question=question, org_id=principal.org_id, user_id=principal.user_id
        )
    except LLMError as exc:
        # The provider failed while generating SQL. Synthesis can still answer from any
        # other branch, so this is reported rather than raised.
        logger.warning(
            "SQL agent could not generate a query for user {user}: {err}",
            user=principal.user_id,
            err=exc,
        )
        return {
            "sql": AgentOutcome(
                source="business_data", ok=False, detail="the language model was unavailable"
            ),
            "steps": ["business_data: generation unavailable"],
        }
    except Exception as exc:
        logger.opt(exception=exc).error("SQL agent failed for user {user}", user=principal.user_id)
        return {
            "sql": AgentOutcome(
                source="business_data", ok=False, detail="the database query did not complete"
            ),
            "steps": ["business_data: query failed"],
        }

    if not answer.ok or answer.result is None:
        # A refusal is a legitimate outcome: the validator rejected every attempt, or the
        # model correctly judged the question unanswerable from the allowed schema.
        return {
            "sql": AgentOutcome(
                source="business_data",
                ok=True,
                content=answer.refusal or "",
                detail="no query could be answered from the available tables",
            ),
            "steps": [f"business_data: no answer after {answer.attempts} attempt(s)"],
        }

    result = answer.result
    rows = result.as_records()[:_MAX_SQL_ROWS_IN_PROMPT]
    lines = [f"Query executed: {answer.sql}", f"Rows returned: {result.row_count}"]
    if result.truncated:
        lines.append(f"NOTE: capped at {result.limit} rows; this may be a partial set.")
    if len(result.rows) > _MAX_SQL_ROWS_IN_PROMPT:
        lines.append(f"Showing the first {_MAX_SQL_ROWS_IN_PROMPT} rows.")
    lines.append("")
    lines.extend(str(row) for row in rows)

    return {
        "sql": AgentOutcome(source="business_data", ok=True, content="\n".join(lines)),
        "steps": [f"business_data: {result.row_count} row(s) in {result.duration_ms} ms"],
    }


@isolated("mcp", "external_code")
async def mcp_agent(state: SupervisorState, llm: LLMRouterProtocol) -> dict[str, object]:
    """Search connected external systems — currently GitHub, read-only.

    Deliberately calls `search_code` and not `read_file`. Reading a specific file requires a
    path, and the only thing that could supply one is either the user (who has not given one)
    or a search result — and letting a search result drive the next call is precisely the
    tool-chaining that CLAUDE.md 4.4 forbids. Snippets are what a synthesis prompt can use;
    whole-file reads stay a deliberate, user-directed action for a later phase.
    """
    principal = state["principal"]
    question = state["question"]

    try:
        args = SearchCodeArgs(query=question[:250])
    except ValueError as exc:
        # The question is unusable as a search query (too short, control characters).
        logger.info(
            "MCP agent rejected query for user {user}: {err}", user=principal.user_id, err=exc
        )
        return {
            "mcp": AgentOutcome(
                source="external_code",
                ok=True,
                detail="the question could not be turned into a code search",
            ),
            "steps": ["external_code: question not searchable"],
        }

    try:
        # Identity bound out of band, exactly as an HTTP caller would (CLAUDE.md 4.6).
        with acting_as(principal):
            content = await search_code(args)
    except MissingToken:
        return {
            "mcp": AgentOutcome(
                source="external_code", ok=False, detail="GitHub access is not configured"
            ),
            "steps": ["external_code: not configured"],
        }
    except Exception as exc:
        logger.opt(exception=exc).error("MCP agent failed for user {user}", user=principal.user_id)
        return {
            "mcp": AgentOutcome(
                source="external_code", ok=False, detail="the code search did not complete"
            ),
            "steps": ["external_code: search failed"],
        }

    # A refusal from the tool (rate limit, no results, bad credentials) is already phrased
    # for a reader; it is passed through as content so synthesis can explain it honestly.
    found = not content.startswith("REFUSED:") and "No code matching" not in content
    return {
        "mcp": AgentOutcome(source="external_code", ok=True, content=content),
        "steps": ["external_code: " + ("results found" if found else "nothing found")],
    }


__all__ = ["isolated", "mcp_agent", "rag_agent", "sql_agent"]
