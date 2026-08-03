"""The three tools the SQL agent may call (CLAUDE.md 4.3, 4.5).

`get_schema`, `describe_table` and `execute_query` are the entire interface. Each argument
is validated against a Pydantic model before anything runs — no LLM-generated string is
passed into a query, a path or a shell (CLAUDE.md 4.5). Phase 6 wraps these same functions
in an MCP server, so the validation lives here rather than in the transport layer.

`get_schema` reports the static allowlist and never introspects the live database. That is
the difference between a schema tool that describes eight columns of one table and one that
describes whatever a future migration happens to add.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.sql_agent.allowlist import (
    ALLOWED_FUNCTIONS,
    ALLOWED_TABLES,
    allowed_schema_text,
    lookup_table,
    table_names,
)
from app.sql_agent.audit import record_query
from app.sql_agent.execution import QueryResult, SQLExecutionError, execute_validated
from app.sql_agent.validation import SQLValidationError, validate_query


class DescribeTableArgs(BaseModel):
    """Arguments for `describe_table`."""

    model_config = ConfigDict(extra="forbid")

    #: Bounded and pattern-free on purpose: the value is only ever used as a dictionary key
    #: against the allowlist, never interpolated into SQL, so an unknown name is a lookup
    #: miss rather than an injection vector.
    table: str = Field(min_length=1, max_length=100)


class ExecuteQueryArgs(BaseModel):
    """Arguments for `execute_query`."""

    model_config = ConfigDict(extra="forbid")

    #: The cap keeps a runaway generation from reaching the parser at all. Real queries over
    #: a single eight-column table are far shorter than this.
    sql: str = Field(min_length=1, max_length=4000)


@dataclass(frozen=True)
class ToolError:
    """A refusal the model is expected to read and correct.

    Distinct from an exception because it is part of the agent's normal loop: a rejected
    query is fed back so the next attempt can be different, not propagated as a fault.
    """

    message: str


def get_schema() -> str:
    """Describe every table the agent may query.

    Returns the static allowlist — never live introspection (CLAUDE.md section 7).
    """
    functions = ", ".join(sorted(ALLOWED_FUNCTIONS))
    return (
        f"{allowed_schema_text()}\n\n"
        f"Permitted functions: {functions}.\n"
        "Only these tables, columns and functions exist for this tool. Rows are already "
        "restricted to the asking user's organization."
    )


def describe_table(args: DescribeTableArgs) -> str | ToolError:
    """Describe one allowlisted table."""
    spec = lookup_table(args.table)
    if spec is None:
        return ToolError(
            f"Table {args.table!r} is not available. Available tables: {', '.join(table_names())}."
        )
    return spec.render()


async def execute_query(
    args: ExecuteQueryArgs, *, org_id: uuid.UUID, user_id: uuid.UUID, question: str
) -> QueryResult | ToolError:
    """Validate, execute and audit one generated query.

    Every path through this function writes an audit row — accepted, rejected or failed —
    because the entries worth reviewing later are exactly the ones that did not succeed.
    """
    try:
        validated = validate_query(args.sql)
    except SQLValidationError as exc:
        await record_query(
            org_id=org_id,
            user_id=user_id,
            question=question,
            status="rejected",
            generated_sql=args.sql,
            rejection_reason=exc.reason,
        )
        return ToolError(exc.reason)

    try:
        result = await execute_validated(validated, org_id=org_id, user_id=user_id)
    except SQLExecutionError as exc:
        await record_query(
            org_id=org_id,
            user_id=user_id,
            question=question,
            status="failed",
            generated_sql=validated.sql,
            rejection_reason=str(exc),
        )
        return ToolError(
            "The query was valid but could not be completed. It may have taken too long; "
            "try a narrower question."
        )

    await record_query(
        org_id=org_id,
        user_id=user_id,
        question=question,
        status="accepted",
        generated_sql=validated.sql,
        row_count=result.row_count,
        duration_ms=result.duration_ms,
    )
    return result


__all__ = [
    "ALLOWED_TABLES",
    "DescribeTableArgs",
    "ExecuteQueryArgs",
    "ToolError",
    "describe_table",
    "execute_query",
    "get_schema",
]
