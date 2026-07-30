"""Database MCP server — read-only SQL over Phase 5's guarded agent.

This server is a **transport wrapper and nothing more**. Every guarantee in CLAUDE.md 4.3
is enforced in `app/sql_agent/`, and this module adds no path around any of them:

- `get_schema` returns the static allowlist from `sql_agent/allowlist.py`. No live
  introspection, so a future migration cannot widen what the agent can see (Risk register).
- `execute_query` delegates to `sql_agent.tools.execute_query`, which validates with sqlglot,
  forces a LIMIT, runs as the read-only `app_sql_agent` role in a READ ONLY transaction under
  a statement timeout, and audits the attempt — accepted, rejected or failed.

The one thing this module must get right is what it *doesn't* do. It never assembles SQL, it
never relaxes the validator, and it never accepts an `org_id` argument: identity comes from
:mod:`app.mcp_servers.identity`, so a prompt-injected "query the other org's rows" has no
argument to land in.

A rejected query comes back as a readable refusal rather than a protocol error, because a
rejection is a normal turn in the agent's loop — the model is expected to read the reason and
try a different query. Only genuine faults raise.
"""

from __future__ import annotations

from loguru import logger
from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.mcp_servers.errors import internal_error, refusal
from app.mcp_servers.identity import current_org_and_user
from app.security.untrusted import neutralize
from app.sql_agent.execution import QueryResult
from app.sql_agent.tools import (
    DescribeTableArgs,
    ExecuteQueryArgs,
    ToolError,
    describe_table,
    execute_query,
    get_schema,
)

SERVER_NAME = "database"

INSTRUCTIONS = """\
Answer questions about structured business data with read-only SQL.

Workflow: call get_schema first to see exactly which tables and columns exist, then send a \
single SELECT to execute_query.

Hard limits, enforced by the database and by a SQL parser — not by your cooperation:
- One plain SELECT statement only. No INSERT/UPDATE/DELETE/DDL, no multiple statements, no \
comments, no CTEs or subqueries outside the allowlist.
- Only the tables and columns get_schema lists. Nothing else exists for this tool.
- Every query is capped at a maximum row count and a short timeout.
- Rows are already restricted to the calling user's organization. There is no organization \
column to filter on and no argument to set.

A rejected query comes back as 'REFUSED:' with the reason. Read it and correct the query; \
do not attempt to work around the restriction."""


class GetSchemaArgs(BaseModel):
    """`get_schema` takes no arguments.

    Declared as an empty `extra="forbid"` model rather than omitted, so passing anything is
    an explicit validation failure instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class ListTablesArgs(BaseModel):
    """`list_tables` takes no arguments."""

    model_config = ConfigDict(extra="forbid")


def _render_result(result: QueryResult) -> str:
    """Render rows as a compact table for the model to read.

    Values are neutralized: they are database contents, and a `filename` column holds
    whatever a user named their upload — an injection payload is a legal filename.
    """
    if result.row_count == 0:
        return "The query succeeded and returned no rows."

    header = " | ".join(result.columns)
    divider = "-+-".join("-" * len(column) for column in result.columns)
    body = "\n".join(
        " | ".join("NULL" if value is None else neutralize(str(value)) for value in row)
        for row in result.rows
    )

    lines = [f"{result.row_count} row(s) in {result.duration_ms} ms:", header, divider, body]
    if result.truncated:
        lines.append(
            f"\nNote: results were capped at {result.limit} rows and may be incomplete. "
            "Say so if you report these numbers as totals."
        )
    return "\n".join(lines)


async def mcp_get_schema(args: GetSchemaArgs) -> str:  # noqa: ARG001 — schema contract
    """Describe every table and column the SQL tool may touch."""
    settings = get_settings()
    return (
        f"{get_schema()}\n\n"
        f"Every query is rewritten to return at most {settings.sql_max_rows} rows and is "
        f"cancelled after {settings.sql_query_timeout_ms} ms."
    )


async def mcp_list_tables(args: ListTablesArgs) -> str:  # noqa: ARG001 — schema contract
    """List the queryable table names."""
    from app.sql_agent.allowlist import table_names

    return "Queryable tables: " + ", ".join(table_names()) + ". Nothing else is available."


async def mcp_describe_table(args: DescribeTableArgs) -> str:
    """Describe one table's columns."""
    outcome = describe_table(args)
    if isinstance(outcome, ToolError):
        return refusal(outcome.message)
    return outcome


async def mcp_execute_query(args: ExecuteQueryArgs) -> str:
    """Validate, run and audit one SELECT statement."""
    org_id, user_id = current_org_and_user()

    try:
        outcome = await execute_query(
            args,
            org_id=org_id,
            user_id=user_id,
            # The question is not available at this boundary — an MCP client sends SQL, not
            # the phrasing behind it. Recorded honestly rather than guessed at, so the audit
            # trail does not imply a user asked something they may not have.
            question="[via database MCP server]",
        )
    except Exception as exc:
        logger.opt(exception=exc).error(
            "Database MCP execute_query failed for user {user}", user=user_id
        )
        raise internal_error("The query could not be executed.") from exc

    if isinstance(outcome, ToolError):
        # Already audited as rejected or failed inside sql_agent.tools.execute_query.
        return refusal(outcome.message)
    return _render_result(outcome)


def build_server() -> MCPServer:
    """The Database MCP server, with its read-only tool set registered."""
    server = MCPServer(
        name=SERVER_NAME,
        title="Business Database (read-only)",
        version="0.1.0",
        instructions=INSTRUCTIONS,
    )
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    for func, name, description in (
        (
            mcp_get_schema,
            "get_schema",
            "List every table and column available to SQL queries, with the row limit and "
            "timeout in force. Call this before writing a query.",
        ),
        (
            mcp_list_tables,
            "list_tables",
            "List just the queryable table names.",
        ),
        (
            mcp_describe_table,
            "describe_table",
            "Describe the columns of one queryable table.",
        ),
        (
            mcp_execute_query,
            "execute_query",
            "Run a single read-only SELECT statement and return the rows. Rejected if it is "
            "not one plain SELECT over the allowlisted tables and columns.",
        ),
    ):
        server.tool(name=name, description=description, annotations=read_only)(func)

    return server


__all__ = [
    "INSTRUCTIONS",
    "SERVER_NAME",
    "GetSchemaArgs",
    "ListTablesArgs",
    "build_server",
    "mcp_describe_table",
    "mcp_execute_query",
    "mcp_get_schema",
    "mcp_list_tables",
]
