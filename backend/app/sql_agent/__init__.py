"""Guarded natural-language SQL over the real application database (CLAUDE.md 4.3).

The layers, outermost first. Each is independent, so a failure in one does not open the
others:

1. `prompts.py`      — tells the model the rules. Guidance only; assumed defeatable.
2. `validation.py`   — parses the generated SQL with sqlglot and refuses anything that is
                       not a single plain SELECT over the allowlist. The real gate.
3. `allowlist.py`    — the static set of tables, columns and functions that exist at all.
4. `execution.py`    — runs it as a read-only role, in a read-only transaction, under a
                       statement timeout, with a forced LIMIT.
5. `audit.py`        — records every query and every refusal against the user who caused it.

Nothing here is proven by automated tests until Phase 14, so the allowlist is kept as narrow
as the phase's goal permits.
"""

from app.sql_agent.agent import SQLAnswer, answer_question, synthesize_answer
from app.sql_agent.allowlist import ALLOWED_TABLES, allowed_schema_text
from app.sql_agent.execution import QueryResult, SQLExecutionError
from app.sql_agent.tools import (
    DescribeTableArgs,
    ExecuteQueryArgs,
    ToolError,
    describe_table,
    execute_query,
    get_schema,
)
from app.sql_agent.validation import SQLValidationError, ValidatedQuery, validate_query

__all__ = [
    "ALLOWED_TABLES",
    "DescribeTableArgs",
    "ExecuteQueryArgs",
    "QueryResult",
    "SQLAnswer",
    "SQLExecutionError",
    "SQLValidationError",
    "ToolError",
    "ValidatedQuery",
    "allowed_schema_text",
    "answer_question",
    "describe_table",
    "execute_query",
    "get_schema",
    "synthesize_answer",
    "validate_query",
]
