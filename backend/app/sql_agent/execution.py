"""Executing a validated query under the read-only role (CLAUDE.md 4.3).

Four independent limits apply to every execution, each of which would be sufficient on its
own for a different failure:

- `SET LOCAL ROLE app_sql_agent` — the database refuses writes and any table or column
  outside the allowlist, so a bypass of the sqlglot layer still reaches nothing.
- `SET TRANSACTION READ ONLY` — blocks writes at the transaction level too, including
  anything a function might attempt.
- `SET LOCAL statement_timeout` — a query can be valid, allowed and still ruinous. A model
  that produces a heavy aggregate must not be able to hold a connection indefinitely.
- The `LIMIT` that :func:`app.sql_agent.validation.validate_query` baked into the SQL.

`SET LOCAL` throughout: every setting reverts when the transaction ends, so a pooled
connection is handed back exactly as it was found. A session-level `SET ROLE` that leaked
into the next borrower of that connection would silently break unrelated requests.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_session_factory
from app.security.rls import set_tenant_claims
from app.sql_agent.validation import ValidatedQuery

#: Role names are validated against this before being interpolated into `SET LOCAL ROLE`.
#: The value comes from settings rather than user input, but `SET ROLE` cannot take a bind
#: parameter, so the one place this project builds SQL by concatenation is also the one
#: place that proves the fragment is a bare identifier.
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class SQLExecutionError(RuntimeError):
    """A validated query failed at the database.

    Carries no driver text: a permission error names the columns and tables that were
    refused, which is precisely the schema information the allowlist exists to withhold.
    The detail is logged server-side instead.
    """


@dataclass(frozen=True)
class QueryResult:
    """Rows from one successful execution."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    duration_ms: int
    #: The row cap that was in force, so a truncated answer can say what it was capped at.
    limit: int

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def truncated(self) -> bool:
        """True when the cap was reached, so the rows may be a partial set."""
        return len(self.rows) >= self.limit

    def as_records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]


def _role_statement(role: str) -> str:
    if not _IDENTIFIER.match(role):
        raise ValueError(f"Refusing to switch to a non-identifier role name: {role!r}")
    return f"SET LOCAL ROLE {role}"


@asynccontextmanager
async def sql_agent_session(
    *, org_id: uuid.UUID, user_id: uuid.UUID
) -> AsyncIterator[AsyncSession]:
    """A read-only session restricted to the SQL agent's role and the caller's org.

    Claims are set before the role switch, matching :func:`app.security.rls.tenant_session`:
    `app_sql_agent` is granted with INHERIT FALSE and holds no privilege on
    `set_config`, so the call has to happen while the connection is still its login role.
    """
    settings = get_settings()

    async with get_session_factory()() as session:
        await session.begin()
        await set_tenant_claims(session, org_id=org_id, user_id=user_id)
        # Before the role switch, while still privileged enough to change it.
        await session.execute(
            text(f"SET LOCAL statement_timeout = {int(settings.sql_query_timeout_ms)}")
        )
        await session.execute(text("SET TRANSACTION READ ONLY"))
        await session.execute(text(_role_statement(settings.sql_agent_role)))
        try:
            yield session
        finally:
            # Always rollback: nothing here may commit, and a read-only transaction has
            # nothing worth keeping.
            await session.rollback()


async def execute_validated(
    query: ValidatedQuery, *, org_id: uuid.UUID, user_id: uuid.UUID
) -> QueryResult:
    """Run an already-validated query and return its rows.

    Takes a :class:`ValidatedQuery` rather than a string so that reaching this function
    with unvalidated SQL requires constructing that type deliberately — the type is the
    reminder that validation is not optional.
    """
    started = time.perf_counter()

    async with sql_agent_session(org_id=org_id, user_id=user_id) as session:
        try:
            result = await session.execute(text(query.sql))
            columns = tuple(result.keys())
            rows = tuple(tuple(row) for row in result.fetchall())
        except DBAPIError as exc:
            # Expected when a bypass is attempted or a query is too slow. Logged in full,
            # reported opaquely.
            logger.warning(
                "SQL agent query failed for user {user}: {error}",
                user=user_id,
                error=str(exc.orig or exc).split("\n")[0],
            )
            raise SQLExecutionError("The query could not be executed.") from exc
        except SQLAlchemyError as exc:
            logger.opt(exception=exc).error(
                "SQL agent execution error for user {user}", user=user_id
            )
            raise SQLExecutionError("The query could not be executed.") from exc

    duration_ms = int((time.perf_counter() - started) * 1000)
    return QueryResult(
        columns=columns,
        rows=rows,
        duration_ms=duration_ms,
        limit=query.limit,
    )


__all__ = [
    "QueryResult",
    "SQLExecutionError",
    "execute_validated",
    "sql_agent_session",
]
