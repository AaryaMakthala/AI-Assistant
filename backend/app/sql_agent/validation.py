"""Structural validation of LLM-generated SQL (CLAUDE.md 4.3).

Nothing reaches the database without passing every check here. The approach is
**allowlist over AST node type**, not pattern-matching over text, because every
string-level defense has a known bypass: casing (`dRoP`), comments
(`SELECT 1 --\\n; DROP`), whitespace, unicode, and nested quoting all defeat a regex while
leaving the parsed meaning intact. A parser sees what Postgres will see.

Two properties make the node-type allowlist strong rather than merely tidy:

1. **sqlglot gives dangerous constructs their own node classes.** `current_user`,
   `version()` and `generate_series()` do not arrive as ordinary function calls — they
   parse to `CurrentUser`, `CurrentVersion` and `ExplodingGenerateSeries`. A check that
   only inspected function *names* would wave all three through.
2. **Anything sqlglot does not recognize becomes `Anonymous` or `Command`.** So
   `pg_read_file(...)`, `current_setting(...)` and `DO $$ ... $$` are rejected by default
   rather than needing to be enumerated. New Postgres versions and new extensions cannot
   widen the surface, which a denylist could never promise.

The query that comes out is regenerated from the validated AST and re-validated, so the
string handed to the driver is one this module built — not one the model wrote.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from app.config import get_settings
from app.sql_agent.allowlist import ALLOWED_TABLES, TableSpec, lookup_table

DIALECT = "postgres"

#: Schema qualifiers a table reference may carry. Anything else — `pg_catalog`,
#: `information_schema`, a cross-database `catalog.db.table` — is refused, so the
#: allowlisted *name* cannot be used to reach a same-named table elsewhere.
_ALLOWED_SCHEMAS = frozenset({"", "public"})

#: The complete set of expression types a validated query may contain. Every entry was
#: added because a legitimate question needs it; the absence of an entry is the defense.
#:
#: Deliberately excluded, with the attack each one enables:
#:   Join, Subquery, CTE/With, Union, Exists — reach a non-allowlisted table in one query
#:   Anonymous, Command                      — any function or statement sqlglot cannot parse
#:   CurrentUser, CurrentVersion, SessionUser — session and server reconnaissance
#:   ExplodingGenerateSeries, Window, Lag     — unbounded row generation, ignoring LIMIT
#:   Lock (FOR UPDATE), Into, Copy            — writes and filesystem access
#:   ObjectIdentifier (`::regclass`, `::oid`) — catalog lookup through a cast
#:   Tuple (GROUP BY grouping sets)           — no question needs it; keeps the surface small
_ALLOWED_NODES: frozenset[type[exp.Expression]] = frozenset(
    {
        # --- Query shape ---
        exp.Select,
        exp.From,
        exp.Table,
        exp.TableAlias,
        exp.Where,
        exp.Group,
        exp.Having,
        exp.Order,
        exp.Ordered,
        exp.Limit,
        exp.Offset,
        exp.Distinct,
        # --- Identifiers and values ---
        exp.Identifier,
        exp.Column,
        exp.Alias,
        exp.Literal,
        exp.Boolean,
        exp.Null,
        exp.Star,  # Constrained further below: only ever as a function argument.
        exp.Interval,
        exp.Var,
        exp.Cast,
        exp.DataType,
        # --- Boolean and comparison operators ---
        exp.And,
        exp.Or,
        exp.Not,
        exp.Paren,
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.In,
        exp.Is,
        exp.Like,
        exp.ILike,
        exp.Between,
        exp.Case,
        exp.If,
        # --- Arithmetic. `Sub` also carries `now() - interval '30 days'`. ---
        exp.Add,
        exp.Sub,
        exp.Mul,
        exp.Div,
        exp.Neg,
        # --- Functions. The prompt-facing list of these is ALLOWED_FUNCTIONS in
        # allowlist.py; these node types are what actually gates execution. ---
        exp.Avg,
        exp.Count,
        exp.Max,
        exp.Min,
        exp.Sum,
        exp.Coalesce,
        exp.Nullif,
        exp.CurrentDate,
        exp.CurrentTimestamp,  # now()
        exp.Extract,  # also date_part()
        exp.TimestampTrunc,  # date_trunc()
        exp.Length,
        exp.Lower,
        exp.Upper,
        exp.Round,
    }
)


class SQLValidationError(ValueError):
    """A generated query was refused.

    `reason` is safe to show the model on a retry: it names the rule that was broken
    without describing how to get around it.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ValidatedQuery:
    """A query cleared for execution, rebuilt from its own parse tree."""

    sql: str
    table: TableSpec
    limit: int


def _meaningful(statements: list[exp.Expression | None]) -> list[exp.Expression]:
    """Drop the artifacts of stray semicolons.

    `parse` represents `SELECT 1;` as two entries and `a; b` as three — the separators
    surface as `None` or `Semicolon`. Filtering them means a harmless trailing semicolon is
    accepted while a genuinely stacked query still counts as two statements.
    """
    return [
        statement
        for statement in statements
        if statement is not None and not isinstance(statement, exp.Semicolon)
    ]


def _reject_comments(statement: exp.Expression) -> None:
    """Refuse any query carrying a comment.

    A comment is the classic vehicle for smuggling a second statement past a naive
    filter, and sqlglot preserves comments when it regenerates SQL — so tolerating them
    would mean shipping attacker-controlled text to the driver. No legitimate generated
    query needs one.
    """
    for node in statement.walk():
        if node.comments:
            raise SQLValidationError("Comments are not permitted in a generated query.")


def _reject_disallowed_nodes(statement: exp.Expression) -> None:
    for node in statement.walk():
        if type(node) not in _ALLOWED_NODES:
            raise SQLValidationError(
                f"{node.sql(dialect=DIALECT)[:60]!r} uses SQL this tool does not allow "
                f"({type(node).__name__}). Only a single plain SELECT over the "
                "documented tables is permitted — no joins, subqueries, CTEs, set "
                "operations or non-documented functions."
            )


def _reject_bare_star(statement: exp.Expression) -> None:
    """Allow `count(*)` but not `SELECT *`.

    `SELECT *` would return every column of the table, including the ones deliberately
    withheld from the allowlist — the grant in migration 0003 blocks it at the database
    too, but failing here gives the model a usable error instead of a driver exception.
    """
    for star in statement.find_all(exp.Star):
        if isinstance(star.parent, exp.Select | exp.Column):
            raise SQLValidationError(
                "Select columns explicitly; '*' is not permitted. Use count(*) for a " "row count."
            )


def _resolve_table(statement: exp.Expression) -> TableSpec:
    """Identify the single allowlisted table the query reads.

    Exactly one is possible by construction: joins, subqueries, CTEs and set operations are
    all rejected above. That is what makes unqualified column names unambiguous below.
    """
    tables = list(statement.find_all(exp.Table))
    if not tables:
        raise SQLValidationError("The query must read from one of the documented tables.")
    if len(tables) > 1:
        raise SQLValidationError("The query may reference only one table.")

    table = tables[0]
    if table.args.get("only"):
        raise SQLValidationError("The ONLY modifier is not permitted.")
    if table.catalog or table.db.lower() not in _ALLOWED_SCHEMAS:
        raise SQLValidationError(
            "Reference tables by their plain name; schema and database qualifiers are "
            "not permitted."
        )

    spec = lookup_table(table.name)
    if spec is None:
        raise SQLValidationError(
            f"Table {table.name!r} is not available to this tool. Available tables: "
            f"{', '.join(ALLOWED_TABLES)}."
        )
    return spec


def _reject_disallowed_columns(statement: exp.Expression, table: TableSpec) -> None:
    """Check every column reference against the table's exposed columns.

    A column may be qualified only by the table's own name or alias. Select-list aliases
    are accepted too: `ORDER BY n` after `count(*) AS n` parses as a column reference, and
    rejecting it would refuse ordinary aggregate queries.
    """
    node = statement.find(exp.Table)
    known_qualifiers = {table.name.lower()}
    if node is not None and node.alias:
        known_qualifiers.add(node.alias.lower())

    select_aliases = {alias.alias.lower() for alias in statement.find_all(exp.Alias) if alias.alias}
    allowed = {name.lower() for name in table.column_names}

    for column in statement.find_all(exp.Column):
        qualifier = column.table.lower()
        if qualifier and qualifier not in known_qualifiers:
            raise SQLValidationError(
                f"Unknown table qualifier {column.table!r}. Use {table.name!r} or its alias."
            )

        name = column.name.lower()
        # An alias only stands in for a column when used unqualified — `d.n` is a real
        # column reference on the table, whatever the select list happens to be named.
        if name in allowed or (not qualifier and name in select_aliases):
            continue
        raise SQLValidationError(
            f"Column {column.name!r} is not available on {table.name!r}. Available "
            f"columns: {', '.join(sorted(table.column_names))}."
        )


def _requested_limit(statement: exp.Expression, *, max_rows: int) -> int | None:
    """The model's own LIMIT, if it is a plain integer within the cap.

    Returns None when there is no limit, when it is not a bare integer literal (`LIMIT ALL`,
    `LIMIT 1+1`), or when it exceeds the cap — in every one of those cases the ceiling is
    what should apply.
    """
    limit = statement.args.get("limit")
    if limit is None or not isinstance(limit.expression, exp.Literal):
        return None
    try:
        value = int(limit.expression.name)
    except ValueError:
        return None
    return value if 1 <= value <= max_rows else None


def _apply_row_limit(statement: exp.Expression, *, limit: int) -> exp.Expression:
    """Set the row limit, replacing whatever was there.

    Always written out, even when it matches what the model asked for: `LIMIT ALL` parses to
    no limit at all, so anything short of an unconditional set would let that through as an
    unlimited query with nothing to correct.
    """
    return statement.limit(limit)


def _validate_tree(statement: exp.Expression) -> TableSpec:
    """Run every structural check. Returns the table the query reads."""
    if not isinstance(statement, exp.Select):
        raise SQLValidationError(
            f"Only SELECT statements are permitted; got {type(statement).__name__.upper()}."
        )
    _reject_comments(statement)
    _reject_disallowed_nodes(statement)
    _reject_bare_star(statement)
    table = _resolve_table(statement)
    _reject_disallowed_columns(statement, table)
    return table


def validate_query(sql: str, *, max_rows: int | None = None) -> ValidatedQuery:
    """Validate `sql` and return the exact string that may be executed.

    Raises :class:`SQLValidationError` for anything that is not a single plain SELECT over
    the allowlist. The returned SQL is regenerated from the parse tree and re-validated,
    so the caller executes a string this module produced rather than one the model wrote.
    """
    if max_rows is None:
        max_rows = get_settings().sql_max_rows

    text = sql.strip()
    if not text:
        raise SQLValidationError("The query is empty.")

    try:
        parsed = sqlglot.parse(text, dialect=DIALECT)
    except sqlglot.ParseError as exc:
        raise SQLValidationError(f"The query is not valid SQL: {exc}") from exc

    statements = _meaningful(parsed)
    if not statements:
        raise SQLValidationError("The query is empty.")
    if len(statements) > 1:
        # The canonical injection shape. Refused before any single statement is inspected,
        # so a benign-looking first statement cannot carry a second one along with it.
        raise SQLValidationError(
            "Submit exactly one statement; multiple statements are not permitted."
        )

    table = _validate_tree(statements[0])
    # A narrower LIMIT the model asked for is kept: "the three largest documents" means
    # LIMIT 3, and overwriting it with the ceiling would return 500 rows for a question that
    # asked for three — leaving the model to trim the excess in prose, which is exactly the
    # kind of quiet mismatch between query and answer this layer exists to prevent. The cap
    # still applies to anything larger, absent, or not a plain integer.
    effective_limit = _requested_limit(statements[0], max_rows=max_rows) or max_rows
    limited = _apply_row_limit(statements[0], limit=effective_limit)
    rendered = limited.sql(dialect=DIALECT, comments=False)

    # The AST is trusted; its serialization is not assumed to be. Re-parsing proves the
    # string about to be executed still means exactly what was approved — and that the
    # limit survived into the output.
    reparsed = _meaningful(sqlglot.parse(rendered, dialect=DIALECT))
    if len(reparsed) != 1:
        raise SQLValidationError("The query could not be safely normalized.")
    _validate_tree(reparsed[0])

    written = reparsed[0].args.get("limit")
    if written is None or written.expression.name != str(effective_limit):
        raise SQLValidationError("The query could not be safely normalized.")

    return ValidatedQuery(sql=rendered, table=table, limit=effective_limit)


__all__ = [
    "DIALECT",
    "SQLValidationError",
    "ValidatedQuery",
    "validate_query",
]
