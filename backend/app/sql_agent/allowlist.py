"""What the SQL agent is allowed to know about (CLAUDE.md 4.3).

This module is the *only* source of truth for the agent's visible surface. Nothing here is
introspected from the live database: automatic introspection is how a schema tool ends up
describing `users` or a credentials table the day someone adds one (CLAUDE.md section 7).
Adding a table is a deliberate edit to this file, reviewed like any other change.

The allowlist is intentionally narrow. The real database currently holds only the
application's own schema — there are no `customers`/`orders`/`employees` business tables —
so `documents` is the one table with data a user could legitimately ask aggregate questions
about, and even there the sensitive columns are withheld:

- `org_id`, `uploaded_by` — tenancy and identity. RLS already scopes rows to the caller's
  org, so exposing the column would let a query filter or group by an org it cannot read,
  turning row-level isolation into an oracle for which org_ids exist.
- `storage_key` — the on-disk name of the raw upload. Not secret by itself, but it is the
  one value that turns "I can read metadata" into "I can name a file on the server".
- `error_message` — ingestion failures embed library exception text, which can quote
  document contents and file paths.

Column-level rather than table-level filtering matters because the DB grant in migration
0003 mirrors this list exactly. Both layers are derived from the same constants, so they
cannot drift out of step. `visibility` was added later, by migration 0007, and carries its
grant there — it is safe to expose because it names a category rather than a person, and
the rows it describes are already filtered to the ones the caller may read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ColumnSpec:
    """One exposed column, as the schema tool describes it to the model."""

    name: str
    type: str
    description: str


@dataclass(frozen=True)
class TableSpec:
    """One exposed table and the subset of its columns the agent may reference."""

    name: str
    description: str
    columns: tuple[ColumnSpec, ...]

    @property
    def column_names(self) -> frozenset[str]:
        return frozenset(column.name for column in self.columns)

    def render(self) -> str:
        """A compact DDL-ish rendering for the prompt.

        Deliberately not real `CREATE TABLE` output: it must not imply the agent may issue
        DDL, and the per-column notes carry more useful signal than exact Postgres types.
        """
        lines = [f"TABLE {self.name} — {self.description}"]
        width = max(len(column.name) for column in self.columns)
        for column in self.columns:
            lines.append(f"  {column.name.ljust(width)}  {column.type:<12} {column.description}")
        return "\n".join(lines)


DOCUMENTS = TableSpec(
    name="documents",
    description=(
        "One row per file uploaded to the knowledge base. Rows are already restricted to "
        "the asking user's own organization; there is no column for filtering by "
        "organization and none is needed."
    ),
    columns=(
        ColumnSpec("id", "uuid", "Primary key."),
        ColumnSpec("filename", "text", "Original display name, e.g. 'refund_policy.pdf'."),
        ColumnSpec("mime_type", "text", "Detected content type, e.g. 'application/pdf'."),
        ColumnSpec("size_bytes", "bigint", "File size in bytes."),
        ColumnSpec(
            "status",
            "text",
            "Ingestion state: 'pending', 'processing', 'ready' or 'failed'.",
        ),
        ColumnSpec("page_count", "integer", "Pages extracted; NULL for non-paginated files."),
        ColumnSpec(
            "visibility",
            "text",
            "'org' for organization-wide files, 'personal' for a single user's own.",
        ),
        ColumnSpec("created_at", "timestamptz", "When the file was uploaded."),
        ColumnSpec("updated_at", "timestamptz", "When the row last changed."),
    ),
)

#: Every table the agent may reference. Keyed by lowercase name — SQL identifiers are
#: case-insensitive unless quoted, so lookups normalize before consulting this map.
ALLOWED_TABLES: Mapping[str, TableSpec] = MappingProxyType({DOCUMENTS.name: DOCUMENTS})

#: Scalar and aggregate functions a generated query may call. An allowlist rather than a
#: denylist: Postgres ships functions that read files (`pg_read_file`), reveal settings
#: (`current_setting`) and identify the session (`current_user`), and a denylist would have
#: to anticipate all of them plus every extension a future migration installs.
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    {
        # Aggregates — the point of the tool.
        "avg",
        "count",
        "max",
        "min",
        "sum",
        # Null handling and conditionals.
        "coalesce",
        "nullif",
        # Dates. `now`/`current_date` are permitted: they reveal nothing and questions like
        # "uploaded in the last 30 days" are unanswerable without them.
        "current_date",
        "date_part",
        "date_trunc",
        "extract",
        "now",
        # Text and numeric shaping for readable output.
        "length",
        "lower",
        "round",
        "upper",
    }
)


def allowed_schema_text() -> str:
    """The whole visible schema, as handed to the model."""
    return "\n\n".join(spec.render() for spec in ALLOWED_TABLES.values())


def table_names() -> tuple[str, ...]:
    return tuple(ALLOWED_TABLES)


def lookup_table(name: str) -> TableSpec | None:
    """Resolve a table name from a query, ignoring case and any schema qualifier."""
    return ALLOWED_TABLES.get(name.strip().strip('"').lower())


__all__ = [
    "ALLOWED_FUNCTIONS",
    "ALLOWED_TABLES",
    "ColumnSpec",
    "DOCUMENTS",
    "TableSpec",
    "allowed_schema_text",
    "lookup_table",
    "table_names",
]
