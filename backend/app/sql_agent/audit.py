"""Audit logging for the SQL agent (CLAUDE.md 4.3).

Every generated query is recorded with the user who caused it — including the ones that
were refused. A log containing only the queries that ran would omit exactly the entries an
incident review needs: the rejected attempts are the evidence that something tried.

Two properties are load-bearing:

- **A failed audit write never fails the request silently.** It is logged at ERROR through
  Loguru, so the record survives even when the database insert does not. Losing the audit
  row is not allowed to be invisible.
- **A failed audit write also never breaks the user's answer.** The write happens after the
  outcome is known, in its own transaction, so a full disk or a policy change degrades
  auditing rather than the product.

The write runs as `app_tenant` under the tenant's own claims, which hold INSERT and nothing
else on this table. Even this module cannot read the log back — that is an operator action
using the administrative role.
"""

from __future__ import annotations

import uuid
from typing import Literal

from loguru import logger
from sqlalchemy import insert

from app.db.models import SqlQueryAudit
from app.security.rls import tenant_session

AuditStatus = Literal["accepted", "rejected", "failed"]

#: Questions and SQL are stored in full up to this length. Truncation is about keeping one
#: pathological input from bloating the table, not about hiding anything.
_MAX_TEXT = 4000


def _clip(value: str | None) -> str | None:
    if value is None:
        return None
    return value if len(value) <= _MAX_TEXT else value[: _MAX_TEXT - 1] + "…"


async def record_query(
    *,
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    question: str,
    status: AuditStatus,
    generated_sql: str | None = None,
    rejection_reason: str | None = None,
    row_count: int | None = None,
    duration_ms: int | None = None,
) -> None:
    """Write one audit row. Never raises.

    `status` distinguishes the three outcomes that matter on review: `accepted` (validated
    and executed), `rejected` (refused by the validation layer, so it never reached the
    database) and `failed` (validated, then errored or timed out at the database).
    """
    # Logged unconditionally, before the insert is attempted: this line is the fallback
    # record if the database write is the thing that fails.
    logger.bind(
        sql_audit=True,
        user_id=str(user_id),
        org_id=str(org_id),
        status=status,
    ).info(
        "SQL agent {status}: {sql}",
        status=status,
        sql=generated_sql or rejection_reason or "<no query generated>",
    )

    try:
        async with tenant_session(org_id=org_id, user_id=user_id) as session:
            await session.execute(
                insert(SqlQueryAudit).values(
                    # Supplied here rather than left to the column's server default: with a
                    # default, SQLAlchemy adds `RETURNING id` to read the generated value
                    # back, and RETURNING requires SELECT — which this table deliberately
                    # does not grant. Every audit write would fail on a permission error.
                    id=uuid.uuid4(),
                    org_id=org_id,
                    user_id=user_id,
                    question=_clip(question),
                    generated_sql=_clip(generated_sql),
                    status=status,
                    rejection_reason=_clip(rejection_reason),
                    row_count=row_count,
                    duration_ms=duration_ms,
                )
            )
    except Exception as exc:
        # The structured log line above is the surviving record. Loud, because a silently
        # broken audit trail is worse than a noisy one.
        logger.opt(exception=exc).error(
            "Could not persist SQL audit row for user {user} (status={status})",
            user=user_id,
            status=status,
        )


__all__ = ["AuditStatus", "record_query"]
