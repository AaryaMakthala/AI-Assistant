"""Turning failures into clear tool errors instead of crashes (CLAUDE.md 4.5).

Phase 6's verification bar is that malformed arguments are "rejected with a clear error,
not a crash". Three things have to be true for that:

1. **Pydantic validates first.** Each tool takes a single validated model, so a bad type or
   an unknown field is refused before any code that touches a database or a network runs.
   The SDK builds each tool's JSON Schema from these models, so the client sees the contract
   too.
2. **Expected refusals return text.** A missing table, an unknown document, a query the
   validator rejected — these are outcomes the model should read and correct, not faults.
   They come back as ordinary tool results, so the agent can adjust and retry.
3. **Unexpected failures raise MCPError.** A protocol-level error tells the client the call
   failed rather than handing back text that reads like an answer. The message is written
   for a caller, never assembled from a driver exception: Postgres permission errors quote
   the exact tables and columns that were refused, which is the schema information the
   allowlist exists to withhold. Details go to the log.
"""

from __future__ import annotations

from typing import TypeVar

from mcp import MCPError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS
from pydantic import BaseModel, ValidationError

from app.security.untrusted import neutralize

ModelT = TypeVar("ModelT", bound=BaseModel)


def refusal(message: str) -> str:
    """An expected, correctable refusal, phrased for the model to act on."""
    return f"REFUSED: {message}"


def invalid_params(message: str) -> MCPError:
    """Arguments that could not be accepted."""
    return MCPError(INVALID_PARAMS, message)


def internal_error(message: str) -> MCPError:
    """A failure the caller cannot fix. `message` must not quote internal detail."""
    return MCPError(INTERNAL_ERROR, message)


def validate_args(model: type[ModelT], raw: object) -> ModelT:
    """Coerce tool arguments into `model`, or raise a clear INVALID_PARAMS error.

    Used where arguments arrive as a mapping rather than through the SDK's own
    signature-derived validation — the Phase 7 agent path calls tools directly.
    """
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'argument'}: {error['msg']}"
            for error in exc.errors()[:5]
        )
        # Neutralized because the echo includes attacker-influenced values, and the result
        # is read back into the model's context.
        raise invalid_params(f"Invalid arguments — {neutralize(problems)}") from exc


__all__ = [
    "internal_error",
    "invalid_params",
    "refusal",
    "validate_args",
]
