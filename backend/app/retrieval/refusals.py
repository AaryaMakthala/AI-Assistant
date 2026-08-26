"""Centralized refusal policy — one reason, one response template.

Every distinct refusal in the system maps to exactly one ``ResponseReason`` value
and one user-facing template.  This prevents scattered refusal strings from
producing contradictory messages (Phase A, step 7).
"""

from __future__ import annotations

from enum import Enum


class ResponseReason(str, Enum):
    """Why the system refused or declined to answer.

    Each value maps to exactly one user-facing response template in
    :func:`refusal_message`.
    """

    #: No document content relevant to the question was found.
    NO_EVIDENCE = "no_evidence"
    #: Documents exist but none contain information about the specific topic.
    NOT_RELEVANT = "not_relevant"
    #: General-knowledge question unrelated to workspace documents.
    OUT_OF_SCOPE = "out_of_scope"
    #: Ambiguous reference that cannot be resolved from context.
    NEEDS_CLARIFICATION = "needs_clarification"
    #: App-help question without an authoritative source in the codebase.
    APP_HELP_UNAVAILABLE = "app_help_unavailable"
    #: Identity information not available from the authenticated session.
    IDENTITY_UNAVAILABLE = "identity_unavailable"
    #: Valid metadata query that returned zero results.
    METADATA_EMPTY = "metadata_empty"


# ---------------------------------------------------------------------------
# Response templates — exactly one per reason
# ---------------------------------------------------------------------------

_RESPONSES: dict[ResponseReason, str] = {
    ResponseReason.NO_EVIDENCE: (
        "I couldn't find any relevant information about that topic "
        "in your uploaded documents."
    ),
    ResponseReason.NOT_RELEVANT: (
        "Your workspace contains documents, but none of them contain "
        "information about that specific topic. Try rephrasing your question "
        "or check that the relevant document has been uploaded."
    ),
    ResponseReason.OUT_OF_SCOPE: (
        "That's outside what I can help with here — I answer questions about "
        "your workspace documents and available workspace information."
    ),
    ResponseReason.NEEDS_CLARIFICATION: (
        "I'm not sure what you're referring to. Could you clarify your question?"
    ),
    ResponseReason.APP_HELP_UNAVAILABLE: (
        "I don't have authoritative information about that."
    ),
    ResponseReason.IDENTITY_UNAVAILABLE: (
        "I don't have your name available."
    ),
    ResponseReason.METADATA_EMPTY: (
        "No results found for that query."
    ),
}


def refusal_message(reason: ResponseReason) -> str:
    """Return the user-facing response for a given refusal reason."""
    return _RESPONSES[reason]


__all__ = ["ResponseReason", "refusal_message"]
