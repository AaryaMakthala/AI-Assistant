"""Shared test helpers for the unit test suite.

Provides standardized fakes for SQLAlchemy sessions/results, the smart
mock LLM router (single source of truth for mock routing behavior), and
the StubLLM provider used across all test files.
"""

from __future__ import annotations

import re as _re
from typing import Any
from collections.abc import AsyncIterator

from app.llm.base import Completion, Message, TokenUsage


# ---------------------------------------------------------------------------
# FakeResult — mimics SQLAlchemy Result for scalar / row queries
# ---------------------------------------------------------------------------

class FakeResult:
    """Mimics a SQLAlchemy Result object for unit tests."""

    def __init__(self, rows: list[Any] | None = None, scalar: Any = None) -> None:
        self._rows = rows or []
        self._scalar = scalar

    def scalar_one(self) -> Any:
        return self._scalar

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def scalars(self) -> FakeResult:
        return self

    def all(self) -> list[Any]:
        return self._rows

    def first(self) -> Any:
        return self._rows[0] if self._rows else None


# ---------------------------------------------------------------------------
# FakeSession — configurable session mock with default fallback
# ---------------------------------------------------------------------------

class FakeSession:
    """Configurable session mock.

    When all explicit responses are consumed, falls back to ``default``
    (defaults to ``FakeResult()`` which returns ``None`` / ``[]``).
    """

    def __init__(
        self,
        responses: list[FakeResult] | None = None,
        default: FakeResult | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._call_index = 0
        self._default = default or FakeResult()
        self.execute_calls: list[Any] = []

    async def execute(self, stmt: Any) -> FakeResult:
        self.execute_calls.append(stmt)
        if self._call_index < len(self._responses):
            result = self._responses[self._call_index]
            self._call_index += 1
            return result
        return self._default

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


def session_with_history(
    *data_responses: FakeResult,
    default: FakeResult | None = None,
) -> FakeSession:
    """Build a FakeSession with a leading _load_recent_history response.

    The LLM-first architecture always calls _load_recent_history before
    any intent-specific handler.  That call needs one response (typically
    FakeResult(scalar=None) = "no session found").  This helper
    prepends that automatically so callers only provide the data responses.
    """
    all_responses = [FakeResult(scalar=None), *data_responses]
    return FakeSession(responses=all_responses, default=default)


# ---------------------------------------------------------------------------
# FakeQuery — minimal SQLAlchemy query builder for metadata tests
# ---------------------------------------------------------------------------

class FakeQuery:
    """Minimal SQLAlchemy-style query builder."""

    def __init__(self, rows: list[Any] | None = None, scalar: Any = None) -> None:
        self._rows = rows or []
        self._scalar = scalar

    def where(self, *args: Any, **kwargs: Any) -> FakeQuery:
        return self

    def order_by(self, *args: Any, **kwargs: Any) -> FakeQuery:
        return self

    def limit(self, *args: Any, **kwargs: Any) -> FakeQuery:
        return self

    def offset(self, *args: Any, **kwargs: Any) -> FakeQuery:
        return self

    def select_from(self, *args: Any, **kwargs: Any) -> FakeQuery:
        return self

    def scalar_one(self) -> Any:
        return self._scalar

    def scalars(self) -> FakeQuery:
        return self

    def all(self) -> list[Any]:
        return self._rows


# ---------------------------------------------------------------------------
# StubLLM — scripted LLM provider
# ---------------------------------------------------------------------------

class StubLLM:
    """Scripted provider: returns fixed text, or raises a configured error."""

    name = "test-provider"
    model = "test-model"

    def __init__(self, text: str = "An answer.") -> None:
        self._text = text
        self._error: Exception | None = None
        self.calls: list[list[Message]] = []

    def fail_with(self, error: Exception) -> None:
        self._error = error

    async def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        completion.provider = self.name
        completion.model = self.model
        if self._error is not None:
            raise self._error
        completion.text = self._text
        completion.usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        yield self._text


# ---------------------------------------------------------------------------
# Smart mock LLM router — single source of truth for test routing
# ---------------------------------------------------------------------------

async def smart_mock_route(
    *, query: str, history: list | None = None, **kw: Any
) -> Any:
    """Smart mock for route_with_llm that classifies common queries.

    Returns a RouteResult with the appropriate route for common query
    patterns.  Falls back to DOCUMENT_CONTENT for unrecognized queries.

    Tests that need specific routes should monkeypatch route_with_llm
    themselves — this is the default mock for tests that don't care about
    routing specifics.
    """
    from app.retrieval.llm_router import RouteResult
    from app.retrieval.intent import normalize_for_classification

    q = normalize_for_classification(query)

    # Greetings
    if _re.match(r"^(?:hi+|hello+|hey+|hola|bonjour|namaste|ciao|bye|thank)", q):
        return RouteResult(route="GREETING", confidence=0.95, reasoning="mock")

    # Out of scope — math, geography, weather, programming
    if _re.search(r"(?:capital\s+of|weather|joke|python|javascript|\d\s+\d)", q):
        return RouteResult(route="OUT_OF_SCOPE", confidence=0.95, reasoning="mock")

    # Identity: assistant
    if _re.search(r"(?:who\s+are\s+you|what\s+(?:is|are)\s+your)", q):
        return RouteResult(route="IDENTITY_ASSISTANT", confidence=0.95, reasoning="mock")

    # Identity: user
    if _re.search(r"(?:my\s+name\s+is|what\s+is\s+my\s+(?:name|info|email))", q):
        return RouteResult(route="IDENTITY_USER", confidence=0.9, reasoning="mock")

    # Permissions
    if _re.search(r"(?:who\s+can|can\s+(?:i|we|members?)\s+(?:upload|add|invite|approve|delete))", q):
        return RouteResult(route="PERMISSIONS", confidence=0.9, reasoning="mock")

    # App help
    if _re.search(r"(?:monitored|tracked|watched|logging)", q):
        return RouteResult(route="APP_HELP", confidence=0.8, reasoning="mock")

    # App help — invite/add member
    if _re.search(r"(?:how\s+(?:do|can|should)\s+(?:i|we)\s+(?:invite|add|onboard)\s+(?:a\s+)?(?:member|user|person|colleague|someone))", q):
        return RouteResult(route="APP_HELP", confidence=0.9, reasoning="mock")

    # Conversation history
    if _re.search(r"(?:what\s+(?:are|were)\s+(?:the\s+)?(?:questions?|things?)|what\s+(?:did|was)\s+(?:my|the)\s+(?:previous|last)|what\s+did\s+(?:i|we)\s+(?:ask|say)|what\s+did\s+you\s+(?:just\s+)?(?:answer|say)|what\s+have\s+(?:i|we)\s+(?:been|discussed|talked)|show\s+(?:me\s+)?(?:my\s+)?(?:previous|recent|last))", q):
        return RouteResult(route="CONVERSATION_HISTORY", confidence=0.9, reasoning="mock")

    # Vague content questions — ask for clarification (must be BEFORE 'about' pattern)
    if _re.search(r"(?:what\s+(?:theu|it|does\s+it)\s+(?:say|sau|mean|about))", q):
        return RouteResult(route="NEEDS_CLARIFICATION", confidence=0.8, reasoning="mock")

    # Bare pronoun references — check history first for document context.
    # 'what are they' after a document question → METADATA (doc_list).
    bare_pronoun_match = _re.search(
        r"^(?:wh(?:at|o|ere|en|ich)|hwat|ho(?:w)?)\s+(?:are|is|was|were)\s+(?:they|them|it|he|she|that|this)\s*[?.!]*\s*$",
        q,
    )
    if bare_pronoun_match:
        # Check if history has a recent document-related question.
        _DOC_HISTORY_RE = _re.compile(
            r"(?:document|file|upload|doc|count)\s*", _re.IGNORECASE,
        )
        if history:
            for turn in history[-4:]:
                if turn.get("role") == "user" and _DOC_HISTORY_RE.search(turn.get("content", "")):
                    return RouteResult(route="METADATA", confidence=0.85, reasoning="mock_pronoun_doc_history")
        return RouteResult(route="NEEDS_CLARIFICATION", confidence=0.85, reasoning="mock")

    # Topic-qualified content questions go to DOCUMENT_CONTENT
    if _re.search(r"(?:about|discuss|cover|mention|regarding)\b", q):
        return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="mock")

    # Metadata — document count
    if _re.search(
        r"(?:how\s+many|number\s+of)\s+(?:uploaded\s+)?(?:my\s+|the\s+|this\s+)?(?:own\s+)?(?:documents?|files?|uploaded)\b",
        q,
    ):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — document list
    if _re.search(
        r"(?:list|show)\s+(?:are\s+the\s+)?(?:me\s+)?(?:all\s+)?(?:my\s+|the\s+|this\s+)?(?:uploaded\s+)?(?:documents?|files?)\b",
        q,
    ):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — "what are the documents" / "what are the names of documents"
    if _re.search(r"^what\s+(?:are|is)\s+(?:the\s+|my\s+)?(?:names?\s+(?:of\s+)?)?(?:uploaded\s+)?(?:documents?|files?)\s*$", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — "what documents have I uploaded"
    if _re.search(r"^what\s+(?:the\s+|my\s+|this\s+)?(?:uploaded\s+)?(?:documents?|files?)\s+(?:have|are|did)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — "list all documents"
    if _re.search(r"^(?:list|show)\s+(?:all\s+)?(?:the\s+|my\s+)?(?:uploaded\s+)?(?:documents?|files?)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — member count (with optional status)
    if _re.search(r"^(?:how\s+many|number\s+of)\s+(?:\w+\s+)?(?:members?|people|users?|employees?)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — member status queries
    if _re.search(r"^how\s+many\s+are\s+(?:invited|pending|active|confirmed|removed)\s*$", q):
        return RouteResult(route="METADATA", confidence=0.85, reasoning="mock")

    if _re.search(r"^who\s+(?:is|are)\s+(?:invited|pending|active)\s*$", q):
        return RouteResult(route="METADATA", confidence=0.85, reasoning="mock")

    # Metadata — member list (with optional status)
    if _re.search(r"(?:list|show)\s+(?:all\s+)?(?:the\s+|my\s+)?(?:\w+\s+)?(?:members?|people|users?|employees?)\b", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — role query
    if _re.search(r"(?:what\s+(?:is|are)\s+my|my)\s+(?:role|access|permission)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — page count
    if _re.search(r"(?:how\s+many|number\s+of|total)\s+\w*\s*(?:pages?|sheets?)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — document description/summary
    if _re.search(r"(?:description|summary|summery|descrption|descriction)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — company/workspace name
    if _re.search(r"(?:company|workspace|organization|org|team)\s+(?:name|is\s+(?:called|named))", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")
    if _re.search(r"what(?:'?s|\s+is)\s+(?:the\s+)?(?:(?:name\s+(?:of|for)\s+(?:the\s+|this\s+|our\s+)?)?(?:company|workspace|organization|org|team)(?:\s+name)?|(?:company|workspace|organization|org|team)\s+name)", q):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Metadata — broad document listing phrasings ("what are documents present",
    # "documents details", "give me 3 document uploaded", etc.)
    if _re.search(
        r"(?:what\s+(?:are|is)\s+(?:.*\s+)?(?:document|file|doc|doucument)s?\s+(?:present|there|presents?)\b"
        r"|document(?:s)?\s+detail"
        r"|give\s+(?:me\s+)?(?:\d+\s+)?(?:.*\s+)?(?:document|file|doc)s?\s+upload"
        r"|(?:name|list|show)\s+(?:any|these|those|the|all|\d+)\s+.*(?:document|file|doc)s?"
        r"|what\s+(?:are|is)\s+(?:.*\s+)?(?:recent|latest|newest)\s+(?:document|file|doc)s?"
        r"|what\s+(?:are|is)\s+(?:those|these|the|my|all)\s+(?:document|file|doc)s?"
        r")",
        q,
    ):
        return RouteResult(route="METADATA", confidence=0.9, reasoning="mock")

    # Default: document content (RAG path)
    return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="mock")
