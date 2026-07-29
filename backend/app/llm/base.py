"""Provider-agnostic chat interface.

Everything above this module works in terms of :class:`LLMProvider` alone, so swapping or
adding a provider never reaches into retrieval, prompting or the API layer. The unit that
crosses the boundary is a :class:`Completion` — text plus usage — because token accounting
is needed per request from Phase 4 onward (CLAUDE.md section 7, cost/quota overrun).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Completion:
    """The result of a generation, filled in as the stream progresses."""

    text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    provider: str = ""


class LLMError(RuntimeError):
    """A provider call failed.

    `retryable` distinguishes a transient fault (timeout, rate limit, 5xx) from a
    permanent one (bad key, malformed request). Only the former justifies failing over —
    retrying a malformed request against a second provider just fails twice as slowly.
    """

    def __init__(self, message: str, *, provider: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


@runtime_checkable
class LLMProvider(Protocol):
    """A streaming chat model.

    Implementations must not raise after yielding their first chunk if it can be avoided:
    the router can only fail over while no token has reached the client.
    """

    name: str
    model: str

    def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        """Yield response text incrementally, recording usage into `completion`."""
        ...


@runtime_checkable
class LLMRouterProtocol(Protocol):
    """What the RAG pipeline needs from a model source.

    Structurally identical to a single provider, which is the point: callers cannot tell
    whether they hold one model or a failover chain, and tests can substitute a scripted
    stub without touching the network.
    """

    def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        ...


__all__ = [
    "Completion",
    "LLMError",
    "LLMProvider",
    "LLMRouterProtocol",
    "Message",
    "Role",
    "TokenUsage",
]
