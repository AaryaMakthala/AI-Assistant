"""Provider-agnostic chat interface.

Everything above this module works in terms of :class:`LLMProvider` alone, so swapping or
adding a provider never reaches into retrieval, prompting or the API layer. The unit that
crosses the boundary is a :class:`Completion` — text plus usage — because token accounting
is needed per request from Phase 4 onward (CLAUDE.md section 7, cost/quota overrun).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

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


def thinking_disable_payload(name: str | None, *, disable: bool) -> dict[str, Any]:
    """Map a generic "disable thinking" request onto provider-specific payload keys.

    Every provider in the configured rotation (Groq, OpenRouter, Gemini,
    NVIDIA NIM) serves a reasoning-capable model, but each disables reasoning
    differently.  This keeps the mapping in one place so callers (e.g. the
    Query Understanding stage) can request a fast, non-thinking response
    without knowing which provider will actually serve the call.

    Returns an empty dict when ``disable`` is False or the provider is
    unknown — callers must then tolerate a thinking response.
    """
    if not disable:
        return {}
    name_l = (name or "").lower()
    if "openrouter" in name_l:
        # OpenRouter: documented ``reasoning`` block; also disables Gemini
        # thinking on OpenRouter-hosted models.
        return {"reasoning": {"enabled": False}}
    if "groq" in name_l:
        # Groq docs (Qwen3 family): ``reasoning_effort: none`` disables
        # reasoning — the model emits no reasoning tokens.
        return {"reasoning_effort": "none"}
    if "gemini" in name_l:
        # Gemini native OpenAI-compat endpoint: ``reasoning_effort: none``
        # turns thinking off for thinking-enabled models.
        return {"reasoning_effort": "none"}
    if "nvidia" in name_l:
        # NVIDIA NIM / vLLM convention for Qwen3-style chat templates.
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


class LLMError(RuntimeError):
    """A provider call failed.

    `retryable` distinguishes a provider-level fault (timeout, rate limit, 5xx,
    auth/permission errors such as 401/403) from a fault in our own request
    (HTTP 400 malformed payload). Only the former justifies failing over to the
    next provider in the chain — a malformed request would be rejected by every
    provider, so retrying it just fails twice as slowly.
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
        self,
        messages: list[Message],
        *,
        completion: Completion,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> AsyncIterator[str]:
        """Yield response text incrementally, recording usage into `completion`.

        Parameters
        ----------
        max_tokens:
            Override the default max output tokens for this call.  When None,
            the provider's configured default is used.
        disable_thinking:
            Request a non-thinking response (reasoning disabled at the API
            level).  Used by the Query Understanding stage, which needs fast
            structured JSON output and has no use for reasoning tokens.
            Implementations translate this into the provider-specific payload
            key (``reasoning_effort`` on Groq, ``reasoning`` on OpenRouter,
            ``chat_template_kwargs`` on NVIDIA NIM).  Providers that cannot
            honor it must fall back to a normal call rather than fail.
        """
        ...


@runtime_checkable
class LLMRouterProtocol(Protocol):
    """What the RAG pipeline needs from a model source.

    Structurally identical to a single provider, which is the point: callers cannot tell
    whether they hold one model or a failover chain, and tests can substitute a scripted
    stub without touching the network.
    """

    def stream(
        self,
        messages: list[Message],
        *,
        completion: Completion,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> AsyncIterator[str]: ...


__all__ = [
    "Completion",
    "LLMError",
    "LLMProvider",
    "LLMRouterProtocol",
    "Message",
    "Role",
    "TokenUsage",
    "thinking_disable_payload",
]
