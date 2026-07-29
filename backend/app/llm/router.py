"""Provider failover.

The rule this module exists to enforce: **failover is only possible before the first
token reaches the client.** Once a partial answer has been streamed, switching providers
would splice two different answers together, producing text that reads fluently and is
wrong in the middle. So a mid-stream failure is surfaced as an error rather than
papered over.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from loguru import logger

from app.config import get_settings
from app.llm.base import Completion, LLMError, LLMProvider, Message


class LLMRouter:
    """Tries each provider in order until one starts producing tokens."""

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        if not providers:
            raise ValueError("LLMRouter needs at least one provider.")
        self._providers = list(providers)

    async def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        timeout = get_settings().llm_timeout_seconds
        last_error: LLMError | None = None

        for index, provider in enumerate(self._providers):
            started = False
            try:
                async with asyncio.timeout(timeout):
                    async for token in provider.stream(messages, completion=completion):
                        started = True
                        yield token
                return
            except asyncio.CancelledError:
                raise
            except (LLMError, TimeoutError) as exc:
                if started:
                    # Past the point of no return: the client already holds a partial
                    # answer, and no second provider can continue someone else's sentence.
                    logger.error(
                        "Provider {provider} failed mid-stream after {n} chars; "
                        "not failing over",
                        provider=provider.name,
                        n=len(completion.text),
                    )
                    raise

                retryable = getattr(exc, "retryable", True)
                last_error = (
                    exc
                    if isinstance(exc, LLMError)
                    else LLMError(
                        f"{provider.name} timed out after {timeout}s", provider=provider.name
                    )
                )
                is_last = index == len(self._providers) - 1
                if not retryable or is_last:
                    raise last_error from exc

                logger.warning(
                    "Provider {provider} unavailable ({error}); falling back",
                    provider=provider.name,
                    error=str(exc),
                )
                # Nothing was emitted, so anything the failed attempt recorded is noise.
                completion.text = ""

        raise last_error or LLMError("No provider produced a response.", provider="router")


def build_default_router() -> LLMRouter:
    """Gemini primary, Groq fallback (CLAUDE.md section 2)."""
    from app.llm.gemini import GeminiProvider
    from app.llm.groq_provider import GroqProvider

    return LLMRouter([GeminiProvider(), GroqProvider()])


__all__ = ["LLMRouter", "build_default_router"]
