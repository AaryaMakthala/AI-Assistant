"""Provider failover.

The rule this module exists to enforce: **failover is only possible before the first
token reaches the client.** Once a partial answer has been streamed, switching providers
would splice two different answers together, producing text that reads fluently and is
wrong in the middle. So a mid-stream failure is surfaced as an error rather than
papered over.

Which providers are tried, and in what order, is configuration (`LLM_PROVIDER_ORDER`) —
no provider name is hardcoded into the chain.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence

from loguru import logger

from app.config import get_settings
from app.llm.base import Completion, LLMError, LLMProvider, Message


def _build_groq() -> LLMProvider:
    from app.llm.groq_provider import GroqProvider

    return GroqProvider()


def _build_gemini() -> LLMProvider:
    from app.llm.gemini import GeminiProvider

    return GeminiProvider()


#: Name → constructor. Imports are deferred into the factories so that building a chain
#: of one provider never imports the other's SDK.
_PROVIDER_FACTORIES: dict[str, Callable[[], LLMProvider]] = {
    "groq": _build_groq,
    "gemini": _build_gemini,
}


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
                        "Provider {provider} failed mid-stream after {n} chars; not failing over",
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
    """Build the failover chain from `LLM_PROVIDER_ORDER`.

    Defaults to `groq,gemini` — Groq first, Gemini as the automatic fallback. The order is
    configuration rather than code so that a provider outage or a quota change is an env
    edit, not a deploy. Providers are constructed lazily by name from `_PROVIDER_FACTORIES`,
    so an unrecognised entry is reported with the set of valid names instead of silently
    dropping a provider out of the chain.
    """
    names = get_settings().llm_providers
    if not names:
        raise ValueError("LLM_PROVIDER_ORDER is empty; name at least one provider.")

    providers: list[LLMProvider] = []
    for name in names:
        factory = _PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown LLM provider {name!r} in LLM_PROVIDER_ORDER. "
                f"Known providers: {', '.join(sorted(_PROVIDER_FACTORIES))}."
            )
        providers.append(factory())

    logger.info("LLM failover order: {order}", order=" → ".join(names))
    return LLMRouter(providers)


__all__ = ["LLMRouter", "build_default_router"]
