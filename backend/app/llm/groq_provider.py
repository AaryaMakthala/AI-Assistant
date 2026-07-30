"""Groq provider.

First in the default failover chain (`LLM_PROVIDER_ORDER`), with Gemini behind it. The
chain itself is wired from Phase 4 rather than added under pressure later: a
single-provider dependency is what turns a free-tier rate limit into an outage
(CLAUDE.md section 7).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app.config import get_settings
from app.llm.base import Completion, LLMError, Message, TokenUsage

PROVIDER_NAME = "groq"

_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate limit",
    "timeout",
    "unavailable",
    "overloaded",
    "connection",
)


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


class GroqProvider:
    name = PROVIDER_NAME

    def __init__(self, *, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.llm_groq_model
        self._api_key = settings.groq_api_key.get_secret_value()
        self._settings = settings
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(
                api_key=self._api_key, timeout=self._settings.llm_timeout_seconds
            )
        return self._client

    async def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        completion.provider = self.name
        completion.model = self.model

        try:
            stream = await self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_output_tokens,
                stream=True,
            )
        except Exception as exc:
            raise LLMError(
                f"Groq request failed: {exc}",
                provider=self.name,
                retryable=_is_retryable(exc),
            ) from exc

        try:
            async for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if choices:
                    content = getattr(choices[0].delta, "content", None)
                    if content:
                        completion.text += content
                        yield content
                # Groq reports usage in an x_groq envelope on the terminal chunk.
                usage = getattr(getattr(chunk, "x_groq", None), "usage", None) or getattr(
                    chunk, "usage", None
                )
                if usage is not None:
                    completion.usage = TokenUsage(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise LLMError(
                f"Groq stream failed: {exc}",
                provider=self.name,
                retryable=_is_retryable(exc),
            ) from exc


__all__ = ["PROVIDER_NAME", "GroqProvider"]
