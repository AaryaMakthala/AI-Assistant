"""Google Gemini provider.

Second in the default failover chain (`LLM_PROVIDER_ORDER`): reached when Groq is
unavailable, rate limited, or times out before its first token.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app.config import get_settings
from app.llm.base import Completion, LLMError, Message, TokenUsage

PROVIDER_NAME = "gemini"

# Errors that mean "try again elsewhere" rather than "this request is wrong". Matched on
# the exception's text because the SDK raises a single APIError type for both classes.
_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "resource_exhausted",
    "unavailable",
    "deadline",
    "internal",
    "overloaded",
)


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _split_system(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Separate system text from the turn list.

    Gemini takes the system prompt as its own config field rather than a message role.
    Keeping it there — instead of prepending it as a user turn — is what preserves the
    priority it is given over conversation content, and with it the injection defenses in
    the RAG prompt (CLAUDE.md 4.4).
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message.content}]})
    return ("\n\n".join(system_parts) or None), contents


class GeminiProvider:
    name = PROVIDER_NAME

    def __init__(self, *, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.llm_gemini_model
        self._api_key = settings.gemini_api_key.get_secret_value()
        self._settings = settings
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        from google.genai import types

        system_instruction, contents = _split_system(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_output_tokens,
        )

        completion.provider = self.name
        completion.model = self.model

        try:
            stream = await self._get_client().aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:
            raise LLMError(
                f"Gemini request failed: {exc}",
                provider=self.name,
                retryable=_is_retryable(exc),
            ) from exc

        try:
            async for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    completion.text += text
                    yield text
                # Usage arrives on the final chunks and is cumulative, so the last
                # non-empty reading is the whole request's total.
                usage = getattr(chunk, "usage_metadata", None)
                if usage is not None:
                    completion.usage = TokenUsage(
                        prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                        completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                    )
        except asyncio.CancelledError:
            # The client disconnected. Not a provider fault, and must not be reported as
            # one or the router would fail over to answer a request nobody is reading.
            raise
        except Exception as exc:
            raise LLMError(
                f"Gemini stream failed: {exc}",
                provider=self.name,
                retryable=_is_retryable(exc),
            ) from exc


__all__ = ["PROVIDER_NAME", "GeminiProvider"]
