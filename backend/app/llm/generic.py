"""The canonical single LLM provider (CLAUDE.md sections 2 and 13).

Section 13 configures exactly ONE model through environment variables —
``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_API_KEY``, ``LLM_BASE_URL`` — treated as a
generic chat-completions endpoint. This provider is that contract's implementation:

* no provider name is hardcoded, no fallback chain exists (CLAUDE.md section 10: one
  provider, one model; if the free tier rate-limits, that is a documented limitation,
  not a reason to add a second provider),
* ``LLM_BASE_URL`` is optional — unset, the provider's default (OpenAI-compatible)
  endpoint is used,
* it implements the same :class:`~app.llm.base.LLMProvider` protocol as the legacy
  Groq/Gemini providers, so everything above the LLM layer is unchanged.

``stream`` consumes the SSE response incrementally and fills the shared
:class:`~app.llm.base.Completion` the same way the legacy providers do, so token
accounting and the mid-stream-failure rules in ``app/llm/router.py`` apply unchanged.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from app.config import get_settings
from app.llm.base import (
    Completion,
    LLMError,
    Message,
    TokenUsage,
    thinking_disable_payload,
)

#: OpenAI-compatible chat-completions endpoint used when LLM_BASE_URL is unset.
_DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: How many characters of an error body are kept for the log. The body may contain
#: provider-specific detail worth diagnosing, and never contains our secrets.
_ERROR_BODY_LIMIT = 500


class GenericProvider:
    """One provider, configured entirely through Section 13 env vars.

    The LLM never gets write access to anything (CLAUDE.md section 0): this client
    only ever POSTs a chat-completions request and streams the reply back.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.name = settings.llm_provider
        self.model = settings.llm_model
        self._api_key = settings.llm_api_key.get_secret_value()
        base_url = settings.llm_base_url or _DEFAULT_BASE_URL
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._settings = settings

    async def stream(
        self,
        messages: list[Message],
        *,
        completion: Completion,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> AsyncIterator[str]:
        completion.provider = self.name
        completion.model = self.model

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._settings.llm_temperature,
            "max_tokens": max_tokens or self._settings.llm_max_output_tokens,
            "stream": True,
        }
        payload.update(thinking_disable_payload(self.name, disable=disable_thinking))
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
                async with client.stream(
                    "POST", self._endpoint, json=payload, headers=headers
                ) as response:
                    if response.status_code == 400 and disable_thinking:
                        # The provider may not accept our thinking-disable key.
                        # Retry the same provider once without it rather than
                        # failing the request — a thinking response we strip is
                        # better than no response at all.
                        body = (await response.aread()).decode("utf-8", "replace")
                        logger.warning(
                            "Provider {provider} rejected thinking-disable payload "
                            "(HTTP 400: {body}); retrying without it",
                            provider=self.name,
                            body=body[:_ERROR_BODY_LIMIT],
                        )
                        payload.pop("reasoning", None)
                        payload.pop("reasoning_effort", None)
                        payload.pop("chat_template_kwargs", None)
                        async with client.stream(
                            "POST", self._endpoint, json=payload, headers=headers
                        ) as response2:
                            if response2.status_code >= 400:
                                body2 = (await response2.aread()).decode("utf-8", "replace")
                                raise LLMError(
                                    f"Provider returned HTTP {response2.status_code}: "
                                    f"{body2[:_ERROR_BODY_LIMIT]}",
                                    provider=self.name,
                                    retryable=response2.status_code >= 500,
                                )
                            async for line in response2.aiter_lines():
                                token = self._parse_line(line, completion)
                                if token is not None:
                                    yield token
                            return
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        # 4xx is a permanent fault (bad key, malformed request) — the
                        # caller must not retry it elsewhere; 5xx is transient.
                        raise LLMError(
                            f"Provider returned HTTP {response.status_code}: "
                            f"{body[:_ERROR_BODY_LIMIT]}",
                            provider=self.name,
                            retryable=response.status_code >= 500,
                        )
                    async for line in response.aiter_lines():
                        token = self._parse_line(line, completion)
                        if token is not None:
                            yield token
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"Provider request failed: {exc}", provider=self.name, retryable=True
            ) from exc

    def _parse_line(self, line: str, completion: Completion) -> str | None:
        """Handle one SSE ``data:`` line, returning the delta text if any.

        OpenRouter and other OpenAI-compatible endpoints emit lines like
        ``data: {"choices":[{"delta":{"content":"..."}}]}`` and terminate with
        ``data: [DONE]``. Anything unrecognizable is skipped rather than fatal —
        a malformed line is noise, not permission to abort a stream that is otherwise
        delivering an answer.
        """
        if not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return None
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return None

        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                completion.text += content
                return content

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            completion.usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
            )
        return None


__all__ = ["GenericProvider"]
