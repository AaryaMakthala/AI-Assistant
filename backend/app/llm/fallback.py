"""Centralized multi-provider LLM fallback chain (CLAUDE.md sections 2, 10).

Implements a sequential failover chain: primary (Groq) → fallback (OpenRouter) →
secondary fallback (Gemini). Providers are tried strictly sequentially, never
in parallel. The fallback triggers on HTTP 429, 500, 502, 503, 504, timeout, or
connection errors — NOT on invalid requests from our own code (those surface
clearly without retrying).

Key constraints (CLAUDE.md section 10):
- No parallel model calls per request — bounded by an overall request timeout.
- If partial content has already been streamed to the client, no failover occurs.
- API keys are never logged; only provider name + status code.
- The chain degrades gracefully: if only the primary provider is configured,
  only that provider runs.

The :class:`FallbackChainProvider` satisfies the same :class:`~app.llm.base.LLMProvider`
protocol that the pipeline and chat endpoints expect, so callers cannot tell whether
they hold one model or a failover chain.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from loguru import logger

from app.config import get_settings
from app.llm.base import Completion, LLMError, Message


class _ProviderConfig:
    """One link in the fallback chain: name + connection details."""

    __slots__ = ("name", "api_key", "model", "base_url")

    def __init__(self, name: str, api_key: str, model: str, base_url: str) -> None:
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url


class FallbackChainProvider:
    """A sequential LLM failover chain that satisfies the LLMProvider protocol.

    Constructed once per request from the configured provider chain.  The chain
    is built from ``settings.fallback_chain_configs`` — only providers whose API
    key is present are included.  If no keys are configured, construction raises
    an error (the app should not start without at least one provider).

    Streaming semantics: the first provider is tried with ``stream=True``.  If it
    fails *before* any token is yielded, the next provider is attempted.  If it
    fails *after* tokens have been yielded, the error is surfaced to the caller
    (we cannot un-send tokens to the client).

    Overall timeout: the sum of individual provider timeouts is capped by the
    ``LLM_TIMEOUT_SECONDS`` setting (which applies per-provider), and the total
    time across all attempts is tracked.  If the overall budget is exceeded, the
    chain returns a final error.  This prevents three sequential timeouts from
    adding up to minutes of hung requests.
    """

    def __init__(self) -> None:
        settings = get_settings()
        chain_configs = settings.fallback_chain_configs
        if not chain_configs:
            raise LLMError(
                "No LLM provider API keys configured. "
                "Set at least one of GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY.",
                provider="none",
                retryable=False,
            )
        self._providers = [
            _ProviderConfig(
                name=cfg["name"],  # type: ignore[arg-type]
                api_key=cfg["api_key"],  # type: ignore[arg-type]
                model=cfg["model"],  # type: ignore[arg-type]
                base_url=cfg["base_url"],  # type: ignore[arg-type]
            )
            for cfg in chain_configs
        ]
        self._timeout_per_provider = settings.llm_timeout_seconds
        # Name/model of the provider that eventually succeeded (filled in during stream).
        self.name = ""
        self.model = ""

    async def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        """Stream from the first available provider, falling back on transient errors.

        Yields tokens from whichever provider succeeds.  If a provider fails
        before any token is emitted, the next provider in the chain is tried.
        If a provider fails after partial streaming, the error is surfaced
        immediately (we cannot un-send tokens).
        """
        total_start = time.monotonic()
        last_error: LLMError | None = None
        content_emitted = False  # tracks whether any token reached the caller

        for i, provider in enumerate(self._providers):
            elapsed = time.monotonic() - total_start
            remaining = self._timeout_per_provider - elapsed
            if remaining <= 0:
                logger.error(
                    "LLM fallback chain exhausted time budget after {elapsed:.1f}s "
                    "across {n} providers",
                    elapsed=elapsed,
                    n=i,
                )
                break

            # Only attempt failover if no content has been streamed yet.
            if content_emitted:
                logger.warning(
                    "Skipping fallback to {provider}: content already streamed",
                    provider=provider.name,
                )
                break

            logger.info(
                "LLM attempting provider={provider} (attempt {attempt}/{total})",
                provider=provider.name,
                attempt=i + 1,
                total=len(self._providers),
            )

            try:
                async for token in self._stream_single(
                    provider, messages, completion, remaining
                ):
                    content_emitted = True
                    yield token

                # Success — record which provider served this request.
                self.name = completion.provider or provider.name
                self.model = completion.model or provider.model
                logger.info(
                    "LLM provider={provider} status=success",
                    provider=provider.name,
                )
                return  # done

            except LLMError as exc:
                last_error = exc
                logger.warning(
                    "LLM provider={provider} status=failed error={error} retryable={retryable}",
                    provider=provider.name,
                    error=str(exc)[:200],
                    retryable=exc.retryable,
                )

                if not exc.retryable:
                    # Permanent fault (bad key, malformed request) — do not retry.
                    raise

                # On 429 (rate limit), retry the same provider once with a short
                # backoff before falling back.  A 429 typically resolves within
                # seconds, and retrying the same provider avoids the ~2s penalty
                # of switching to a fallback provider.
                if exc.retryable and hasattr(exc, "status_code") and exc.status_code == 429 and not getattr(exc, "_retried", False):
                    backoff = getattr(exc, "retry_after", 2.0)
                    logger.info(
                        "LLM retrying provider={provider} after 429 backoff={backoff:.1f}s",
                        provider=provider.name,
                        backoff=backoff,
                    )
                    exc._retried = True  # type: ignore[attr-defined]
                    await asyncio.sleep(backoff)
                    continue  # retry same provider

                if i < len(self._providers) - 1:
                    logger.info(
                        "LLM fallback provider={next} reason=previous_provider_failure",
                        next=self._providers[i + 1].name,
                    )
                continue

        # All providers failed or timed out.
        logger.error("All configured LLM providers failed or timed out")
        if last_error is not None:
            raise last_error
        raise LLMError(
            "All configured LLM providers failed or timed out.",
            provider="chain",
            retryable=True,
        )

    async def _stream_single(
        self,
        provider: _ProviderConfig,
        messages: list[Message],
        completion: Completion,
        timeout: float,
    ) -> AsyncIterator[str]:
        """Stream from a single provider, enforcing a per-provider timeout."""
        import httpx

        import json

        payload = {
            "model": provider.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": get_settings().llm_temperature,
            "max_tokens": get_settings().llm_max_output_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{provider.base_url.rstrip('/')}/chat/completions"

        completion.provider = provider.name
        completion.model = provider.model

        error_body_limit = 500

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", endpoint, json=payload, headers=headers
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        retryable = response.status_code == 429 or response.status_code >= 500
                        exc = LLMError(
                            f"Provider returned HTTP {response.status_code}: "
                            f"{body[:error_body_limit]}",
                            provider=provider.name,
                            retryable=retryable,
                        )
                        # Attach metadata for the 429 retry logic in the fallback chain.
                        exc.status_code = response.status_code  # type: ignore[attr-defined]
                        if response.status_code == 429:
                            retry_after_header = response.headers.get("retry-after")
                            try:
                                exc.retry_after = float(retry_after_header) if retry_after_header else 2.0  # type: ignore[attr-defined]
                            except (ValueError, TypeError):
                                exc.retry_after = 2.0  # type: ignore[attr-defined]
                        raise exc
                    async for line in response.aiter_lines():
                        token = self._parse_line(line, completion)
                        if token is not None:
                            yield token
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"Provider request failed: {exc}",
                provider=provider.name,
                retryable=True,
            ) from exc

    @staticmethod
    def _parse_line(line: str, completion: Completion) -> str | None:
        """Handle one SSE ``data:`` line, returning the delta text if any."""
        import json

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
            from app.llm.base import TokenUsage

            completion.usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
            )
        return None


__all__ = ["FallbackChainProvider"]
