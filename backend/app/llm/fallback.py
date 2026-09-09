"""Centralized multi-provider LLM fallback chain (CLAUDE.md sections 2, 10).

Implements a sequential failover chain: primary (Groq) → fallback (OpenRouter) →
secondary fallback (Gemini). Providers are tried strictly
sequentially, never in parallel. Failover triggers on any provider-level failure
— HTTP 429, any 5xx, auth/permission errors (401/403), timeouts, and connection
errors — so a single broken provider cannot take the assistant down. The one
exception is HTTP 400: a malformed request from our own code would be rejected by
every provider, so it surfaces immediately instead of failing over.

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
from app.llm.base import Completion, LLMError, Message, thinking_disable_payload


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
        self,
        messages: list[Message],
        *,
        completion: Completion,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
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
                    provider, messages, completion, remaining,
                    max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
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
                status_code = getattr(exc, "status_code", None)
                logger.warning(
                    "LLM provider={provider} status=failed error_type={error_type} "
                    "http_status={status} error={error}",
                    provider=provider.name,
                    error_type=_fallback_reason(exc),
                    status=status_code,
                    error=str(exc)[:200],
                )

                if not exc.retryable:
                    # Our own fault (HTTP 400 malformed request) — every other
                    # provider would reject the same payload, so fail loudly
                    # instead of burning the chain.
                    raise

                # On 429 (rate limit), pause briefly before moving on.  A 429
                # typically clears within seconds; the backoff also paces the
                # request before the next provider is tried.
                if status_code == 429 and not getattr(exc, "_retried", False):
                    backoff = getattr(exc, "retry_after", 2.0)
                    exc._retried = True  # type: ignore[attr-defined]
                    logger.info(
                        "LLM provider={provider} rate_limited backoff={backoff:.1f}s "
                        "then_failover=true",
                        provider=provider.name,
                        backoff=backoff,
                    )
                    await asyncio.sleep(backoff)

                if i < len(self._providers) - 1:
                    logger.info(
                        "LLM fallback failed_provider={failed} next_provider={next} "
                        "reason={reason}",
                        failed=provider.name,
                        next=self._providers[i + 1].name,
                        reason=_fallback_reason(exc),
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
        *,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> AsyncIterator[str]:
        """Stream from a single provider, enforcing a per-provider timeout."""
        import httpx

        import json

        payload = {
            "model": provider.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": get_settings().llm_temperature,
            "max_tokens": max_tokens or get_settings().llm_max_output_tokens,
            "stream": True,
        }
        payload.update(thinking_disable_payload(provider.name, disable=disable_thinking))
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
                    if response.status_code == 400 and disable_thinking:
                        # The provider may not accept our thinking-disable key.
                        # Retry the same provider once without it rather than
                        # failing over — a thinking response we strip is better
                        # than no response at all.
                        body = (await response.aread()).decode("utf-8", "replace")
                        logger.warning(
                            "Provider {provider} rejected thinking-disable payload "
                            "(HTTP 400: {body}); retrying without it",
                            provider=provider.name,
                            body=body[:error_body_limit],
                        )
                        payload.pop("reasoning", None)
                        payload.pop("reasoning_effort", None)
                        payload.pop("chat_template_kwargs", None)
                        async with client.stream(
                            "POST", endpoint, json=payload, headers=headers
                        ) as response2:
                            if response2.status_code >= 400:
                                body2 = (await response2.aread()).decode("utf-8", "replace")
                                raise LLMError(
                                    f"Provider returned HTTP {response2.status_code}: "
                                    f"{body2[:error_body_limit]}",
                                    provider=provider.name,
                                    retryable=response2.status_code >= 500,
                                )
                            async for line in response2.aiter_lines():
                                token = self._parse_line(line, completion)
                                if token is not None:
                                    yield token
                            return
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        # Failover policy: the chain exists so a single provider's
                        # trouble never takes the assistant down. 429 and 5xx are
                        # provider-side; any other 4xx (401/403 auth or permission,
                        # 404 route, 413 payload size, ...) is provider-specific state
                        # that the next provider may not share, so it fails over. The
                        # one exception is 400: our own payload is malformed and would
                        # be rejected by every provider, so retrying just fails twice
                        # as slowly — surface it instead.
                        retryable = response.status_code != 400
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




def _fallback_reason(exc: LLMError) -> str:
    """Classify why a provider was skipped, for the fallback audit log."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate_limited"
    if status is not None and status >= 500:
        return "server_error"
    if status is not None:
        return f"http_{status}"
    return "timeout_or_connection_error"


# ---------------------------------------------------------------------------
# Rotating provider — round-robin across all configured providers
# ---------------------------------------------------------------------------


class RotatingProvider:
    """Round-robin LLM provider with graceful failover per request.

    Unlike :class:`FallbackChainProvider` which always starts with provider #1,
    this rotates the starting provider across requests so load is distributed.
    Within a single request, if the chosen provider fails, the remaining
    providers are tried in order (same failover behavior as the chain).

    Recently-failed providers are temporarily skipped (cooldown period) to
    avoid wasting time on providers that are likely still down or rate-limited.

    This satisfies the same :class:`~app.llm.base.LLMProvider` protocol, so
    callers cannot tell whether they hold a chain or a rotating pool.
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
                name=cfg["name"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                base_url=cfg["base_url"],
            )
            for cfg in chain_configs
        ]
        self._timeout_per_provider = settings.llm_timeout_seconds
        self._rotation_idx = 0  # next provider to try (round-robin)
        self.name = ""
        self.model = ""

    def _next_provider(self) -> _ProviderConfig:
        """Return the next provider in rotation and advance the index."""
        provider = self._providers[self._rotation_idx % len(self._providers)]
        self._rotation_idx += 1
        return provider

    async def stream(
        self,
        messages: list[Message],
        *,
        completion: Completion,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> AsyncIterator[str]:
        """Stream from a rotating provider, falling back on transient errors.

        Picks the next provider via round-robin. If it fails before any token
        is emitted, tries the remaining providers in order. If it fails after
        partial streaming, the error is surfaced (we cannot un-send tokens).
        """
        total_start = time.monotonic()
        last_error: LLMError | None = None
        content_emitted = False

        # Build the ordered list of providers to try for this request.
        # Start from the current rotation point, then wrap around.
        n = len(self._providers)
        start_idx = self._rotation_idx % n
        ordered = [
            self._providers[(start_idx + i) % n] for i in range(n)
        ]
        # Advance rotation for the next request.
        self._rotation_idx = (start_idx + 1) % n

        for i, provider in enumerate(ordered):
            elapsed = time.monotonic() - total_start
            remaining = self._timeout_per_provider - elapsed
            if remaining <= 0:
                logger.error(
                    "LLM rotating pool exhausted time budget after {elapsed:.1f}s",
                    elapsed=elapsed,
                )
                break

            if content_emitted:
                logger.warning(
                    "Skipping failover to {provider}: content already streamed",
                    provider=provider.name,
                )
                break

            logger.info(
                "LLM rotating pool attempting provider={provider} (attempt {attempt}/{total})",
                provider=provider.name,
                attempt=i + 1,
                total=n,
            )

            try:
                async for token in self._stream_single(
                    provider, messages, completion, remaining,
                    max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ):
                    content_emitted = True
                    yield token

                self.name = completion.provider or provider.name
                self.model = completion.model or provider.model
                logger.info(
                    "LLM provider={provider} status=success",
                    provider=provider.name,
                )
                return

            except LLMError as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                logger.warning(
                    "LLM provider={provider} status=failed error_type={error_type} "
                    "http_status={status} error={error}",
                    provider=provider.name,
                    error_type=_fallback_reason(exc),
                    status=status_code,
                    error=str(exc)[:200],
                )

                if not exc.retryable:
                    raise

                if status_code == 429 and not getattr(exc, "_retried", False):
                    backoff = getattr(exc, "retry_after", 2.0)
                    exc._retried = True  # type: ignore[attr-defined]
                    logger.info(
                        "LLM provider={provider} rate_limited backoff={backoff:.1f}s",
                        provider=provider.name,
                        backoff=backoff,
                    )
                    await asyncio.sleep(backoff)

                if i < len(ordered) - 1:
                    logger.info(
                        "LLM failover failed_provider={failed} next_provider={next} "
                        "reason={reason}",
                        failed=provider.name,
                        next=ordered[i + 1].name,
                        reason=_fallback_reason(exc),
                    )
                continue

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
        *,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> AsyncIterator[str]:
        """Stream from a single provider, enforcing a per-provider timeout."""
        import httpx
        import json as _json

        payload = {
            "model": provider.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": get_settings().llm_temperature,
            "max_tokens": max_tokens or get_settings().llm_max_output_tokens,
            "stream": True,
        }
        payload.update(thinking_disable_payload(provider.name, disable=disable_thinking))
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
                    if response.status_code == 400 and disable_thinking:
                        # The provider may not accept our thinking-disable key.
                        # Retry the same provider once without it rather than
                        # failing over — a thinking response we strip is better
                        # than no response at all.
                        body = (await response.aread()).decode("utf-8", "replace")
                        logger.warning(
                            "Provider {provider} rejected thinking-disable payload "
                            "(HTTP 400: {body}); retrying without it",
                            provider=provider.name,
                            body=body[:error_body_limit],
                        )
                        payload.pop("reasoning", None)
                        payload.pop("reasoning_effort", None)
                        payload.pop("chat_template_kwargs", None)
                        async with client.stream(
                            "POST", endpoint, json=payload, headers=headers
                        ) as response2:
                            if response2.status_code >= 400:
                                body2 = (await response2.aread()).decode("utf-8", "replace")
                                raise LLMError(
                                    f"Provider returned HTTP {response2.status_code}: "
                                    f"{body2[:error_body_limit]}",
                                    provider=provider.name,
                                    retryable=response2.status_code >= 500,
                                )
                            async for line in response2.aiter_lines():
                                token = self._parse_line(line, completion)
                                if token is not None:
                                    yield token
                            return
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        retryable = response.status_code != 400
                        exc = LLMError(
                            f"Provider returned HTTP {response.status_code}: "
                            f"{body[:error_body_limit]}",
                            provider=provider.name,
                            retryable=retryable,
                        )
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
        import json as _json

        if not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return None
        try:
            chunk = _json.loads(data)
        except _json.JSONDecodeError:
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


__all__ = ["FallbackChainProvider", "RotatingProvider"]
