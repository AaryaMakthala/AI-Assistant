"""Test the LLM fallback chain: Groq 429 → OpenRouter succeeds.

Verifies that when the primary provider (Groq) returns a 429 rate-limit,
the chain falls back to the secondary provider (OpenRouter) and completes
the request successfully.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.base import Completion, Message
from app.llm.fallback import FallbackChainProvider


def _make_http_429_response():
    """Simulate an HTTP 429 response from the LLM provider."""
    resp = AsyncMock()
    resp.status_code = 429
    resp.aread = AsyncMock(return_value=b'{"error": "rate_limit_exceeded"}')
    resp.aiter_lines = AsyncMock(return_value=iter([]))
    return resp


def _make_success_response(tokens: list[str]):
    """Simulate a successful streaming response."""
    resp = AsyncMock()
    resp.status_code = 200

    async def _aiter_lines():
        for token in tokens:
            chunk = {
                "choices": [{"delta": {"content": token}}],
                "usage": None,
            }
            yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"

    resp.aiter_lines = _aiter_lines
    return resp


@pytest.mark.asyncio
async def test_groq_429_falls_back_to_openrouter():
    """When Groq returns 429, the chain should fall back to OpenRouter."""
    settings = MagicMock()
    settings.groq_api_key = MagicMock(get_secret_value=MagicMock(return_value="groq-key"))
    settings.openrouter_api_key = MagicMock(get_secret_value=MagicMock(return_value="or-key"))
    settings.gemini_api_key = None
    settings.llm_timeout_seconds = 30.0
    settings.llm_temperature = 0.2
    settings.llm_max_output_tokens = 4096
    settings.fallback_chain_configs = [
        {
            "name": "groq",
            "api_key": "groq-key",
            "model": "qwen/qwen3.6-27b",
            "base_url": "https://api.groq.com/openai/v1",
        },
        {
            "name": "openrouter",
            "api_key": "or-key",
            "model": "google/gemini-2.5-flash",
            "base_url": "https://openrouter.ai/api/v1",
        },
    ]

    call_log = []

    async def _mock_stream(self, provider_cfg, messages, completion, timeout):
        call_log.append(provider_cfg.name)
        if provider_cfg.name == "groq":
            # Simulate 429
            from app.llm.fallback import LLMError
            raise LLMError(
                "Provider returned HTTP 429: rate_limit_exceeded",
                provider="groq",
                retryable=True,
            )
            yield  # pragma: no cover
        else:
            # OpenRouter succeeds
            completion.text = "OpenRouter answered successfully."
            completion.provider = "openrouter"
            completion.model = "google/gemini-2.5-flash"
            yield "OpenRouter "
            yield "answered "
            yield "successfully."

    with patch("app.llm.fallback.get_settings", return_value=settings), \
         patch.object(FallbackChainProvider, "_stream_single", _mock_stream):
        provider = FallbackChainProvider()
        completion = Completion()
        messages = [Message(role="user", content="What is the leave policy?")]

        tokens = []
        async for token in provider.stream(messages, completion=completion):
            tokens.append(token)

    # Groq was attempted first
    assert call_log[0] == "groq"
    # OpenRouter was tried as fallback
    assert call_log[1] == "openrouter"
    # Only two providers attempted (no Gemini)
    assert len(call_log) == 2
    # The response is complete
    assert "".join(tokens) == "OpenRouter answered successfully."
    # Completion metadata recorded the successful provider
    assert completion.provider == "openrouter"
    assert completion.model == "google/gemini-2.5-flash"
    # The provider instance tracks the winner
    assert provider.name == "openrouter"
    assert provider.model == "google/gemini-2.5-flash"


@pytest.mark.asyncio
async def test_groq_429_with_openrouter_model_id():
    """Confirm the OpenRouter fallback model ID is google/gemini-2.5-flash."""
    settings = MagicMock()
    settings.groq_api_key = MagicMock(get_secret_value=MagicMock(return_value="groq-key"))
    settings.openrouter_api_key = MagicMock(get_secret_value=MagicMock(return_value="or-key"))
    settings.gemini_api_key = None
    settings.openrouter_model = None  # use default preset
    settings.llm_timeout_seconds = 30.0
    settings.llm_temperature = 0.2
    settings.llm_max_output_tokens = 4096
    settings.fallback_chain_configs = [
        {
            "name": "groq",
            "api_key": "groq-key",
            "model": "qwen/qwen3.6-27b",
            "base_url": "https://api.groq.com/openai/v1",
        },
        {
            "name": "openrouter",
            "api_key": "or-key",
            "model": "google/gemini-2.5-flash",
            "base_url": "https://openrouter.ai/api/v1",
        },
    ]

    with patch("app.llm.fallback.get_settings", return_value=settings):
        provider = FallbackChainProvider()

    # Verify the OpenRouter config uses the correct model
    or_config = provider._providers[1]
    assert or_config.name == "openrouter"
    assert or_config.model == "google/gemini-2.5-flash"
    assert or_config.base_url == "https://openrouter.ai/api/v1"
