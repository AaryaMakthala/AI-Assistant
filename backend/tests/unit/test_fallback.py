"""FallbackChainProvider tests with mocked HTTP calls.

All tests mock httpx.AsyncClient to simulate provider failures and successes
without calling any real API.  Tests verify:
- Exact provider call order
- Context preservation across failover
- Partial stream behavior (no failover after content emitted)
- All-provider failure with no key leakage
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.llm.base import Completion, LLMError, Message, TokenUsage
from app.llm.fallback import FallbackChainProvider, _ProviderConfig

pytestmark = pytest.mark.usefixtures("valid_env")


@pytest.fixture
def _valid_env() -> None:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(name: str, model: str, base_url: str = "https://fake.api") -> _ProviderConfig:
    return _ProviderConfig(name=name, api_key=f"fake-{name}-key", model=model, base_url=base_url)


def _sse(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})


class _MockResponse:
    """Simulates an httpx streaming response used as ``async with client.stream()``."""

    def __init__(self, status_code: int, lines: list[str] | None = None, body: bytes = b"error") -> None:
        self.status_code = status_code
        self._lines = lines or []
        self._body = body

    async def aread(self) -> bytes:
        return self._body

    async def __aenter__(self) -> "_MockResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


def _build_chain(*providers: _ProviderConfig) -> FallbackChainProvider:
    """Build a FallbackChainProvider with pre-configured providers (no env vars needed)."""
    chain = FallbackChainProvider.__new__(FallbackChainProvider)
    chain._providers = list(providers)
    chain._timeout_per_provider = 60.0
    chain.name = ""
    chain.model = ""
    return chain


def _make_mock_client(side_effect: Any) -> MagicMock:
    """Build a mock httpx.AsyncClient that supports ``async with`` and ``client.stream()``."""
    mock_client = MagicMock()
    mock_client.stream = side_effect
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# ---------------------------------------------------------------------------
# TEST A: Gemini 503 -> Grok success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_503_grok_success() -> None:
    """Gemini -> HTTP 503, Grok -> success.

    Asserts:
    - Gemini called first
    - Grok called second
    - OpenRouter NOT called
    - returned content comes from Grok
    """
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    grok = _make_provider("grok", "grok-3-mini", base_url="https://api.x.ai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")

    chain = _build_chain(gemini, grok, openrouter)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "generativelanguage" in url:
            call_log.append("gemini")
            return _MockResponse(503)
        elif "x.ai" in url:
            call_log.append("grok")
            return _MockResponse(200, lines=[
                _sse("Kanban is a visual workflow method."),
                "data: [DONE]",
            ])
        elif "openrouter" in url:
            call_log.append("openrouter")
            return _MockResponse(200)  # should not be reached
        return _MockResponse(500)

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        tokens = []
        async for token in chain.stream(messages, completion=completion):
            tokens.append(token)

    assert call_log == ["gemini", "grok"], f"Expected [gemini, grok], got {call_log}"
    assert "openrouter" not in call_log, "OpenRouter should NOT be called"
    assert "".join(tokens) == "Kanban is a visual workflow method."
    assert completion.text == "Kanban is a visual workflow method."
    assert completion.provider == "grok"
    assert completion.model == "grok-3-mini"


# ---------------------------------------------------------------------------
# TEST B: Gemini 503 -> Grok 503 -> OpenRouter success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_503_grok_503_openrouter_success() -> None:
    """Gemini -> 503, Grok -> 503, OpenRouter -> success.

    Asserts exact order: Gemini -> Grok -> OpenRouter.
    Asserts calls are sequential, OpenRouter is reached, final response from OpenRouter.
    """
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    grok = _make_provider("grok", "grok-3-mini", base_url="https://api.x.ai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")

    chain = _build_chain(gemini, grok, openrouter)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "generativelanguage" in url:
            call_log.append("gemini")
            return _MockResponse(503)
        elif "x.ai" in url:
            call_log.append("grok")
            return _MockResponse(503)
        elif "openrouter" in url:
            call_log.append("openrouter")
            return _MockResponse(200, lines=[
                _sse("OpenRouter says: Kanban is a workflow."),
                "data: [DONE]",
            ])
        return _MockResponse(500)

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        tokens = []
        async for token in chain.stream(messages, completion=completion):
            tokens.append(token)

    assert call_log == ["gemini", "grok", "openrouter"], f"Expected sequential order, got {call_log}"
    assert "".join(tokens) == "OpenRouter says: Kanban is a workflow."
    assert completion.text == "OpenRouter says: Kanban is a workflow."
    assert completion.provider == "openrouter"
    assert completion.model == "google/gemini-2.0-flash-001"


# ---------------------------------------------------------------------------
# TEST C: All providers fail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_providers_fail_no_key_leakage() -> None:
    """Gemini -> 503, Grok -> 503, OpenRouter -> 503.

    Asserts:
    - All three providers are attempted
    - Final error is the expected provider-unavailable error
    - No API key is present in exception text
    """
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    grok = _make_provider("grok", "grok-3-mini", base_url="https://api.x.ai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")

    chain = _build_chain(gemini, grok, openrouter)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "generativelanguage" in url:
            call_log.append("gemini")
        elif "x.ai" in url:
            call_log.append("grok")
        elif "openrouter" in url:
            call_log.append("openrouter")
        return _MockResponse(503, body=json.dumps({"error": "server overloaded"}).encode())

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        with pytest.raises(LLMError) as exc_info:
            async for _token in chain.stream(messages, completion=completion):
                pass

    assert call_log == ["gemini", "grok", "openrouter"], f"Expected all three attempted, got {call_log}"
    error_text = str(exc_info.value)
    assert "503" in error_text or "failed" in error_text.lower()
    # No API key leakage
    assert "fake-gemini-key" not in error_text
    assert "fake-grok-key" not in error_text
    assert "fake-openrouter-key" not in error_text


# ---------------------------------------------------------------------------
# Context preservation test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_preserves_context() -> None:
    """When Gemini fails and Grok succeeds, Grok receives the same context.

    Creates a request with system prompt, user query, conversation history,
    retrieved chunk text, and source metadata.  Forces Gemini failure, Grok
    success.  Captures the actual payload given to Grok.
    """
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    grok = _make_provider("grok", "grok-3-mini", base_url="https://api.x.ai/v1")

    chain = _build_chain(gemini, grok)

    captured_payloads: list[dict] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:
        if "generativelanguage" in url:
            return _MockResponse(503)
        elif "x.ai" in url:
            captured_payloads.append(kwargs.get("json", {}))
            return _MockResponse(200, lines=[
                _sse("The policy allows 20 days."),
                "data: [DONE]",
            ])
        return _MockResponse(500)

    # Build messages with system prompt, history, and user query with context.
    system_prompt = (
        "You are an enterprise knowledge assistant. Answer only from the supplied material."
    )
    history = [
        Message(role="user", content="What is the vacation policy?"),
        Message(role="assistant", content="The policy allows 20 days per year."),
    ]
    context_block = (
        "BEGIN_UNTRUSTED_DOCUMENT_CONTEXT-abc123\n"
        "[1] source: handbook.pdf · page 2\n"
        "Vacation accrues at 20 days per year.\n"
        "END_UNTRUSTED_DOCUMENT_CONTEXT-abc123"
    )
    user_query = (
        f"{context_block}\n\n"
        "Using only the quoted material above, answer this question.\n\n"
        "Question: Can I carry over unused days?"
    )

    messages = [
        Message(role="system", content=system_prompt),
        *history,
        Message(role="user", content=user_query),
    ]

    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        tokens = []
        async for token in chain.stream(messages, completion=completion):
            tokens.append(token)

    # Verify Grok received the correct payload.
    assert len(captured_payloads) == 1, f"Expected 1 Grok payload, got {len(captured_payloads)}"
    payload = captured_payloads[0]

    assert payload["model"] == "grok-3-mini"

    sent_messages = payload["messages"]
    assert len(sent_messages) == 4  # system + 2 history + user

    # System prompt preserved.
    assert sent_messages[0]["role"] == "system"
    assert "enterprise knowledge assistant" in sent_messages[0]["content"]

    # History preserved.
    assert sent_messages[1]["role"] == "user"
    assert "vacation policy" in sent_messages[1]["content"]
    assert sent_messages[2]["role"] == "assistant"
    assert "20 days" in sent_messages[2]["content"]

    # User query with context preserved.
    assert sent_messages[3]["role"] == "user"
    assert "BEGIN_UNTRUSTED_DOCUMENT_CONTEXT" in sent_messages[3]["content"]
    assert "handbook.pdf" in sent_messages[3]["content"]
    assert "Can I carry over unused days?" in sent_messages[3]["content"]


# ---------------------------------------------------------------------------
# Partial stream behavior tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_stream_no_failover() -> None:
    """Gemini emits meaningful content then fails -> Grok is NOT called.

    Asserts:
    - Gemini is called
    - Grok is NOT called
    - OpenRouter is NOT called
    - partial output is not concatenated with another provider's output
    """
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    grok = _make_provider("grok", "grok-3-mini", base_url="https://api.x.ai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")

    chain = _build_chain(gemini, grok, openrouter)

    call_log: list[str] = []

    class _PartialGeminiResponse:
        status_code = 200

        async def aread(self) -> bytes:
            return b""

        async def __aenter__(self) -> "_PartialGeminiResponse":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def aiter_lines(self) -> AsyncIterator[str]:
            yield _sse("Kanban is a")
            yield _sse(" visual workflow")
            # Simulate mid-stream failure.
            raise ConnectionError("Connection reset by peer")

    def _mock_stream(method: str, url: str, **kwargs: Any) -> Any:  # noqa: ARG001
        if "generativelanguage" in url:
            call_log.append("gemini")
            return _PartialGeminiResponse()
        elif "x.ai" in url:
            call_log.append("grok")
            return _MockResponse(200, lines=[_sse("Grok's answer"), "data: [DONE]"])
        elif "openrouter" in url:
            call_log.append("openrouter")
            return _MockResponse(200)
        return _MockResponse(500)

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        with pytest.raises(LLMError) as exc_info:
            async for _token in chain.stream(messages, completion=completion):
                pass

    # Only Gemini was called — no failover after partial content.
    assert call_log == ["gemini"], f"Expected only Gemini, got {call_log}"
    assert "grok" not in call_log
    assert "openrouter" not in call_log
    # Partial content was captured before the failure.
    assert completion.text == "Kanban is a visual workflow"


@pytest.mark.asyncio
async def test_pre_stream_failure_allows_failover() -> None:
    """Gemini fails before any content -> Grok IS attempted.

    This is the counterpart to test_partial_stream_no_failover: failover
    IS allowed when no content has been emitted.
    """
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    grok = _make_provider("grok", "grok-3-mini", base_url="https://api.x.ai/v1")

    chain = _build_chain(gemini, grok)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "generativelanguage" in url:
            call_log.append("gemini")
            return _MockResponse(503, body=b"server overloaded")
        elif "x.ai" in url:
            call_log.append("grok")
            return _MockResponse(200, lines=[
                _sse("Grok says: Kanban is a workflow."),
                "data: [DONE]",
            ])
        return _MockResponse(500)

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        tokens = []
        async for token in chain.stream(messages, completion=completion):
            tokens.append(token)

    assert call_log == ["gemini", "grok"], f"Expected [gemini, grok], got {call_log}"
    assert "".join(tokens) == "Grok says: Kanban is a workflow."
    assert completion.provider == "grok"


# ---------------------------------------------------------------------------
# Non-retryable error test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_retryable_error_no_failover() -> None:
    """Gemini returns 401 (non-retryable) -> Grok is NOT attempted."""
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    grok = _make_provider("grok", "grok-3-mini", base_url="https://api.x.ai/v1")

    chain = _build_chain(gemini, grok)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "generativelanguage" in url:
            call_log.append("gemini")
            return _MockResponse(401, body=b"invalid api key")
        elif "x.ai" in url:
            call_log.append("grok")
            return _MockResponse(200)
        return _MockResponse(500)

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        with pytest.raises(LLMError) as exc_info:
            async for _token in chain.stream(messages, completion=completion):
                pass

    # Only Gemini called — 401 is non-retryable, no failover.
    assert call_log == ["gemini"], f"Expected only Gemini, got {call_log}"
    assert exc_info.value.retryable is False
