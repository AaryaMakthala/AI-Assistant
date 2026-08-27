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
# TEST A: Groq 503 -> OpenRouter success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_groq_503_openrouter_success() -> None:
    """Groq -> HTTP 503, OpenRouter -> success.

    Asserts:
    - Groq called first
    - OpenRouter called second
    - Gemini NOT called
    - returned content comes from OpenRouter
    """
    groq = _make_provider("groq", "qwen/qwen3.6-27b", base_url="https://api.groq.com/openai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")

    chain = _build_chain(groq, openrouter, gemini)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "api.groq.com" in url:
            call_log.append("groq")
            return _MockResponse(503)
        elif "openrouter" in url:
            call_log.append("openrouter")
            return _MockResponse(200, lines=[
                _sse("Kanban is a visual workflow method."),
                "data: [DONE]",
            ])
        elif "generativelanguage" in url:
            call_log.append("gemini")
            return _MockResponse(200)  # should not be reached
        return _MockResponse(500)

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        tokens = []
        async for token in chain.stream(messages, completion=completion):
            tokens.append(token)

    assert call_log == ["groq", "openrouter"], f"Expected [groq, openrouter], got {call_log}"
    assert "gemini" not in call_log, "Gemini should NOT be called"
    assert "".join(tokens) == "Kanban is a visual workflow method."
    assert completion.text == "Kanban is a visual workflow method."
    assert completion.provider == "openrouter"
    assert completion.model == "google/gemini-2.0-flash-001"


# ---------------------------------------------------------------------------
# TEST B: Groq 503 -> OpenRouter 503 -> Gemini success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_groq_503_openrouter_503_gemini_success() -> None:
    """Groq -> 503, OpenRouter -> 503, Gemini -> success.

    Asserts exact order: Groq -> OpenRouter -> Gemini.
    Asserts calls are sequential, Gemini is reached, final response from Gemini.
    """
    groq = _make_provider("groq", "qwen/qwen3.6-27b", base_url="https://api.groq.com/openai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")

    chain = _build_chain(groq, openrouter, gemini)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "api.groq.com" in url:
            call_log.append("groq")
            return _MockResponse(503)
        elif "openrouter" in url:
            call_log.append("openrouter")
            return _MockResponse(503)
        elif "generativelanguage" in url:
            call_log.append("gemini")
            return _MockResponse(200, lines=[
                _sse("Gemini says: Kanban is a workflow."),
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

    assert call_log == ["groq", "openrouter", "gemini"], f"Expected sequential order, got {call_log}"
    assert "".join(tokens) == "Gemini says: Kanban is a workflow."
    assert completion.text == "Gemini says: Kanban is a workflow."
    assert completion.provider == "gemini"
    assert completion.model == "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# TEST C: All providers fail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_providers_fail_no_key_leakage() -> None:
    """Groq -> 503, OpenRouter -> 503, Gemini -> 503.

    Asserts:
    - All three providers are attempted
    - Final error is the expected provider-unavailable error
    - No API key is present in exception text
    """
    groq = _make_provider("groq", "qwen/qwen3.6-27b", base_url="https://api.groq.com/openai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")

    chain = _build_chain(groq, openrouter, gemini)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "api.groq.com" in url:
            call_log.append("groq")
        elif "openrouter" in url:
            call_log.append("openrouter")
        elif "generativelanguage" in url:
            call_log.append("gemini")
        return _MockResponse(503, body=json.dumps({"error": "server overloaded"}).encode())

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        with pytest.raises(LLMError) as exc_info:
            async for _token in chain.stream(messages, completion=completion):
                pass

    assert call_log == ["groq", "openrouter", "gemini"], f"Expected all three attempted, got {call_log}"
    error_text = str(exc_info.value)
    assert "503" in error_text or "failed" in error_text.lower()
    # No API key leakage
    assert "fake-groq-key" not in error_text
    assert "fake-openrouter-key" not in error_text
    assert "fake-gemini-key" not in error_text


# ---------------------------------------------------------------------------
# Context preservation test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_preserves_context() -> None:
    """When Groq fails and OpenRouter succeeds, OpenRouter receives the same context.

    Creates a request with system prompt, user query, conversation history,
    retrieved chunk text, and source metadata.  Forces Groq failure, OpenRouter
    success.  Captures the actual payload given to OpenRouter.
    """
    groq = _make_provider("groq", "qwen/qwen3.6-27b", base_url="https://api.groq.com/openai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")

    chain = _build_chain(groq, openrouter)

    captured_payloads: list[dict] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:
        if "api.groq.com" in url:
            return _MockResponse(503)
        elif "openrouter" in url:
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

    # Verify OpenRouter received the correct payload.
    assert len(captured_payloads) == 1, f"Expected 1 OpenRouter payload, got {len(captured_payloads)}"
    payload = captured_payloads[0]

    assert payload["model"] == "google/gemini-2.0-flash-001"

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
    """Groq emits meaningful content then fails -> OpenRouter is NOT called.

    Asserts:
    - Groq is called
    - OpenRouter is NOT called
    - Gemini is NOT called
    - partial output is not concatenated with another provider's output
    """
    groq = _make_provider("groq", "qwen/qwen3.6-27b", base_url="https://api.groq.com/openai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")
    gemini = _make_provider("gemini", "gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai")

    chain = _build_chain(groq, openrouter, gemini)

    call_log: list[str] = []

    class _PartialGroqResponse:
        status_code = 200

        async def aread(self) -> bytes:
            return b""

        async def __aenter__(self) -> "_PartialGroqResponse":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def aiter_lines(self) -> AsyncIterator[str]:
            yield _sse("Kanban is a")
            yield _sse(" visual workflow")
            # Simulate mid-stream failure.
            raise ConnectionError("Connection reset by peer")

    def _mock_stream(method: str, url: str, **kwargs: Any) -> Any:  # noqa: ARG001
        if "api.groq.com" in url:
            call_log.append("groq")
            return _PartialGroqResponse()
        elif "openrouter" in url:
            call_log.append("openrouter")
            return _MockResponse(200, lines=[_sse("OpenRouter answer"), "data: [DONE]"])
        elif "generativelanguage" in url:
            call_log.append("gemini")
            return _MockResponse(200)
        return _MockResponse(500)

    messages = [Message(role="user", content="What is Kanban?")]
    completion = Completion()

    mock_client = _make_mock_client(_mock_stream)
    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        with pytest.raises(LLMError) as exc_info:
            async for _token in chain.stream(messages, completion=completion):
                pass

    # Only Groq was called — no failover after partial content.
    assert call_log == ["groq"], f"Expected only Groq, got {call_log}"
    assert "openrouter" not in call_log
    assert "gemini" not in call_log
    # Partial content was captured before the failure.
    assert completion.text == "Kanban is a visual workflow"


@pytest.mark.asyncio
async def test_pre_stream_failure_allows_failover() -> None:
    """Groq fails before any content -> OpenRouter IS attempted.

    This is the counterpart to test_partial_stream_no_failover: failover
    IS allowed when no content has been emitted.
    """
    groq = _make_provider("groq", "qwen/qwen3.6-27b", base_url="https://api.groq.com/openai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")

    chain = _build_chain(groq, openrouter)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "api.groq.com" in url:
            call_log.append("groq")
            return _MockResponse(503, body=b"server overloaded")
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

    assert call_log == ["groq", "openrouter"], f"Expected [groq, openrouter], got {call_log}"
    assert "".join(tokens) == "OpenRouter says: Kanban is a workflow."
    assert completion.provider == "openrouter"


# ---------------------------------------------------------------------------
# Non-retryable error test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_retryable_error_no_failover() -> None:
    """Groq returns 401 (non-retryable) -> OpenRouter is NOT attempted."""
    groq = _make_provider("groq", "qwen/qwen3.6-27b", base_url="https://api.groq.com/openai/v1")
    openrouter = _make_provider("openrouter", "google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1")

    chain = _build_chain(groq, openrouter)

    call_log: list[str] = []

    def _mock_stream(method: str, url: str, **kwargs: Any) -> _MockResponse:  # noqa: ARG001
        if "api.groq.com" in url:
            call_log.append("groq")
            return _MockResponse(401, body=b"invalid api key")
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

    # Only Groq called — 401 is non-retryable, no failover.
    assert call_log == ["groq"], f"Expected only Groq, got {call_log}"
    assert exc_info.value.retryable is False
