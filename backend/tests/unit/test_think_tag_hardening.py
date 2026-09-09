"""Unit tests for think-block hardening (QU truncation fix).

Covers:
- ``detect_unclosed_think_block``: distinguishes a max-tokens cutoff mid-reasoning
  from a complete response and from a genuinely empty response.
- ``thinking_disable_payload``: maps the generic disable-thinking request onto the
  provider-specific payload key for each configured provider.
"""

from __future__ import annotations

from app.llm.base import thinking_disable_payload
from app.llm.utils import detect_unclosed_think_block, strip_think_tags


class TestDetectUnclosedThinkBlock:
    def test_unclosed_think_block_detected(self) -> None:
        text = "\n thinking\nHere's a thinking process:\n1. Analyze user input..."
        assert detect_unclosed_think_block(text) is True

    def test_closed_think_block_not_detected(self) -> None:
        # The closer is the bare-word line "thinks" (Qwen3's actual closer), not
        # the word "response".  A complete tagged response has an opener line,
        # reasoning, a closer line, then the answer.
        text = "\n thinking\nreasoning here\n\nthinks\n{\"intent\": \"greeting\"}"
        assert detect_unclosed_think_block(text) is False

    def test_empty_text_not_detected(self) -> None:
        assert detect_unclosed_think_block("") is False
        assert detect_unclosed_think_block("just a plain answer") is False

    def test_strip_then_detect_agree(self) -> None:
        """strip_think_tags returning empty + unclosed block = truncation, not emptiness."""
        text = "\n thinking\nreasoning cut off by max_tokens"
        assert strip_think_tags(text) == ""
        assert detect_unclosed_think_block(text) is True


class TestThinkingDisablePayload:
    def test_disabled_returns_empty(self) -> None:
        assert thinking_disable_payload("groq", disable=False) == {}

    def test_groq_uses_reasoning_effort(self) -> None:
        assert thinking_disable_payload("groq", disable=True) == {
            "reasoning_effort": "none"
        }

    def test_openrouter_uses_reasoning_block(self) -> None:
        assert thinking_disable_payload("openrouter", disable=True) == {
            "reasoning": {"enabled": False}
        }

    def test_gemini_uses_reasoning_effort(self) -> None:
        assert thinking_disable_payload("gemini", disable=True) == {
            "reasoning_effort": "none"
        }

    def test_nvidia_uses_chat_template_kwargs(self) -> None:
        assert thinking_disable_payload("nvidia", disable=True) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_unknown_provider_returns_empty(self) -> None:
        assert thinking_disable_payload("mystery-cloud", disable=True) == {}
        assert thinking_disable_payload(None, disable=True) == {}


class TestStripThinkTags:
    """Test strip_think_tags for correct think-tag removal (ISSUE 1)."""

    def test_inline_think_tags(self) -> None:
        """<think>reasoning</think>Final answer → Final answer"""
        result = strip_think_tags("<think>reasoning</think>Final answer")
        assert result == "Final answer"

    def test_multiline_think_tags(self) -> None:
        """<think>\\nreasoning\\n</think>\\nFinal answer → Final answer"""
        result = strip_think_tags("<think>\nreasoning\n</think>\nFinal answer")
        assert result == "Final answer"

    def test_no_think_tags_unchanged(self) -> None:
        """Response with no think tags → unchanged."""
        result = strip_think_tags("Just a normal answer")
        assert result == "Just a normal answer"

    def test_think_only_response_becomes_empty(self) -> None:
        """Think-only response → empty string after cleanup."""
        result = strip_think_tags("<think>\nreasoning\n</think>")
        assert result.strip() == ""

    def test_think_only_inline_becomes_empty(self) -> None:
        """Think-only inline → empty string after cleanup."""
        result = strip_think_tags("<think>reasoning</think>")
        assert result.strip() == ""

    def test_whitespace_around_tags(self) -> None:
        """Whitespace around tags preserved for content after closer."""
        result = strip_think_tags("<think> reasoning </think>  Final answer  ")
        assert result == "Final answer"

    def test_unclosed_think_drops_everything(self) -> None:
        """Unclosed think block drops all content (max-tokens cutoff)."""
        result = strip_think_tags("<think>reasoning cut off")
        assert result == ""

    def test_pipe_style_tags(self) -> None:
        """<|start_of_thought|>...<|end_of_thought|> also stripped."""
        result = strip_think_tags(
            "<|start_of_thought|>reasoning<|end_of_thought|>The answer"
        )
        assert result == "The answer"

    def test_closing_thinking_tag(self) -> None:
        """</thinking> variant also works."""
        result = strip_think_tags("<think>reasoning</thinking>The answer")
        assert result == "The answer"

    def test_empty_string_passthrough(self) -> None:
        """Empty string stays empty."""
        result = strip_think_tags("")
        assert result == ""

    def test_whitespace_only_passthrough(self) -> None:
        """Whitespace-only stays empty after strip."""
        result = strip_think_tags("   \n  ")
        assert result == ""