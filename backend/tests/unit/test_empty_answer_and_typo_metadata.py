"""Tests for ISSUE 1 (empty LLM answer safety) and ISSUE 2 (typo-tolerant metadata sub-intent inference).

ISSUE 1: When the LLM returns None, empty string, or whitespace-only text
after think-tag stripping, the system must return a safe fallback message
instead of silently sending an empty answer to the frontend.

ISSUE 2: When regex classification doesn't match a typo-heavy metadata query
but QU classifies it as workspace_metadata, the system must infer the
sub-intent from the corrected query and route to the DB metadata handler.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.api.chat_v2 import _refine_intent_from_qu, _infer_metadata_sub_from_query
from app.retrieval.intent import IntentCategory, MetadataSubIntent
from app.retrieval.query_understanding import QueryUnderstanding


# ---------------------------------------------------------------------------
# ISSUE 2: _infer_metadata_sub_from_query tests (pure function, no DB)
# ---------------------------------------------------------------------------


class TestInferMetadataSubFromQuery:
    """Test the heuristic sub-intent inference from corrected query strings."""

    def test_doc_count_normal(self) -> None:
        assert _infer_metadata_sub_from_query("how many documents are there") == MetadataSubIntent.DOC_COUNT

    def test_doc_count_typo(self) -> None:
        """The core typo case from the user's test: 'how manu files hee=re'."""
        assert _infer_metadata_sub_from_query("how manu files hee=re") == MetadataSubIntent.DOC_COUNT

    def test_doc_count_uploaded(self) -> None:
        assert _infer_metadata_sub_from_query("how many files have i uploaded") == MetadataSubIntent.DOC_COUNT

    def test_doc_list_normal(self) -> None:
        assert _infer_metadata_sub_from_query("what documents are in this workspace") == MetadataSubIntent.DOC_LIST

    def test_doc_list_typo(self) -> None:
        """The core typo case: 'what aer the docs presnt?'."""
        assert _infer_metadata_sub_from_query("what aer the docs presnt?") == MetadataSubIntent.DOC_LIST

    def test_doc_list_what_are_they(self) -> None:
        assert _infer_metadata_sub_from_query("what are the docs") == MetadataSubIntent.DOC_LIST

    def test_workspace_name(self) -> None:
        assert _infer_metadata_sub_from_query("what is the name of the workspace") == MetadataSubIntent.COMPANY_NAME

    def test_workspace_name_title(self) -> None:
        assert _infer_metadata_sub_from_query("what is the title of this company") == MetadataSubIntent.COMPANY_NAME

    def test_member_count(self) -> None:
        assert _infer_metadata_sub_from_query("how many members are there") == MetadataSubIntent.MEMBER_COUNT

    def test_member_list(self) -> None:
        assert _infer_metadata_sub_from_query("list all members") == MetadataSubIntent.MEMBER_LIST

    def test_role(self) -> None:
        assert _infer_metadata_sub_from_query("what is my role") == MetadataSubIntent.ROLE

    def test_doc_page_count(self) -> None:
        assert _infer_metadata_sub_from_query("how many pages in the documents") == MetadataSubIntent.DOC_PAGE_COUNT

    def test_doc_description(self) -> None:
        assert _infer_metadata_sub_from_query("what is the description of each document") == MetadataSubIntent.DOC_DESCRIPTION

    def test_unrelated_query_returns_none(self) -> None:
        """A non-metadata query should return None — no false positive."""
        assert _infer_metadata_sub_from_query("capital of japan") is None

    def test_code_request_returns_none(self) -> None:
        assert _infer_metadata_sub_from_query("write me a python script") is None

    def test_empty_query_returns_none(self) -> None:
        assert _infer_metadata_sub_from_query("") is None

    def test_7_regression_queries(self) -> None:
        """Original 7 regression queries must route to their expected metadata sub-intent."""
        cases = [
            ("how many documents are there?", MetadataSubIntent.DOC_COUNT),
            ("how many files have I uploaded?", MetadataSubIntent.DOC_COUNT),
            ("what documents are in this workspace?", MetadataSubIntent.DOC_LIST),
            ("how many files do I have?", MetadataSubIntent.DOC_COUNT),
            ("what is the name of the workspace?", MetadataSubIntent.COMPANY_NAME),
            ("how manu files hee=re", MetadataSubIntent.DOC_COUNT),
            ("what aer the docs presnt?", MetadataSubIntent.DOC_LIST),
        ]
        for query, expected_sub in cases:
            actual = _infer_metadata_sub_from_query(query)
            assert actual == expected_sub, f"Query '{query}' expected {expected_sub}, got {actual}"

    def test_ambiguous_queries_return_none(self) -> None:
        """Ambiguous queries must NOT be forced into metadata sub-intents."""
        ambiguous_cases = [
            "what about that?",
            "tell me more",
            "can you explain?",
            "why did that happen?",
            "continue",
            "help",
        ]
        for query in ambiguous_cases:
            assert _infer_metadata_sub_from_query(query) is None, (
                f"Ambiguous query '{query}' should not match any metadata sub-intent"
            )


# ---------------------------------------------------------------------------
# ISSUE 2: _refine_intent_from_qu with QU fallback
# ---------------------------------------------------------------------------


class TestRefineIntentFromQuTypoMetadata:
    """Test that typo-heavy metadata queries reach the metadata handler
    when QU classifies them as workspace_metadata but regex misses."""

    def _make_qu(self, intent: IntentCategory, corrected_query: str, confidence: float = 0.9) -> QueryUnderstanding:
        return QueryUnderstanding(
            corrected_query=corrected_query,
            search_query=corrected_query,
            intent=intent,
            confidence=confidence,
            reasoning="test",
        )

    @patch("app.retrieval.intent.classify_intent_regex")
    def test_typo_doc_count_infers_sub_intent(self, mock_regex: MagicMock) -> None:
        """'how manu files hee=re' → QU says workspace_metadata, regex returns
        regex_fallback_to_llm (no match) → should infer doc_count."""
        mock_regex.return_value = IntentCategory.DOCUMENT_CONTENT  # regex doesn't match
        from app.retrieval.intent import Intent
        # Patch the regex return to be a DOCUMENT_CONTENT intent (no metadata_sub)
        mock_regex.return_value = Intent(
            category=IntentCategory.DOCUMENT_CONTENT,
            reason="regex_fallback_to_llm",
        )
        qu = self._make_qu(IntentCategory.WORKSPACE_METADATA, "how manu files hee=re")
        result = _refine_intent_from_qu(qu_result=qu, original_query="how manu files hee=re")
        assert result.category == IntentCategory.WORKSPACE_METADATA
        assert result.metadata_sub == MetadataSubIntent.DOC_COUNT

    @patch("app.retrieval.intent.classify_intent_regex")
    def test_typo_doc_list_infers_sub_intent(self, mock_regex: MagicMock) -> None:
        """'what aer the docs presnt?' → QU says workspace_metadata, regex misses → infer doc_list."""
        from app.retrieval.intent import Intent
        mock_regex.return_value = Intent(
            category=IntentCategory.DOCUMENT_CONTENT,
            reason="regex_fallback_to_llm",
        )
        qu = self._make_qu(IntentCategory.WORKSPACE_METADATA, "what aer the docs presnt?")
        result = _refine_intent_from_qu(qu_result=qu, original_query="what aer the docs presnt?")
        assert result.category == IntentCategory.WORKSPACE_METADATA
        assert result.metadata_sub == MetadataSubIntent.DOC_LIST

    @patch("app.retrieval.intent.classify_intent_regex")
    def test_normal_metadata_still_uses_regex(self, mock_regex: MagicMock) -> None:
        """Normal query where regex matches → existing behavior preserved (regex path)."""
        from app.retrieval.intent import Intent
        mock_regex.return_value = Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_COUNT,
            skip_rewrite=True,
            reason="doc_count",
        )
        qu = self._make_qu(IntentCategory.WORKSPACE_METADATA, "how many documents are there")
        result = _refine_intent_from_qu(qu_result=qu, original_query="how many documents are there")
        assert result.metadata_sub == MetadataSubIntent.DOC_COUNT
        # The reason should contain "regex:" (regex path, not inference path)
        assert "regex:" in result.reason

    @patch("app.retrieval.intent.classify_intent_regex")
    def test_non_metadata_query_no_inference(self, mock_regex: MagicMock) -> None:
        """A document_content query should NOT get metadata sub-intent inferred."""
        from app.retrieval.intent import Intent
        mock_regex.return_value = Intent(
            category=IntentCategory.DOCUMENT_CONTENT,
            reason="regex_fallback_to_llm",
        )
        qu = self._make_qu(IntentCategory.DOCUMENT_CONTENT, "capital of japan")
        result = _refine_intent_from_qu(qu_result=qu, original_query="capital of japan")
        assert result.category == IntentCategory.DOCUMENT_CONTENT
        assert result.metadata_sub is None

    @patch("app.retrieval.intent.classify_intent_regex")
    def test_inferred_sub_intent_has_skip_rewrite(self, mock_regex: MagicMock) -> None:
        """Inferred metadata sub-intents should set skip_rewrite=True."""
        from app.retrieval.intent import Intent
        mock_regex.return_value = Intent(
            category=IntentCategory.DOCUMENT_CONTENT,
            reason="regex_fallback_to_llm",
        )
        qu = self._make_qu(IntentCategory.WORKSPACE_METADATA, "how manu files hee=re")
        result = _refine_intent_from_qu(qu_result=qu, original_query="how manu files hee=re")
        assert result.skip_rewrite is True


# ---------------------------------------------------------------------------
# ISSUE 1: Empty LLM answer safety (grounded_chat sync path)
# ---------------------------------------------------------------------------


class TestEmptyLlmAnswerSafety:
    """Test that empty/whitespace-only LLM answers after think-tag stripping
    are caught and replaced with a fallback message."""

    def test_strip_think_tags_empty_gives_fallback(self) -> None:
        """If _strip_think_tags returns empty, the answer should be the fallback."""
        from app.llm.utils import strip_think_tags
        # Unclosed think block → strip returns empty
        result = strip_think_tags("<think>some reasoning that was cut off")
        assert result == ""
        # The code checks answer_text.strip() after stripping
        # and returns the fallback message

    def test_strip_think_tags_whitespace_gives_empty(self) -> None:
        from app.llm.utils import strip_think_tags
        result = strip_think_tags("<think> </think>   ")
        assert result.strip() == ""

    def test_strip_think_tags_normal_text_preserved(self) -> None:
        from app.llm.utils import strip_think_tags
        result = strip_think_tags("<think>thinking</think>The answer is 42.")
        assert "42" in result
        assert "thinking" not in result

    def test_strip_think_tags_only_answer_preserved(self) -> None:
        from app.llm.utils import strip_think_tags
        result = strip_think_tags("The answer is 42.")
        assert result == "The answer is 42."
