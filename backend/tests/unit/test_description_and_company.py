"""Tests for per-document description routing and company name routing.

Covers:
- Description/summary questions route to DOC_DESCRIPTION metadata operation
- Company name questions route to COMPANY_NAME metadata operation
- Specific document description returns that document's description
- Generic description returns all documents' descriptions
- Graceful handling of documents with no description
- Regression: content questions still go through retrieval
"""

from __future__ import annotations

import re
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.retrieval.intent import (
    IntentCategory,
    MetadataSubIntent,
    classify_intent_regex,
)


# ---------------------------------------------------------------------------
# Regex classification tests
# ---------------------------------------------------------------------------


class TestDescriptionRouting:
    """Verify description/summary queries route to DOC_DESCRIPTION."""

    @pytest.mark.parametrize(
        "query",
        [
            "description",
            "descriction",
            "summary",
            "give me a description",
            "give a description of each document",
            "can you give descriction on each document",
            "summary of 5 lines of each document",
            "summary of each document",
            "description of all documents",
            "summery",
        ],
    )
    def test_generic_description_routes_to_doc_description(self, query: str) -> None:
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.metadata_sub == MetadataSubIntent.DOC_DESCRIPTION

    @pytest.mark.parametrize(
        "query",
        [
            "summary of DevOps",
            "description of test_doc.txt",
            "summary of the resume",
            "give me a summary of the DATA MINING",
            "description of CRNS",
        ],
    )
    def test_specific_description_routes_to_doc_description(self, query: str) -> None:
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.metadata_sub == MetadataSubIntent.DOC_DESCRIPTION
        assert "specific" in intent.reason

    def test_regression_content_question_not_description(self) -> None:
        """'what does devops contain' must NOT route as description."""
        intent = classify_intent_regex("what does devops contain")
        assert intent.category == IntentCategory.DOCUMENT_CONTENT
        assert intent.metadata_sub != MetadataSubIntent.DOC_DESCRIPTION

    @pytest.mark.parametrize(
        "query",
        [
            "explain any one document",
            "explain any one documetn",
            "explain the documents",
            "explain any file",
            "describe each document",
            "can you explain every document",
        ],
    )
    def test_explain_document_routes_to_doc_description(self, query: str) -> None:
        """'explain/describe <doc-noun>' routes to DOC_DESCRIPTION (typo-tolerant)."""
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.metadata_sub == MetadataSubIntent.DOC_DESCRIPTION
        assert intent.reason == "doc_description_explain"

    @pytest.mark.parametrize(
        "query",
        [
            "explain the vacation policy",
            "explain the Kanban section",
            "explain",
            "explain everything about workforce",
            "describe what the handbook says",
        ],
    )
    def test_explain_content_question_stays_document_content(self, query: str) -> None:
        """Content questions must NOT be hijacked by the doc-noun explain pattern."""
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.DOCUMENT_CONTENT
        assert intent.metadata_sub != MetadataSubIntent.DOC_DESCRIPTION

    def test_explain_anaphoric_stays_ambiguous(self) -> None:
        """'explain that' remains the anaphoric/ambiguous path, not metadata."""
        intent = classify_intent_regex("explain that")
        assert intent.category == IntentCategory.AMBIGUOUS

    def test_pick_any_five_not_doc_description(self) -> None:
        """'pick any five' must not become a metadata description request."""
        intent = classify_intent_regex("pick any five")
        assert intent.metadata_sub != MetadataSubIntent.DOC_DESCRIPTION


class TestAnaphoricTypoDescription:
    """Verify describe typos + anaphoric pronouns route to DOC_DESCRIPTION."""

    @pytest.mark.parametrize(
        "query",
        [
            "descrivbe them",
            "descibe it",
            "descrption that",
            "describe this",
        ],
    )
    def test_describe_typo_anaphoric_routes_to_doc_description(self, query: str) -> None:
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.metadata_sub == MetadataSubIntent.DOC_DESCRIPTION
        assert "anaphoric_typo" in intent.reason

    @pytest.mark.parametrize(
        "query",
        [
            "explain that",
            "how about it",
            "what about them",
        ],
    )
    def test_other_anaphoric_stays_ambiguous(self, query: str) -> None:
        """Non-describe anaphoric references remain AMBIGUOUS (LLM disambiguates)."""
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.AMBIGUOUS
        assert intent.metadata_sub is None


class TestTellAboutNumberFiles:
    """Verify 'tell about' + number + document nouns routes to DOC_DESCRIPTION."""

    @pytest.mark.parametrize(
        "query",
        [
            "tell about any five files",
            "tell about 5 files",
            "tell me about any three documents",
            "tell about the seven docs",
            "tell about some two files",
        ],
    )
    def test_tell_about_number_files_routes_to_doc_description(self, query: str) -> None:
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.metadata_sub == MetadataSubIntent.DOC_DESCRIPTION
        assert "tell_about" in intent.reason

    @pytest.mark.parametrize(
        "query",
        [
            "tell about any fuve files",
            "tell about fiv docs",
            "tell about fie files",
        ],
    )
    def test_tell_about_typo_number_routes_to_doc_description(self, query: str) -> None:
        """Typo-tolerant number words should still route to DOC_DESCRIPTION."""
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.metadata_sub == MetadataSubIntent.DOC_DESCRIPTION

    @pytest.mark.parametrize(
        "query",
        [
            "tell about",
            "tell me more",
            "tell me about the vacation policy",
            "tell about any five",
        ],
    )
    def test_tell_about_no_hijack(self, query: str) -> None:
        """Bare 'tell about' or content-qualified 'tell about' must NOT become DOC_DESCRIPTION."""
        intent = classify_intent_regex(query)
        assert intent.metadata_sub != MetadataSubIntent.DOC_DESCRIPTION


class TestCompanyNameRouting:
    """Verify company/workspace name queries route to COMPANY_NAME."""

    @pytest.mark.parametrize(
        "query",
        [
            "what is the name of this company",
            "what is the company name",
            "what's our company name",
            "what is the workspace called",
            "what is the workspace name",
            "what is the organization name",
        ],
    )
    def test_company_name_queries(self, query: str) -> None:
        intent = classify_intent_regex(query)
        assert intent.category == IntentCategory.WORKSPACE_METADATA
        assert intent.metadata_sub == MetadataSubIntent.COMPANY_NAME

    def test_regression_not_company_name(self) -> None:
        """'what is my role' must NOT route as company name."""
        intent = classify_intent_regex("what is my role")
        assert intent.metadata_sub != MetadataSubIntent.COMPANY_NAME


# ---------------------------------------------------------------------------
# Metadata handler tests (integration-level with mocks)
# ---------------------------------------------------------------------------


class TestDescriptionHandler:
    """Test the _answer_metadata_question handler for DOC_DESCRIPTION."""

    @pytest.mark.asyncio
    async def test_generic_description_returns_all(self) -> None:
        """Generic description returns all READY documents with descriptions."""
        from app.api.chat_v2 import _answer_metadata_question
        from app.retrieval.intent import Intent

        mock_docs = [
            MagicMock(filename="resume.pdf", description="A resume for Aarya."),
            MagicMock(filename="devops.pdf", description="DevOps question bank."),
            MagicMock(filename="test.txt", description=None),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = mock_docs
        mock_session.execute.return_value = mock_result

        intent = Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_DESCRIPTION,
            reason="doc_description",
        )

        with patch("app.api.chat_v2.tenant_session") as mock_ts:
            mock_ts.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ts.return_value.__aexit__ = AsyncMock(return_value=False)
            answer, reason = await _answer_metadata_question(
                intent=intent,
                question="description",
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
            )

        assert "resume.pdf" in answer
        assert "A resume for Aarya" in answer
        assert "devops.pdf" in answer
        assert "No description available" in answer
        assert "test.txt" in answer
        assert reason is None

    @pytest.mark.asyncio
    async def test_specific_description_returns_match(self) -> None:
        """Specific document description returns that document's description."""
        from app.api.chat_v2 import _answer_metadata_question
        from app.retrieval.intent import Intent

        mock_docs = [
            MagicMock(filename="Makthala Aarya Resume.pdf", description="A subject resume."),
            MagicMock(filename="DevOps 4-1 AIML QUESTION BANK.pdf", description="DevOps questions."),
            MagicMock(filename="DATA MINING doc.pdf", description="Data mining material."),
            MagicMock(filename="CRNS doc.pdf", description="CRNS material."),
            MagicMock(filename="test_doc.txt", description=None),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = mock_docs
        mock_session.execute.return_value = mock_result

        intent = Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_DESCRIPTION,
            reason="doc_description_specific",
        )

        with patch("app.api.chat_v2.tenant_session") as mock_ts:
            mock_ts.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ts.return_value.__aexit__ = AsyncMock(return_value=False)
            answer, reason = await _answer_metadata_question(
                intent=intent,
                question="summary of DevOps",
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
            )

        assert "DevOps" in answer
        assert "DevOps questions" in answer
        assert reason is None

    @pytest.mark.asyncio
    async def test_no_description_available(self) -> None:
        """Document with no description shows a clear message."""
        from app.api.chat_v2 import _answer_metadata_question
        from app.retrieval.intent import Intent

        mock_docs = [
            MagicMock(filename="test_doc.txt", description=None),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = mock_docs
        mock_session.execute.return_value = mock_result

        intent = Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.DOC_DESCRIPTION,
            reason="doc_description_specific",
        )

        with patch("app.api.chat_v2.tenant_session") as mock_ts:
            mock_ts.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ts.return_value.__aexit__ = AsyncMock(return_value=False)
            answer, reason = await _answer_metadata_question(
                intent=intent,
                question="summary of test_doc",
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
            )

        assert "no generated description" in answer.lower() or "does not have a generated description" in answer


class TestCompanyNameHandler:
    """Test the _answer_metadata_question handler for COMPANY_NAME."""

    @pytest.mark.asyncio
    async def test_company_name_returns_workspace_name(self) -> None:
        """Company name query returns the workspace name."""
        from app.api.chat_v2 import _answer_metadata_question
        from app.retrieval.intent import Intent

        mock_ws = MagicMock()
        mock_ws.name = "Acme Corp"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_ws.name
        mock_session.execute.return_value = mock_result

        intent = Intent(
            category=IntentCategory.WORKSPACE_METADATA,
            metadata_sub=MetadataSubIntent.COMPANY_NAME,
            reason="company_name",
        )

        with patch("app.api.chat_v2.tenant_session") as mock_ts:
            mock_ts.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ts.return_value.__aexit__ = AsyncMock(return_value=False)
            answer, reason = await _answer_metadata_question(
                intent=intent,
                question="what is the company name",
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
            )

        assert "Acme Corp" in answer
        assert reason is None


class TestRegressionChecks:
    """Ensure existing behavior doesn't regress."""

    def test_greeting_still_works(self) -> None:
        intent = classify_intent_regex("hey")
        assert intent.category == IntentCategory.GREETING

    def test_doc_list_still_works(self) -> None:
        intent = classify_intent_regex("what documents you haeve")
        assert intent.category == IntentCategory.DOCUMENT_LIST

    def test_leave_policy_goes_to_retrieval(self) -> None:
        intent = classify_intent_regex("what is our leave policy?")
        assert intent.category == IntentCategory.DOCUMENT_CONTENT

    def test_devops_content_goes_to_retrieval(self) -> None:
        intent = classify_intent_regex("what does devops contain")
        assert intent.category == IntentCategory.DOCUMENT_CONTENT

    def test_out_of_scope_still_refused(self) -> None:
        intent = classify_intent_regex("capital of India")
        assert intent.category == IntentCategory.OUT_OF_SCOPE
