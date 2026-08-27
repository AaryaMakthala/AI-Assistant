"""Phase B-2: Grounding thresholds + filename-aware retrieval tests.

Tests behavioral requirements against the actual implementation.
All external database calls are mocked.
Never hits a real database.

Key contract:
- Cross-encoder scores are raw logits (~[-12, +12]), NOT probabilities.
- Overview grounding uses absolute thresholds (OVERVIEW_MIN_SCORE, OVERVIEW_AGGREGATE_MIN).
- Fact-lookup uses the existing retrieval_relevance_threshold.
- High-confidence document targeting relaxes the grounding floor.
- Filename matching is workspace-scoped and READY-only.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.retrieval.grounding as grounding_module
from app.retrieval.grounding import is_grounded, is_overview_grounded
from app.retrieval.hybrid import (
    _extract_filename_tokens,
    _filename_matches_query,
    _normalize_filename_for_match,
)

pytestmark = pytest.mark.usefixtures("valid_env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides):
    """Create a settings object with calibrated defaults for testing."""
    defaults = {
        "retrieval_relevance_threshold": 0.3,
        "overview_min_score": -4.0,
        "overview_aggregate_min": -7.5,
        "doc_target_high_confidence": 0.90,
        "doc_target_relaxed_score": -3.0,
        "filename_match_relaxed_score": -15.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ===========================================================================
# SECTION 1: Fact-Lookup Grounding (absolute threshold)
# ===========================================================================

class TestFactLookupGrounding:
    """Fact-lookup grounding uses retrieval_relevance_threshold (on [0,1] scale)."""

    def test_positive_above_threshold_grounds(self, monkeypatch):
        """Score above threshold = grounded."""
        settings = _make_settings(retrieval_relevance_threshold=0.3)
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_grounded(0.7558) is True
        assert is_grounded(0.3578) is True
        assert is_grounded(0.3) is True  # boundary

    def test_below_threshold_not_grounded(self, monkeypatch):
        """Score below threshold = not grounded."""
        settings = _make_settings(retrieval_relevance_threshold=0.3)
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_grounded(0.2) is False
        assert is_grounded(0.0) is False
        assert is_grounded(-0.4767) is False
        assert is_grounded(-2.0) is False

    def test_none_never_grounds(self, monkeypatch):
        """No retrieval = no grounding."""
        settings = _make_settings()
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_grounded(None) is False


# ===========================================================================
# SECTION 2: Fact-Lookup with High-Confidence Doc-Target Relaxation
# ===========================================================================

class TestDocTargetRelaxation:
    """High-confidence document targeting relaxes the grounding floor."""

    def test_high_conf_with_target_chunk_relaxes(self, monkeypatch):
        """High-confidence target + chunk from target doc = relaxed threshold."""
        settings = _make_settings(
            retrieval_relevance_threshold=0.3,
            doc_target_high_confidence=0.90,
            doc_target_relaxed_score=-3.0,
        )
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        # Negative score that would normally fail, but passes with relaxation.
        assert is_grounded(
            -0.4767,
            doc_target_high_confidence=True,
            has_target_chunk=True,
        ) is True

    def test_high_conf_without_target_chunk_uses_normal(self, monkeypatch):
        """High-confidence target but NO chunk from target = normal threshold."""
        settings = _make_settings(
            retrieval_relevance_threshold=0.3,
            doc_target_high_confidence=0.90,
            doc_target_relaxed_score=-3.0,
        )
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        # Without has_target_chunk, the normal threshold applies.
        assert is_grounded(
            -0.4767,
            doc_target_high_confidence=True,
            has_target_chunk=False,
        ) is False

    def test_low_conf_uses_normal(self, monkeypatch):
        """Low-confidence target = normal threshold, even with target chunk."""
        settings = _make_settings(
            retrieval_relevance_threshold=0.3,
            doc_target_high_confidence=0.90,
            doc_target_relaxed_score=-3.0,
        )
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_grounded(
            -0.4767,
            doc_target_high_confidence=False,
            has_target_chunk=True,
        ) is False

    def test_very_negative_still_fails_with_relaxation(self, monkeypatch):
        """Strongly negative scores (~-8 to -10) must NOT pass even with relaxation."""
        settings = _make_settings(
            retrieval_relevance_threshold=0.3,
            doc_target_high_confidence=0.90,
            doc_target_relaxed_score=-3.0,
        )
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_grounded(
            -8.0,
            doc_target_high_confidence=True,
            has_target_chunk=True,
        ) is False
        assert is_grounded(
            -10.0,
            doc_target_high_confidence=True,
            has_target_chunk=True,
        ) is False

    def test_positive_scores_still_pass_with_relaxation(self, monkeypatch):
        """Known-good positive scores continue to pass regardless."""
        settings = _make_settings(
            retrieval_relevance_threshold=0.3,
            doc_target_high_confidence=0.90,
            doc_target_relaxed_score=-3.0,
        )
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_grounded(
            0.7558,
            doc_target_high_confidence=True,
            has_target_chunk=True,
        ) is True
        assert is_grounded(
            0.3578,
            doc_target_high_confidence=True,
            has_target_chunk=True,
        ) is True


# ===========================================================================
# SECTION 3: Overview Grounding (absolute thresholds)
# ===========================================================================

class TestOverviewGroundingAbsolute:
    """Overview grounding uses absolute cross-encoder score thresholds."""

    def test_positive_cluster_grounds(self, monkeypatch):
        """Tight positive cluster should ground."""
        settings = _make_settings()
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_overview_grounded([0.25, 0.22, 0.18, 0.15, 0.10]) is True

    def test_negative_above_threshold_grounds(self, monkeypatch):
        """Negative scores above absolute minimum should ground."""
        settings = _make_settings()
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        # Top score -2.45 is above -4.0 -> passes condition 1.
        assert is_overview_grounded([-2.45, -3.80, -4.24, -5.94, -6.11]) is True

    def test_top_score_below_min_does_not_ground(self, monkeypatch):
        """Top chunk below absolute minimum = not grounded."""
        settings = _make_settings()
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        # Top score -4.5 is below -4.0.
        assert is_overview_grounded([-4.5, -5.0, -5.5, -8.0, -9.0]) is False

    def test_mean_below_aggregate_min_does_not_ground(self, monkeypatch):
        """Top-k mean below aggregate minimum = not grounded."""
        settings = _make_settings()
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        # Top score -3.0 is above -4.0, but mean of top 3 = (-3.0 + -10.0 + -11.0)/3 = -8.0
        # which is below -7.5.
        assert is_overview_grounded([-3.0, -10.0, -11.0, -12.0, -13.0]) is False

    def test_strongly_negative_does_not_ground(self, monkeypatch):
        """Clearly irrelevant scores (-8, -10) must not ground."""
        settings = _make_settings()
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_overview_grounded([-8.0, -9.0, -10.0, -11.0]) is False

    def test_single_chunk_does_not_ground(self, monkeypatch):
        """Overview needs multiple chunks for diffuse relevance."""
        settings = _make_settings()
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_overview_grounded([-1.5]) is False

    def test_empty_does_not_ground(self, monkeypatch):
        settings = _make_settings()
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        assert is_overview_grounded([]) is False

    def test_high_conf_target_relaxes_overview(self, monkeypatch):
        """High-confidence doc target + target chunk relaxes overview thresholds."""
        settings = _make_settings(
            overview_min_score=-4.0,
            overview_aggregate_min=-5.0,
            doc_target_high_confidence=0.90,
            doc_target_relaxed_score=-5.0,
        )
        monkeypatch.setattr(grounding_module, "get_settings", lambda: settings)
        # Top score -4.5 is below normal -4.0 but above relaxed -5.0.
        # Mean of top 3 = (-4.5 + -5.0 + -5.5)/3 = -5.0 which is at relaxed -5.0.
        assert is_overview_grounded(
            [-4.5, -5.0, -5.5, -8.0, -9.0],
            doc_target_high_confidence=True,
            has_target_chunk=True,
        ) is True


# ===========================================================================
# SECTION 4: Filename Normalization
# ===========================================================================

class TestFilenameNormalization:
    """Verify normalization matches doc_targeting.py conventions."""

    def test_strip_extension(self):
        assert _normalize_filename_for_match("resume.pdf") == "resume"
        assert _normalize_filename_for_match("handbook.docx") == "handbook"

    def test_lowercase(self):
        assert _normalize_filename_for_match("Resume.PDF") == "resume"

    def test_replace_punctuation(self):
        result = _normalize_filename_for_match(
            "DevOps 4-1 AIML QUESTION BANK_SUBJECTIVE_CIE-I.docx"
        )
        assert "devops" in result
        assert "aiml" in result
        assert "question" in result
        assert "bank" in result
        assert "subjective" in result
        assert "cie" in result

    def test_replace_hyphens_with_spaces(self):
        assert _normalize_filename_for_match("my-file.pdf") == "my file"

    def test_replace_underscores_with_spaces(self):
        assert _normalize_filename_for_match("my_file.pdf") == "my file"

    def test_collapse_whitespace(self):
        assert _normalize_filename_for_match("resume  with   spaces.pdf") == "resume with spaces"

    def test_real_filename_aarya(self):
        result = _normalize_filename_for_match("Makthala Aarya Resume.pdf")
        assert result == "makthala aarya resume"


# ===========================================================================
# SECTION 5: Filename Token Extraction
# ===========================================================================

class TestFilenameTokenExtraction:
    """Extract meaningful tokens from queries for filename matching."""

    def test_stop_words_stripped(self):
        tokens = _extract_filename_tokens("do you have any resume")
        assert "resume" in tokens
        assert "do" not in tokens
        assert "you" not in tokens
        assert "have" not in tokens
        assert "any" not in tokens

    def test_single_token(self):
        tokens = _extract_filename_tokens("resume")
        assert tokens == {"resume"}

    def test_name_tokens(self):
        tokens = _extract_filename_tokens("aarya document you have")
        assert "aarya" in tokens
        assert "document" not in tokens  # stop word
        assert "you" not in tokens
        assert "have" not in tokens

    def test_multi_token_phrase(self):
        tokens = _extract_filename_tokens("aarya resume")
        assert tokens == {"aarya", "resume"}

    def test_empty_query(self):
        tokens = _extract_filename_tokens("the have you any what")
        assert len(tokens) == 0


# ===========================================================================
# SECTION 6: Filename Matching
# ===========================================================================

class TestFilenameMatching:
    """Verify filename matching against normalized filenames."""

    def test_resume_matches(self):
        assert _filename_matches_query("Makthala Aarya Resume.pdf", {"resume"}) is True

    def test_name_matches(self):
        assert _filename_matches_query("Makthala Aarya Resume.pdf", {"aarya"}) is True

    def test_multi_token_matches(self):
        assert _filename_matches_query(
            "Makthala Aarya Resume.pdf", {"aarya", "resume"}
        ) is True

    def test_punctuation_filename_matches(self):
        tokens = _extract_filename_tokens("devops aiml")
        assert _filename_matches_query(
            "DevOps 4-1 AIML QUESTION BANK_SUBJECTIVE_CIE-I.docx", tokens
        ) is True

    def test_unrelated_no_match(self):
        assert _filename_matches_query("handbook.pdf", {"resume"}) is False
        assert _filename_matches_query("policy.docx", {"kanban"}) is False

    def test_stop_words_no_match(self):
        """Stop words alone should not match."""
        assert _filename_matches_query("resume.pdf", {"the"}) is False
        assert _filename_matches_query("resume.pdf", {"document"}) is False

    def test_case_insensitive(self):
        assert _filename_matches_query("RESUME.PDF", {"resume"}) is True

    def test_underscore_filename_matches(self):
        assert _filename_matches_query("my_resume_v2.pdf", {"resume"}) is True


# ===========================================================================
# SECTION 7: Filename Search (Integration-like, mocked DB)
# ===========================================================================

class TestFilenameSearch:
    """Test the filename_search function against a mocked database."""

    @pytest.mark.asyncio
    async def test_returns_chunks_from_matched_docs(self):
        """Filename match returns chunks from matching documents."""
        ws_id = uuid.uuid4()

        # Mock READY documents in workspace — tuples like SQLAlchemy Row objects.
        resume_doc_id = uuid.uuid4()
        handbook_doc_id = uuid.uuid4()
        doc_rows = [
            SimpleNamespace(id=resume_doc_id, filename="Makthala Aarya Resume.pdf"),
            SimpleNamespace(id=handbook_doc_id, filename="handbook.pdf"),
        ]
        # Use a mock that unpacks like a Row (each element can be iterated as tuples).
        class _FakeDocResult:
            def __init__(self, rows):
                self._rows = rows
            def all(self):
                # Return objects that support tuple unpacking.
                return [(r.id, r.filename) for r in self._rows]
        doc_result = _FakeDocResult(doc_rows)

        # Mock chunks from the matched doc.
        class _FakeChunkResult:
            def __init__(self, rows):
                self._rows = rows
            def all(self):
                return self._rows
        chunk_result = _FakeChunkResult([
            SimpleNamespace(
                id=uuid.uuid4(),
                document_id=resume_doc_id,
                content="Aarya is a software engineer.",
                page_number=1,
                section_title="Profile",
                chunk_index=0,
                filename="Makthala Aarya Resume.pdf",
            ),
        ])

        session = AsyncMock()
        session.execute = AsyncMock(side_effect=[doc_result, chunk_result])

        from app.retrieval.hybrid import filename_search

        results = await filename_search(
            session,
            query="do you have any resume",
            workspace_id=ws_id,
            limit=15,
        )

        assert len(results) == 1
        assert results[0].document_id == resume_doc_id
        assert results[0].filename == "Makthala Aarya Resume.pdf"
        assert "Aarya" in results[0].content

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        """Unrelated query returns no results."""
        ws_id = uuid.uuid4()

        class _FakeDocResult:
            def __init__(self, rows):
                self._rows = rows
            def all(self):
                return self._rows
        doc_result = _FakeDocResult([
            (uuid.uuid4(), "handbook.pdf"),
        ])

        session = AsyncMock()
        session.execute = AsyncMock(return_value=doc_result)

        from app.retrieval.hybrid import filename_search

        results = await filename_search(
            session,
            query="resume",
            workspace_id=ws_id,
            limit=15,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_stop_word_query_returns_empty(self):
        """Query with only stop words returns no results."""
        ws_id = uuid.uuid4()

        session = AsyncMock()
        session.execute = AsyncMock()  # Should not be called.

        from app.retrieval.hybrid import filename_search

        results = await filename_search(
            session,
            query="do you have any what",
            workspace_id=ws_id,
            limit=15,
        )

        assert results == []
        # No DB call should have been made.
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_workspace_scoped(self):
        """Filename search only queries the specified workspace."""
        ws_id = uuid.uuid4()

        class _FakeDocResult:
            def __init__(self, rows):
                self._rows = rows
            def all(self):
                return self._rows
        doc_result = _FakeDocResult([])  # No docs in this workspace.

        session = AsyncMock()
        session.execute = AsyncMock(return_value=doc_result)

        from app.retrieval.hybrid import filename_search

        results = await filename_search(
            session,
            query="resume",
            workspace_id=ws_id,
            limit=15,
        )

        assert results == []
        # The query should have been made (with the workspace_id filter).
        session.execute.assert_called_once()


# ===========================================================================
# SECTION 8: Tenant Isolation for Filename Search
# ===========================================================================

class TestFilenameSearchTenantIsolation:
    """Verify workspace A cannot see workspace B's filenames."""

    @pytest.mark.asyncio
    async def test_cross_workspace_no_leak(self):
        """Query in workspace A must not return chunks from workspace B."""
        ws_a = uuid.uuid4()
        ws_b = uuid.uuid4()

        class _FakeDocResult:
            def __init__(self, rows):
                self._rows = rows
            def all(self):
                return self._rows
        ws_a_docs = _FakeDocResult([])  # Workspace A has no resume docs.

        session = AsyncMock()
        session.execute = AsyncMock(return_value=ws_a_docs)

        from app.retrieval.hybrid import filename_search

        results = await filename_search(
            session,
            query="resume",
            workspace_id=ws_a,  # Searching in workspace A.
            limit=15,
        )

        assert results == []
        # The only query should be for workspace A's documents.
        call_args = session.execute.call_args[0][0]
        # Verify the query includes the workspace_id filter.
        # (SQLAlchemy WHERE clauses are in the statement's whereclause.)
        where_str = str(call_args.whereclause)
        assert str(ws_a) in where_str or "workspace_id" in where_str

    @pytest.mark.asyncio
    async def test_non_ready_docs_excluded(self):
        """PENDING/REJECTED docs must not participate in filename search."""
        ws_id = uuid.uuid4()

        class _FakeDocResult:
            def __init__(self, rows):
                self._rows = rows
            def all(self):
                return self._rows
        doc_result = _FakeDocResult([])  # Query filters by READY, so no results.

        session = AsyncMock()
        session.execute = AsyncMock(return_value=doc_result)

        from app.retrieval.hybrid import filename_search

        results = await filename_search(
            session,
            query="resume",
            workspace_id=ws_id,
            limit=15,
        )

        assert results == []
        # Verify the query included the status=READY filter.
        call_args = session.execute.call_args[0][0]
        where_str = str(call_args.whereclause)
        assert "READY" in where_str or "status" in where_str.lower()
