"""Text normalization: what reaches the embedder must match what a human reads.

Normalization is not cosmetic. Invisible characters split a token the model has seen into
two it has not, and whitespace runs change where the splitter breaks — so these behaviours
are load-bearing for retrieval quality, not tidiness.
"""

from __future__ import annotations

import pytest

from app.rag.extraction import count_words, normalize_text

pytestmark = pytest.mark.usefixtures("valid_env")


def test_collapses_runs_of_horizontal_whitespace() -> None:
    assert normalize_text("Refunds    are\t\tissued") == "Refunds are issued"


def test_collapses_more_than_one_blank_line() -> None:
    assert normalize_text("First\n\n\n\n\nSecond") == "First\n\nSecond"


def test_preserves_a_single_blank_line() -> None:
    """The paragraph break is the splitter's preferred separator — it must survive."""
    assert normalize_text("First\n\nSecond") == "First\n\nSecond"


def test_lines_of_only_whitespace_become_blank_lines() -> None:
    assert normalize_text("First\n   \n   \nSecond") == "First\n\nSecond"


def test_strips_trailing_whitespace_from_each_line() -> None:
    assert normalize_text("First   \nSecond   ") == "First\nSecond"


def test_removes_zero_width_characters() -> None:
    """A zero-width space inside a word is invisible but splits it for the tokenizer."""
    assert normalize_text("refund​policy") == "refundpolicy"


def test_removes_bidirectional_and_bom_characters() -> None:
    assert normalize_text("﻿Refunds‮are issued") == "Refundsare issued"


def test_normalizes_windows_and_classic_mac_line_endings() -> None:
    assert normalize_text("First\r\nSecond\rThird") == "First\nSecond\nThird"


def test_trims_leading_and_trailing_whitespace() -> None:
    assert normalize_text("\n\n  Refunds are issued.  \n\n") == "Refunds are issued."


def test_whitespace_only_input_normalizes_to_empty() -> None:
    """Extraction drops pages that normalize to nothing, so this is the signal it uses."""
    assert normalize_text("   \n\n \t \n  ") == ""


def test_word_count_is_whitespace_delimited() -> None:
    assert count_words("Refunds are issued within 30 days.") == 6


def test_word_count_of_empty_text_is_zero() -> None:
    assert count_words("") == 0
    assert count_words("   \n  ") == 0


def test_word_count_ignores_collapsed_whitespace() -> None:
    """Counting after normalization must not double-count a run of spaces."""
    assert count_words(normalize_text("one    two\n\n\nthree")) == 3
