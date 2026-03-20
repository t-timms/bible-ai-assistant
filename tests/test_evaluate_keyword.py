"""Unit tests for evaluate.py keyword scoring logic (no network)."""
import sys
from pathlib import Path

# Add project root so we can import training.evaluate
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate import (
    BIBLE_BOOKS,
    check_hallucination,
    check_verse_accuracy,
    has_citation,
)


class TestHasCitation:
    """Tests for has_citation()."""

    def test_returns_true_when_verse_ref_present(self) -> None:
        assert has_citation("John 3:16 says that God so loved the world.")
        assert has_citation("As Romans 8:28 indicates, all things work together.")
        assert has_citation("1 Corinthians 13:4 describes love as patient.")

    def test_returns_false_when_no_ref(self) -> None:
        assert not has_citation("The Bible teaches about love.")
        assert not has_citation("")
        assert not has_citation("Genesis chapter 1")


class TestCheckVerseAccuracy:
    """Tests for check_verse_accuracy()."""

    def test_full_match_returns_one(self) -> None:
        expected = "For God so loved the world, that he gave his only Son."
        response = "For God so loved the world, that he gave his only Son. John 3:16."
        assert check_verse_accuracy(response, expected) == 1.0

    def test_partial_match_returns_fraction(self) -> None:
        expected = "First phrase. Second phrase here. Third phrase."
        response = "First phrase and Second phrase here."
        # Key phrases (len>10): "Second phrase here", "Third phrase"; "First phrase" too short
        # Actually expected.split(".") -> ["First phrase", " Second phrase here", " Third phrase", ""]
        # len > 10: " Second phrase here".strip() = "Second phrase here", " Third phrase".strip() = "Third phrase"
        # So key_phrases = [" second phrase here", " third phrase"] after strip/lower - wait, split gives " Second phrase here"
        # strip() makes "Second phrase here" (11 chars), "Third phrase" (12 chars). "First phrase" is 12 chars.
        # len(p.strip()) > 10: "First phrase" = 12, " Second phrase here" = 18, " Third phrase" = 12
        # So key_phrases = ["first phrase", "second phrase here", "third phrase"]
        # response has "first phrase" and "second phrase here" -> 2/3 = 0.666...
        result = check_verse_accuracy(response, expected)
        assert 0 < result < 1

    def test_empty_expected_returns_zero(self) -> None:
        assert check_verse_accuracy("John 3:16 says...", "") == 0.0

    def test_no_overlap_returns_zero(self) -> None:
        expected = "Completely different phrase about something else."
        response = "John 3:16 is about love."
        assert check_verse_accuracy(response, expected) == 0.0


class TestCheckHallucination:
    """Tests for check_hallucination()."""

    def test_real_book_not_hallucination(self) -> None:
        assert not check_hallucination("John 3:16 says God loved the world.")
        assert not check_hallucination("Romans 8:28 and 1 Corinthians 13:4.")
        assert not check_hallucination("Psalm 23:1 and Psalms 27:1.")
        assert not check_hallucination("Psalm 23:1. The Lord is my shepherd.")

    def test_fake_book_is_hallucination(self) -> None:
        assert check_hallucination("As Fakebook 1:2 says, this is made up.")
        assert check_hallucination("According to Invalid 5:10...")

    def test_no_ref_not_hallucination(self) -> None:
        assert not check_hallucination("The Bible teaches love.")
        assert not check_hallucination("")


class TestBibleBooks:
    """Sanity check BIBLE_BOOKS constant."""

    def test_contains_major_books(self) -> None:
        assert "genesis" in BIBLE_BOOKS
        assert "john" in BIBLE_BOOKS
        assert "revelation" in BIBLE_BOOKS
        assert "1 corinthians" in BIBLE_BOOKS
        assert "psalm" in BIBLE_BOOKS or "psalms" in BIBLE_BOOKS
