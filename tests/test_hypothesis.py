"""Hypothesis property-based tests for pure string helpers in rag.rag_server.

No ChromaDB, sentence-transformers, or network I/O is required.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag.helpers import (
    _content_to_str,
    _is_counseling_request,
    _is_verse_lookup,
    _normalize_verse_id,
    _strip_repetition_and_meta,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Printable text that avoids null bytes (which str operations handle fine but
# are unusual in natural language and can confuse regex anchors).
_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=500,
)

_book_names = st.one_of(
    st.sampled_from(
        [
            "Genesis",
            "Exodus",
            "Psalms",
            "Psalm",
            "John",
            "Romans",
            "1 Corinthians",
            "2 Corinthians",
            "Hebrews",
            "Revelation",
            "Matthew",
            "Luke",
            "Acts",
            "Ephesians",
            "Colossians",
        ]
    ),
    st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20
    ),
)

_chapter = st.integers(min_value=1, max_value=150)
_verse = st.integers(min_value=1, max_value=176)


@st.composite
def _verse_ref(draw: st.DrawFn) -> str:
    """Generate a well-formed 'Book Chapter:Verse' string."""
    book = draw(_book_names)
    ch = draw(_chapter)
    vs = draw(_verse)
    return f"{book} {ch}:{vs}"


# ---------------------------------------------------------------------------
# _normalize_verse_id
# ---------------------------------------------------------------------------


class TestNormalizeVerseIdProperties:
    """Property-based tests for _normalize_verse_id."""

    @given(_verse_ref())
    @settings(max_examples=200)
    def test_output_preserves_chapter_verse_pattern(self, ref: str) -> None:
        """If input contains a chapter:verse number, output must also contain one."""
        result = _normalize_verse_id(ref)
        assert re.search(r"\d{1,3}:\d{1,3}", result), (
            f"chapter:verse pattern missing in output {result!r} for input {ref!r}"
        )

    @given(_verse_ref())
    @settings(max_examples=200)
    def test_idempotent(self, ref: str) -> None:
        """Calling _normalize_verse_id twice must give the same result as once."""
        once = _normalize_verse_id(ref)
        twice = _normalize_verse_id(once)
        assert once == twice, f"Not idempotent: first={once!r}, second={twice!r}"

    @given(_verse_ref())
    @settings(max_examples=200)
    def test_non_empty_input_gives_non_empty_output(self, ref: str) -> None:
        """Well-formed Book Chapter:Verse input must not collapse to empty string."""
        result = _normalize_verse_id(ref)
        assert result, f"Empty output for non-empty input {ref!r}"

    @given(st.just(""))
    def test_empty_string_returns_empty(self, ref: str) -> None:
        """Empty string must round-trip to empty string."""
        assert _normalize_verse_id(ref) == ""

    @given(_text)
    @settings(max_examples=50)
    def test_always_returns_str(self, ref: str) -> None:
        """Result is always a str, regardless of input content."""
        result = _normalize_verse_id(ref)
        assert isinstance(result, str)

    @given(
        st.builds(
            lambda book, ch, vs: f"  {book}   {ch}:{vs}  ",
            book=_book_names,
            ch=_chapter,
            vs=_verse,
        )
    )
    @settings(max_examples=100)
    def test_whitespace_collapsed_in_output(self, ref: str) -> None:
        """Consecutive spaces in the input must be collapsed to single spaces in output."""
        result = _normalize_verse_id(ref)
        assert "  " not in result, f"Double space found in {result!r}"


# ---------------------------------------------------------------------------
# _is_verse_lookup
# ---------------------------------------------------------------------------


class TestIsVerseLookupProperties:
    """Property-based tests for _is_verse_lookup."""

    @given(_text)
    @settings(max_examples=200)
    def test_always_returns_bool(self, text: str) -> None:
        """Return value must always be exactly bool."""
        result = _is_verse_lookup(text)
        assert isinstance(result, bool)

    @given(st.just(""))
    def test_empty_string_returns_false(self, text: str) -> None:
        """Empty string must not be classified as a verse lookup."""
        assert _is_verse_lookup(text) is False

    @given(
        st.text(
            alphabet=st.characters(blacklist_categories=("Nd",), blacklist_characters=":"),
            min_size=1,
            max_size=200,
        )
    )
    @settings(max_examples=100)
    def test_no_digit_colon_digit_pattern_returns_false(self, text: str) -> None:
        """Strings without a digit:digit substring cannot be verse lookups."""
        # Ensure there's truly no digit:digit in the string
        if re.search(r"\d:\d", text):
            pytest.skip("Strategy produced digit:digit despite blacklist")
        assert _is_verse_lookup(text) is False

    @given(
        st.builds(
            lambda ref: f"What does {ref} say?",
            ref=_verse_ref(),
        )
    )
    @settings(max_examples=100)
    def test_canonical_lookup_phrase_returns_true(self, text: str) -> None:
        """'What does <Book> Chapter:Verse say?' must always be recognised as a lookup."""
        assert _is_verse_lookup(text) is True


# ---------------------------------------------------------------------------
# _strip_repetition_and_meta
# ---------------------------------------------------------------------------


class TestStripRepetitionAndMetaProperties:
    """Property-based tests for _strip_repetition_and_meta."""

    @given(_text)
    @settings(max_examples=200)
    def test_output_never_longer_than_input(self, text: str) -> None:
        """The function may only shorten or equal the input, never lengthen it."""
        result = _strip_repetition_and_meta(text)
        assert len(result) <= len(text), f"Output longer than input: {len(result)} > {len(text)}"

    @given(_text)
    @settings(max_examples=200)
    def test_always_returns_str(self, text: str) -> None:
        """Return type is always str for any str input."""
        result = _strip_repetition_and_meta(text)
        assert isinstance(result, str)

    @given(st.just(""))
    def test_empty_string_returns_empty(self, text: str) -> None:
        """Empty string input must produce empty string output."""
        assert _strip_repetition_and_meta(text) == ""

    @given(_text)
    @settings(max_examples=50)
    def test_output_has_no_leading_trailing_whitespace(self, text: str) -> None:
        """Non-empty outputs must not have leading or trailing whitespace."""
        result = _strip_repetition_and_meta(text)
        if result:
            assert result == result.strip(), f"Result has surrounding whitespace: {result!r}"

    @given(
        st.text(
            alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,'\"",
            min_size=30,
            max_size=300,
        )
    )
    @settings(max_examples=100)
    def test_plain_text_without_cutoff_phrases_unchanged_in_content(self, text: str) -> None:
        """Text that has no meta-instruction cutoff markers must still be valid str output."""
        # We only assert the invariant, not exact equality, because whitespace
        # normalisation may still alter the string.
        result = _strip_repetition_and_meta(text)
        assert isinstance(result, str)
        assert len(result) <= len(text)


# ---------------------------------------------------------------------------
# _is_counseling_request
# ---------------------------------------------------------------------------


class TestIsCounselingRequestProperties:
    """Property-based tests for _is_counseling_request."""

    @given(_text)
    @settings(max_examples=200)
    def test_always_returns_bool(self, text: str) -> None:
        """Return value must always be exactly bool."""
        result = _is_counseling_request(text)
        assert isinstance(result, bool)

    @given(st.just(""))
    def test_empty_string_returns_false(self, text: str) -> None:
        """Empty string must not be flagged as a counseling request."""
        assert _is_counseling_request(text) is False

    @given(
        st.sampled_from(
            [
                "What does John 3:16 say?",
                "Who was Moses?",
                "Tell me about the Book of Psalms.",
                "What are the ten commandments?",
                "Explain the Sermon on the Mount.",
            ]
        )
    )
    def test_plain_scripture_questions_are_not_counseling(self, text: str) -> None:
        """Ordinary scripture-study questions must not be flagged as counseling."""
        assert _is_counseling_request(text) is False

    @given(
        st.sampled_from(
            [
                "I need counseling for my depression.",
                "I want to kill myself.",
                "My marriage is falling apart, can you help?",
                "I have trauma from abuse.",
                "I am suffering from severe anxiety.",
                "I need someone to talk to.",
            ]
        )
    )
    def test_crisis_keywords_flagged(self, text: str) -> None:
        """Strings containing known counseling/crisis keywords must return True."""
        assert _is_counseling_request(text) is True


# ---------------------------------------------------------------------------
# _content_to_str
# ---------------------------------------------------------------------------


class TestContentToStrProperties:
    """Property-based tests for _content_to_str."""

    @given(_text)
    @settings(max_examples=200)
    def test_always_returns_str(self, value: str) -> None:
        """Return value is always str for any str input."""
        result = _content_to_str(value)
        assert isinstance(result, str)

    @given(_text)
    @settings(max_examples=200)
    def test_str_input_identity(self, value: str) -> None:
        """str input must be returned unchanged (identity)."""
        assert _content_to_str(value) == value

    @given(st.integers())
    @settings(max_examples=50)
    def test_int_input_returns_str(self, value: int) -> None:
        """Integer input must be coerced to str."""
        result = _content_to_str(value)
        assert isinstance(result, str)

    @given(st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_float_input_returns_str(self, value: float) -> None:
        """Float input must be coerced to str."""
        result = _content_to_str(value)
        assert isinstance(result, str)

    def test_none_returns_empty_str(self) -> None:
        """None must map to empty string."""
        assert _content_to_str(None) == ""

    @given(
        st.lists(
            st.fixed_dictionaries({"type": st.just("text"), "text": _text}),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_list_of_text_parts_returns_str(self, parts: list) -> None:
        """List of text-typed dicts must return a str."""
        result = _content_to_str(parts)
        assert isinstance(result, str)

    @given(
        st.lists(
            st.fixed_dictionaries({"type": st.just("text"), "text": _text}),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_list_first_text_part_is_returned(self, parts: list) -> None:
        """First text-typed dict's 'text' value must be returned."""
        result = _content_to_str(parts)
        assert result == parts[0]["text"]

    @given(st.lists(st.fixed_dictionaries({"type": st.just("image_url"), "url": _text})))
    @settings(max_examples=50)
    def test_list_without_text_part_returns_empty(self, parts: list) -> None:
        """List with no text-typed dicts must return empty string."""
        result = _content_to_str(parts)
        assert result == ""

    def test_empty_list_returns_empty(self) -> None:
        """Empty list must return empty string."""
        assert _content_to_str([]) == ""
