"""Unit tests for rag/verification.py (pure functions, no ChromaDB/network)."""

from rag.verification import (
    CitationIssue,
    annotate_unverified_citations,
    extract_verse_refs,
    verify_citations,
)

_FIXTURE_VERSES = {
    "John 3:16": (
        "For God so loved the world, that he gave his one and only Son, that "
        "whoever believes in him should not perish, but have eternal life."
    ),
    "Psalms 23:1": "Yahweh is my shepherd; I shall lack nothing.",
    "Romans 8:28": (
        "We know that all things work together for good for those who love "
        "God, to those who are called according to his purpose."
    ),
}


def _lookup(ref: str) -> str | None:
    return _FIXTURE_VERSES.get(ref)


class TestExtractVerseRefs:
    def test_extracts_single_ref(self) -> None:
        assert extract_verse_refs("John 3:16 says this.") == ["John 3:16"]

    def test_extracts_multiple_refs(self) -> None:
        refs = extract_verse_refs("See John 3:16 and Romans 8:28 for more.")
        assert "John 3:16" in refs
        assert "Romans 8:28" in refs

    def test_filters_connective_false_positive(self) -> None:
        # "and Psalms 27:1" should not extract as book "and psalms"
        refs = extract_verse_refs("Read Psalm 23:1, and Psalms 27:1 too.")
        assert all(not r.lower().startswith("and ") for r in refs)

    def test_no_refs_in_plain_text(self) -> None:
        assert extract_verse_refs("The Bible teaches about love.") == []


class TestVerifyCitations:
    def test_real_verified_reference_no_issue(self) -> None:
        issues = verify_citations("John 3:16 speaks of God's love.", _lookup)
        assert issues == []

    def test_unknown_book_flagged(self) -> None:
        issues = verify_citations("As Hezekiah 3:5 says, be faithful.", _lookup)
        assert len(issues) == 1
        assert issues[0].ref == "Hezekiah 3:5"
        assert issues[0].reason == "unknown_reference"

    def test_real_book_fake_verse_number_flagged(self) -> None:
        # Real book (would pass a book-name-only check), fabricated verse number.
        issues = verify_citations("1 Corinthians 47:99 teaches patience.", _lookup)
        assert len(issues) == 1
        assert issues[0].reason == "unknown_reference"

    def test_accurate_quote_not_flagged_as_misquote(self) -> None:
        text = (
            'John 3:16 says: "For God so loved the world, that he gave his one '
            "and only Son, that whoever believes in him should not perish, but "
            'have eternal life."'
        )
        assert verify_citations(text, _lookup) == []

    def test_paraphrase_without_quotes_not_flagged(self) -> None:
        # No quoted span — paraphrase should never be scored as a misquote.
        text = "John 3:16 teaches that God's love for the world led him to give his Son."
        assert verify_citations(text, _lookup) == []

    def test_fabricated_quote_flagged_as_misquote(self) -> None:
        text = 'John 3:16 says: "The moon is made of cheese and everyone knows it."'
        issues = verify_citations(text, _lookup)
        assert len(issues) == 1
        assert issues[0].reason == "possible_misquote"

    def test_psalm_singular_alias_resolves(self) -> None:
        # Response cites "Psalm" (singular); fixture is keyed "Psalms 23:1".
        # verify_citations itself does not alias — that's the lookup's job
        # (see rag.retrieval._fetch_verses_by_refs / training.evaluate's
        # normalized lookup) — confirm the raw ref is passed through as-is.
        issues = verify_citations('Psalms 23:1 says: "Yahweh is my shepherd."', _lookup)
        assert issues == []

    def test_no_duplicate_issues_for_repeated_ref(self) -> None:
        text = "Hezekiah 3:5 says one thing. Hezekiah 3:5 says it again."
        issues = verify_citations(text, _lookup)
        assert len(issues) == 1

    def test_empty_text_no_issues(self) -> None:
        assert verify_citations("", _lookup) == []


class TestAnnotateUnverifiedCitations:
    def test_appends_marker_for_unknown_reference(self) -> None:
        text = "As Hezekiah 3:5 says, be faithful."
        issues = [CitationIssue(ref="Hezekiah 3:5", reason="unknown_reference")]
        out = annotate_unverified_citations(text, issues)
        assert "Hezekiah 3:5" in out
        assert "not found in indexed text" in out

    def test_does_not_annotate_misquote_issues(self) -> None:
        text = 'John 3:16 says: "wrong text here."'
        issues = [CitationIssue(ref="John 3:16", reason="possible_misquote")]
        out = annotate_unverified_citations(text, issues)
        assert out == text

    def test_no_issues_returns_text_unchanged(self) -> None:
        text = "John 3:16 speaks of God's love."
        assert annotate_unverified_citations(text, []) == text
