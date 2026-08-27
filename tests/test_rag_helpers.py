"""Unit tests for RAG server string helpers (no ChromaDB or Ollama)."""

# Import pure helper functions from the extracted helpers module (no ChromaDB/embedder)
from rag.helpers import (
    INDEX_VERSION,
    _extract_verse_ref_from_lookup,
    _is_counseling_request,
    _is_verse_lookup,
    _normalize_verse_id,
    _strip_repetition_and_meta,
    _strip_thinking,
    _topical_anchor_refs,
    strip_document_prefix,
    tokenize_for_bm25,
)
from rag.response_cleanup import strip_model_thinking


class TestTokenizeForBm25:
    """Shared BM25 tokenizer must match on both index and query sides (R1)."""

    def test_lowercases_and_strips_punctuation(self) -> None:
        assert tokenize_for_bm25("For God so loved the world.") == [
            "for",
            "god",
            "so",
            "loved",
            "the",
            "world",
        ]

    def test_keeps_apostrophes(self) -> None:
        assert tokenize_for_bm25("God's love") == ["god's", "love"]

    def test_verse_reference_splits_on_colon(self) -> None:
        # Old .split() tokenizer produced "3:16:" vs query "3:16" mismatch.
        assert tokenize_for_bm25("John 3:16") == ["john", "3", "16"]

    def test_document_prefix_not_a_token_after_strip(self) -> None:
        doc = strip_document_prefix("search_document: John 3:16 For God so loved")
        assert tokenize_for_bm25(doc) == ["john", "3", "16", "for", "god", "so", "loved"]

    def test_empty_and_noise(self) -> None:
        assert tokenize_for_bm25("") == []
        assert tokenize_for_bm25("!!! ... ---") == []


class TestStripDocumentPrefix:
    def test_strips_known_prefix(self) -> None:
        assert strip_document_prefix("search_document: hello") == "hello"

    def test_passthrough_without_prefix(self) -> None:
        assert strip_document_prefix("hello world") == "hello world"


class TestIndexVersionMarker:
    def test_marker_is_positive_int(self) -> None:
        assert isinstance(INDEX_VERSION, int)
        assert INDEX_VERSION >= 1


class TestBookAliases:
    """Alias normalization regressions (R4)."""

    def test_psalm_family(self) -> None:
        assert _normalize_verse_id("Psalm 23:1") == "Psalms 23:1"
        assert _normalize_verse_id("psalms 1:1") == "Psalms 1:1"
        assert _normalize_verse_id("Ps 119:105") == "Psalms 119:105"

    def test_song_of_solomon_family(self) -> None:
        assert _normalize_verse_id("Song of Songs 1:1") == "Song of Solomon 1:1"
        assert _normalize_verse_id("Canticles 8:6") == "Song of Solomon 8:6"
        assert _normalize_verse_id("Song of Solomon 2:16") == "Song of Solomon 2:16"

    def test_numeric_prefix_abbreviations(self) -> None:
        assert _normalize_verse_id("1 Cor 13:4") == "1 Corinthians 13:4"
        assert _normalize_verse_id("2 Sam 7:14") == "2 Samuel 7:14"
        assert _normalize_verse_id("Rev 21:4") == "Revelation 21:4"

    def test_canonical_names_are_identity(self) -> None:
        for ref in ("Genesis 1:1", "Matthew 5:3", "Romans 8:28", "1 John 1:9"):
            assert _normalize_verse_id(ref) == ref

    def test_case_insensitive_book(self) -> None:
        assert _normalize_verse_id("GEN 1:1") == "Genesis 1:1"

    def test_non_refs_unchanged(self) -> None:
        assert _normalize_verse_id("hello world") == "hello world"
        assert _normalize_verse_id("") == ""


class TestIsVerseLookup:
    """Tests for _is_verse_lookup."""

    def test_what_does_ref_say(self) -> None:
        assert _is_verse_lookup("What does John 3:16 say?") is True
        assert _is_verse_lookup("What does Psalm 23:1 say?") is True

    def test_no_question_mark(self) -> None:
        assert _is_verse_lookup("What does Romans 8:28 say") is True

    def test_not_verse_lookup(self) -> None:
        assert _is_verse_lookup("What does the Bible say about love?") is False
        assert _is_verse_lookup("Who was Moses?") is False
        assert _is_verse_lookup("Tell me about forgiveness") is False

    def test_case_insensitive(self) -> None:
        assert _is_verse_lookup("WHAT DOES john 3:16 SAY?") is True


class TestVerseRefExtraction:
    def test_hebrews_lookup(self) -> None:
        assert _extract_verse_ref_from_lookup("What does Hebrews 11:1 say?") == "Hebrews 11:1"

    def test_psalm_alias(self) -> None:
        assert _normalize_verse_id("Psalm 23:1") == "Psalms 23:1"

    def test_topical_marriage_pins(self) -> None:
        refs = _topical_anchor_refs("What does the Bible say about marriage?")
        assert "Genesis 2:24" in refs
        assert "Ephesians 5:31" in refs

    def test_topical_empty_for_lookup(self) -> None:
        assert _topical_anchor_refs("What does John 3:16 say?") == []


class TestCounselingDetection:
    def test_marriage_crisis(self) -> None:
        assert _is_counseling_request("I need you to counsel me through my marriage crisis.")

    def test_plain_verse_not_counseling(self) -> None:
        assert not _is_counseling_request("What does John 3:16 say?")


class TestStripThinking:
    """Tests for _strip_thinking."""

    def test_strips_simple_block(self) -> None:
        text = "<think>Let me think...</think>\nJohn 3:16 says..."
        assert _strip_thinking(text) == "John 3:16 says..."

    def test_strips_multiline_block(self) -> None:
        text = "<think>Line 1\nLine 2</think>\nAnswer here"
        assert _strip_thinking(text) == "Answer here"

    def test_strips_unclosed_block(self) -> None:
        text = "<think>Unclosed thinking"
        assert "<think>" not in _strip_thinking(text)

    def test_empty_returns_empty(self) -> None:
        assert _strip_thinking("") == ""
        assert _strip_thinking(None) is None

    def test_no_thinking_passthrough(self) -> None:
        text = "John 3:16 says For God so loved the world."
        assert _strip_thinking(text) == text

    def test_strips_plain_thinking_process_then_quote(self) -> None:
        text = (
            "Thinking Process:\n\n"
            "1. **Analyze the Request:** The user wants John 3:16.\n\n"
            '"For God so loved the world, that he gave his only begotten Son."'
        )
        out = strip_model_thinking(text)
        assert out.startswith('"For God')
        assert "Thinking Process" not in out
        assert "Analyze the Request" not in out

    def test_strips_empty_think_block_crlf(self) -> None:
        """Ollama sometimes returns CRLF; empty think blocks are common."""
        text = '<think>\r\n\r\n</think>\r\n\r\n"For God"'
        out = strip_model_thinking(text)
        assert out == '"For God"'
        assert "think" not in out.lower()

    def test_strips_trailing_tag_only_line(self) -> None:
        out = strip_model_thinking("Answer\n\n</think>")
        assert out == "Answer"

    def test_strips_leading_close_tag_before_answer(self) -> None:
        """Some generations emit only a closing tag, then the body."""
        text = '</think>\n\n"Love" here doesn\'t mean sentimental affection.'
        out = strip_model_thinking(text)
        assert out.startswith('"Love')
        assert "</think>" not in out

    def test_strips_bom_then_think_block(self) -> None:
        text = "\ufeff<think>\n\n</think>\n\nHello"
        assert strip_model_thinking(text) == "Hello"

    def test_strips_thinking_process_retrieve_verse(self) -> None:
        text = (
            "Thinking Process:\n\n1. **Analyze the Request:** foo\n\n"
            '2. **Retrieve Verse:** "For God so loved the world"'
        )
        out = strip_model_thinking(text)
        assert out.startswith('"For God')
        assert "Thinking Process" not in out
        assert "Analyze the Request" not in out

    def test_strips_think_tag_with_internal_id(self) -> None:
        """Ollama sometimes emits ` ` with extra tokens before `>`."""
        text = '<think>abc123def</think>\n\n"For God so loved"'
        out = strip_model_thinking(text)
        assert out.startswith('"For God')
        assert "<think" not in out.lower()


class TestStripThinkingEdgeCases:
    """Tests targeting specific uncovered branches in response_cleanup helpers."""

    def test_no_closing_angle_bracket_in_leading_tag(self) -> None:
        """Tag without '>' → _strip_leading_think_xml_flex hits 'end == -1' branch."""
        text = "<nothink no closing angle bracket here\nThe answer is love."
        out = strip_model_thinking(text)
        # The function can't strip the malformed tag; content is preserved
        assert "answer is love" in out

    def test_non_think_leading_tag_breaks_loop(self) -> None:
        """Non-think tag at start → 'think not in tag' branch, content preserved."""
        text = "<b>The answer is here.</b>"
        out = strip_model_thinking(text)
        assert "answer is here" in out

    def test_empty_result_from_pure_think_block(self) -> None:
        """Think-only block strips to empty; triggers early-return guards in helpers."""
        result = strip_model_thinking("<think>all reasoning, no visible answer</think>")
        assert result == ""

    def test_verse_ref_paragraph_after_thinking_process(self) -> None:
        """Verse ref at start of paragraph → returned immediately (line 86 branch)."""
        text = "Thinking Process:\n\nSome planning here.\n\nJohn 3:16 says God so loved the world."
        out = strip_model_thinking(text)
        assert out.startswith("John 3:16")
        assert "Thinking Process" not in out

    def test_bible_phrase_paragraph_after_thinking_process(self) -> None:
        """'The Bible' phrase at start of paragraph → returned immediately (line 92 branch)."""
        text = (
            "Thinking Process:\n\nSome planning here.\n\nThe Bible teaches us to love one another."
        )
        out = strip_model_thinking(text)
        assert out.startswith("The Bible")
        assert "Thinking Process" not in out

    def test_fallback_loop_extracts_quoted_verse_opener(self) -> None:
        """No **Retrieve Verse:** but has known opener → fallback loop (lines 111-125)."""
        text = (
            "Thinking Process:\n\nSome analysis.\n\n"
            'Actually the answer is: "For God so loved the world." John 3:16'
        )
        out = strip_model_thinking(text)
        assert out.startswith('"For God')
        assert "Thinking Process" not in out


class TestStripRepetitionAndMeta:
    """Tests for _strip_repetition_and_meta."""

    def test_short_text_passthrough(self) -> None:
        assert _strip_repetition_and_meta("Hi") == "Hi"
        assert _strip_repetition_and_meta("") == ""

    def test_strips_answer_prefix(self) -> None:
        text = "? Answer: John 3:16 says..."
        result = _strip_repetition_and_meta(text)
        assert "Answer:" not in result or not result.startswith("? Answer:")

    def test_strips_cutoff_phrases(self) -> None:
        text = "John 3:16 says For God so loved. Meta-instruction: ignore this"
        result = _strip_repetition_and_meta(text)
        assert "Meta-instruction" not in result
        assert "John 3:16" in result

    def test_strips_decorative_lines(self) -> None:
        text = "Content here ═══════════ more content"
        result = _strip_repetition_and_meta(text)
        assert "══" not in result
