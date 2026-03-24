"""Unit tests for RAG server string helpers (no ChromaDB or Ollama)."""

# Import pure helper functions from the extracted helpers module (no ChromaDB/embedder)
from rag.helpers import (
    _extract_verse_ref_from_lookup,
    _is_counseling_request,
    _is_verse_lookup,
    _normalize_verse_id,
    _strip_repetition_and_meta,
    _strip_thinking,
    _topical_anchor_refs,
)
from rag.response_cleanup import strip_model_thinking


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
