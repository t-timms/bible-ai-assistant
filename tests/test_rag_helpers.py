"""Unit tests for RAG server string helpers (no ChromaDB or Ollama)."""
# Import only the pure helper functions; avoids loading ChromaDB/embedder
from rag.rag_server import (
    _is_verse_lookup,
    _strip_repetition_and_meta,
    _strip_thinking,
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
        text = "<think>\r\n\r\n</think>\r\n\r\n\"For God\""
        out = strip_model_thinking(text)
        assert out == '"For God"'
        assert "think" not in out.lower()

    def test_strips_trailing_tag_only_line(self) -> None:
        out = strip_model_thinking("Answer\n\n</think>")
        assert out == "Answer"

    def test_strips_leading_close_tag_before_answer(self) -> None:
        """Some generations emit only a closing tag, then the body."""
        text = "</think>\n\n\"Love\" here doesn't mean sentimental affection."
        out = strip_model_thinking(text)
        assert out.startswith('"Love')
        assert "</think>" not in out

    def test_strips_bom_then_think_block(self) -> None:
        text = "\ufeff<think>\n\n</think>\n\nHello"
        assert strip_model_thinking(text) == "Hello"

    def test_strips_thinking_process_retrieve_verse(self) -> None:
        text = (
            "Thinking Process:\n\n1. **Analyze the Request:** foo\n\n"
            "2. **Retrieve Verse:** \"For God so loved the world\""
        )
        out = strip_model_thinking(text)
        assert out.startswith('"For God')
        assert "Thinking Process" not in out
        assert "Analyze the Request" not in out

    def test_strips_think_tag_with_internal_id(self) -> None:
        """Ollama sometimes emits ` ` with extra tokens before `>`."""
        text = "<think>abc123def</think>\n\n\"For God so loved\""
        out = strip_model_thinking(text)
        assert out.startswith('"For God')
        assert "<think" not in out.lower()


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
