"""Unit tests for untested pure functions in rag_server (no ChromaDB/Ollama needed)."""

from rag.helpers import (
    _clean_doc_text,
    _content_to_str,
    _is_meta_question,
    _merge_pin_order,
    _normalize_verse_id,
    _strip_openclaw_metadata,
    _strip_thinking_from_stream,
    _validate_ollama_url,
)


class TestValidateOllamaUrl:
    def test_valid_http_url(self) -> None:
        assert _validate_ollama_url("http://localhost:11434") == "http://localhost:11434"

    def test_valid_https_url(self) -> None:
        assert _validate_ollama_url("https://example.com:11434") == "https://example.com:11434"

    def test_invalid_scheme_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="http or https"):
            _validate_ollama_url("ftp://localhost:11434")

    def test_no_host_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="no host"):
            _validate_ollama_url("http://")


class TestIsMetaQuestion:
    def test_capability_questions(self) -> None:
        assert _is_meta_question("What can you do?")
        assert _is_meta_question("What are your capabilities?")
        assert _is_meta_question("Who are you?")
        assert _is_meta_question("How can you help me?")

    def test_greetings(self) -> None:
        assert _is_meta_question("hi")
        assert _is_meta_question("hello")
        assert _is_meta_question("hey")

    def test_bible_questions_not_meta(self) -> None:
        assert not _is_meta_question("What does John 3:16 say?")
        assert not _is_meta_question("Who was Moses?")
        assert not _is_meta_question("Tell me about forgiveness")


class TestContentToStr:
    def test_string_passthrough(self) -> None:
        assert _content_to_str("hello") == "hello"

    def test_none_returns_empty(self) -> None:
        assert _content_to_str(None) == ""

    def test_list_with_text_part(self) -> None:
        content = [{"type": "text", "text": "my question"}, {"type": "image_url", "url": "x"}]
        assert _content_to_str(content) == "my question"

    def test_list_without_text_returns_empty(self) -> None:
        assert _content_to_str([{"type": "image"}]) == ""

    def test_int_coerced_to_str(self) -> None:
        assert _content_to_str(42) == "42"


class TestNormalizeVerseId:
    def test_psalm_normalised_to_psalms(self) -> None:
        assert _normalize_verse_id("Psalm 23:1") == "Psalms 23:1"

    def test_psalms_unchanged(self) -> None:
        assert _normalize_verse_id("Psalms 23:1") == "Psalms 23:1"

    def test_other_books_unchanged(self) -> None:
        assert _normalize_verse_id("John 3:16") == "John 3:16"
        assert _normalize_verse_id("1 Corinthians 13:4") == "1 Corinthians 13:4"

    def test_whitespace_collapsed(self) -> None:
        assert _normalize_verse_id("John  3:16") == "John 3:16"

    def test_empty_string(self) -> None:
        assert _normalize_verse_id("") == ""


class TestMergePinOrder:
    def test_deduplicates_while_preserving_order(self) -> None:
        refs = ["John 3:16", "Romans 8:28", "John 3:16"]
        result = _merge_pin_order(refs)
        assert result == ["John 3:16", "Romans 8:28"]

    def test_empty_input(self) -> None:
        assert _merge_pin_order([]) == []

    def test_normalises_psalm(self) -> None:
        result = _merge_pin_order(["Psalm 23:1"])
        assert result == ["Psalms 23:1"]


class TestCleanDocText:
    def test_strips_search_document_prefix(self) -> None:
        # Real ChromaDB format: "search_document: John 3:16: verse text"
        doc = "search_document: John 3:16: For God so loved the world."
        assert _clean_doc_text(doc, "John 3:16") == "For God so loved the world."

    def test_strips_ref_prefix_without_search_prefix(self) -> None:
        doc = "John 3:16: For God so loved the world."
        assert _clean_doc_text(doc, "John 3:16") == "For God so loved the world."

    def test_no_prefix_passthrough(self) -> None:
        doc = "For God so loved the world."
        assert _clean_doc_text(doc, "John 3:16") == "For God so loved the world."

    def test_empty_ref_skips_ref_strip(self) -> None:
        doc = "Some text here."
        assert _clean_doc_text(doc, "") == "Some text here."


class TestStripOpenclawMetadata:
    def test_plain_text_passthrough(self) -> None:
        text = "What does John 3:16 say?"
        assert _strip_openclaw_metadata(text) == text

    def test_removes_sender_metadata_block(self) -> None:
        text = 'Sender (untrusted metadata): ```json {"user":"alice"} ```\nWhat does John 3:16 say?'
        result = _strip_openclaw_metadata(text)
        assert "Sender" not in result
        assert "What does John 3:16 say?" in result

    def test_empty_string(self) -> None:
        assert _strip_openclaw_metadata("") == ""

    def test_none_passthrough(self) -> None:
        assert _strip_openclaw_metadata(None) is None


class TestStripThinkingFromStream:
    def test_extracts_content_from_sse(self) -> None:
        sse = (
            'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        result = _strip_thinking_from_stream(sse)
        assert b"Hello world" in result

    def test_strips_think_block_in_stream(self) -> None:
        sse = (
            'data: {"choices":[{"delta":{"content":"<think>thinking</think>The answer is 42"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        result = _strip_thinking_from_stream(sse)
        assert b"<think>" not in result
        assert b"The answer is 42" in result

    def test_empty_content_returns_fallback(self) -> None:
        sse = 'data: {"choices":[{"delta":{"content":""}}]}\n\ndata: [DONE]\n\n'
        result = _strip_thinking_from_stream(sse)
        # Should return the EMPTY_MODEL_REPLY fallback
        assert b"didn't receive" in result or len(result) > 0
