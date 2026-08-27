"""Tests for rag/prompt_format.py — canonical RAG wrapper shared by serving and training."""

from rag.prompt_format import (
    CONTEXT_HEADER,
    QUESTION_SEPARATOR,
    augment_question,
    build_context_block,
    extract_question,
    format_context_entry,
)


class TestFormatContextEntry:
    def test_renders_bullet_with_bold_ref(self) -> None:
        assert (
            format_context_entry("John 3:16", "For God so loved")
            == "- **John 3:16**: For God so loved"
        )

    def test_strips_search_document_prefix(self) -> None:
        assert (
            format_context_entry("John 3:16", "search_document: For God so loved")
            == "- **John 3:16**: For God so loved"
        )


class TestBuildContextBlock:
    def test_header_then_one_line_per_entry(self) -> None:
        entries = [("John 3:16", "a"), ("Romans 8:28", "b")]
        assert build_context_block(entries) == (
            "Context:\n- **John 3:16**: a\n- **Romans 8:28**: b"
        )

    def test_empty_entries_yields_header_and_trailing_newline(self) -> None:
        assert build_context_block([]) == CONTEXT_HEADER + "\n"


class TestAugmentQuestion:
    def test_production_format_matches_module_contract(self) -> None:
        out = augment_question("What does John 3:16 say?", [("John 3:16", "For God so loved...")])
        assert out == (
            "Context:\n- **John 3:16**: For God so loved...\n\nQ: What does John 3:16 say?"
        )

    def test_constants_are_the_single_source_of_truth(self) -> None:
        assert CONTEXT_HEADER == "Context:"
        assert QUESTION_SEPARATOR == "\n\nQ: "


class TestExtractQuestion:
    def test_recovers_bare_question_from_augmented_text(self) -> None:
        augmented = "Context:\n- **John 3:16**: text\n\nQ: What does John 3:16 say?"
        assert extract_question(augmented) == "What does John 3:16 say?"

    def test_falls_back_to_original_without_separator(self) -> None:
        assert extract_question("plain question") == "plain question"

    def test_prefers_last_separator_for_nested_questions(self) -> None:
        augmented = "Context:\n- **A 1:1**: x\n\nQ: note\n\nQ: real question"
        assert extract_question(augmented) == "real question"

    def test_round_trip_is_identity(self) -> None:
        question = "Who was Moses?"
        entries = [("Exodus 2:1", "text")]
        assert extract_question(augment_question(question, entries)) == question
