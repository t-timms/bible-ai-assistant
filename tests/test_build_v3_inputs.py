"""Offline unit tests for training/build_v3_inputs.py — the v3 distillation
input builder (keeps only context + question; the teacher regenerates answers)."""

from __future__ import annotations

from training.build_v3_inputs import REGEN, split_context_question


def test_split_basic() -> None:
    user = (
        "Context:\n- **Romans 8:28**: And we know that all things work\n\nQ: What does this teach?"
    )
    context, question = split_context_question(user)
    assert context == "- **Romans 8:28**: And we know that all things work"
    assert question == "What does this teach?"


def test_split_strips_context_header_variants() -> None:
    for header in ("Context:\n", "Context: "):
        context, question = split_context_question(f"{header}body line\n\nQ: q?")
        assert context == "body line"
        assert question == "q?"


def test_split_no_separator_returns_empty_context() -> None:
    context, question = split_context_question("just a bare question with no marker")
    assert context == ""
    assert question == "just a bare question with no marker"


def test_split_uses_last_separator() -> None:
    # a question that itself contains "Q: " must still split on the final marker
    user = "Context:\nverse\n\nQ: In Q: form, what is meant?"
    context, question = split_context_question(user)
    assert context == "verse"
    assert question == "In Q: form, what is meant?"


def test_regen_targets_the_four_templated_categories() -> None:
    assert set(REGEN) == {
        "topical_collections",
        "cross_reference_chains",
        "chapter_context",
        "grounded_exegesis",
    }
    assert all(isinstance(n, int) and n > 0 for n in REGEN.values())
