"""Canonical RAG prompt-format builders shared by serving and training.

Single source of truth for the wrapped-question format used at inference
(`rag/rag_server.py`) and mirrored by SFT/ORPO example generation
(`training/dataset_builder.py`, `training/build_preference_data.py`).

Production format (must not drift):

    Context:
    - **John 3:16**: For God so loved...
    - **John 3:17**: ...

    Q: What does John 3:16 say?

Training data MUST use this exact wrapper so the fine-tune matches what the
RAG server injects (audit finding F-2/F-3: prior SFT data used a divergent
"Relevant Bible verses:" format that never occurs at inference).
"""

from __future__ import annotations

from collections.abc import Iterable

from rag.helpers import _clean_doc_text

__all__ = [
    "CONTEXT_HEADER",
    "QUESTION_SEPARATOR",
    "format_context_entry",
    "build_context_block",
    "augment_question",
    "extract_question",
]

CONTEXT_HEADER = "Context:"
QUESTION_SEPARATOR = "\n\nQ: "


def format_context_entry(ref: str, text: str) -> str:
    """Render one retrieved verse as ``- **<ref>**: <text>``."""
    return f"- **{ref}**: {_clean_doc_text(text, ref)}"


def build_context_block(entries: Iterable[tuple[str, str]]) -> str:
    """Build the full ``Context:\\n...`` block from ``(ref, text)`` pairs."""
    lines = [format_context_entry(ref, text) for ref, text in entries]
    return CONTEXT_HEADER + "\n" + "\n".join(lines)


def augment_question(question: str, entries: Iterable[tuple[str, str]]) -> str:
    """Wrap a raw user question with the retrieval context block."""
    return build_context_block(entries) + QUESTION_SEPARATOR + question


def extract_question(augmented: str) -> str:
    """Recover the bare user question from an augmented message.

    Used to strip stale ``Context:`` blocks out of prior conversation turns so
    multi-turn history does not accumulate dead context (audit finding F-13).
    Falls back to the original text when no separator is present.
    """
    idx = augmented.rfind(QUESTION_SEPARATOR)
    if idx == -1:
        return augmented
    return augmented[idx + len(QUESTION_SEPARATOR) :]
