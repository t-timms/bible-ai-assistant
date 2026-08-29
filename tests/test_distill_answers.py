"""Unit tests for the v3 distillation harness + the book-name alias fix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.distill_answers import (
    _first_context_verse,
    _looks_like_list_dump,
    validate,
)
from training.train_grpo import build_verse_lookup

CORPUS = Path(__file__).resolve().parents[1] / "data" / "raw" / "bible_web.json"
_HAVE_CORPUS = CORPUS.exists()
needs_corpus = pytest.mark.skipif(not _HAVE_CORPUS, reason="data/raw/bible_web.json not present")


@pytest.fixture(scope="module")
def lookup():
    return build_verse_lookup(CORPUS)


@needs_corpus
def test_alias_psalm_singular_resolves(lookup):
    # corpus stores "Psalms"; training data + citations say "Psalm"
    assert lookup("Psalm 84:4") is not None
    assert lookup("Psalm 84:4") == lookup("Psalms 84:4")


@needs_corpus
def test_alias_song_of_songs_resolves(lookup):
    assert lookup("Song of Songs 2:1") is not None
    assert lookup("Song of Songs 2:1") == lookup("Song of Solomon 2:1")


@needs_corpus
def test_unknown_reference_is_none(lookup):
    assert lookup("Hezekiah 3:5") is None
    assert lookup("Psalm 999:1") is None


@needs_corpus
def test_validate_flags_unknown_reference(lookup):
    bad = "This idea appears in Hezekiah 3:5, which teaches patience."
    problems = validate(bad, lookup)
    assert any(p.startswith("unknown_reference:") for p in problems)


@needs_corpus
def test_validate_passes_clean_synthesized_answer(lookup):
    good = (
        "Scripture ties endurance to hope: Romans 5:3 says suffering produces "
        "perseverance, and perseverance character. The point is formation, not "
        "mere survival."
    )
    assert validate(good, lookup) == []


@needs_corpus
def test_validate_flags_list_dump(lookup):
    dump = "\n".join(
        [
            "Here are passages on wisdom:",
            "• Proverbs 1:7",
            "• Proverbs 2:6",
            "• Proverbs 3:13",
            "• Proverbs 4:7",
        ]
    )
    assert "list_dump" in validate(dump, lookup)


def test_looks_like_list_dump_threshold():
    assert _looks_like_list_dump("a\n- one\n- two\n- three\n- four")
    assert not _looks_like_list_dump("a\n- one\n- two")


def test_first_context_verse_parses_entry():
    ctx = "CONTEXT:\n- **Romans 8:28**: And we know that all things work together for good\n\nQUESTION: x"
    ref, text = _first_context_verse(ctx)
    assert ref == "Romans 8:28"
    assert text.startswith("And we know")


def test_thematic_questions_file_is_wellformed():
    p = Path(__file__).resolve().parents[1] / "training" / "v3_thematic_questions.json"
    data = json.loads(p.read_text())
    qs = data["questions"]
    assert len(qs) >= 40
    assert all(q["q"].strip().endswith("?") for q in qs)
    assert all(q["theme"] for q in qs)
