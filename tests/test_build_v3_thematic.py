"""Offline unit tests for training/build_v3_thematic.py — the v3.1 thematic_qa
distillation-input builder (pure functions only; the RAG retrieval + async
build loop are exercised by an integration run, not here)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from training.build_v3_thematic import (
    _RE_CHAR,
    _RE_CTX,
    _RE_XREF,
    _fill,
    _passage_ref,
    _persona,
    _shape_of,
    _variants,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rng() -> random.Random:
    return random.Random(1234)


# ── _persona ─────────────────────────────────────────────────────────────


def test_persona_empty_prefix_returns_question_unchanged() -> None:
    # seed chosen so the first choice is one of the "" prefixes
    q = "Who was Moses?"
    outs = {_persona(q, random.Random(s)) for s in range(30)}
    assert q in outs  # at least one draw picks an empty prefix


def test_persona_lowercases_after_midsentence_prefix() -> None:
    # every non-empty prefix ends in " " or "— ", so the question's first letter
    # is lowercased and the question is appended verbatim after it
    for s in range(200):
        out = _persona("What is grace?", random.Random(s))
        if out != "What is grace?":
            assert out.endswith("what is grace?"), out
            return
    raise AssertionError("never drew a non-empty prefix")


def test_persona_keeps_leading_I() -> None:
    q = "I need the context of Psalm 23."
    for s in range(50):
        assert _persona(q, random.Random(s)).count("I need the context") == 1


# ── _shape_of ────────────────────────────────────────────────────────────


def test_shape_of_defaults_to_topical() -> None:
    assert _shape_of({"q": "What is faith?"}) == "topical"
    assert _shape_of({"q": "x", "shape": "character"}) == "character"
    assert _shape_of({"q": "x", "shape": None}) == "topical"


# ── regexes ──────────────────────────────────────────────────────────────


def test_regexes_extract_expected_groups() -> None:
    assert _RE_CHAR.search("Who was the Apostle Paul?").group(1) == "the Apostle Paul"
    assert _RE_CTX.search("What is the context of Psalm 23?").group(1) == "Psalm 23"
    m = _RE_XREF.search("How does Isaiah 53:5 relate to 1 Peter 2:24?")
    assert m.group(1) == "Isaiah 53:5"
    assert m.group(2) == "1 Peter 2:24"


# ── _fill ────────────────────────────────────────────────────────────────


def test_fill_character() -> None:
    stem = {"shape": "character", "q": "Who was Deborah?"}
    assert _fill("Tell me about {name}.", stem) == "Tell me about Deborah."


def test_fill_context() -> None:
    stem = {"shape": "context", "q": "What is the context of Romans 8:28?"}
    assert _fill("background of {ref}?", stem) == "background of Romans 8:28?"


def test_fill_cross_reference() -> None:
    stem = {"shape": "cross_reference", "q": "How does A 1:1 relate to B 2:2?"}
    assert _fill("{a} <-> {b}", stem) == "A 1:1 <-> B 2:2"


def test_fill_topical_lowercases_for_ql_slot() -> None:
    stem = {"q": "What does the Bible say about hope?"}
    assert _fill("According to Scripture, {ql}", stem) == (
        "According to Scripture, what does the Bible say about hope?"
    )
    assert _fill("{q}", stem) == "What does the Bible say about hope?"


def test_fill_returns_none_when_pattern_missing() -> None:
    assert _fill("{name}", {"shape": "character", "q": "Explain the Trinity."}) is None
    assert _fill("{ref}", {"shape": "context", "q": "What is grace?"}) is None
    assert _fill("{a}{b}", {"shape": "cross_reference", "q": "How are these linked?"}) is None


# ── _passage_ref ─────────────────────────────────────────────────────────


def test_passage_ref_chapter_only_gets_verse_1() -> None:
    assert _passage_ref("Psalm 23") == "Psalms 23:1"
    assert _passage_ref("Genesis 1") == "Genesis 1:1"
    assert _passage_ref("1 Corinthians 13") == "1 Corinthians 13:1"


def test_passage_ref_keeps_explicit_verse() -> None:
    assert _passage_ref("Romans 8:28") == "Romans 8:28"
    assert _passage_ref("Ephesians 2:8-9") == "Ephesians 2:8"  # range -> first verse


def test_passage_ref_psalm_alias_and_leading_the() -> None:
    assert _passage_ref("the Psalm 51") == "Psalms 51:1"


def test_passage_ref_named_sections_return_none() -> None:
    assert _passage_ref("the Sermon on the Mount") is None
    assert _passage_ref("the book of Job") is None
    assert _passage_ref("the Ten Commandments") is None


# ── _variants ────────────────────────────────────────────────────────────


def test_variants_are_unique_and_hit_target() -> None:
    stem = {"shape": "character", "q": "Who was Ruth?"}
    vs = _variants(stem, _rng())
    assert len(vs) == 25
    assert len({v.lower() for v in vs}) == len(vs)
    assert all("Ruth" in v for v in vs)


def test_variants_topical_stem() -> None:
    stem = {"q": "What does the Bible teach about worship?"}
    vs = _variants(stem, _rng())
    assert 1 <= len(vs) <= 22
    assert any("worship" in v.lower() for v in vs)


def test_variants_deterministic_under_seed() -> None:
    stem = {"shape": "cross_reference", "q": "How does Genesis 15:6 relate to Romans 4:3?"}
    assert _variants(stem, random.Random(7)) == _variants(stem, random.Random(7))


# ── the shipped stem file ────────────────────────────────────────────────


def test_stem_file_is_valid_and_shapes_parse() -> None:
    meta = json.loads((PROJECT_ROOT / "training" / "v3_thematic_questions.json").read_text())
    stems = meta["questions"]
    assert len(stems) >= 100
    for s in stems:
        shape = _shape_of(s)
        # every non-topical stem must yield a fillable base with its first template
        if shape == "character":
            assert _RE_CHAR.search(s["q"]), s["q"]
        elif shape == "context":
            assert _RE_CTX.search(s["q"]), s["q"]
        elif shape == "cross_reference":
            assert _RE_XREF.search(s["q"]), s["q"]
