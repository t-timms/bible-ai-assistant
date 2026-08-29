"""Tests for the v2 dataset engine (training/build_dataset_v2.py).

All tests are offline: they use tiny synthetic corpora fixtures and never hit
the network. Focus: parser correctness, deterministic generation, contamination
integration, and manifest integrity.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from training import build_dataset_v2 as v2


@pytest.fixture(autouse=True)
def _seed():
    random.seed(v2.RANDOM_SEED)
    yield
    random.seed(v2.RANDOM_SEED)


def make_corpus(triples_texts: dict[tuple[str, int, int], str]) -> dict:
    """Synthetic 1-translation corpus in the shape generators expect."""
    return {"KJV": {"verses": dict(triples_texts)}}


SAMPLE_VERSES = {
    (
        "Genesis",
        1,
        i + 1,
    ): f"In the beginning day {i + 1} God created light and the evening was morning number {i + 1} indeed."
    for i in range(14)
}
SAMPLE_VERSES.update(
    {
        (
            "John",
            3,
            16,
        ): "For God so loved the world that he gave his only begotten Son that whosoever believeth in him should not perish but have everlasting life.",
        (
            "John",
            3,
            17,
        ): "For God sent not his Son into the world to condemn the world but that the world through him might be saved.",
        (
            "Romans",
            8,
            28,
        ): "And we know that all things work together for good to them that love God to them who are the called according to his purpose.",
        ("Psalms", 23, 1): "The LORD is my shepherd I shall not want.",
    }
)

SYSTEM_PROMPT = "You are a Bible AI Assistant."


class TestParsing:
    def test_parse_translation_text_basic(self):
        raw = "HEADER\nHEADER\nGenesis 1:1\tFirst verse.\nGenesis 1:2\tSecond verse.\n"
        verses = v2.parse_translation_text(raw)
        assert verses[("Genesis", 1, 1)] == "First verse."
        assert verses[("Genesis", 1, 2)] == "Second verse."

    def test_parse_skips_malformed_lines(self):
        raw = "H\nH\nGenesis 1:1\tgood\nnot a verse line\nGenesis x:y\tbad\n"
        assert len(v2.parse_translation_text(raw)) == 1

    def test_parse_normalizes_ordinal_books(self):
        raw = "H\nH\nFirst Samuel 15:22\tobedience text here\n"
        parsed = v2.parse_translation_text(raw)
        keys = list(parsed)
        assert book_norm(keys[0]) == ("1 samuel", 15, 22)


def book_norm(key):
    return v2.book_key(key)


class TestOsisMapping:
    def test_plain_book(self):
        assert v2.osis_to_ref("Gen.1.1") == ("Genesis", 1, 1)

    def test_numbered_book(self):
        assert v2.osis_to_ref("1Sam.15.22") == ("1 Samuel", 15, 22)
        assert v2.osis_to_ref("2Cor.5.17") == ("2 Corinthians", 5, 17)

    def test_invalid_returns_none(self):
        assert v2.osis_to_ref("Nonsense.9.9") is None
        assert v2.osis_to_ref("") is None

    def test_key_equivalence_across_parsers(self):
        # 'First Samuel' (TSV books) must match '1Sam' (OSIS) under book_key
        assert v2.book_key(("1 Samuel", 15, 22)) == v2.book_key(("First Samuel", 15, 22))


class TestClipSnippet:
    def test_short_passthrough(self):
        assert v2.clip_snippet("short verse") == "short verse"

    def test_long_truncates_on_word_boundary(self):
        long = " ".join(["word"] * 40)
        out = v2.clip_snippet(long, max_chars=60)
        assert out.endswith("...")
        assert not out[:-3].endswith("wordword")


class TestGenerators:
    def setup_method(self):
        self.corpus = make_corpus(SAMPLE_VERSES)

    def test_verse_recall_shape_and_citation_in_output(self):
        out = v2.gen_verse_recall(self.corpus, SYSTEM_PROMPT, n=10)
        assert out
        user = out[0]["messages"][1]["content"]
        assistant = out[0]["messages"][2]["content"]
        assert "Context:" in user  # production prompt contract
        assert "(KJV)" in assistant

    def test_generation_deterministic_from_same_seed(self):
        random.seed(v2.RANDOM_SEED)
        first = [
            json.dumps(e, sort_keys=True)
            for e in v2.gen_reverse_lookup(self.corpus, SYSTEM_PROMPT, 5)
        ]
        random.seed(v2.RANDOM_SEED)
        second = [
            json.dumps(e, sort_keys=True)
            for e in v2.gen_reverse_lookup(self.corpus, SYSTEM_PROMPT, 5)
        ]
        assert first == second

    def test_near_miss_labels_next_verse_correctly(self):
        random.seed(v2.RANDOM_SEED)
        out = v2.gen_near_miss_guard(self.corpus, SYSTEM_PROMPT, n=4)
        assert out, "corpus with 14-verse chapters must yield near-miss traps"
        assistant = out[0]["messages"][2]["content"]
        assert "Not quite" in assistant

    def test_passage_recall_cites_range(self):
        out = v2.gen_passage_recall(self.corpus, SYSTEM_PROMPT, n=3)
        assert out
        q = out[0]["messages"][1]["content"]
        assert "-" in q  # 'Gen-style' range citation present

    def test_topical_collections_hit_known_theme(self):
        out = v2.gen_topical_collections(self.corpus, SYSTEM_PROMPT, n=2)
        assert out  # 'light'/'love' etc. present in synthetic verses

    def test_empty_corpus_yields_nothing(self):
        assert v2.gen_cross_reference_chains([], {}, SYSTEM_PROMPT, 5) == []
        assert v2.gen_near_miss_guard({}, SYSTEM_PROMPT, 5) == []


class TestContaminationIntegration:
    def test_contaminated_question_filtered_via_v1_contract(self):
        from training.dataset_builder import (
            filter_contaminated,
            normalize_question,
        )

        ex = {
            "category": "verse_recall",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": SAMPLE_VERSES[("John", 3, 16)][:30]},
                {"role": "assistant", "content": "a"},
            ],
        }
        contaminated = {normalize_question(ex["messages"][1]["content"])}
        clean, removed = filter_contaminated([ex], contaminated)
        assert removed == 1
        assert clean == []


class TestManifestIntegrity:
    def test_finalize_writes_examples_and_manifest(self, tmp_path: Path):
        exs = [v2._msg(SYSTEM_PROMPT, f"question {i}?", f"answer {i}!") for i in range(2)]
        examples = {"verse_recall": exs}
        dest = tmp_path / "train_v2.json"
        manifest = v2.finalize(examples, dest)
        data = json.loads(dest.read_text(encoding="utf-8"))
        man = json.loads(dest.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        # train_v2.json is a flat array of chat examples (v1-compatible shape).
        assert isinstance(data, list)
        assert len(data) == 2
        assert all("messages" in ex and "category" in ex for ex in data)
        # Provenance lives entirely in the sidecar manifest.
        assert man["protocol_id"] == "bible_assistant_v2_train"
        assert man["total"] == 2
        assert "seed" in man and "counts_dropped_contamination_or_dupes" in man
        assert manifest == man


class TestPersona:
    def test_persona_sometimes_prefixes_and_keeps_grammar(self):
        random.seed(v2.RANDOM_SEED)
        outs = {v2._persona("What does John 3:16 say?") for _ in range(200)}
        # at least one bare and one prefixed variant appear
        assert "What does John 3:16 say?" in outs
        assert any(o != "What does John 3:16 say?" for o in outs)
        # a mid-sentence prefix lowercases the following word
        assert all("— What" not in o for o in outs)


class TestCleanTurns:
    def test_strips_think_blocks(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "content": "<think>add them</think>The answer is 4."},
        ]
        turns = v2._clean_turns(msgs)
        assert turns == [("Solve 2+2.", "The answer is 4.")]

    def test_multi_turn_preserved_system_dropped(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
        ]
        assert v2._clean_turns(msgs) == [("hi", "hello"), ("bye", "goodbye")]

    def test_empty_after_strip_is_unusable(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>only reasoning</think>"},
        ]
        assert v2._clean_turns(msgs) is None


class TestPastoralTriage:
    def test_mixes_all_three_behaviours(self):
        random.seed(v2.RANDOM_SEED)
        out = v2.gen_pastoral_triage(SYSTEM_PROMPT, n=30)
        assert len(out) == 30
        answers = " ".join(e["messages"][2]["content"] for e in out)
        assert "pastor" in answers.lower()  # an escalation answer surfaced
        assert "not a Bible verse" in answers  # a calibrated-abstention answer surfaced
        assert "differ" in answers  # a tradition-aware answer surfaced

    def test_full_pool_covers_crisis_escalation(self):
        out = v2.gen_pastoral_triage(SYSTEM_PROMPT, n=10_000)
        answers = " ".join(e["messages"][2]["content"] for e in out)
        assert "988" in answers  # the suicide-crisis escalation is in the pool

    def test_all_examples_well_formed(self):
        out = v2.gen_pastoral_triage(SYSTEM_PROMPT, n=50)
        for e in out:
            roles = [m["role"] for m in e["messages"]]
            assert roles == ["system", "user", "assistant"]
            assert all(m["content"].strip() for m in e["messages"])


class TestGroundedExegesis:
    def _cache(self, tmp_path: Path) -> Path:
        blob = {
            "meta": {"source": "MHC", "license": "CC0", "sha256": "abc"},
            "records": [
                {
                    "book": "John",
                    "chapter": 3,
                    "verse_start": 16,
                    "verse_end": 17,
                    "text": "This is a long stretch of grounded exposition about the love of "
                    "God shown in giving the Son, repeated enough to clear the 200-"
                    "character floor the generator enforces before it will emit an "
                    "example at all, and then some more for good measure.",
                }
            ],
        }
        p = tmp_path / "mhc_commentary.json"
        p.write_text(json.dumps(blob), encoding="utf-8")
        return p

    def test_emits_grounded_answer_with_commentary_in_context(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v2, "_MHC_CACHE", self._cache(tmp_path))
        out = v2.gen_grounded_exegesis(make_corpus(SAMPLE_VERSES), SYSTEM_PROMPT, n=2)
        assert out
        user = out[0]["messages"][1]["content"]
        assistant = out[0]["messages"][2]["content"]
        assert "Matthew Henry" in user  # commentary injected as context, matches RAG wrapper
        assert "John 3:16" in user
        assert "public domain" in assistant

    def test_missing_cache_is_soft_skip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v2, "_MHC_CACHE", tmp_path / "nope.json")
        assert v2.gen_grounded_exegesis(make_corpus(SAMPLE_VERSES), SYSTEM_PROMPT, n=2) == []
