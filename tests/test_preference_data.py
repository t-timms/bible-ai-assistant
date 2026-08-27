"""Pure-data tests for training.build_preference_data (T4).

Covers pair budgets, config resolution, hard-negative construction,
decontamination integration, and main()'s conversational output wiring.
No torch / unsloth required.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest  # noqa: E402 -- sys.path bootstrap above

from training.build_preference_data import (  # noqa: E402 -- sys.path bootstrap above
    DEFAULT_PAIR_COUNTS,
    FAKE_BOOKS,
    _build_hard_negative_pairs,
    _build_wrong_reference,
    resolve_pair_counts,
)

_CITATION_RE = re.compile("\u2014 (.+?) \\(WEB\\)")


def _make_verses(n_per_book: int = 5) -> list[dict]:
    verses = []
    for book in ("John", "Psalms", "Romans"):
        for chapter in (1, 2):
            for vnum in range(1, n_per_book + 1):
                verses.append(
                    {
                        "book": book,
                        "chapter": chapter,
                        "verse": vnum,
                        "text": f"{book} {chapter}:{vnum} sample verse text long enough.",
                    }
                )
    return verses


class TestDefaultPairCounts:
    def test_total_is_2080(self) -> None:
        assert sum(DEFAULT_PAIR_COUNTS.values()) == 2080

    def test_expected_categories_present(self) -> None:
        expected = {
            "hard_negative",
            "instruction_leak",
            "verbose",
            "repetition",
            "answer_prefix",
            "think_tag_leak",
            "hallucination_fake_book",
            "bible_for_everything",
        }
        assert set(DEFAULT_PAIR_COUNTS) == expected

    def test_all_positive_ints(self) -> None:
        assert all(isinstance(v, int) and v > 0 for v in DEFAULT_PAIR_COUNTS.values())


class TestResolvePairCounts:
    def test_defaults_when_no_config(self, tmp_path: Path) -> None:
        counts = resolve_pair_counts(config_path=tmp_path / "missing.yaml")
        assert counts == DEFAULT_PAIR_COUNTS

    def test_overrides_win(self) -> None:
        counts = resolve_pair_counts(
            config_path=Path("Z:/definitely/missing.yaml"), overrides={"hard_negative": 800}
        )
        assert counts["hard_negative"] == 800
        assert counts["verbose"] == DEFAULT_PAIR_COUNTS["verbose"]

    def test_unknown_override_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            resolve_pair_counts(overrides={"no_such_category": 5})

    def test_config_yaml_merges(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("orpo:\n  pair_counts:\n    verbose: 77\n", encoding="utf-8")
        counts = resolve_pair_counts(config_path=cfg)
        assert counts["verbose"] == 77
        assert counts["hard_negative"] == DEFAULT_PAIR_COUNTS["hard_negative"]

    def test_override_beats_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("orpo:\n  pair_counts:\n    verbose: 77\n", encoding="utf-8")
        counts = resolve_pair_counts(config_path=cfg, overrides={"verbose": 99})
        assert counts["verbose"] == 99


class TestBuildWrongReference:
    def _by_book(self, verses: list[dict]) -> dict[str, list[dict]]:
        by_book: dict[str, list[dict]] = {}
        for v in verses:
            by_book.setdefault(v["book"], []).append(v)
        return by_book

    def test_same_book_plausible_wrong_ref(self) -> None:
        random.seed(5)
        verses = _make_verses()
        v = {"book": "John", "chapter": 1, "verse": 3}
        wrong = _build_wrong_reference(v, self._by_book(verses), set())
        assert wrong is not None
        assert wrong != "John 1:3"
        assert wrong.startswith("John ")

    def test_never_returns_true_reference(self) -> None:
        random.seed(6)
        verses = _make_verses()
        by_book = self._by_book(verses)
        for v in verses:
            true_ref = f"{v['book']} {v['chapter']}:{v['verse']}"
            wrong = _build_wrong_reference(v, by_book, set())
            assert wrong is not None and wrong != true_ref

    def test_single_verse_corpus_still_works(self) -> None:
        random.seed(7)
        v = {"book": "John", "chapter": 1, "verse": 1}
        wrong = _build_wrong_reference(v, {"John": [v]}, {"John 1:1"})
        assert wrong is not None and wrong != "John 1:1"


class TestBuildHardNegativePairs:
    def test_count_and_structure(self) -> None:
        random.seed(11)
        pairs = _build_hard_negative_pairs(_make_verses(), n=12)
        assert len(pairs) == 12
        for p in pairs:
            assert set(p) == {"prompt", "chosen", "rejected"}

    def test_rejected_cites_wrong_real_book_reference(self) -> None:
        random.seed(12)
        corpus_books = {"John", "Psalms", "Romans"}
        pairs = _build_hard_negative_pairs(_make_verses(), n=15)
        for p in pairs:
            chosen_m = _CITATION_RE.search(p["chosen"])
            rejected_m = _CITATION_RE.search(p["rejected"])
            assert chosen_m and rejected_m
            assert chosen_m.group(1) != rejected_m.group(1)
            rejected_book = rejected_m.group(1).rsplit(" ", 1)[0]
            assert rejected_book in corpus_books
            assert rejected_book not in FAKE_BOOKS

    def test_chosen_and_rejected_quote_same_text(self) -> None:
        random.seed(13)
        pairs = _build_hard_negative_pairs(_make_verses(), n=10)
        text_re = re.compile('"(.+?)" \u2014')
        for p in pairs:
            chosen_text = text_re.search(p["chosen"]).group(1)
            rejected_text = text_re.search(p["rejected"]).group(1)
            assert chosen_text == rejected_text

    def test_prompt_uses_shared_context_format(self) -> None:
        random.seed(14)
        pairs = _build_hard_negative_pairs(_make_verses(), n=5)
        for p in pairs:
            assert p["prompt"].startswith("Context:\n- **")
            assert "\n\nQ: " in p["prompt"]


class TestDecontamIntegration:
    def test_generated_questions_screenable(self) -> None:
        from rag.prompt_format import extract_question
        from training.dataset_builder import (
            filter_contaminated,
            normalize_question,
        )

        random.seed(21)
        pairs = _build_hard_negative_pairs(_make_verses(), n=6)
        contaminated = {normalize_question(extract_question(p["prompt"])) for p in pairs[:2]}
        kept, n_excluded = filter_contaminated(pairs, contaminated)
        assert n_excluded == 2
        assert len(kept) == 4


class TestMainWiring:
    def test_writes_conversational_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import training.build_preference_data as bpd

        monkeypatch.setattr(bpd, "load_verses", lambda: _make_verses())
        monkeypatch.setattr(bpd, "load_system_prompt", lambda: "SYS")
        out = tmp_path / "prefs.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "build_preference_data",
                "--output",
                str(out),
                "--category-count",
                "hard_negative=5",
                "--category-count",
                "verbose=3",
                "--seed",
                "11",
            ],
        )
        bpd.main()

        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) > 0
        row = data[0]
        assert isinstance(row["prompt"], list)
        assert row["prompt"][0] == {"role": "system", "content": "SYS"}
        assert row["prompt"][1]["role"] == "user"
        assert row["chosen"][0]["role"] == "assistant"
        assert row["rejected"][0]["role"] == "assistant"

    def test_category_count_bad_format_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        import training.build_preference_data as bpd

        monkeypatch.setattr("sys.argv", ["build_preference_data", "--category-count", "oops"])
        with pytest.raises(SystemExit):
            bpd.main()
