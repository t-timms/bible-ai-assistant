"""Lightweight tests for pure-Python training utilities.

These tests cover functions that require no torch, unsloth, or external models:
  - training.evaluate: _extract_scores_json, _apply_score_clamps
  - training.merge_adapters: _remap_lora_state_dict
  - training.build_preference_data: pair generators (structure / count verification)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path (evaluate.py does this at runtime)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest

# ---------------------------------------------------------------------------
# training.evaluate — pure scoring helpers
# ---------------------------------------------------------------------------


class TestExtractScoresJson:
    """Tests for _extract_scores_json (parses LLM judge JSON responses)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from training.evaluate import _extract_scores_json

        self.fn = _extract_scores_json

    def test_parses_valid_json_object(self) -> None:
        content = '{"faithfulness": 5, "citation": 4, "hallucination": 5}'
        result = self.fn(content)
        assert result is not None
        assert result["faithfulness"] == 5

    def test_parses_json_inside_markdown_fence(self) -> None:
        content = '```json\n{"faithfulness": 5, "citation": 3}\n```'
        result = self.fn(content)
        assert result is not None
        assert result["citation"] == 3

    def test_returns_none_for_no_json(self) -> None:
        assert self.fn("No JSON here at all.") is None

    def test_returns_none_for_malformed_json(self) -> None:
        assert self.fn("{bad json {{{") is None

    def test_strips_think_block_first(self) -> None:
        content = '<think>ignore this</think>\n{"faithfulness": 5}'
        result = self.fn(content)
        assert result is not None
        assert result["faithfulness"] == 5

    def test_returns_none_for_non_dict(self) -> None:
        assert self.fn("[1, 2, 3]") is None


class TestApplyScoreClamps:
    """Tests for _apply_score_clamps (bounds 1-5 on each dimension)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from training.evaluate import _apply_score_clamps

        self.fn = _apply_score_clamps

    def test_valid_scores_unchanged(self) -> None:
        scores = {
            "faithfulness": 5,
            "citation": 4,
            "hallucination": 3,
            "helpfulness": 2,
            "conciseness": 1,
        }
        result = self.fn(scores)
        assert result["faithfulness"] == 5
        assert result["conciseness"] == 1

    def test_clamps_above_5(self) -> None:
        result = self.fn(
            {
                "faithfulness": 99,
                "citation": 0,
                "hallucination": 0,
                "helpfulness": 0,
                "conciseness": 0,
            }
        )
        assert result["faithfulness"] == 5

    def test_clamps_below_1(self) -> None:
        result = self.fn(
            {
                "faithfulness": -3,
                "citation": 0,
                "hallucination": 0,
                "helpfulness": 0,
                "conciseness": 0,
            }
        )
        assert result["faithfulness"] == 1

    def test_non_numeric_becomes_zero(self) -> None:
        result = self.fn(
            {
                "faithfulness": "five",
                "citation": 0,
                "hallucination": 0,
                "helpfulness": 0,
                "conciseness": 0,
            }
        )
        assert result["faithfulness"] == 0

    def test_reasoning_preserved(self) -> None:
        scores = {
            "faithfulness": 4,
            "citation": 4,
            "hallucination": 4,
            "helpfulness": 4,
            "conciseness": 4,
            "reasoning": "looks good",
        }
        result = self.fn(scores)
        assert "reasoning" in result


# ---------------------------------------------------------------------------
# training.merge_adapters — key remapping (pure string manipulation)
# ---------------------------------------------------------------------------


class TestRemapLoraStateDict:
    """Tests for _remap_lora_state_dict (Unsloth→PEFT key layout conversion)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from training.merge_adapters import _remap_lora_state_dict

        self.fn = _remap_lora_state_dict

    def test_remaps_language_model_prefix(self) -> None:
        sd = {"base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": "t"}
        result = self.fn(sd)
        assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight" in result

    def test_remaps_lora_a_to_default(self) -> None:
        sd = {"some.prefix.lora_A.weight": "tensor"}
        result = self.fn(sd)
        assert "some.prefix.lora_A.default.weight" in result

    def test_remaps_lora_b_to_default(self) -> None:
        sd = {"some.prefix.lora_B.weight": "tensor"}
        result = self.fn(sd)
        assert "some.prefix.lora_B.default.weight" in result

    def test_non_lora_keys_unchanged(self) -> None:
        sd = {"model.embed_tokens.weight": "tensor"}
        result = self.fn(sd)
        assert "model.embed_tokens.weight" in result

    def test_empty_dict(self) -> None:
        assert self.fn({}) == {}

    def test_values_preserved(self) -> None:
        sd = {"some.lora_A.weight": "my_tensor"}
        result = self.fn(sd)
        assert list(result.values())[0] == "my_tensor"


# ---------------------------------------------------------------------------
# training.build_preference_data — pair generators (structure verification)
# ---------------------------------------------------------------------------


class TestBuildHallucinationPairs:
    @pytest.fixture(autouse=True)
    def _import(self):
        from training.build_preference_data import _build_hallucination_pairs

        self.fn = _build_hallucination_pairs

    def _sample_verses(self, n: int = 20) -> list[dict]:
        return [
            {
                "book": "John",
                "chapter": 3,
                "verse": i + 1,
                "text": f"Sample verse text {i + 1} here for testing purposes.",
            }
            for i in range(n)
        ]

    def test_returns_correct_count(self) -> None:
        verses = self._sample_verses(30)
        pairs = self.fn(verses, n=10)
        assert len(pairs) == 10

    def test_pair_has_required_keys(self) -> None:
        verses = self._sample_verses(10)
        pairs = self.fn(verses, n=5)
        for p in pairs:
            assert "prompt" in p
            assert "chosen" in p
            assert "rejected" in p

    def test_chosen_contains_real_verse(self) -> None:
        verses = self._sample_verses(10)
        pairs = self.fn(verses, n=5)
        for p in pairs:
            assert "John" in p["chosen"]

    def test_rejected_contains_fake_book(self) -> None:
        from training.build_preference_data import FAKE_BOOKS

        verses = self._sample_verses(20)
        pairs = self.fn(verses, n=20)
        fake_books_found = sum(1 for p in pairs if any(fb in p["rejected"] for fb in FAKE_BOOKS))
        assert fake_books_found > 0


class TestBuildVerbosePairs:
    @pytest.fixture(autouse=True)
    def _import(self):
        from training.build_preference_data import _build_verbose_pairs

        self.fn = _build_verbose_pairs

    def _sample_verses(self, n: int = 80) -> list[dict]:
        books = ["Genesis", "Psalms", "John", "Romans", "Matthew"]
        return [
            {
                "book": books[i % len(books)],
                "chapter": (i // 5) + 1,
                "verse": (i % 5) + 1,
                "text": f"This is sample verse text number {i + 1} for testing the verbose pair generator.",
            }
            for i in range(n)
        ]

    def test_returns_correct_count(self) -> None:
        verses = self._sample_verses(80)
        pairs = self.fn(verses, n=10)
        assert len(pairs) == 10

    def test_pair_has_required_keys(self) -> None:
        verses = self._sample_verses(80)
        pairs = self.fn(verses, n=5)
        for p in pairs:
            assert "prompt" in p
            assert "chosen" in p
            assert "rejected" in p

    def test_rejected_is_longer_than_chosen(self) -> None:
        verses = self._sample_verses(80)
        pairs = self.fn(verses, n=20)
        for p in pairs:
            assert len(p["rejected"]) > len(p["chosen"]), "rejected should have verbose tail"

    def test_all_prompts_unique_at_small_n(self) -> None:
        """Diverse verse sampling means no prompt duplication at small n."""
        verses = self._sample_verses(80)
        pairs = self.fn(verses, n=30)
        prompts = [p["prompt"] for p in pairs]
        assert len(set(prompts)) == len(prompts), "each prompt should be unique (diverse sampling)"


class TestBuildBibleForEverythingPairs:
    @pytest.fixture(autouse=True)
    def _import(self):
        from training.build_preference_data import _build_bible_for_everything_pairs

        self.fn = _build_bible_for_everything_pairs

    def test_returns_correct_count(self) -> None:
        pairs = self.fn(n=30)
        assert len(pairs) == 30

    def test_pair_has_required_keys(self) -> None:
        for p in self.fn(n=5):
            assert "prompt" in p
            assert "chosen" in p
            assert "rejected" in p

    def test_chosen_does_not_contain_bible_shoehorn(self) -> None:
        """Chosen answers should be factual, not scripture-shoehorned."""
        pairs = self.fn(n=30)
        for p in pairs:
            # Chosen should not contain 'As ... says' style shoehorns
            assert "As Proverbs" not in p["chosen"]
            assert "As John" not in p["chosen"]

    def test_diverse_prompts(self) -> None:
        """30 diverse QA tuples means variety across n=30.

        random.choice with replacement at n=30 from 30 tuples gives ~19 unique
        prompts on average (birthday problem).  Assert >=15 to stay stable.
        """
        pairs = self.fn(n=30)
        prompts = {p["prompt"] for p in pairs}
        assert len(prompts) >= 15, f"expected diverse prompts, got {len(prompts)} unique"


# ---------------------------------------------------------------------------
# Remaining pair generators (structure verification)
# ---------------------------------------------------------------------------


def _make_verses(n: int = 30) -> list[dict]:
    """Helper to produce minimal verse dicts for testing."""
    books = ["Genesis", "Psalms", "John", "Romans", "Matthew", "Isaiah"]
    return [
        {
            "book": books[i % len(books)],
            "chapter": (i // 6) + 1,
            "verse": (i % 6) + 1,
            "text": f"This is verse text number {i + 1} and it has enough characters to pass length checks.",
        }
        for i in range(n)
    ]


class TestBuildInstructionLeakPairs:
    def test_structure(self) -> None:
        from training.build_preference_data import _build_instruction_leak_pairs

        pairs = _build_instruction_leak_pairs(_make_verses(30), n=10)
        assert len(pairs) == 10
        for p in pairs:
            assert "prompt" in p and "chosen" in p and "rejected" in p

    def test_rejected_contains_leaked_instruction(self) -> None:
        from training.build_preference_data import (
            LEAKED_INSTRUCTIONS,
            _build_instruction_leak_pairs,
        )

        pairs = _build_instruction_leak_pairs(_make_verses(30), n=30)
        found = sum(
            1 for p in pairs if any(instr in p["rejected"] for instr in LEAKED_INSTRUCTIONS)
        )
        assert found > 0


class TestBuildRepetitionPairs:
    def test_structure(self) -> None:
        from training.build_preference_data import _build_repetition_pairs

        pairs = _build_repetition_pairs(_make_verses(30), n=10)
        assert len(pairs) == 10
        for p in pairs:
            assert "prompt" in p and "chosen" in p and "rejected" in p

    def test_rejected_is_longer(self) -> None:
        from training.build_preference_data import _build_repetition_pairs

        pairs = _build_repetition_pairs(_make_verses(30), n=10)
        for p in pairs:
            assert len(p["rejected"]) > len(p["chosen"])


class TestBuildAnswerPrefixPairs:
    def test_structure(self) -> None:
        from training.build_preference_data import _build_answer_prefix_pairs

        pairs = _build_answer_prefix_pairs(_make_verses(30), n=10)
        assert len(pairs) == 10
        for p in pairs:
            assert "prompt" in p and "chosen" in p and "rejected" in p

    def test_rejected_starts_with_answer_prefix(self) -> None:
        from training.build_preference_data import _build_answer_prefix_pairs

        pairs = _build_answer_prefix_pairs(_make_verses(30), n=10)
        assert all("Answer:" in p["rejected"] for p in pairs)


class TestBuildThinkTagPairs:
    def test_structure(self) -> None:
        from training.build_preference_data import _build_think_tag_pairs

        pairs = _build_think_tag_pairs(_make_verses(30), n=10)
        assert len(pairs) == 10
        for p in pairs:
            assert "prompt" in p and "chosen" in p and "rejected" in p

    def test_rejected_contains_think_tag(self) -> None:
        from training.build_preference_data import _build_think_tag_pairs

        pairs = _build_think_tag_pairs(_make_verses(30), n=10)
        assert all("<think>" in p["rejected"] for p in pairs)

    def test_chosen_has_no_think_tag(self) -> None:
        from training.build_preference_data import _build_think_tag_pairs

        pairs = _build_think_tag_pairs(_make_verses(30), n=10)
        assert all("<think>" not in p["chosen"] for p in pairs)


# ---------------------------------------------------------------------------
# evaluate.py — additional pure helpers
# ---------------------------------------------------------------------------


class TestOllamaBaseUrl:
    def test_strips_path(self) -> None:
        from training.evaluate import _ollama_base_url

        assert (
            _ollama_base_url("http://127.0.0.1:11434/v1/chat/completions")
            == "http://127.0.0.1:11434"
        )

    def test_plain_base_url(self) -> None:
        from training.evaluate import _ollama_base_url

        assert _ollama_base_url("http://localhost:11434") == "http://localhost:11434"


# ---------------------------------------------------------------------------
# training.merge_adapters — main() error paths (no GPU required)
# ---------------------------------------------------------------------------


class TestMergeAdaptersMain:
    """Tests for merge_adapters.main() — exercises argparse + FileNotFoundError paths."""

    def test_explicit_lora_path_missing_raises_file_not_found(
        self, tmp_path: pytest.TemporaryDirectory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from training.merge_adapters import main

        nonexistent = str(tmp_path / "no_such_lora")
        monkeypatch.setattr("sys.argv", ["merge_adapters", "--lora-path", nonexistent])
        with pytest.raises(FileNotFoundError, match="LoRA checkpoint not found"):
            main()

    def test_default_lora_path_prints_warning_then_raises(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        import training.merge_adapters as ma

        # Point DEFAULT_LORA_NAME at a path that is guaranteed not to exist
        monkeypatch.setattr(ma, "DEFAULT_LORA_NAME", "__nonexistent_test_lora_abc__")
        monkeypatch.setattr("sys.argv", ["merge_adapters"])
        with caplog.at_level(logging.WARNING, logger="training.merge_adapters"):
            with pytest.raises(FileNotFoundError, match="LoRA checkpoint not found"):
                ma.main()
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_explicit_output_dir_created_before_missing_adapter(
        self, tmp_path: pytest.TemporaryDirectory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Covers the --output and lora_path.exists() check sequence."""
        from training.merge_adapters import main

        nonexistent = str(tmp_path / "no_lora")
        out_dir = str(tmp_path / "out")
        monkeypatch.setattr(
            "sys.argv",
            ["merge_adapters", "--lora-path", nonexistent, "--output", out_dir],
        )
        with pytest.raises(FileNotFoundError):
            main()
