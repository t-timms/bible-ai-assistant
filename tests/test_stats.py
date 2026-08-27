"""Unit tests for scripts/benchmark_stats.py — hand-computed expectations, no scipy.

Wilson references: standard Wilson score interval tables
  0/10  → (0.0,    0.2775)
  7/10  → (0.3967, 0.8922)
  10/10 → (0.7225, 1.0)
McNemar exact two-sided: p = 2 * sum_{i<=min(b,c)} C(b+c,i) / 2^(b+c).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_stats import (  # noqa: E402
    format_rate,
    mcnemar_pvalue,
    normalize_question,
    paired_bootstrap_delta,
    wilson_interval,
)


class TestWilsonInterval:
    def test_zero_successes_upper_bound(self) -> None:
        lo, hi = wilson_interval(0, 10)
        assert lo == 0.0
        assert hi == pytest.approx(0.2775, abs=1e-3)

    def test_all_successes_lower_bound(self) -> None:
        lo, hi = wilson_interval(10, 10)
        assert hi == 1.0
        assert lo == pytest.approx(0.7225, abs=1e-3)

    def test_seven_of_ten_matches_published_table(self) -> None:
        lo, hi = wilson_interval(7, 10)
        assert lo == pytest.approx(0.3967, abs=1e-3)
        assert hi == pytest.approx(0.8922, abs=1e-3)

    def test_half_has_symmetric_bracket(self) -> None:
        lo, hi = wilson_interval(50, 100)
        assert lo == pytest.approx(0.4038, abs=1e-3)
        assert hi == pytest.approx(0.5962, abs=1e-3)

    def test_interval_contains_point_estimate(self) -> None:
        for successes in range(0, 11):
            lo, hi = wilson_interval(successes, 10)
            assert lo - 1e-12 <= successes / 10 <= hi + 1e-12

    def test_zero_n_returns_zero_band(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_custom_z_widens_interval(self) -> None:
        lo95, hi95 = wilson_interval(7, 10)
        lo99, hi99 = wilson_interval(7, 10, z=2.576)
        assert lo99 < lo95 < hi95 < hi99

    def test_negative_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            wilson_interval(-1, 10)
        with pytest.raises(ValueError):
            wilson_interval(0, -10)

    def test_successes_exceeding_n_raises(self) -> None:
        with pytest.raises(ValueError):
            wilson_interval(11, 10)


class TestMcnemarPvalue:
    def test_b1_c9_exact_value(self) -> None:
        # p = 2*(C(10,0)+C(10,1))/2^10 = 22/1024
        assert mcnemar_pvalue(1, 9) == pytest.approx(22 / 1024)

    def test_b0_c5_exact_value(self) -> None:
        # p = 2*C(5,0)/2^5 = 2/32
        assert mcnemar_pvalue(0, 5) == pytest.approx(2 / 32)

    def test_b3_c15_exact_value(self) -> None:
        # p = 2*(1+18+153+816)/2^18 = 1976/262144
        expected = 2 * (1 + 18 + 153 + 816) / 2**18
        assert mcnemar_pvalue(3, 15) == pytest.approx(expected)

    def test_equal_discordant_counts_is_not_significant(self) -> None:
        assert mcnemar_pvalue(5, 5) == 1.0

    def test_no_discordance_is_not_significant(self) -> None:
        assert mcnemar_pvalue(0, 0) == 1.0

    def test_extreme_imbalance_is_significant(self) -> None:
        # b=0, c=20: p = 2/2^20 ≈ 1.9e-6
        assert mcnemar_pvalue(0, 20) == pytest.approx(2 / 2**20)
        assert mcnemar_pvalue(0, 20) < 0.001

    def test_symmetric_in_arguments(self) -> None:
        assert mcnemar_pvalue(2, 7) == mcnemar_pvalue(7, 2)

    def test_never_exceeds_one(self) -> None:
        for b in range(0, 6):
            for c in range(0, 6):
                assert 0.0 < mcnemar_pvalue(b, c) <= 1.0

    def test_negative_counts_raise(self) -> None:
        with pytest.raises(ValueError):
            mcnemar_pvalue(-1, 5)


class TestPairedBootstrapDelta:
    def test_perfect_uplift_degenerate_ci(self) -> None:
        a = [0.0] * 20
        b = [1.0] * 20
        delta, lo, hi = paired_bootstrap_delta(a, b, B=500, seed=42)
        assert delta == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_identical_outcomes_zero_delta(self) -> None:
        xs = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        delta, lo, hi = paired_bootstrap_delta(xs, list(xs), B=500, seed=42)
        assert delta == pytest.approx(0.0)
        assert lo <= delta <= hi

    def test_deterministic_for_fixed_seed(self) -> None:
        a = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        first = paired_bootstrap_delta(a, b, B=300, seed=7)
        second = paired_bootstrap_delta(a, b, B=300, seed=7)
        assert first == second

    def test_clear_improvement_ci_excludes_zero(self) -> None:
        a = [0.0] * 30 + [1.0] * 10
        b = [1.0] * 30 + [1.0] * 10  # B better on exactly the A-failures
        delta, lo, hi = paired_bootstrap_delta(a, b, B=2000, seed=42)
        assert delta == pytest.approx(0.75)
        assert lo > 0.5
        assert hi <= 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap_delta([1.0, 0.0], [1.0])

    def test_empty_lists_raise(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap_delta([], [])


class TestNormalizeQuestion:
    def test_lowercase_and_whitespace_collapse(self) -> None:
        assert normalize_question("  What   DOES\nJohn\t3:16  Say? ") == ("what does john 3:16 say")

    def test_trailing_punctuation_stripped_repeatedly(self) -> None:
        assert normalize_question("Who was Moses?!.") == "who was moses"
        assert normalize_question("Who was Moses...") == "who was moses"

    def test_interior_punctuation_preserved(self) -> None:
        # Only *trailing* punctuation is stripped — interior chars stay.
        assert normalize_question("What does John 3:16 say?") == "what does john 3:16 say"

    def test_empty_and_whitespace_only(self) -> None:
        assert normalize_question("") == ""
        assert normalize_question("   ") == ""
        assert normalize_question("?") == ""

    def test_same_normalized_form_matches(self) -> None:
        variants = [
            "Who was Moses?",
            "who was moses",
            "  Who was Moses?  ",
            "Who \t was\nMoses!",
        ]
        normalized = {normalize_question(v) for v in variants}
        assert len(normalized) == 1


class TestNormalizeQuestionEquivalence:
    """benchmark_stats.normalize_question must match dataset_builder's byte-for-byte.

    The training-side decontamination filter and the eval-side overlap checker
    MUST agree on normalization or contamination slips through the gap.
    """

    def test_matches_dataset_builder_implementation(self) -> None:
        try:
            from training.dataset_builder import normalize_question as db_normalize
        except Exception as e:  # pragma: no cover - mid-flight edits upstream
            pytest.skip(f"training.dataset_builder not importable: {e}")

        samples = [
            "Who was Moses?",
            "  What   does John 3:16 say?!.. ",
            "Explain Romans 8:28.",
            "Tell me about; 'faith'...",
            "hello\u2026",  # ellipsis is in the trailing strip set
            "",
            "A question with, commas; and: colons?",
        ]
        for s in samples:
            assert normalize_question(s) == db_normalize(s), f"drift on {s!r}"


class TestFormatRate:
    def test_carries_rate_ci_and_n(self) -> None:
        lo, hi = wilson_interval(7, 10)
        text = format_rate(0.7, lo, hi, 10)
        assert "70.0%" in text
        assert "n=10" in text
        assert "[" in text and "]" in text


class _MathSanity:
    """Guard against silent stdlib behavior changes the formulas rely on."""

    def test_comb_available(self) -> None:
        assert math.comb(5, 2) == 10
