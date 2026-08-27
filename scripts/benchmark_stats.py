"""Pure-python statistics + shared normalization for the benchmark harness.

No scipy/numpy (blocked by machine policy) — everything here uses only the
standard library (math, random). This module is the single source of truth for:

- ``normalize_question``  — the train/eval decontamination normalization
  contract (identical semantics to ``training.dataset_builder.normalize_question``;
  pinned together by tests/test_stats.py so the two cannot drift).
- ``wilson_interval``     — 95% CI for binomial rates (every printed rate in the
  benchmark tooling carries one).
- ``mcnemar_pvalue``      — exact two-sided binomial test on paired binary
  outcomes (discordant pairs b, c).
- ``paired_bootstrap_delta`` — percentile CI for the paired mean delta (B-A).

Used by training/evaluate.py, scripts/compare_benchmark_runs.py,
scripts/check_train_eval_overlap.py, and scripts/build_qrels.py.
"""

from __future__ import annotations

import math
import random
import re

# Normalization contract (shared with training/dataset_builder.py — keep in sync;
# equivalence is enforced by tests/test_stats.py::TestNormalizeQuestionEquivalence):
#   1. lowercase
#   2. collapse every whitespace run to a single space
#   3. strip leading/trailing whitespace
#   4. repeatedly strip trailing characters in _TRAILING_STRIP_CHARS, then strip again.
_TRAILING_STRIP_CHARS = "?.!,;:'\"…"

_Z_DEFAULT = 1.96  # two-sided 95%


def normalize_question(text: str) -> str:
    """Canonical normalized form of a question for contamination/dedup comparison."""
    t = re.sub(r"\s+", " ", text.lower()).strip()
    while t and t[-1] in _TRAILING_STRIP_CHARS:
        t = t[:-1].rstrip()
    return t


def wilson_interval(successes: int, n: int, z: float = _Z_DEFAULT) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — returns ``(lo, hi)``.

    Chosen over the normal approximation because it stays inside [0, 1] for
    extreme rates and small n (e.g. per-category counts of 8-30).

    Raises ValueError on negative inputs or successes > n. n == 0 → (0.0, 0.0).
    """
    if successes < 0 or n < 0:
        raise ValueError(f"successes and n must be non-negative, got {successes}/{n}")
    if successes > n:
        raise ValueError(f"successes ({successes}) cannot exceed n ({n})")
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def mcnemar_pvalue(b: int, c: int) -> float:
    """Exact two-sided McNemar test p-value on discordant pair counts.

    ``b`` = count of (A success, B failure), ``c`` = count of (A failure, B
    success). Under the null the discordant counts are exchangeable, so the
    p-value is the two-sided binomial tail of min(b, c) against Bin(b + c, 0.5):

        p = 2 * sum_{i=0..min(b,c)} C(b+c, i) / 2^(b+c),  clamped to 1.0.

    b + c == 0 → 1.0 (no evidence of any difference).
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2**n))


def paired_bootstrap_delta(
    paired_outcomes_a: list[float],
    paired_outcomes_b: list[float],
    B: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the paired mean delta (mean(B) - mean(A)).

    Resamples *indices* (never values independently) so each replicate preserves
    within-question pairing. Returns ``(delta_mean, ci_lo, ci_hi)`` where the CI
    bounds are the 2.5th / 97.5th percentiles of the replicate deltas.

    Raises ValueError when the outcome lists are empty or differ in length
    (pairing is the whole point). Deterministic for a fixed seed.
    """
    if len(paired_outcomes_a) != len(paired_outcomes_b):
        raise ValueError(
            "paired outcome lists must have equal length: "
            f"{len(paired_outcomes_a)} != {len(paired_outcomes_b)}"
        )
    n = len(paired_outcomes_a)
    if n == 0:
        raise ValueError("paired outcome lists must be non-empty")
    mean_a = sum(paired_outcomes_a) / n
    mean_b = sum(paired_outcomes_b) / n
    delta_mean = mean_b - mean_a

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(B):
        sa = 0.0
        sb = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            sa += paired_outcomes_a[i]
            sb += paired_outcomes_b[i]
        deltas.append((sb - sa) / n)

    deltas.sort()
    ci_lo = deltas[max(0, int(math.floor(0.025 * B)))]
    ci_hi = deltas[min(B - 1, int(math.ceil(0.975 * B)) - 1)]
    return (delta_mean, ci_lo, ci_hi)


def format_rate(rate: float, lo: float, hi: float, n: int) -> str:
    """Render a rate with its Wilson CI and n — every printed rate carries all three."""
    return f"{rate:.1%} [{lo:.1%}, {hi:.1%}] (n={n})"
