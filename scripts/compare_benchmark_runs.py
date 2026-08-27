#!/usr/bin/env python3
"""
Compare two benchmark JSON artifacts (keyword or judge mode).

Refuses to compare runs with different benchmark_protocol_id unless --force is
passed (scores across protocols are not comparable). Keyword comparisons carry
Wilson 95% CIs, an exact McNemar p-value, and a paired-bootstrap delta CI for
the verse_accuracy / citation_rate / hallucination_rate columns, computed over
per-item outcomes paired by normalized question text.

  python scripts/compare_benchmark_runs.py docs/benchmark_runs/20260825_a_keyword.json docs/benchmark_runs/20260825_b_keyword.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

FUZZY_PASS_THRESHOLD = 0.85


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _protocol_guard(a: dict, b: dict, force: bool) -> None:
    proto_a = a.get("benchmark_protocol_id")
    proto_b = b.get("benchmark_protocol_id")
    if proto_a == proto_b:
        return
    print("\n!!! PROTOCOL MISMATCH !!!", file=sys.stderr)
    print(f"    A: {proto_a or '<unspecified>'}", file=sys.stderr)
    print(f"    B: {proto_b or '<unspecified>'}", file=sys.stderr)
    print(
        "Scores from different protocols are NOT comparable (different suites,\n"
        "    rubrics, or metric semantics). Re-run both models under one protocol.",
        file=sys.stderr,
    )
    if not force:
        print("Pass --force to compare anyway.", file=sys.stderr)
        raise SystemExit(2)


def _index_by_question(results: list | None) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for r in results or []:
        key = normalize_question(str(r.get("question", "")))
        if key and key not in indexed:
            indexed[key] = r
    return indexed


def pair_results(a: dict, b: dict) -> list[tuple[dict, dict]]:
    """Per-item pairs matched by normalized question (order-independent)."""
    ia = _index_by_question(a.get("results"))
    ib = _index_by_question(b.get("results"))
    return [(ia[k], ib[k]) for k in sorted(set(ia) & set(ib))]


def _fuzzy_pass(item: dict) -> bool | None:
    if "fuzzy_pass" in item:
        return bool(item["fuzzy_pass"])
    fuzzy = item.get("verse_accuracy_fuzzy")
    if isinstance(fuzzy, (int, float)):
        return float(fuzzy) >= FUZZY_PASS_THRESHOLD
    return None


def _verse_outcome(item: dict) -> float:
    """Binary verse-accuracy outcome: fuzzy pass at threshold when fuzzy data
    exists, else exact-substring full credit (legacy artifacts)."""
    passed = _fuzzy_pass(item)
    if passed is None:
        return 1.0 if float(item.get("verse_accuracy", 0.0)) >= 1.0 else 0.0
    return 1.0 if passed else 0.0


def _bool_outcome(item: dict, key: str) -> float:
    return 1.0 if bool(item.get(key, False)) else 0.0


METRIC_EXTRACTORS = {
    "verse_accuracy": (_verse_outcome, "pass-rate at fuzzy threshold 0.85"),
    "citation_rate": (
        lambda item: _bool_outcome(item, "citation_present"),
        "regex citation present",
    ),
    "hallucination_rate": (
        lambda item: _bool_outcome(item, "hallucination_detected"),
        "fabricated-reference detected",
    ),
}


def _rate_line(label: str, outcomes: list[float]) -> str:
    n = len(outcomes)
    successes = int(sum(outcomes))
    lo, hi = wilson_interval(successes, n)
    return f"{label}: {format_rate(successes / n if n else 0.0, lo, hi, n)}"


def _paired_stats_block(
    metric: str,
    description: str,
    outcomes_a: list[float],
    outcomes_b: list[float],
    bootstrap_b: int,
    seed: int,
) -> None:
    n = len(outcomes_a)
    print(f"\n{metric}  ({description})")
    print("  " + _rate_line("A", outcomes_a))
    print("  " + _rate_line("B", outcomes_b))
    if not n:
        print("  no paired items — cannot compute delta statistics")
        return

    delta_pp = (sum(outcomes_b) - sum(outcomes_a)) / n * 100
    print(f"  Delta(B-A): {delta_pp:+.1f}pp over n={n} paired questions")

    b_discordant = sum(
        1 for xa, xb in zip(outcomes_a, outcomes_b, strict=True) if xa == 1.0 and xb == 0.0
    )
    c_discordant = sum(
        1 for xa, xb in zip(outcomes_a, outcomes_b, strict=True) if xa == 0.0 and xb == 1.0
    )
    p = mcnemar_pvalue(b_discordant, c_discordant)
    print(
        f"  McNemar exact: b(A-pass,B-fail)={b_discordant} c(A-fail,B-pass)={c_discordant}  p={p:.3g}"
    )

    delta_mean, ci_lo, ci_hi = paired_bootstrap_delta(
        outcomes_a, outcomes_b, B=bootstrap_b, seed=seed
    )
    print(
        f"  Paired bootstrap delta mean={delta_mean:+.3f} "
        f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]  (B={bootstrap_b}, seed={seed})"
    )


def _compare_keyword(a: dict, b: dict, bootstrap_b: int, seed: int) -> None:
    pairs = pair_results(a, b)
    if not pairs:
        print(
            "\nNo overlapping questions between runs' per-item results — "
            "only summary-level comparison possible."
        )
        acc_a = a.get("overall_verse_accuracy", 0)
        acc_b = b.get("overall_verse_accuracy", 0)
        print(f"Verse accuracy (summary): A={acc_a:.3f}  B={acc_b:.3f}  Δ={acc_b - acc_a:+.3f}")
        return

    for metric, (extractor, description) in METRIC_EXTRACTORS.items():
        outcomes_a = [extractor(ra) for ra, _ in pairs]
        outcomes_b = [extractor(rb) for _, rb in pairs]
        _paired_stats_block(metric, description, outcomes_a, outcomes_b, bootstrap_b, seed)

    # Fuzzy similarity means (not an accuracy — reported alongside the pass-rate).
    fuzzy_a = [float(ra.get("verse_accuracy_fuzzy", 0.0)) for ra, _ in pairs]
    fuzzy_b = [float(rb.get("verse_accuracy_fuzzy", 0.0)) for _, rb in pairs]
    print(
        f"\nfuzzy mean similarity: A={sum(fuzzy_a) / len(fuzzy_a):.3f}  "
        f"B={sum(fuzzy_b) / len(fuzzy_b):.3f}  "
        f"(mean is not an accuracy; see pass-rates above)"
    )

    modes = {run.get("hallucination_verification_mode", "<legacy>") for run in (a, b)}
    if len(modes) > 1:
        print(
            f"\nWARNING: hallucination verification_mode differs ({sorted(modes)}) — "
            "hallucination_rate is not comparable across these runs."
        )


def _compare_judge(a: dict, b: dict) -> None:
    sa, sb = a.get("overall_scores", {}), b.get("overall_scores", {})
    keys = sorted(set(sa) | set(sb))
    print(f"\n{'Dimension':<14} {'A':>8} {'B':>8} {'Delta(B-A)':>10}")
    print("-" * 44)
    for k in keys:
        va, vb = float(sa.get(k, 0)), float(sb.get(k, 0))
        print(f"{k:<14} {va:>8.2f} {vb:>8.2f} {vb - va:>+10.2f}")
    avg_a = sum(sa.values()) / len(sa) if sa else 0
    avg_b = sum(sb.values()) / len(sb) if sb else 0
    print("-" * 44)
    print(f"{'mean(5 dims)':<14} {avg_a:>8.2f} {avg_b:>8.2f} {avg_b - avg_a:>+10.2f}")

    pf_a = a.get("parse_failure_rate", {}).get("value")
    pf_b = b.get("parse_failure_rate", {}).get("value")
    if pf_a is not None or pf_b is not None:

        def _fmt(v: object) -> str:
            return f"{v:.1%}" if isinstance(v, (int, float)) else "-"

        print(f"\nJudge parse-failure rate: A={_fmt(pf_a)}  B={_fmt(pf_b)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark run JSON files.")
    parser.add_argument("run_a", type=Path, help="First run JSON")
    parser.add_argument("run_b", type=Path, help="Second run JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even when the two runs use different benchmark_protocol_id values",
    )
    parser.add_argument(
        "--bootstrap-B", type=int, default=10000, help="Bootstrap resamples (default 10000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed (default 42)")
    args = parser.parse_args()

    a, b = _load(args.run_a), _load(args.run_b)
    mode_a = a.get("eval_mode", "?")
    mode_b = b.get("eval_mode", "?")

    print("\n" + "=" * 72)
    print("BENCHMARK A/B COMPARISON")
    print("=" * 72)
    print(f"A: {args.run_a.name}")
    print(
        f"   ollama_model={a.get('ollama_model', '?')}  tag={a.get('model_tag', '-')}  mode={mode_a}"
    )
    print(f"   protocol={a.get('benchmark_protocol_id', '-')}")
    print(f"B: {args.run_b.name}")
    print(
        f"   ollama_model={b.get('ollama_model', '?')}  tag={b.get('model_tag', '-')}  mode={mode_b}"
    )
    print(f"   protocol={b.get('benchmark_protocol_id', '-')}")

    _protocol_guard(a, b, args.force)

    if mode_a != mode_b:
        print("\nWarning: eval_mode differs — comparison is approximate.")

    if mode_a == "keyword" and mode_b == "keyword":
        _compare_keyword(a, b, args.bootstrap_B, args.seed)
    elif mode_a == "llm-as-judge" and mode_b == "llm-as-judge":
        _compare_judge(a, b)
    else:
        print("\nPrint raw keys for manual compare:")
        print("  A overall_scores / overall_verse_accuracy:", list(a.keys()))
        print("  B overall_scores / overall_verse_accuracy:", list(b.keys()))

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
