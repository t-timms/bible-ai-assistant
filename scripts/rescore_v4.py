#!/usr/bin/env python3
"""Re-score existing protocol-v3 keyword runs under protocol v4 (no model re-run).

v4 splits verse_lookup -> verse_quote / verse_exposition (see
benchmarks/manifest.v4.yaml). The per-item responses and their
verse_accuracy / verse_accuracy_fuzzy / fuzzy_pass / citation / hallucination
values are UNCHANGED — only the category label moves and the aggregates are
recomputed under the new buckets, using training.evaluate's own aggregation
(_save_keyword_results) so formulas match byte-for-byte.

Inputs (protocol-v3 keyword artifacts already on disk):
  docs/benchmark_runs/20260901_v2-4b_keyword.json
  docs/benchmark_runs/20260901_v3-sft_keyword.json
  docs/benchmark_runs/20260901_v3-grpo_keyword.json

Outputs:
  docs/benchmark_runs/20260902_<label>_v4keyword.json   (one per input)
  + a printed v2-4b vs v3-sft comparison table, including the
    exposition-excluded overall fuzzy mean (the v4 headline number).

Usage:
  python scripts/rescore_v4.py            # write v4 JSONs + print table
  python scripts/rescore_v4.py --print-only   # table only, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate import _save_keyword_results  # noqa: E402

PROTOCOL_V4 = "bible_assistant_baseline_v4"
EXPOSITION_RE = re.compile(r"(teach|about)\?\s*$", re.IGNORECASE)

RUNS = [
    ("v2-4b", "20260901_v2-4b_keyword.json"),
    ("v3-sft", "20260901_v3-sft_keyword.json"),
    ("v3-grpo", "20260901_v3-grpo_keyword.json"),
]
BENCH_DIR = PROJECT_ROOT / "docs/benchmark_runs"

# categories that are counted only, never rate-scored (mirrors evaluate.py / protocol v3)
COUNT_ONLY = {"refusal"}


def rebucket(category: str, question: str) -> str:
    if category != "verse_lookup":
        return category
    return "verse_exposition" if EXPOSITION_RE.search(question.strip()) else "verse_quote"


def build_category_scores(results: list[dict]) -> dict:
    """Reconstruct the `category_scores` dict shape _save_keyword_results expects."""
    cs: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        d = cs.setdefault(
            cat,
            {
                "total": 0,
                "verse_accuracy_sum": 0.0,
                "verse_accuracy_fuzzy_sum": 0.0,
                "fuzzy_passes": 0,
                "citations": 0,
                "hallucinations": 0,
            },
        )
        d["total"] += 1
        if cat in COUNT_ONLY:
            d["count_only"] = True
            continue
        d["verse_accuracy_sum"] += float(r.get("verse_accuracy", 0.0))
        d["verse_accuracy_fuzzy_sum"] += float(r.get("verse_accuracy_fuzzy", 0.0))
        d["fuzzy_passes"] += 1 if r.get("fuzzy_pass") else 0
        d["citations"] += 1 if r.get("citation_present") else 0
        d["hallucinations"] += 1 if r.get("hallucination_detected") else 0
    return cs


def load_run(fname: str) -> dict:
    return json.loads((BENCH_DIR / fname).read_text(encoding="utf-8"))


def rescored_results(src: dict) -> list[dict]:
    out = []
    for r in src["results"]:
        nr = dict(r)
        nr["category"] = rebucket(r["category"], r["question"])
        out.append(nr)
    return out


def overall_fuzzy(results: list[dict], *, exclude: set[str]) -> tuple[float, int]:
    vals = [
        float(r.get("verse_accuracy_fuzzy", 0.0))
        for r in results
        if r["category"] not in COUNT_ONLY and r["category"] not in exclude
    ]
    n = len(vals)
    return (sum(vals) / n if n else 0.0), n


def cat_exact(results: list[dict], cat: str) -> tuple[float, int]:
    vals = [float(r.get("verse_accuracy", 0.0)) for r in results if r["category"] == cat]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def cat_fuzzy_pass(results: list[dict], cat: str) -> tuple[float, float, int]:
    items = [r for r in results if r["category"] == cat]
    n = len(items)
    passes = sum(1 for r in items if r.get("fuzzy_pass"))
    fmean = sum(float(r.get("verse_accuracy_fuzzy", 0.0)) for r in items) / n if n else 0.0
    return (passes / n if n else 0.0), fmean, n


def mcnemar_verse_quote(a: list[dict], b: list[dict]) -> str:
    """Paired pass/fail flip on verse_quote (fuzzy_pass), a=v2-4b vs b=v3-sft."""
    from scripts.benchmark_stats import mcnemar_pvalue, normalize_question

    ia = {normalize_question(r["question"]): r for r in a if r["category"] == "verse_quote"}
    ib = {normalize_question(r["question"]): r for r in b if r["category"] == "verse_quote"}
    keys = sorted(set(ia) & set(ib))
    b_only = sum(1 for k in keys if ia[k].get("fuzzy_pass") and not ib[k].get("fuzzy_pass"))
    c_only = sum(1 for k in keys if not ia[k].get("fuzzy_pass") and ib[k].get("fuzzy_pass"))
    p = mcnemar_pvalue(b_only, c_only)
    return f"n={len(keys)}  v2-only-pass={b_only}  v3-only-pass={c_only}  McNemar p={p:.3f}"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-only", action="store_true", help="print the table, write no files")
    args = ap.parse_args()

    loaded = {}
    for label, fname in RUNS:
        src = load_run(fname)
        res = rescored_results(src)
        loaded[label] = (src, res)

        if not args.print_only:
            cs = build_category_scores(res)
            out_path = BENCH_DIR / f"20260902_{label}_v4keyword.json"
            _save_keyword_results(
                cs,
                res,
                out_path,
                src.get("ollama_model", label),
                PROTOCOL_V4,
                src.get("hallucination_verification_mode", "corpus"),
            )

    v2 = loaded["v2-4b"][1]
    v3 = loaded["v3-sft"][1]
    grpo = loaded["v3-grpo"][1]

    def row(name: str, f):
        return f"  {name:<34} {f(v2):>14} {f(v3):>14} {f(grpo):>14}"

    print("\n" + "=" * 82)
    print("  protocol v4 re-score (from 20260901 protocol-v3 keyword runs; no model re-run)")
    print("=" * 82)
    print(f"  {'metric':<34} {'v2-4b':>14} {'v3-sft':>14} {'v3-grpo':>14}")
    print("-" * 82)

    print(row("verse_quote  exact-match", lambda r: pct(cat_exact(r, "verse_quote")[0])))
    print(
        row(
            "verse_quote  fuzzy pass@.85",
            lambda r: pct(cat_fuzzy_pass(r, "verse_quote")[0]),
        )
    )
    print(
        row(
            "verse_exposition  fuzzy pass@.85",
            lambda r: pct(cat_fuzzy_pass(r, "verse_exposition")[0]),
        )
    )
    print(
        row(
            "verse_exposition  fuzzy mean",
            lambda r: f"{cat_fuzzy_pass(r, 'verse_exposition')[1]:.3f}",
        )
    )
    print(
        row(
            "verse_exposition  exact (non-primary)",
            lambda r: pct(cat_exact(r, "verse_exposition")[0]),
        )
    )
    print("-" * 82)
    print(
        row(
            "overall fuzzy mean  (all-in)",
            lambda r: f"{overall_fuzzy(r, exclude=set())[0]:.3f}",
        )
    )
    print(
        row(
            "overall fuzzy mean  (expo-excluded)",
            lambda r: f"{overall_fuzzy(r, exclude={'verse_exposition'})[0]:.3f}",
        )
    )
    print("-" * 82)
    print("  bar: overall fuzzy mean >= 0.520 ; verse_quote exact >= 74% (was verse_lookup)")
    print("  n(all-in)      =", overall_fuzzy(v3, exclude=set())[1])
    print("  n(expo-excl)   =", overall_fuzzy(v3, exclude={"verse_exposition"})[1])
    print()
    print("  verse_quote recall hold (v2-4b vs v3-sft):")
    print("   ", mcnemar_verse_quote(v2, v3))
    print("=" * 82)
    if args.print_only:
        print("  (--print-only: no v4 JSON files written)")
    else:
        print("  wrote docs/benchmark_runs/20260902_{v2-4b,v3-sft,v3-grpo}_v4keyword.json")
        print("  compare: python scripts/compare_benchmark_runs.py \\")
        print("             docs/benchmark_runs/20260902_v2-4b_v4keyword.json \\")
        print("             docs/benchmark_runs/20260902_v3-sft_v4keyword.json")


if __name__ == "__main__":
    main()
