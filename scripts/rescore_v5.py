#!/usr/bin/env python3
"""Re-score the 4 shipped-candidate keyword runs under protocol v5 (no model
re-run) -- adds verse_accuracy_semantic per item and prints a ranked table.

v5 does not change the question suite or any existing metric (see
benchmarks/manifest.v5.yaml). It: (1) re-buckets verse_lookup ->
verse_quote/verse_exposition where a run hasn't already been through
rescore_v4.py's split (v3.1 and v3.2's native re-eval used the unsplit
category), using the identical deterministic rule scripts/rescore_v4.py
uses, so all 4 candidates compare on the same category axis; (2) computes
verse_accuracy_semantic for every item via training.evaluate's cross-encoder
metric; (3) aggregates fuzzy AND semantic overall means two ways (all-in,
exposition-excluded), plus a paired bootstrap delta between the top two
candidates by semantic (expo-excl) mean.

Inputs (best available protocol-v4-labeled keyword run per candidate --
v2-4b/v3-sft already went through a native re-eval with the split category;
v3.1/v3.2 have not, so they're re-bucketed here):
  docs/benchmark_runs/20260903_v2-4b_v4keyword.json
  docs/benchmark_runs/20260903_v3-sft_v4keyword.json
  docs/benchmark_runs/20260904_v3.1_keyword.json
  docs/benchmark_runs/20260904_v3.2_keyword.json

Outputs:
  docs/benchmark_runs/20260904_<label>_v5semantic.json   (one per candidate)
  + a printed ranked comparison table.

Also supports ad-hoc single-file mode, for backfilling the semantic metric onto
an external comparator's fresh v4 keyword run after scripts/run_external_baselines.sh
(same re-bucket + semantic-score logic, no model re-run; naming convention
scripts/sota_scoreboard.py expects: <run>_keyword.json -> <run>_v5semantic.json):

  python scripts/rescore_v5.py \
    --file docs/benchmark_runs/20260905_ext-qwen3-8b-instruct_keyword.json \
    --out  docs/benchmark_runs/20260905_ext-qwen3-8b-instruct_v5semantic.json

Usage:
  python scripts/rescore_v5.py                # write v5 JSONs + print table
  python scripts/rescore_v5.py --print-only    # table only, write nothing
  python scripts/rescore_v5.py --file IN --out OUT   # ad-hoc single-file backfill
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_stats import normalize_question, paired_bootstrap_delta  # noqa: E402
from scripts.rescore_v4 import COUNT_ONLY, rebucket  # noqa: E402
from training.evaluate import _get_semantic_scorer, check_verse_accuracy_semantic  # noqa: E402

PROTOCOL_V5 = "bible_assistant_baseline_v5"
BENCH_DIR = PROJECT_ROOT / "docs/benchmark_runs"
EXPOSITION_CATEGORIES = {"verse_exposition"}

RUNS = [
    ("v2-4b", "20260903_v2-4b_v4keyword.json"),
    ("v3-sft", "20260903_v3-sft_v4keyword.json"),
    ("v3.1", "20260904_v3.1_keyword.json"),
    ("v3.2", "20260904_v3.2_keyword.json"),
]


def load_run(fname: str) -> dict:
    return json.loads((BENCH_DIR / fname).read_text(encoding="utf-8"))


def rescore(src: dict, scorer: object) -> list[dict]:
    """Re-bucket categories (idempotent -- a no-op on already-split runs) and
    add verse_accuracy_semantic to every result, in place on copies."""
    out = []
    for r in src["results"]:
        nr = dict(r)
        nr["category"] = rebucket(r["category"], r["question"])
        nr["verse_accuracy_semantic"] = check_verse_accuracy_semantic(
            nr["response"], nr["expected_answer"], scorer=scorer
        )
        out.append(nr)
    return out


def mean(results: list[dict], field: str, *, exclude: set[str]) -> tuple[float, int]:
    vals = [
        float(r[field])
        for r in results
        if r["category"] not in COUNT_ONLY
        and r["category"] not in exclude
        and r.get(field) is not None
    ]
    n = len(vals)
    return (sum(vals) / n if n else 0.0), n


def cat_mean(results: list[dict], cat: str, field: str) -> tuple[float, int]:
    vals = [float(r[field]) for r in results if r["category"] == cat and r.get(field) is not None]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def build_output(label: str, src: dict, results: list[dict]) -> dict:
    n_semantic = sum(1 for r in results if r.get("verse_accuracy_semantic") is not None)
    fuzzy_all, n_fuzzy_all = mean(results, "verse_accuracy_fuzzy", exclude=set())
    fuzzy_excl, n_fuzzy_excl = mean(results, "verse_accuracy_fuzzy", exclude=EXPOSITION_CATEGORIES)
    sem_all, n_sem_all = mean(results, "verse_accuracy_semantic", exclude=set())
    sem_excl, n_sem_excl = mean(results, "verse_accuracy_semantic", exclude=EXPOSITION_CATEGORIES)
    return {
        "benchmark_protocol_id": PROTOCOL_V5,
        "source_run": src.get("ollama_model", label),
        "candidate_label": label,
        "total_questions": len(results),
        "semantic_scored_count": n_semantic,
        "overall_verse_accuracy_fuzzy_all_in": round(fuzzy_all, 4),
        "overall_verse_accuracy_fuzzy_expo_excl": round(fuzzy_excl, 4),
        "overall_verse_accuracy_semantic_all_in": round(sem_all, 4),
        "overall_verse_accuracy_semantic_expo_excl": round(sem_excl, 4),
        "n_all_in": n_fuzzy_all,
        "n_expo_excl": n_fuzzy_excl,
        "n_semantic_all_in": n_sem_all,
        "n_semantic_expo_excl": n_sem_excl,
        "results": results,
    }


def paired_semantic(
    a: list[dict], b: list[dict], *, exclude: set[str]
) -> tuple[float, float, float, int]:
    """Paired bootstrap delta (mean(b) - mean(a)) on verse_accuracy_semantic,
    matched by normalized question text, restricted to scored/non-excluded items."""
    ia = {
        normalize_question(r["question"]): r
        for r in a
        if r["category"] not in COUNT_ONLY
        and r["category"] not in exclude
        and r.get("verse_accuracy_semantic") is not None
    }
    ib = {
        normalize_question(r["question"]): r
        for r in b
        if r["category"] not in COUNT_ONLY
        and r["category"] not in exclude
        and r.get("verse_accuracy_semantic") is not None
    }
    keys = sorted(set(ia) & set(ib))
    va = [float(ia[k]["verse_accuracy_semantic"]) for k in keys]
    vb = [float(ib[k]["verse_accuracy_semantic"]) for k in keys]
    if not keys:
        return (0.0, 0.0, 0.0, 0)
    delta, lo, hi = paired_bootstrap_delta(va, vb)
    return (delta, lo, hi, len(keys))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def run_single_file(in_path: Path, out_path: Path, scorer: object) -> None:
    """Ad-hoc mode: backfill verse_accuracy_semantic onto one arbitrary saved
    keyword run (e.g. an external comparator) -- same logic as the batch mode,
    scoped to a single file so it can be called once per comparator after
    scripts/run_external_baselines.sh, with no dependency on the 4 hardcoded
    RUNS labels."""
    src = json.loads(in_path.read_text(encoding="utf-8"))
    label = in_path.stem
    results = rescore(src, scorer)
    out = build_output(label, src, results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    fuzzy_excl, _ = mean(results, "verse_accuracy_fuzzy", exclude=EXPOSITION_CATEGORIES)
    sem_excl, n = mean(results, "verse_accuracy_semantic", exclude=EXPOSITION_CATEGORIES)
    print(
        f"  {label}: fuzzy(expo-excl)={fuzzy_excl:.3f}  semantic(expo-excl)={sem_excl:.3f} (n={n})"
    )
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-only", action="store_true", help="print the table, write no files")
    ap.add_argument(
        "--file", type=Path, default=None, help="ad-hoc mode: one keyword.json to backfill"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="ad-hoc mode: output path (required with --file)"
    )
    args = ap.parse_args()

    scorer = _get_semantic_scorer()
    if scorer is None:
        sys.exit(
            "ERROR: semantic scorer unavailable (sentence-transformers / bge-reranker-v2-m3 "
            "not loadable). Run inside .venv-rag with the model reachable."
        )

    if args.file:
        if not args.out:
            sys.exit("ERROR: --file requires --out")
        run_single_file(args.file, args.out, scorer)
        return

    loaded: dict[str, tuple[dict, list[dict]]] = {}
    t0 = time.time()
    for label, fname in RUNS:
        src = load_run(fname)
        results = rescore(src, scorer)
        loaded[label] = (src, results)
        print(f"  scored {label:<8} n={len(results)}  ({time.time() - t0:.1f}s elapsed)")

        if not args.print_only:
            out = build_output(label, src, results)
            out_path = BENCH_DIR / f"20260904_{label}_v5semantic.json"
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 96)
    print("  protocol v5 re-score (semantic metric added; no model re-run)")
    print("=" * 96)
    header = f"  {'candidate':<10} {'fuzzy all-in':>13} {'fuzzy expo-excl':>16} {'sem all-in':>12} {'sem expo-excl':>14}"
    print(header)
    print("-" * 96)

    ranked = []
    for label, _fname in RUNS:
        _src, res = loaded[label]
        fuzzy_all, _ = mean(res, "verse_accuracy_fuzzy", exclude=set())
        fuzzy_excl, _ = mean(res, "verse_accuracy_fuzzy", exclude=EXPOSITION_CATEGORIES)
        sem_all, _ = mean(res, "verse_accuracy_semantic", exclude=set())
        sem_excl, n_excl = mean(res, "verse_accuracy_semantic", exclude=EXPOSITION_CATEGORIES)
        ranked.append((label, fuzzy_all, fuzzy_excl, sem_all, sem_excl, n_excl))
        print(
            f"  {label:<10} {fuzzy_all:>13.3f} {fuzzy_excl:>16.3f} {sem_all:>12.3f} {sem_excl:>14.3f}"
        )

    print("-" * 96)
    ranked_by_sem = sorted(ranked, key=lambda t: t[4], reverse=True)
    print("  ranked by semantic (expo-excl), highest first:")
    for i, (label, _fa, _fe, _sa, se, n) in enumerate(ranked_by_sem, start=1):
        print(f"    {i}. {label:<10} semantic_expo_excl={se:.3f}  (n={n})")

    if len(ranked_by_sem) >= 2:
        top, second = ranked_by_sem[0][0], ranked_by_sem[1][0]
        a_res = loaded[second][1]
        b_res = loaded[top][1]
        delta, lo, hi, n = paired_semantic(a_res, b_res, exclude=EXPOSITION_CATEGORIES)
        print(
            f"\n  paired bootstrap delta, semantic expo-excl, {top} vs {second} "
            f"(n={n}): {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        )
        if lo <= 0.0 <= hi:
            print("    -> 95% CI spans 0: not distinguishable at this n.")
        else:
            print(f"    -> {top} beats {second}, CI excludes 0.")

    print("=" * 96)
    if args.print_only:
        print("  (--print-only: no v5 JSON files written)")
    else:
        labels = ",".join(label for label, _ in RUNS)
        print(f"  wrote docs/benchmark_runs/20260904_{{{labels}}}_v5semantic.json")


if __name__ == "__main__":
    main()
