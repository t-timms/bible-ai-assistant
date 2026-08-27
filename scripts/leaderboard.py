#!/usr/bin/env python3
"""
Print a leaderboard of model evaluations from docs/evaluation_results_*.json.
Run from project root: python scripts/leaderboard.py

Judge-scale (1-5 rubric means) and keyword-scale (rates) results are reported
in separate sections — mixing them in one sort compares incomparable scales.
Each row shows its benchmark protocol_id; scores across protocols are not
comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _fmt_rate(rate: dict | None) -> str:
    """'56.7% [38.2%, 73.8%] n=30' from a v3 rate block; '-' for legacy artifacts."""
    if not isinstance(rate, dict) or "value" not in rate:
        return "-"
    value = rate["value"]
    ci = rate.get("wilson95") or {}
    lo, hi = ci.get("lo"), ci.get("hi")
    n = rate.get("n", "?")
    base = f"{value:.1%}"
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        base += f" [{lo:.1%}, {hi:.1%}]"
    return f"{base} n={n}"


def _load_rows() -> list[dict]:
    rows = []
    for f in sorted(DOCS.glob("evaluation_results*.json")):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        tag = f.stem.replace("evaluation_results", "").lstrip("_") or "default"
        rows.append(
            {
                "tag": tag,
                "mode": data.get("eval_mode", "?"),
                "protocol": data.get("benchmark_protocol_id") or "<unspecified>",
                "data": data,
            }
        )
    return rows


def _print_judge_section(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("JUDGE SCALE (LLM-as-judge, 1-5 rubric means) — not comparable to keyword rates")
    print("=" * 78)
    judge_rows = [r for r in rows if r["mode"] == "llm-as-judge"]
    if not judge_rows:
        print("  (no judge-mode artifacts)")
        return

    def sort_key(r: dict) -> float:
        scores = r["data"].get("overall_scores", {})
        return -(sum(scores.values()) / len(scores) if scores else 0)

    for i, r in enumerate(sorted(judge_rows, key=sort_key), 1):
        scores = r["data"].get("overall_scores", {})
        avg = sum(scores.values()) / len(scores) if scores else 0
        scored = r["data"].get("scored_questions", r["data"].get("total_questions", "?"))
        pf = (r["data"].get("parse_failure_rate") or {}).get("value")
        pf_text = f"  parse_fail={pf:.0%}" if isinstance(pf, (int, float)) else ""
        print(f"\n#{i}  {r['tag']}  protocol={r['protocol']}")
        print(f"    Overall avg: {avg:.2f}/5  (scored n={scored}{pf_text})")
        for k, v in scores.items():
            print(f"      {k}: {v:.2f}")


def _print_keyword_section(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("KEYWORD SCALE (rates) — not comparable to judge rubric means")
    print("=" * 78)
    kw_rows = [r for r in rows if r["mode"] != "llm-as-judge"]
    if not kw_rows:
        print("  (no keyword-mode artifacts)")
        return

    def pass_rate(r: dict) -> float:
        rate = r["data"].get("overall_fuzzy_pass_rate")
        if isinstance(rate, dict) and "value" in rate:
            return -float(rate["value"])
        return -(
            r["data"].get(
                "overall_verse_accuracy_fuzzy", r["data"].get("overall_verse_accuracy", 0)
            )
            or 0
        )

    for i, r in enumerate(sorted(kw_rows, key=pass_rate), 1):
        d = r["data"]
        threshold = d.get("fuzzy_pass_threshold", 0.85)
        fuzzy_pass = _fmt_rate(d.get("overall_fuzzy_pass_rate"))
        cite = _fmt_rate(d.get("overall_citation_rate"))
        hall = _fmt_rate(d.get("overall_hallucination_rate"))
        mode_note = (
            f"  verification={d['hallucination_verification_mode']}"
            if "hallucination_verification_mode" in d
            else ""
        )
        print(f"\n#{i}  {r['tag']}  protocol={r['protocol']}{mode_note}")
        print(
            f"    verse_accuracy mean: {d.get('overall_verse_accuracy', 0):.0%}  "
            f"fuzzy mean: {d.get('overall_verse_accuracy_fuzzy', 0):.3f}"
        )
        print(f"    fuzzy pass >= {threshold}: {fuzzy_pass}")
        print(f"    citations: {cite}   hallucinations: {hall}")


def main() -> None:
    rows = _load_rows()
    if not rows:
        print("No evaluation results found in docs/. Run evaluate.py first.")
        return

    print("\n" + "=" * 78)
    print("BIBLE MODEL EVALUATION LEADERBOARD")
    print("=" * 78)
    _print_judge_section(rows)
    _print_keyword_section(rows)
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
