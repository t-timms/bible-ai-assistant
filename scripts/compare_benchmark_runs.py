#!/usr/bin/env python3
"""
Compare two benchmark JSON artifacts (keyword or judge mode).

  python scripts/compare_benchmark_runs.py docs/benchmark_runs/20260319_orpo-q4_judge.json docs/benchmark_runs/20260319_orpo-f16_judge.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark run JSON files.")
    parser.add_argument("run_a", type=Path, help="First run JSON")
    parser.add_argument("run_b", type=Path, help="Second run JSON")
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

    if mode_a != mode_b:
        print("\nWarning: eval_mode differs — comparison is approximate.")

    if mode_a == "llm-as-judge" and mode_b == "llm-as-judge":
        sa, sb = a.get("overall_scores", {}), b.get("overall_scores", {})
        keys = sorted(set(sa) | set(sb))
        print(f"\n{'Dimension':<14} {'A':>8} {'B':>8} {'Δ(B-A)':>10}")
        print("-" * 44)
        for k in keys:
            va, vb = float(sa.get(k, 0)), float(sb.get(k, 0))
            print(f"{k:<14} {va:>8.2f} {vb:>8.2f} {vb - va:>+10.2f}")
        avg_a = sum(sa.values()) / len(sa) if sa else 0
        avg_b = sum(sb.values()) / len(sb) if sb else 0
        print("-" * 44)
        print(f"{'mean(5 dims)':<14} {avg_a:>8.2f} {avg_b:>8.2f} {avg_b - avg_a:>+10.2f}")

    elif mode_a == "keyword" and mode_b == "keyword":
        acc_a = a.get("overall_verse_accuracy", 0)
        acc_b = b.get("overall_verse_accuracy", 0)
        ha, hb = a.get("total_hallucinations", 0), b.get("total_hallucinations", 0)
        print(f"\nVerse accuracy:  A={acc_a:.3f}  B={acc_b:.3f}  Δ={acc_b - acc_a:+.3f}")
        print(f"Hallucinations:  A={ha}  B={hb}  Δ={hb - ha:+d}")

    else:
        print("\nPrint raw keys for manual compare:")
        print("  A overall_scores / overall_verse_accuracy:", list(a.keys()))
        print("  B overall_scores / overall_verse_accuracy:", list(b.keys()))

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
