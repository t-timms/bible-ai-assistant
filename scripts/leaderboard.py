#!/usr/bin/env python3
"""
Print a leaderboard of model evaluations from docs/evaluation_results_*.json.
Run from project root: python scripts/leaderboard.py
"""

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main():
    results_files = sorted(DOCS.glob("evaluation_results*.json"))
    if not results_files:
        print("No evaluation results found in docs/. Run evaluate.py first.")
        return

    rows = []
    for f in results_files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        tag = f.stem.replace("evaluation_results", "").lstrip("_") or "default"
        mode = data.get("eval_mode", "?")
        if mode == "llm-as-judge":
            scores = data.get("overall_scores", {})
            avg = sum(scores.values()) / len(scores) if scores else 0
            rows.append((tag, avg, mode, scores))
        else:
            acc = data.get("overall_verse_accuracy", 0)
            cite = data.get("total_citations", 0)
            total = data.get("total_questions", 1)
            hall = data.get("total_hallucinations", 0)
            rows.append(
                (
                    tag,
                    acc,
                    mode,
                    {"verse_accuracy": acc, "citations": f"{cite}/{total}", "hallucinations": hall},
                )
            )

    # Sort: judge avg desc, then keyword verse_accuracy desc
    def key(r):
        tag, val, mode, _ = r
        return (0 if mode == "llm-as-judge" else 1, -val)

    rows.sort(key=key)

    print("\n" + "=" * 70)
    print("BIBLE MODEL EVALUATION LEADERBOARD")
    print("=" * 70)
    for i, (tag, val, mode, extra) in enumerate(rows, 1):
        print(f"\n#{i}  {tag}  ({mode})")
        if mode == "llm-as-judge":
            print(f"    Overall avg: {val:.2f}/5")
            for k, v in (extra or {}).items():
                print(f"    {k}: {v}")
        else:
            print(f"    Verse accuracy: {val:.0%}")
            print(
                f"    Citations: {extra.get('citations', '?')}  Hallucinations: {extra.get('hallucinations', '?')}"
            )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
