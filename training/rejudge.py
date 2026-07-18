#!/usr/bin/env python3
"""Re-judge existing eval results with a different judge model.

Takes saved responses from a previous eval run and re-scores them
with a new judge model — no RAG server needed.

Usage:
    python training/rejudge.py \
        --input docs/evaluation_results_sft+orpo.json \
        --output docs/evaluation_results_sft+orpo_gemma4judge.json \
        --judge-model gemma4:e4b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Add repo root to path for imports
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from training.evaluate import JUDGE_URL_DEFAULT, judge_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge eval results with a different model.")
    parser.add_argument("--input", required=True, help="Path to existing eval results JSON")
    parser.add_argument("--output", required=True, help="Path to write re-judged results")
    parser.add_argument("--judge-model", default="gemma4:e4b", help="Judge model name")
    parser.add_argument("--judge-url", default=JUDGE_URL_DEFAULT, help="Ollama judge URL")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    dims = ["faithfulness", "citation", "hallucination", "helpfulness", "conciseness"]
    category_scores: dict[str, dict] = {}

    _logger.info("Re-judging %d responses with %s", len(results), args.judge_model)

    for i, r in enumerate(results):
        question = r["question"]
        expected = r["expected_answer"]
        response = r["response"]
        category = r.get("category", "unknown")

        # Skip error responses
        if "[ERROR" in response:
            _logger.warning("[%d/%d] Skipping error response for: %s", i + 1, len(results), question)
            r["judge_scores"] = dict.fromkeys(dims, 0)
            r["judge_scores"]["reasoning"] = "Skipped: original response was an error"
            continue

        print(f"[{i + 1}/{len(results)}] ({category}) {question[:80]}")
        scores = judge_response(question, expected, response, args.judge_url, args.judge_model)
        print(
            f"  F={scores.get('faithfulness', '?')} C={scores.get('citation', '?')} "
            f"H={scores.get('hallucination', '?')} He={scores.get('helpfulness', '?')} "
            f"Co={scores.get('conciseness', '?')}"
        )
        r["judge_scores"] = scores

        if category not in category_scores:
            category_scores[category] = {"total": 0}
            for d in dims:
                category_scores[category][f"{d}_sum"] = 0.0
        cs = category_scores[category]
        cs["total"] += 1
        for d in dims:
            cs[f"{d}_sum"] += scores.get(d, 0)

    # Build summary
    overall = {}
    total = sum(cs["total"] for cs in category_scores.values())
    for d in dims:
        s = sum(cs[f"{d}_sum"] for cs in category_scores.values())
        overall[d] = round(s / total, 3) if total else 0

    cat_summary = {}
    for cat, cs in sorted(category_scores.items()):
        cat_summary[cat] = {"count": cs["total"]}
        for d in dims:
            cat_summary[cat][d] = round(cs[f"{d}_sum"] / cs["total"], 2) if cs["total"] else 0

    output = {
        "eval_mode": "llm-as-judge (re-judged)",
        "judge_model": args.judge_model,
        "original_judge": data.get("judge_model", "unknown"),
        "ollama_model": data.get("ollama_model", "unknown"),
        "model_tag": data.get("model_tag", "unknown"),
        "total_questions": len(results),
        "overall_scores": overall,
        "category_summary": cat_summary,
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"{'Category':<18} {'N':>3}", end="")
    for d in dims:
        print(f" {d[:8]:>9}", end="")
    print(f" {'avg':>7}")
    print("-" * 80)
    for cat in sorted(cat_summary):
        cs = cat_summary[cat]
        vals = [cs[d] for d in dims]
        print(f"{cat:<18} {cs['count']:>3}", end="")
        for v in vals:
            print(f" {v:>9.2f}", end="")
        print(f" {sum(vals)/len(vals):>7.2f}")

    overall_vals = [overall[d] for d in dims]
    print("-" * 80)
    print(f"{'OVERALL':<18} {total:>3}", end="")
    for v in overall_vals:
        print(f" {v:>9.2f}", end="")
    print(f" {sum(overall_vals)/len(overall_vals):>7.3f}")

    _logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
