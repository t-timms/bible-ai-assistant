"""Batch LLM-as-judge scoring for logged interactions.

Pulls unscored Q&A pairs from the eval store, sends each to an LLM judge,
and records the scores back.

Usage:
    python scripts/batch_judge.py                           # default judge
    python scripts/batch_judge.py --judge-model gemma3:12b  # custom judge
    python scripts/batch_judge.py --limit 20                # score 20 at a time
    python scripts/batch_judge.py --check-regressions       # also run regression check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.eval_store import get_eval_store

logger = logging.getLogger(__name__)

JUDGE_TEMPLATE = """\
You are an expert evaluator for a Bible Q&A assistant. Score the following \
response on five dimensions (1-5 each):

**User Question:** {question}

**Assistant Response:** {response}

Score each dimension:
1. **faithfulness** (1-5): Is the response faithful to Christian scripture and theology?
2. **citation** (1-5): Does the response cite specific Bible verses with book/chapter/verse?
3. **hallucination** (1-5): Does the response fabricate verses or misattribute quotes? \
(1=no hallucination, 5=severe hallucination)
4. **helpfulness** (1-5): Is the response helpful and relevant to the question?
5. **conciseness** (1-5): Is the response appropriately concise without losing substance?

Respond ONLY with valid JSON (no markdown fences):
{{"faithfulness": N, "citation": N, "hallucination": N, "helpfulness": N, \
"conciseness": N, "reasoning": "brief explanation"}}
"""


def score_interaction(
    question: str,
    response: str,
    judge_model: str,
    ollama_url: str,
) -> dict | None:
    """Score a single interaction using LLM-as-judge."""
    prompt = JUDGE_TEMPLATE.format(question=question, response=response)

    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{ollama_url}/v1/chat/completions",
                json={
                    "model": judge_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 300,
                },
            )
            if r.status_code != 200:
                logger.error("Judge request failed: %s", r.text)
                return None

            content = r.json()["choices"][0]["message"]["content"]
            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return json.loads(content)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error("Failed to parse judge response: %s", e)
        return None
    except httpx.HTTPError as e:
        logger.error("Judge HTTP error: %s", e)
        return None


def main() -> None:
    """Run batch LLM-as-judge scoring on unscored interactions."""
    parser = argparse.ArgumentParser(description="Batch LLM-as-judge scoring")
    parser.add_argument(
        "--judge-model",
        default="qwen3:8b",
        help="Ollama model for judging",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max interactions to score",
    )
    parser.add_argument(
        "--db-path",
        default="data/eval_store.db",
        help="Eval store database path",
    )
    parser.add_argument(
        "--check-regressions",
        action="store_true",
        help="Run regression detection after scoring",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    store = get_eval_store(args.db_path)
    unscored = store.get_unscored(limit=args.limit)

    if not unscored:
        logger.info("No unscored interactions found")
        return

    logger.info("Scoring %d interactions with %s", len(unscored), args.judge_model)

    scored = 0
    failed = 0
    for i, interaction in enumerate(unscored, 1):
        logger.info(
            "[%d/%d] Scoring interaction %d...",
            i,
            len(unscored),
            interaction["id"],
        )

        scores = score_interaction(
            question=interaction["query"],
            response=interaction["response"],
            judge_model=args.judge_model,
            ollama_url=args.ollama_url,
        )

        if scores:
            store.record_score(
                interaction_id=interaction["id"],
                judge_model=args.judge_model,
                faithfulness=scores.get("faithfulness", 0),
                citation=scores.get("citation", 0),
                hallucination=scores.get("hallucination", 0),
                helpfulness=scores.get("helpfulness", 0),
                conciseness=scores.get("conciseness", 0),
                reasoning=scores.get("reasoning", ""),
            )
            scored += 1
        else:
            failed += 1

        # Rate limit to avoid overwhelming Ollama
        if i < len(unscored):
            time.sleep(1)

    logger.info("Scoring complete: %d scored, %d failed", scored, failed)

    # Print summary
    summary = store.get_summary()
    logger.info("Store summary: %s", json.dumps(summary, indent=2))

    if args.check_regressions:
        regressions = store.detect_regressions()
        if regressions:
            logger.warning("REGRESSIONS DETECTED:")
            for r in regressions:
                logger.warning(
                    "  %s: %.2f -> %.2f (delta=%.2f)",
                    r["metric"],
                    r["baseline_value"],
                    r["current_value"],
                    r["delta"],
                )
        else:
            logger.info("No regressions detected")


if __name__ == "__main__":
    main()
