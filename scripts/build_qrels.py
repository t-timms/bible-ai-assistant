#!/usr/bin/env python3
"""Build qrels (relevance judgments) for retrieval evaluation from suite snapshots.

Gold references come from the frozen benchmark suites themselves: every verse
reference extractable from a question or its expected answer becomes a binary-
relevant document id for that question. This measures whether retrieval surfaces
the verses the suite authors deemed correct — it says nothing about generation.

Output (default docs/qrels.json):
  {"format": "bible-qrels-v1",
   "sources": [...],
   "qrels": {"q_<sha256[:12]>": {"question", "category", "relevant": {"John 3:16": 1}}}}

Usage:
  python scripts/build_qrels.py [--output docs/qrels.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_stats import normalize_question  # noqa: E402
from scripts.check_train_eval_overlap import read_json_tolerant  # noqa: E402

try:
    from rag.verification import extract_verse_refs
except ImportError as e:
    raise SystemExit(f"rag.verification is required to build qrels ({e})") from e

try:
    from rag.helpers import _normalize_verse_id
except ImportError:

    def _normalize_verse_id(ref: str) -> str:
        return " ".join((ref or "").split())


DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "qrels.json"


def _collect_records(node: object, found: list[dict]) -> None:
    """Collect dicts carrying a question string, keeping category/expected_answer."""
    if isinstance(node, dict):
        question = node.get("question")
        if isinstance(question, str) and question.strip():
            found.append(
                {
                    "question": question,
                    "category": node.get("category", "unknown"),
                    "expected_answer": node.get("expected_answer"),
                }
            )
        else:
            for value in node.values():
                _collect_records(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_records(item, found)


def question_id(question: str) -> str:
    digest = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
    return f"q_{digest[:12]}"


def extract_gold_refs(question: str, expected_answer: object) -> list[str]:
    """Ordered unique canonical verse ids mentioned by question or gold answer."""
    texts = [question]
    if isinstance(expected_answer, str):
        texts.append(expected_answer)
    refs: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for raw_ref in extract_verse_refs(text):
            ref = _normalize_verse_id(raw_ref)
            if ref and ref.lower() not in seen:
                seen.add(ref.lower())
                refs.append(ref)
    return refs


def build_qrels(suites_dir: Path) -> dict:
    sources: list[str] = []
    records: list[dict] = []
    for path in sorted(suites_dir.glob("*.json")):
        sources.append(path.name)
        _collect_records(read_json_tolerant(path), records)

    deduped: dict[str, dict] = {}
    for record in records:
        qid = question_id(record["question"])
        if qid not in deduped:
            deduped[qid] = record

    qrels: dict[str, dict] = {}
    for qid, record in sorted(deduped.items()):
        refs = extract_gold_refs(record["question"], record["expected_answer"])
        qrels[qid] = {
            "question": record["question"],
            "category": record["category"],
            "relevant": dict.fromkeys(refs, 1),
        }

    return {
        "format": "bible-qrels-v1",
        "sources": sources,
        "num_questions": len(qrels),
        "num_with_gold": sum(1 for entry in qrels.values() if entry["relevant"]),
        "qrels": qrels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "suites",
        help="Directory of frozen suite snapshots",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.suites_dir.is_dir():
        print(f"No suites directory: {args.suites_dir}")
        return 1

    qrels_doc = build_qrels(args.suites_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(qrels_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {args.output}")
    print(
        f"  questions: {qrels_doc['num_questions']} (with gold refs: {qrels_doc['num_with_gold']})"
    )
    categories: dict[str, int] = {}
    for entry in qrels_doc["qrels"].values():
        categories[entry["category"]] = categories.get(entry["category"], 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
