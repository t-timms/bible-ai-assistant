#!/usr/bin/env python3
"""Evaluate retrieval variants against qrels built by scripts/build_qrels.py.

Metrics (binary relevance): Recall@{5,10}, MRR@10, nDCG@10.
Variants are probed from rag.retrieval internals and skipped individually when
the underlying components (ChromaDB index, embedding model, BM25 data,
cross-encoder) are unavailable on this machine.

Usage:
  python scripts/build_qrels.py            # first
  python scripts/retrieval_metrics.py [--qrels docs/qrels.json]
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

K_VALUES = (5, 10)
FUSION_DEPTH = 20

VARIANTS = ("dense", "bm25", "fused", "fused_rerank")


def load_qrels(path: Path) -> list[dict]:
    import json

    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") != "bible-qrels-v1":
        raise SystemExit(f"Unexpected qrels format in {path}: {doc.get('format')!r}")
    return [{"qid": qid, **entry} for qid, entry in doc["qrels"].items() if entry.get("relevant")]


def binary_metrics(ranked_ids: list[str], relevant: set[str]) -> dict[str, float]:
    """Recall@k, MRR@10 and nDCG@10 with binary gains."""
    top10 = ranked_ids[:10]
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        hits = sum(1 for vid in ranked_ids[:k] if vid in relevant)
        metrics[f"recall@{k}"] = hits / len(relevant)
    mrr = 0.0
    for rank, vid in enumerate(top10, start=1):
        if vid in relevant:
            mrr = 1.0 / rank
            break
    metrics["mrr"] = mrr
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, vid in enumerate(top10, start=1) if vid in relevant
    )
    ideal_depth = min(len(relevant), 10)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_depth + 1))
    metrics["ndcg"] = dcg / idcg if idcg else 0.0
    return metrics


def _variant_available(retrieval_mod: object, variant: str) -> str | None:
    required = {
        "dense": ("_get_rag", "_dense_search"),
        "bm25": ("_bm25_search",),
        "fused": ("_get_rag", "_dense_search", "_bm25_search", "_reciprocal_rank_fusion"),
        "fused_rerank": (
            "_get_rag",
            "_dense_search",
            "_bm25_search",
            "_reciprocal_rank_fusion",
            "_rerank",
        ),
    }[variant]
    missing = [name for name in required if not hasattr(retrieval_mod, name)]
    if missing:
        return f"rag.retrieval lacks {', '.join(missing)}"
    return None


def rank_for_query(retrieval_mod: object, variant: str, query: str, depth: int) -> list[str]:
    """Ranked verse ids for one query under one variant."""
    if variant == "dense":
        collection, _passages, embedder = retrieval_mod._get_rag()
        hits = retrieval_mod._dense_search(query, collection, embedder, depth)
        return [hit.verse_id for hit in hits]

    if variant == "bm25":
        hits = retrieval_mod._bm25_search(query, depth)
        return [hit.verse_id for hit in hits]

    collection, _passages, embedder = retrieval_mod._get_rag()
    dense_hits = retrieval_mod._dense_search(query, collection, embedder, FUSION_DEPTH)
    bm25_hits = retrieval_mod._bm25_search(query, FUSION_DEPTH)
    fused_hits = retrieval_mod._reciprocal_rank_fusion(dense_hits, bm25_hits)

    if variant == "fused":
        return [hit.verse_id for hit in fused_hits[:depth]]

    reranked = asyncio.run(retrieval_mod._rerank(query, fused_hits, depth))
    return [hit.verse_id for hit in reranked]


def evaluate_variant(
    retrieval_mod: object,
    variant: str,
    records: list[dict],
    depth: int,
) -> tuple[dict[str, dict], str | None]:
    """Run a variant over all records; returns ({qid: metrics}, skip_reason)."""
    reason = _variant_available(retrieval_mod, variant)
    if reason:
        return {}, reason

    per_question: dict[str, dict] = {}
    try:
        for record in records:
            ranked = rank_for_query(retrieval_mod, variant, record["question"], depth)
            relevant = set(record["relevant"])
            per_question[record["qid"]] = binary_metrics(ranked, relevant)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        return {}, f"failed during evaluation ({detail})"

    return per_question, None


def aggregate(per_question: dict[str, dict], subset: list[dict]) -> dict[str, float]:
    if not subset:
        return {}
    keys = next(iter(per_question.values())).keys()
    return {key: sum(per_question[r["qid"]][key] for r in subset) / len(subset) for key in keys}


def _fmt_row(label: str, n: int, means: dict[str, float], key_order: list[str]) -> str:
    """`key_order` must match the printed header — `sorted(means)` alphabetizes
    keys ("recall@10" < "recall@5" as strings), silently misaligning every
    column against its header. Caller passes the same `metric_keys` used to
    build the header so the two can never drift apart again."""
    cells = "  ".join(f"{means[key]:.3f}" for key in key_order)
    return f"  {label:<28} n={n:<4} {cells}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "docs" / "qrels.json")
    parser.add_argument("--depth", type=int, default=max(K_VALUES))
    args = parser.parse_args()

    if not args.qrels.exists():
        print(f"qrels not found: {args.qrels}. Run scripts/build_qrels.py first.")
        return 1

    records = load_qrels(args.qrels)
    print(f"Loaded {len(records)} judged questions from {args.qrels}")

    try:
        import rag.retrieval as retrieval_mod
    except ImportError as e:
        print(f"SKIPPED: rag.retrieval unimportable: {e}")
        return 0

    metric_keys = [f"recall@{k}" for k in K_VALUES] + ["mrr", "ndcg"]
    header = "  " + "  ".join(f"{key:>9}" for key in metric_keys)
    any_variant_ran = False

    for variant in VARIANTS:
        per_question, reason = evaluate_variant(retrieval_mod, variant, records, args.depth)
        if reason:
            print(f"\n[{variant}] SKIPPED: {reason}")
            continue
        any_variant_ran = True
        overall = aggregate(per_question, records)
        print(f"\n[{variant}] overall")
        print(header)
        print(_fmt_row("all", len(records), overall, metric_keys))

        categories: dict[str, list[dict]] = {}
        for record in records:
            categories.setdefault(record.get("category", "unknown"), []).append(record)
        for cat, cat_records in sorted(categories.items()):
            print(
                _fmt_row(cat, len(cat_records), aggregate(per_question, cat_records), metric_keys)
            )

    if not any_variant_ran:
        print(
            "\nNo retrieval variant could run on this machine "
            "(missing index/models are expected outside the RAG dev environment)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
