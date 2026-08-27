"""Tests for retrieval evaluation: binary metrics, qrels building.

Metric expectations are hand-computed; retrieval itself is exercised only with
synthetic ranked lists (no index/models required).
"""

import json
import math
from pathlib import Path

import pytest

from scripts.retrieval_metrics import binary_metrics


def _import_build_qrels():
    try:
        import scripts.build_qrels as bq

        return bq
    except (ImportError, SystemExit):
        return None


build_qrels_mod = _import_build_qrels()


class TestBinaryMetrics:
    def test_perfect_ranking_scores_one(self) -> None:
        metrics = binary_metrics(["A", "B", "C"], {"A"})
        assert metrics["recall@5"] == 1.0
        assert metrics["recall@10"] == 1.0
        assert metrics["mrr"] == 1.0
        assert metrics["ndcg"] == 1.0

    def test_single_relevant_at_rank_two_ndcg(self) -> None:
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1 -> nDCG = 1/log2(3)
        metrics = binary_metrics(["B", "A", "C"], {"A"})
        assert metrics["mrr"] == pytest.approx(0.5)
        assert metrics["ndcg"] == pytest.approx(1.0 / math.log2(3))

    def test_mrr_third_position(self) -> None:
        metrics = binary_metrics(["X", "Y", "A"], {"A"})
        assert metrics["mrr"] == pytest.approx(1.0 / 3)

    def test_no_relevant_hit_gives_zeroes(self) -> None:
        metrics = binary_metrics(["X", "Y", "Z"], {"A"})
        assert metrics["recall@5"] == 0.0
        assert metrics["recall@10"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["ndcg"] == 0.0

    def test_recall_at_k_boundary(self) -> None:
        relevant = {"A", "B", "C", "D"}
        ranked = ["A", "x", "B", "y", "z", "w", "v", "u", "t", "s"]
        metrics = binary_metrics(ranked, relevant)
        assert metrics["recall@5"] == pytest.approx(0.5)
        assert metrics["recall@10"] == pytest.approx(0.5)

    def test_multiple_relevant_perfect_order_ndcg_one(self) -> None:
        metrics = binary_metrics(["B", "A"], {"A", "B"})
        assert metrics["ndcg"] == pytest.approx(1.0)


@pytest.mark.skipif(build_qrels_mod is None, reason="scripts.build_qrels not importable")
class TestBuildQrels:
    def test_question_id_stable_under_normalization(self) -> None:
        assert build_qrels_mod.question_id("What is love?") == build_qrels_mod.question_id(
            "  what is love?  "
        )
        assert build_qrels_mod.question_id("What is love?") != build_qrels_mod.question_id(
            "What is faith?"
        )

    def test_extract_gold_refs_from_question_and_answer(self) -> None:
        pytest.importorskip("rag.verification")
        refs = build_qrels_mod.extract_gold_refs(
            "What does John say about God's love in John 3:16?",
            "The answer quotes John 3:16 and echoes Psalm 23:1.",
        )
        assert refs[0] == "John 3:16"
        assert "Psalms 23:1" in refs

    def test_extract_gold_refs_empty_without_references(self) -> None:
        pytest.importorskip("rag.verification")
        assert build_qrels_mod.extract_gold_refs("What is love?", "Love is patient.") == []

    def test_build_qrels_synthetic_suite(self, tmp_path: Path) -> None:
        pytest.importorskip("rag.verification")
        suite = {
            "questions": [
                {
                    "question": "What does John 3:16 say?",
                    "expected_answer": "For God so loved the world...",
                    "category": "verse_lookup",
                },
                {
                    "question": "what does john 3:16 say?",  # normalizes to duplicate
                    "expected_answer": "duplicate",
                    "category": "verse_lookup",
                },
                {
                    "question": "How should I treat my enemies?",
                    "expected_answer": "Love them; see Matthew 5:44.",
                    "category": "topical",
                },
            ]
        }
        (tmp_path / "suite.json").write_text(json.dumps(suite), encoding="utf-8")

        doc = build_qrels_mod.build_qrels(tmp_path)

        assert doc["format"] == "bible-qrels-v1"
        assert doc["sources"] == ["suite.json"]
        assert doc["num_questions"] == 2  # duplicate collapsed by normalization
        assert doc["num_with_gold"] == 2
        relevant_maps = [entry["relevant"] for entry in doc["qrels"].values()]
        assert any("John 3:16" in m for m in relevant_maps)
        assert any("Matthew 5:44" in m for m in relevant_maps)
        categories = {entry["category"] for entry in doc["qrels"].values()}
        assert categories == {"verse_lookup", "topical"}
