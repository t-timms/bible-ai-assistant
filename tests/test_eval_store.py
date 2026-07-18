"""Tests for the online evaluation store."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.eval_store import EvalStore


class TestEvalStore:
    """Tests for EvalStore SQLite operations."""

    @pytest.fixture()
    def store(self, tmp_path: Path) -> EvalStore:
        """Create a temporary EvalStore for testing."""
        s = EvalStore(tmp_path / "test_eval.db")
        yield s
        s.close()

    def test_log_interaction(self, store: EvalStore) -> None:
        iid = store.log_interaction(
            request_id="req-001",
            model="test-model",
            query="What does John 3:16 say?",
            response="For God so loved the world...",
            latency_ms=150.0,
        )
        assert iid > 0

    def test_get_unscored(self, store: EvalStore) -> None:
        store.log_interaction(
            request_id="req-001",
            model="test",
            query="Question 1",
            response="Answer 1",
        )
        store.log_interaction(
            request_id="req-002",
            model="test",
            query="Question 2",
            response="Answer 2",
        )
        unscored = store.get_unscored()
        assert len(unscored) == 2

    def test_record_score_marks_as_scored(self, store: EvalStore) -> None:
        iid = store.log_interaction(
            request_id="req-001",
            model="test",
            query="Q",
            response="A",
        )
        store.record_score(
            interaction_id=iid,
            judge_model="judge-v1",
            faithfulness=4,
            citation=3,
            hallucination=1,
            helpfulness=5,
            conciseness=4,
            reasoning="Good response",
        )
        unscored = store.get_unscored()
        assert len(unscored) == 0

    def test_overall_score_calculation(self, store: EvalStore) -> None:
        iid = store.log_interaction(
            request_id="req-001",
            model="test",
            query="Q",
            response="A",
        )
        store.record_score(
            interaction_id=iid,
            judge_model="judge",
            faithfulness=4,
            citation=3,
            hallucination=2,  # higher = worse, so 6-2=4 in overall
            helpfulness=5,
            conciseness=4,
        )
        # overall = (4 + 3 + (6-2) + 5 + 4) / 5 = 20/5 = 4.0
        row = store.conn.execute(
            "SELECT overall FROM scores WHERE interaction_id = ?", (iid,)
        ).fetchone()
        assert abs(row["overall"] - 4.0) < 0.01

    def test_summary(self, store: EvalStore) -> None:
        iid = store.log_interaction(
            request_id="req-001",
            model="test",
            query="Q",
            response="A",
        )
        store.log_interaction(
            request_id="req-002",
            model="test",
            query="Q2",
            response="A2",
        )
        store.record_score(
            interaction_id=iid,
            judge_model="judge",
            faithfulness=4,
            citation=3,
            hallucination=1,
            helpfulness=5,
            conciseness=4,
        )
        summary = store.get_summary()
        assert summary["total_interactions"] == 2
        assert summary["scored_interactions"] == 1
        assert summary["unscored_interactions"] == 1

    def test_empty_store_summary(self, store: EvalStore) -> None:
        summary = store.get_summary()
        assert summary["total_interactions"] == 0
        assert summary["avg_overall"] == 0

    def test_log_interaction_returns_sequential_ids(self, store: EvalStore) -> None:
        id1 = store.log_interaction(
            request_id="req-001",
            model="test",
            query="Q1",
            response="A1",
        )
        id2 = store.log_interaction(
            request_id="req-002",
            model="test",
            query="Q2",
            response="A2",
        )
        assert id2 == id1 + 1

    def test_unscored_respects_limit(self, store: EvalStore) -> None:
        for i in range(5):
            store.log_interaction(
                request_id=f"req-{i:03d}",
                model="test",
                query=f"Q{i}",
                response=f"A{i}",
            )
        unscored = store.get_unscored(limit=3)
        assert len(unscored) == 3
