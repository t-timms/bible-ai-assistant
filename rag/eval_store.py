"""Online evaluation store — logs live Q&A interactions for offline analysis.

Records every user query and model response to SQLite for:
- Batch LLM-as-judge scoring
- Regression detection across model versions
- Quality trend analysis

WAL mode for non-blocking writes during inference.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EvalStore:
    """SQLite-backed interaction log for online evaluation."""

    def __init__(self, db_path: Path | str = "data/eval_store.db") -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._create_tables()
        logger.info("eval_store_initialized path=%s", db_path)

    def _create_tables(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context_used TEXT DEFAULT '',
                    latency_ms REAL DEFAULT 0.0,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    judge_model TEXT NOT NULL,
                    scored_at TEXT NOT NULL,
                    faithfulness INTEGER DEFAULT 0,
                    citation INTEGER DEFAULT 0,
                    hallucination INTEGER DEFAULT 0,
                    helpfulness INTEGER DEFAULT 0,
                    conciseness INTEGER DEFAULT 0,
                    overall REAL DEFAULT 0.0,
                    reasoning TEXT DEFAULT '',
                    FOREIGN KEY (interaction_id) REFERENCES interactions(id)
                );

                CREATE TABLE IF NOT EXISTS regressions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    baseline_value REAL NOT NULL,
                    delta REAL NOT NULL,
                    window_days INTEGER NOT NULL,
                    details TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_interactions_timestamp
                    ON interactions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_interactions_request_id
                    ON interactions(request_id);
                CREATE INDEX IF NOT EXISTS idx_scores_interaction
                    ON scores(interaction_id);
                CREATE INDEX IF NOT EXISTS idx_regressions_detected
                    ON regressions(detected_at);
            """)
            self.conn.commit()

    def log_interaction(
        self,
        request_id: str,
        model: str,
        query: str,
        response: str,
        context_used: str = "",
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> int:
        """Log a Q&A interaction. Returns the interaction ID."""
        try:
            with self._lock:
                cursor = self.conn.execute(
                    """INSERT INTO interactions
                       (timestamp, request_id, model, query, response,
                        context_used, latency_ms, tokens_in, tokens_out)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(UTC).isoformat(),
                        request_id,
                        model,
                        query,
                        response,
                        context_used,
                        latency_ms,
                        tokens_in,
                        tokens_out,
                    ),
                )
                self.conn.commit()
                return cursor.lastrowid or 0
        except sqlite3.Error:
            logger.error("eval_store_log_failed request_id=%s", request_id, exc_info=True)
            return 0

    def record_score(
        self,
        interaction_id: int,
        judge_model: str,
        faithfulness: int,
        citation: int,
        hallucination: int,
        helpfulness: int,
        conciseness: int,
        reasoning: str = "",
    ) -> None:
        """Record LLM-as-judge scores for an interaction."""
        overall = (faithfulness + citation + (6 - hallucination) + helpfulness + conciseness) / 5.0
        try:
            with self._lock:
                self.conn.execute(
                    """INSERT INTO scores
                       (interaction_id, judge_model, scored_at,
                        faithfulness, citation, hallucination,
                        helpfulness, conciseness, overall, reasoning)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        interaction_id,
                        judge_model,
                        datetime.now(UTC).isoformat(),
                        faithfulness,
                        citation,
                        hallucination,
                        helpfulness,
                        conciseness,
                        overall,
                        reasoning,
                    ),
                )
                self.conn.commit()
        except sqlite3.Error:
            logger.error(
                "eval_score_record_failed interaction_id=%s",
                interaction_id,
                exc_info=True,
            )

    def get_unscored(self, limit: int = 50) -> list[dict]:
        """Get interactions that haven't been scored yet."""
        rows = self.conn.execute(
            """SELECT i.* FROM interactions i
               LEFT JOIN scores s ON i.id = s.interaction_id
               WHERE s.id IS NULL
               ORDER BY i.timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_score_trends(self, days: int = 7) -> dict[str, Any]:
        """Get scoring trends over recent days."""
        rows = self.conn.execute(
            """SELECT
                 date(i.timestamp) as date,
                 COUNT(*) as interactions,
                 AVG(s.overall) as avg_overall,
                 AVG(s.faithfulness) as avg_faithfulness,
                 AVG(s.citation) as avg_citation,
                 AVG(s.hallucination) as avg_hallucination,
                 AVG(s.helpfulness) as avg_helpfulness,
                 AVG(s.conciseness) as avg_conciseness
               FROM interactions i
               JOIN scores s ON i.id = s.interaction_id
               WHERE i.timestamp >= datetime('now', ?)
               GROUP BY date(i.timestamp)
               ORDER BY date(i.timestamp)""",
            (f"-{days} days",),
        ).fetchall()
        return {"daily_trends": [dict(r) for r in rows]}

    def detect_regressions(
        self,
        baseline_days: int = 14,
        recent_days: int = 3,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Detect score regressions by comparing recent vs baseline windows.

        A regression is flagged when a metric drops by more than ``threshold``
        points between the baseline and recent windows.
        """
        metrics = [
            "faithfulness",
            "citation",
            "hallucination",
            "helpfulness",
            "conciseness",
            "overall",
        ]
        regressions: list[dict] = []

        for metric in metrics:
            # metric names are from a hardcoded list, not user input
            baseline = self.conn.execute(
                f"""SELECT AVG(s.{metric}) as val FROM scores s
                    JOIN interactions i ON i.id = s.interaction_id
                    WHERE i.timestamp >= datetime('now', ?)
                      AND i.timestamp < datetime('now', ?)""",
                (f"-{baseline_days} days", f"-{recent_days} days"),
            ).fetchone()

            recent = self.conn.execute(
                f"""SELECT AVG(s.{metric}) as val FROM scores s
                    JOIN interactions i ON i.id = s.interaction_id
                    WHERE i.timestamp >= datetime('now', ?)""",
                (f"-{recent_days} days",),
            ).fetchone()

            if baseline and recent and baseline["val"] and recent["val"]:
                delta = recent["val"] - baseline["val"]
                # For hallucination, higher is worse, so flip the comparison
                is_regression = (
                    delta > threshold if metric == "hallucination" else delta < -threshold
                )
                if is_regression:
                    reg = {
                        "metric": metric,
                        "current_value": round(recent["val"], 2),
                        "baseline_value": round(baseline["val"], 2),
                        "delta": round(delta, 2),
                        "window_days": recent_days,
                    }
                    regressions.append(reg)
                    # Persist
                    with self._lock:
                        self.conn.execute(
                            """INSERT INTO regressions
                               (detected_at, metric, current_value,
                                baseline_value, delta, window_days, details)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                datetime.now(UTC).isoformat(),
                                metric,
                                recent["val"],
                                baseline["val"],
                                delta,
                                recent_days,
                                f"baseline={baseline_days}d recent={recent_days}d",
                            ),
                        )
                        self.conn.commit()

        return regressions

    def get_summary(self) -> dict[str, Any]:
        """Summary statistics for the eval store."""
        total = self.conn.execute("SELECT COUNT(*) as c FROM interactions").fetchone()["c"]
        scored = self.conn.execute(
            "SELECT COUNT(DISTINCT interaction_id) as c FROM scores"
        ).fetchone()["c"]
        avg_scores = self.conn.execute(
            """SELECT
                 AVG(overall) as avg_overall,
                 AVG(faithfulness) as avg_faithfulness,
                 AVG(hallucination) as avg_hallucination
               FROM scores"""
        ).fetchone()

        return {
            "total_interactions": total,
            "scored_interactions": scored,
            "unscored_interactions": total - scored,
            "avg_overall": round(avg_scores["avg_overall"] or 0, 2),
            "avg_faithfulness": round(avg_scores["avg_faithfulness"] or 0, 2),
            "avg_hallucination": round(avg_scores["avg_hallucination"] or 0, 2),
        }

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self.conn.close()


# ── Module singleton ──────────────────────────────────────────────────────────

_store: EvalStore | None = None
_store_lock = threading.Lock()


def get_eval_store(db_path: str = "data/eval_store.db") -> EvalStore:
    """Get or create the module-level EvalStore singleton."""
    global _store  # noqa: PLW0603
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = EvalStore(db_path)
    return _store
