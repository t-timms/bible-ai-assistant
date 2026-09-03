"""Tests for rag/retrieval.py's async orchestration (mocked — no real ChromaDB/models).

Covers the dense-search timeout fallback added alongside the H-8 fix (a hung
ChromaDB must not hang the request indefinitely), the switch from
sequential sync calls to genuinely concurrent thread-pool calls for
dense + BM25 search, and the R1/R2/R3/R5/R6 index-version, tokenizer,
normalization, passage-delimiter, lookup-probe, and context-budget behaviors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import types
from unittest.mock import patch

import pytest

import rag.retrieval as retrieval
from rag.helpers import INDEX_VERSION
from rag.retrieval import RetrievalHit, _retrieve
from rag.settings import settings


def _run(coro):
    return asyncio.run(coro)


class TestRetrieveDenseTimeout:
    def test_dense_timeout_excludes_dense_results_no_crash(self, monkeypatch) -> None:
        """A slow dense search must not surface its (late) results or raise.

        Note: this is a *soft* timeout — asyncio.wait_for cannot forcibly kill a
        blocking call already running in a worker thread (see comment in
        rag/retrieval.py), so this does not assert a wall-clock bound on
        _retrieve() itself, only that the timed-out dense results are excluded
        and BM25 results still come through.
        """

        def slow_dense_search(query, collection, embedder, n):
            time.sleep(0.2)  # longer than the patched timeout below
            return [RetrievalHit(verse_id="Should Not Appear 1:1", document="x", score=0.0)]

        def fast_bm25_search(query, n):
            return [RetrievalHit(verse_id="John 3:16", document="search_document: text", score=0.0)]

        monkeypatch.setattr(retrieval, "_get_rag", lambda: (object(), None, object()))
        monkeypatch.setattr(retrieval, "_dense_search", slow_dense_search)
        monkeypatch.setattr(retrieval, "_bm25_search", fast_bm25_search)
        monkeypatch.setattr(retrieval, "_fetch_verses_by_refs", lambda refs: [])

        with patch.object(settings, "chroma_query_timeout_seconds", 0.02):
            result = _run(_retrieve("What does John 3:16 say?", top_k=3))

        assert "Should Not Appear" not in result
        assert "John 3:16" in result

    def test_fast_dense_and_bm25_both_contribute(self, monkeypatch) -> None:
        def fast_dense_search(query, collection, embedder, n):
            return [
                RetrievalHit(verse_id="Romans 8:28", document="search_document: text", score=0.0)
            ]

        def fast_bm25_search(query, n):
            return [RetrievalHit(verse_id="John 3:16", document="search_document: text", score=0.0)]

        monkeypatch.setattr(retrieval, "_get_rag", lambda: (object(), None, object()))
        monkeypatch.setattr(retrieval, "_dense_search", fast_dense_search)
        monkeypatch.setattr(retrieval, "_bm25_search", fast_bm25_search)
        monkeypatch.setattr(retrieval, "_fetch_verses_by_refs", lambda refs: [])

        with patch.object(settings, "chroma_query_timeout_seconds", 5.0):
            result = _run(_retrieve("What does the Bible say about love?", top_k=5))

        assert "Romans 8:28" in result
        assert "John 3:16" in result

    def test_no_index_falls_back_to_pinned_only(self, monkeypatch) -> None:
        def raise_not_found():
            raise FileNotFoundError("no index")

        monkeypatch.setattr(retrieval, "_get_rag", raise_not_found)
        monkeypatch.setattr(
            retrieval,
            "_fetch_verses_by_refs",
            lambda refs: [("John 3:16", "For God so loved the world...")],
        )

        result = _run(_retrieve("What does John 3:16 say?", pin_refs=["John 3:16"]))
        assert "John 3:16" in result
        assert "For God so loved the world" in result


class TestBm25TokenizerWiring:
    """R1: corpus and query tokenization must use the shared tokenizer."""

    def test_bm25_search_tokenizes_query_with_shared_tokenizer(self, monkeypatch) -> None:
        captured: dict[str, list[str]] = {}

        class FakeBM25:
            def get_scores(self, tokens):
                captured["tokens"] = list(tokens)
                return [0.0]

        monkeypatch.setattr(
            retrieval,
            "_get_bm25",
            lambda: {
                "bm25": FakeBM25(),
                "ids": ["John 3:16"],
                "documents": ["search_document: For God so loved"],
            },
        )
        hits = retrieval._bm25_search("What does John 3:16 say?", n=1)
        assert captured["tokens"] == ["what", "does", "john", "3", "16", "say"]
        assert hits == []

    def test_get_bm25_stale_version_disables_sparse_with_error_log(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        fake_rank_bm25 = types.ModuleType("rank_bm25")
        fake_rank_bm25.BM25Okapi = lambda tokens: object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "rank_bm25", fake_rank_bm25)

        db_dir = tmp_path / "chroma_db"
        db_dir.mkdir()
        (db_dir / "bm25_index.json").write_text(
            json.dumps(
                {
                    "index_version": INDEX_VERSION - 1,
                    "ids": ["John 3:16"],
                    "documents": ["search_document: For God so loved"],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(retrieval, "_get_chroma_db_path", lambda: db_dir)
        monkeypatch.setattr(retrieval, "_bm25_data", None)

        with caplog.at_level(logging.ERROR, logger="rag.retrieval"):
            assert retrieval._get_bm25() is None
        assert "Rebuild required" in caplog.text

    def test_get_bm25_current_version_loads_without_prefix_token(
        self, tmp_path, monkeypatch
    ) -> None:
        created: dict[str, list[list[str]]] = {}

        class FakeBM25:
            def __init__(self, tokenized_corpus):
                created["corpus"] = tokenized_corpus

        fake_rank_bm25 = types.ModuleType("rank_bm25")
        fake_rank_bm25.BM25Okapi = FakeBM25  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "rank_bm25", fake_rank_bm25)

        db_dir = tmp_path / "chroma_db"
        db_dir.mkdir()
        (db_dir / "bm25_index.json").write_text(
            json.dumps(
                {
                    "index_version": INDEX_VERSION,
                    "ids": ["John 3:16"],
                    "documents": ["search_document: John 3:16 For God so loved"],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(retrieval, "_get_chroma_db_path", lambda: db_dir)
        monkeypatch.setattr(retrieval, "_bm25_data", None)

        data = retrieval._get_bm25()
        assert data is not None
        assert created["corpus"] == [["john", "3", "16", "for", "god", "so", "loved"]]


class TestDenseSearchNormalization:
    """R2: dense queries must be L2-normalized like the indexed embeddings."""

    def test_encode_uses_normalized_embeddings(self) -> None:
        captured: dict[str, dict] = {}

        class FakeEncoded:
            def __init__(self, rows):
                self._rows = rows

            def tolist(self):
                return self._rows

        class FakeEmbedder:
            def encode(self, texts, **kwargs):
                captured["kwargs"] = kwargs
                return FakeEncoded([[0.0, 1.0]])

        class FakeCollection:
            def query(self, **kwargs):
                return {
                    "ids": [["John 3:16"]],
                    "documents": [["search_document: For God so loved"]],
                    "metadatas": [[{"reference": "John 3:16"}]],
                    "distances": [[0.1]],
                }

        hits = retrieval._dense_search("love", FakeCollection(), FakeEmbedder(), n=3)
        assert captured["kwargs"].get("normalize_embeddings") is True
        assert hits[0].verse_id == "John 3:16"


class TestPassageExpansionDelimiter:
    """R3: child_ids lookup must use pipe delimiters to avoid substring collisions."""

    def test_where_clause_uses_pipe_delimiters(self) -> None:
        captured: dict[str, object] = {}

        class FakePassages:
            def get(self, where=None, include=None):
                captured["where"] = where
                return {"ids": [], "documents": [], "metadatas": []}

        out = retrieval._expand_to_passages(["John 3:16"], FakePassages())
        assert captured["where"] == {"child_ids": {"$contains": "|John 3:16|"}}
        assert out == {}


class TestVerseTextLookupIndexProbe:
    """R5: infrastructure outage must surface as IndexUnavailableError, not None."""

    def test_raises_when_index_unavailable(self, monkeypatch) -> None:
        def raise_no_index():
            raise retrieval.IndexUnavailableError("stale index")

        monkeypatch.setattr(retrieval, "_get_rag", raise_no_index)
        with pytest.raises(retrieval.IndexUnavailableError):
            retrieval.verse_text_lookup("John 3:16")


class TestHybridCandidatesWiring:
    """R8: candidate count must come from settings, not a dead constant."""

    def test_settings_value_passed_to_both_searches(self, monkeypatch) -> None:
        seen: dict[str, int] = {}

        def dense(query, collection, embedder, n):
            seen["dense_n"] = n
            return []

        def sparse(query, n):
            seen["bm25_n"] = n
            return []

        monkeypatch.setattr(retrieval, "_get_rag", lambda: (object(), None, object()))
        monkeypatch.setattr(retrieval, "_dense_search", dense)
        monkeypatch.setattr(retrieval, "_bm25_search", sparse)
        monkeypatch.setattr(retrieval, "_fetch_verses_by_refs", lambda refs: [])

        with patch.object(settings, "hybrid_candidates", 7):
            entries = _run(retrieval._retrieve_entries("anything at all", top_k=2))

        assert seen == {"dense_n": 7, "bm25_n": 7}
        assert entries == []


class TestSearchQueryOverride:
    """Exposition path: dense + BM25 search against `search_query` (the verse text),
    not `user_message` (the bare reference, which retrieves verse-number coincidences)."""

    def _capture(self, monkeypatch):
        seen: dict[str, str] = {}

        def dense(query, collection, embedder, n):
            seen["dense_q"] = query
            return []

        def sparse(query, n):
            seen["bm25_q"] = query
            return []

        monkeypatch.setattr(retrieval, "_get_rag", lambda: (object(), None, object()))
        monkeypatch.setattr(retrieval, "_dense_search", dense)
        monkeypatch.setattr(retrieval, "_bm25_search", sparse)
        monkeypatch.setattr(retrieval, "_fetch_verses_by_refs", lambda refs: [])
        return seen

    def test_search_query_used_for_candidates(self, monkeypatch) -> None:
        seen = self._capture(monkeypatch)
        _run(
            retrieval._retrieve_entries(
                "What is 1 Chronicles 9:17 about?",
                top_k=2,
                search_query="The gatekeepers: Shallum, Akkub, Talmon, Ahiman",
            )
        )
        assert seen["dense_q"] == "The gatekeepers: Shallum, Akkub, Talmon, Ahiman"
        assert seen["bm25_q"] == "The gatekeepers: Shallum, Akkub, Talmon, Ahiman"

    def test_none_search_query_falls_back_to_user_message(self, monkeypatch) -> None:
        seen = self._capture(monkeypatch)
        _run(retrieval._retrieve_entries("What does John 3:16 say?", top_k=2))
        assert seen["dense_q"] == "What does John 3:16 say?"
        assert seen["bm25_q"] == "What does John 3:16 say?"

    def test_blank_search_query_falls_back(self, monkeypatch) -> None:
        seen = self._capture(monkeypatch)
        _run(retrieval._retrieve_entries("real question", top_k=2, search_query="   "))
        assert seen["dense_q"] == "real question"


class TestContextBudget:
    """R6b: lowest-ranked unpinned entries are dropped under the char budget."""

    def test_lowest_ranked_dropped_when_budget_exhausted(self, monkeypatch) -> None:
        text = "y" * 50

        def fast_dense(query, collection, embedder, n):
            return [
                RetrievalHit(
                    verse_id=f"Book {i}:1", document=f"search_document: {text}", score=float(i)
                )
                for i in range(4)
            ]

        monkeypatch.setattr(retrieval, "_get_rag", lambda: (object(), None, object()))
        monkeypatch.setattr(retrieval, "_dense_search", fast_dense)
        monkeypatch.setattr(retrieval, "_bm25_search", lambda query, n: [])
        monkeypatch.setattr(retrieval, "_fetch_verses_by_refs", lambda refs: [])

        async def no_rerank(query, candidates, top_k):
            return candidates[:top_k]

        monkeypatch.setattr(retrieval, "_rerank", no_rerank)

        with patch.object(settings, "context_max_chars", 150):
            entries = _run(retrieval._retrieve_entries("topic question", top_k=10))

        # Each entry costs len(ref)=8 + len(text)=50 + overhead=9 = 67 chars;
        # a 150-char budget admits exactly two.
        assert [ref for ref, _ in entries] == ["Book 0:1", "Book 1:1"]
