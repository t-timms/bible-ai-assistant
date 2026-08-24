"""Tests for rag/retrieval.py's async orchestration (mocked — no real ChromaDB/models).

Covers the dense-search timeout fallback added alongside the H-8 fix (a hung
ChromaDB must not hang the request indefinitely) and the switch from
sequential sync calls to genuinely concurrent asyncio.to_thread calls for
dense + BM25 search.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import rag.retrieval as retrieval
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
