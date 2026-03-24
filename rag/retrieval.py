"""Thread-safe lazy loaders and hybrid retrieval pipeline for the RAG server.

Imports from rag.helpers for pure helper functions.
"""

from __future__ import annotations

import json as _json
import logging
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from rag.helpers import (
    HYBRID_CANDIDATES,
    PASSAGES_COLLECTION,
    QUERY_PREFIX,
    RRF_K,
    VERSES_COLLECTION,
    _clean_doc_text,
    _is_verse_lookup,
    _merge_pin_order,
    _normalize_verse_id,
)

logger = logging.getLogger(__name__)


class RetrievalHit(NamedTuple):
    """A single result from the hybrid retrieval pipeline.

    Attributes:
        verse_id:  ChromaDB document id (e.g. ``"John 3:16"``).
        document:  Raw stored document text (may include search_document prefix).
        score:     Rank position (dense/BM25) or RRF / cross-encoder score.
    """

    verse_id: str
    document: str
    score: float

# ---------------------------------------------------------------------------
# Lazy-loaded globals with thread safety (double-checked locking)
# ---------------------------------------------------------------------------

_chroma_client = None
_verse_collection = None
_passage_collection = None
_embedder = None
_bm25_data = None
_reranker = None

_rag_lock = threading.Lock()
_bm25_lock = threading.Lock()
_reranker_lock = threading.Lock()


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_rag():
    """Load ChromaDB collections and embedding model (thread-safe, initialises once)."""
    global _chroma_client, _verse_collection, _passage_collection, _embedder
    if _verse_collection is not None and _embedder is not None:
        return _verse_collection, _passage_collection, _embedder
    with _rag_lock:
        if _verse_collection is not None and _embedder is not None:
            return _verse_collection, _passage_collection, _embedder
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("RAG requires chromadb and sentence-transformers.") from e

        db_path = _get_project_root() / "rag" / "chroma_db"
        if not db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB index not found at {db_path}. Run: python rag/build_index.py"
            )
        _chroma_client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )
        _verse_collection = _chroma_client.get_collection(VERSES_COLLECTION)
        try:
            _passage_collection = _chroma_client.get_collection(PASSAGES_COLLECTION)
        except (ValueError, KeyError) as e:
            logger.warning("Passage collection not found: %s", e)
            _passage_collection = None
        # trust_remote_code required by nomic-embed-text-v1.5 for custom pooling
        _embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        return _verse_collection, _passage_collection, _embedder


def _get_bm25():
    """Load BM25 index from JSON (thread-safe, initialises once)."""
    global _bm25_data
    if _bm25_data is not None:
        return _bm25_data
    with _bm25_lock:
        if _bm25_data is not None:
            return _bm25_data
        db_dir = _get_project_root() / "rag" / "chroma_db"
        json_path = db_dir / "bm25_index.json"
        if not json_path.exists():
            logger.warning("BM25 index not found at %s — sparse retrieval disabled", json_path)
            return None
        from rank_bm25 import BM25Okapi

        with open(json_path, encoding="utf-8") as f:
            data = _json.load(f)

        # Validate schema — malformed or tampered index must not reach the retrieval pipeline
        if not isinstance(data, dict):
            raise ValueError(f"BM25 index must be a JSON object, got {type(data).__name__}")
        for key in ("ids", "documents"):
            if key not in data:
                raise KeyError(f"BM25 index missing required key: {key!r}")
            if not isinstance(data[key], list):
                raise TypeError(f"BM25 index '{key}' must be a list, got {type(data[key]).__name__}")
        if len(data["ids"]) != len(data["documents"]):
            raise ValueError(
                f"BM25 index length mismatch: {len(data['ids'])} ids vs "
                f"{len(data['documents'])} documents"
            )
        if not all(isinstance(i, str) for i in data["ids"]):
            raise TypeError("BM25 index 'ids' must contain only strings")
        if not all(isinstance(d, str) for d in data["documents"]):
            raise TypeError("BM25 index 'documents' must contain only strings")

        tokenized = [doc.lower().split() for doc in data["documents"]]
        bm25 = BM25Okapi(tokenized)
        _bm25_data = {"bm25": bm25, "ids": data["ids"], "documents": data["documents"]}
        logger.info("Loaded BM25 index from JSON (%d docs)", len(data["ids"]))
        return _bm25_data


def _get_reranker():
    """Load cross-encoder reranker model (thread-safe, initialises once)."""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            logger.info("Loaded cross-encoder reranker (bge-reranker-v2-m3)")
            return _reranker
        except (ImportError, OSError) as e:
            logger.warning("Reranker unavailable: %s", e)
            return None


def release_resources() -> None:
    """Release all heavy model objects (called on shutdown)."""
    global _chroma_client, _verse_collection, _passage_collection, _embedder, _bm25_data, _reranker
    _chroma_client = None
    _verse_collection = None
    _passage_collection = None
    _embedder = None
    _bm25_data = None
    _reranker = None
    logger.info("RAG server resources released")


# ---------------------------------------------------------------------------
# Hybrid retrieval pipeline
# ---------------------------------------------------------------------------


def _dense_search(query: str, collection, embedder, n: int) -> list[RetrievalHit]:
    """Dense vector search via ChromaDB. Returns ranked RetrievalHits."""
    embedding = embedder.encode([QUERY_PREFIX + query], show_progress_bar=False)
    results = collection.query(
        query_embeddings=embedding.tolist(),
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    out: list[RetrievalHit] = []
    if results and results["ids"] and results["ids"][0]:
        ids_list = results["ids"][0]
        docs_list = results["documents"][0]
        for i, (vid, doc) in enumerate(zip(ids_list, docs_list, strict=True)):
            out.append(RetrievalHit(verse_id=vid, document=doc, score=float(i)))
    return out


def _bm25_search(query: str, n: int) -> list[RetrievalHit]:
    """BM25 sparse search. Returns ranked RetrievalHits."""
    bm25_data = _get_bm25()
    if bm25_data is None:
        return []
    bm25 = bm25_data["bm25"]
    ids = bm25_data["ids"]
    documents = bm25_data["documents"]
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:n]
    return [
        RetrievalHit(verse_id=ids[i], document=documents[i], score=float(rank))
        for rank, i in enumerate(top_indices)
        if scores[i] > 0
    ]


def _reciprocal_rank_fusion(
    *result_lists: list[RetrievalHit], k: int = RRF_K
) -> list[RetrievalHit]:
    """Merge multiple ranked lists using RRF. Returns sorted RetrievalHits."""
    scores: dict[str, float] = {}
    docs: dict[str, str] = {}
    for results in result_lists:
        for hit in results:
            scores[hit.verse_id] = scores.get(hit.verse_id, 0.0) + 1.0 / (k + hit.score + 1)
            docs[hit.verse_id] = hit.document
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [RetrievalHit(verse_id=vid, document=docs[vid], score=scores[vid]) for vid in sorted_ids]


def _rerank(query: str, candidates: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    """Cross-encoder reranking. Falls back to RRF order if reranker unavailable."""
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return candidates[:top_k]
    pairs = [(query, hit.document) for hit in candidates]
    ce_scores = reranker.predict(pairs)
    ranked = sorted(
        zip(candidates, ce_scores, strict=True), key=lambda x: x[1], reverse=True
    )
    return [
        RetrievalHit(verse_id=hit.verse_id, document=hit.document, score=float(s))
        for hit, s in ranked[:top_k]
    ]


def _expand_to_passages(verse_ids: list[str], passage_collection) -> dict[str, str]:
    """For thematic queries, look up parent passages for matched verses."""
    if passage_collection is None:
        return {}
    expanded: dict[str, str] = {}
    for vid in verse_ids:
        try:
            results = passage_collection.get(
                where={"child_ids": {"$contains": vid}},
                include=["documents", "metadatas"],
            )
            if results and results["ids"]:
                doc = results["documents"][0]
                meta = results["metadatas"][0]
                ref = meta.get("reference", "")
                expanded[vid] = _clean_doc_text(doc, ref)
        except (ValueError, KeyError) as e:
            logger.debug("Passage expansion failed for %s: %s", vid, e)
    return expanded


def _fetch_verses_by_refs(refs: list[str]) -> list[tuple[str, str]]:
    """Load verse text by Chroma id; try Psalm/Psalms alias if needed."""

    refs = [_normalize_verse_id(r) for r in refs if r and str(r).strip()]
    if not refs:
        return []
    try:
        verse_collection, _, _ = _get_rag()
    except FileNotFoundError:
        return []

    results: list[tuple[str, str]] = []
    seen_chroma_ids: set[str] = set()
    for raw in refs:
        candidates = [raw]
        low = raw.lower()
        if low.startswith("psalms ") and len(raw.split(" ", 1)) == 2:
            candidates.append("Psalm " + raw.split(" ", 1)[1])
        elif low.startswith("psalm ") and len(raw.split(" ", 1)) == 2:
            candidates.append("Psalms " + raw.split(" ", 1)[1])

        for cid in candidates:
            if cid in seen_chroma_ids:
                break
            try:
                res = verse_collection.get(ids=[cid], include=["documents"])
                ids_r = res.get("ids") or []
                docs_r = res.get("documents") or []
            except (ValueError, KeyError) as e:
                logger.debug("Verse lookup failed for %s: %s", cid, e)
                continue
            if not ids_r or not docs_r:
                continue
            vid, doc = ids_r[0], docs_r[0]
            seen_chroma_ids.add(vid)
            text = _clean_doc_text(doc, vid)
            results.append((vid, text))
            break

    return results


def _retrieve(
    user_message: str,
    top_k: int = 5,
    pin_refs: list[str] | None = None,
) -> str:
    """Hybrid retrieval: Dense + BM25 -> RRF -> Rerank -> format context string.

    pin_refs: verse ids (e.g. Hebrews 11:1) prepended so explicit lookups are never
    dropped when hybrid search ranks other verses higher.
    """

    pin_refs = _merge_pin_order(pin_refs or [])
    pinned = _fetch_verses_by_refs(pin_refs)
    pinned_ids = {vid for vid, _ in pinned}

    try:
        verse_collection, passage_collection, embedder = _get_rag()
    except FileNotFoundError:
        if not pinned:
            return ""
        lines = [f"- **{vid}**: {text}" for vid, text in pinned]
        return "\n".join(lines)

    _t0 = time.monotonic()

    # Stage 1: Parallel dense + BM25 search
    dense_results = _dense_search(user_message, verse_collection, embedder, HYBRID_CANDIDATES)
    bm25_results = _bm25_search(user_message, HYBRID_CANDIDATES)
    logger.debug("RAG stage1 dense+BM25: %.3fs", time.monotonic() - _t0)

    # Stage 2: Reciprocal Rank Fusion
    _t1 = time.monotonic()
    fused = _reciprocal_rank_fusion(dense_results, bm25_results) if bm25_results else dense_results
    logger.debug("RAG stage2 RRF: %.3fs", time.monotonic() - _t1)

    if not fused and not pinned:
        return ""

    # Stage 3: Cross-encoder reranking (skip ids we already pinned)
    _t2 = time.monotonic()
    if fused:
        fused_filtered = [h for h in fused if h.verse_id not in pinned_ids]
        reranked = _rerank(user_message, fused_filtered, top_k) if fused_filtered else []
    else:
        reranked = []
    logger.debug("RAG stage3 rerank: %.3fs", time.monotonic() - _t2)

    # Stage 4: Format context (with passage expansion for thematic queries)
    is_lookup = _is_verse_lookup(user_message)
    verse_ids = [h.verse_id for h in reranked]

    if not is_lookup and passage_collection is not None:
        passages = _expand_to_passages(verse_ids, passage_collection)
    else:
        passages = {}

    lines = [f"- **{vid}**: {text}" for vid, text in pinned]
    seen_passages = set()
    for hit in reranked:
        if hit.verse_id in pinned_ids:
            continue
        if hit.verse_id in passages and passages[hit.verse_id] not in seen_passages:
            seen_passages.add(passages[hit.verse_id])
            lines.append(f"- **{hit.verse_id} (passage)**: {passages[hit.verse_id]}")
        else:
            text = _clean_doc_text(hit.document, hit.verse_id)
            lines.append(f"- **{hit.verse_id}**: {text}")

    logger.debug("RAG retrieve total: %.3fs", time.monotonic() - _t0)
    return "\n".join(lines)
