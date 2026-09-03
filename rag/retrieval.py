"""Thread-safe lazy loaders and hybrid retrieval pipeline for the RAG server.

Imports from rag.helpers for pure helper functions.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np

from rag.helpers import (
    INDEX_VERSION,
    PASSAGES_COLLECTION,
    QUERY_PREFIX,
    RRF_K,
    VERSES_COLLECTION,
    _clean_doc_text,
    _is_verse_lookup,
    _merge_pin_order,
    _normalize_verse_id,
    strip_document_prefix,
    tokenize_for_bm25,
)
from rag.settings import settings

logger = logging.getLogger(__name__)


class IndexUnavailableError(FileNotFoundError):
    """ChromaDB index is missing or was built by an incompatible index version.

    Subclasses FileNotFoundError so existing graceful-degradation paths (e.g.
    `_retrieve` falling back to pinned-only results) keep working, while callers
    that must distinguish infrastructure failure from an unknown reference
    (citation verification) can catch this type specifically.
    """


# Dedicated small pool for the dense search (R7g): a hung ChromaDB call can
# occupy at most these threads and cannot starve the default executor used by
# BM25 scoring / cross-encoder reranking.
_dense_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-dense")


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


def _get_chroma_db_path() -> Path:
    """Return the ChromaDB directory, respecting CHROMA_DB_PATH env var."""
    env_path = os.getenv("CHROMA_DB_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    return Path(__file__).resolve().parents[1] / "rag" / "chroma_db"


def _read_index_meta(db_path: Path) -> dict[str, object] | None:
    meta_path = db_path / "index_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = _json.load(f)
    except (_json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _check_index_marker(db_path: Path) -> None:
    """Raise IndexUnavailableError when the on-disk index predates INDEX_VERSION."""
    meta = _read_index_meta(db_path)
    found = meta.get("index_version") if meta else None
    if found != INDEX_VERSION:
        raise IndexUnavailableError(
            f"Index at {db_path} is missing or was built with an incompatible "
            f"version (found {found!r}, expected {INDEX_VERSION}). Rebuild required: "
            "python rag/build_index.py"
        )


def _collection_index_version(collection: object) -> object:
    meta = getattr(collection, "metadata", None) or {}
    return meta.get("index_version") if isinstance(meta, dict) else None


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

        db_path = _get_chroma_db_path()
        if not db_path.exists():
            raise IndexUnavailableError(
                f"ChromaDB index not found at {db_path}. Run: python rag/build_index.py"
            )
        try:
            _check_index_marker(db_path)
        except IndexUnavailableError as e:
            logger.error("Stale/incompatible RAG index detected: %s", e)
            raise

        _chroma_client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )
        _verse_collection = _chroma_client.get_collection(VERSES_COLLECTION)
        verse_version = _collection_index_version(_verse_collection)
        if verse_version != INDEX_VERSION:
            logger.error(
                "Verse collection built with incompatible index version (%r != %d) — "
                "results would be silently wrong (pre-cosine/L2 space or old child_ids "
                "encoding). Refusing to serve from it. Rebuild: python rag/build_index.py",
                verse_version,
                INDEX_VERSION,
            )
            raise IndexUnavailableError(
                f"Verse collection index_version {verse_version!r} != expected "
                f"{INDEX_VERSION}. Rebuild required: python rag/build_index.py"
            )
        try:
            _passage_collection = _chroma_client.get_collection(PASSAGES_COLLECTION)
            passage_version = _collection_index_version(_passage_collection)
            if passage_version != INDEX_VERSION:
                logger.error(
                    "Passage collection built with incompatible index version (%r != %d) "
                    "— passage expansion disabled until rebuild: python rag/build_index.py",
                    passage_version,
                    INDEX_VERSION,
                )
                _passage_collection = None
        except (ValueError, KeyError) as e:
            logger.warning("Passage collection not found: %s", e)
            _passage_collection = None
        # trust_remote_code required by nomic-embed-text-v1.5 for custom pooling.
        # revision pinned (H-5): trust_remote_code on an unpinned "main" is a
        # supply-chain risk — see rag/settings.py for details and re-verify date.
        _embedder = SentenceTransformer(
            settings.embed_model,
            revision=settings.embed_model_revision or None,
            trust_remote_code=True,
        )
        return _verse_collection, _passage_collection, _embedder


def _get_bm25():
    """Load BM25 index from JSON (thread-safe, initialises once)."""
    global _bm25_data
    if _bm25_data is not None:
        return _bm25_data
    with _bm25_lock:
        if _bm25_data is not None:
            return _bm25_data
        db_dir = _get_chroma_db_path()
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
                raise TypeError(
                    f"BM25 index '{key}' must be a list, got {type(data[key]).__name__}"
                )
        if len(data["ids"]) != len(data["documents"]):
            raise ValueError(
                f"BM25 index length mismatch: {len(data['ids'])} ids vs "
                f"{len(data['documents'])} documents"
            )
        if not all(isinstance(i, str) for i in data["ids"]):
            raise TypeError("BM25 index 'ids' must contain only strings")
        if not all(isinstance(d, str) for d in data["documents"]):
            raise TypeError("BM25 index 'documents' must contain only strings")

        version = data.get("index_version")
        if version != INDEX_VERSION:
            logger.error(
                "BM25 index at %s has index_version %r but this code expects %d — it was "
                "built with an older tokenizer. Sparse retrieval is DISABLED (never "
                "silently wrong). Rebuild required: python rag/build_index.py",
                json_path,
                version,
                INDEX_VERSION,
            )
            return None

        tokenized = [tokenize_for_bm25(strip_document_prefix(doc)) for doc in data["documents"]]
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

            _reranker = CrossEncoder(
                settings.reranker_model, revision=settings.reranker_model_revision or None
            )
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
    # normalize_embeddings=True must match rag/build_index.py — the collection is
    # built from L2-normalized embeddings and served in cosine space (R2).
    embedding = embedder.encode(
        [QUERY_PREFIX + query], show_progress_bar=False, normalize_embeddings=True
    )
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
    tokenized_query = tokenize_for_bm25(query)
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
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [RetrievalHit(verse_id=vid, document=docs[vid], score=scores[vid]) for vid in sorted_ids]


async def _rerank(query: str, candidates: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    """Cross-encoder reranking. Falls back to RRF order if reranker unavailable.

    Runs the CPU-bound cross-encoder in a thread pool to avoid blocking the
    async event loop.
    """
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return candidates[:top_k]
    pairs = [(query, hit.document) for hit in candidates]
    # Cross-encoder predict is CPU-bound; run in thread pool so the event loop
    # stays responsive under concurrent load.
    ce_scores = await asyncio.to_thread(reranker.predict, pairs)
    ranked = sorted(zip(candidates, ce_scores, strict=True), key=lambda x: x[1], reverse=True)
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
                # child_ids is stored pipe-delimited ("|John 3:16||1 John 3:16|") by
                # rag/build_index.py — matching with the delimiters prevents
                # "John 3:16" from substring-matching "1 John 3:16".
                where={"child_ids": {"$contains": f"|{vid}|"}},
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


def verse_text_lookup(ref: str) -> str | None:
    """Single-reference lookup for citation verification (rag.verification).

    Thin wrapper over `_fetch_verses_by_refs` — returns the real verse text for
    `ref` (handling Psalm/Psalms aliasing), or None if it doesn't resolve in the
    index (nonexistent book, or a real book with a chapter:verse that doesn't
    exist).

    Probes `_get_rag()` first so IndexUnavailableError (missing/stale index)
    propagates to callers instead of being swallowed as "reference not found" —
    an infrastructure outage must not be reported as unverified citations.
    """
    _get_rag()
    hits = _fetch_verses_by_refs([ref])
    if not hits:
        return None
    return hits[0][1]


# Per-entry rendering overhead: "- **" + "**: " + joining "\n" (R6b).
_ENTRY_OVERHEAD = 9


async def _retrieve_entries(
    user_message: str,
    top_k: int = 5,
    pin_refs: list[str] | None = None,
    search_query: str | None = None,
) -> list[tuple[str, str]]:
    """Hybrid retrieval: Dense + BM25 -> RRF -> Rerank -> (ref, text) entries.

    pin_refs: verse ids (e.g. Hebrews 11:1) prepended so explicit lookups are never
    dropped when hybrid search ranks other verses higher. Pinned entries are
    protected from the settings.context_max_chars budget; lowest-ranked unpinned
    entries are skipped first when the budget is exhausted.

    search_query: when given, dense + BM25 candidate retrieval run against this
    string instead of `user_message` (the cross-encoder rerank still scores
    against `user_message`). Used for exposition questions — searching with the
    verse's own text finds thematic neighbours, whereas the bare "Book chap:verse"
    reference in the raw question retrieves verse-number coincidences.
    """

    retrieval_query = (
        search_query.strip() if search_query and search_query.strip() else user_message
    )
    pin_refs = _merge_pin_order(pin_refs or [])
    pinned = _fetch_verses_by_refs(pin_refs)
    pinned_ids = {vid for vid, _ in pinned}

    try:
        verse_collection, passage_collection, embedder = _get_rag()
    except FileNotFoundError:
        return list(pinned)

    _t0 = time.monotonic()

    # Stage 1: Parallel dense + BM25 search. Both are sync/CPU-or-IO-bound calls
    # (ChromaDB query, BM25Okapi scoring) — run in threads via asyncio.gather so
    # they actually run concurrently and don't block the event loop, matching
    # the pattern already used for reranking (_rerank). The dense search alone
    # gets a wall-clock timeout so a slow/unresponsive ChromaDB doesn't dominate
    # response latency indefinitely.
    #
    # This is a *soft* timeout, not a hard kill: asyncio.wait_for cannot forcibly
    # interrupt a synchronous call already running in a worker thread (Future.cancel()
    # is a no-op once the thread has started; CPython gives no safe way to abort a
    # running thread). Once the timeout elapses, this coroutine stops *waiting* on the
    # slow result and proceeds without it — but the worker thread itself keeps running
    # in the background until the underlying call returns on its own. A true hard
    # timeout would need the query to run in a separate process (killable), or the
    # underlying client to support its own cancellable timeout — out of scope here.
    #
    # The dense search runs on the dedicated _dense_executor pool rather than the
    # default executor: a hung ChromaDB call then occupies at most that small pool
    # and cannot starve BM25 scoring / cross-encoder reranking.
    #
    # return_exceptions=True + per-result handling (not a shared try/except
    # around gather) matters here: asyncio.gather does not cancel sibling
    # awaitables when one raises, so a naive except-and-retry would re-run
    # (and potentially queue behind) the BM25 thread instead of reusing the
    # result it already computed concurrently.
    loop = asyncio.get_running_loop()
    dense_task = asyncio.wait_for(
        loop.run_in_executor(
            _dense_executor,
            _dense_search,
            retrieval_query,
            verse_collection,
            embedder,
            settings.hybrid_candidates,
        ),
        timeout=settings.chroma_query_timeout_seconds,
    )
    bm25_task = asyncio.to_thread(_bm25_search, retrieval_query, settings.hybrid_candidates)
    dense_outcome, bm25_outcome = await asyncio.gather(
        dense_task, bm25_task, return_exceptions=True
    )

    if isinstance(dense_outcome, BaseException):
        logger.warning(
            "RAG dense search failed (%.1fs timeout): %s — falling back to BM25 only",
            settings.chroma_query_timeout_seconds,
            dense_outcome,
        )
        dense_results: list[RetrievalHit] = []
    else:
        dense_results = dense_outcome

    if isinstance(bm25_outcome, BaseException):
        logger.warning("RAG BM25 search failed: %s", bm25_outcome)
        bm25_results: list[RetrievalHit] = []
    else:
        bm25_results = bm25_outcome
    logger.debug("RAG stage1 dense+BM25: %.3fs", time.monotonic() - _t0)

    # Stage 2: Reciprocal Rank Fusion
    _t1 = time.monotonic()
    fused = _reciprocal_rank_fusion(dense_results, bm25_results) if bm25_results else dense_results
    logger.debug("RAG stage2 RRF: %.3fs", time.monotonic() - _t1)

    if not fused and not pinned:
        return []

    # Stage 3: Cross-encoder reranking (skip ids we already pinned)
    _t2 = time.monotonic()
    if fused:
        fused_filtered = [h for h in fused if h.verse_id not in pinned_ids]
        reranked = await _rerank(user_message, fused_filtered, top_k) if fused_filtered else []
    else:
        reranked = []
    logger.debug("RAG stage3 rerank: %.3fs", time.monotonic() - _t2)

    # Stage 4: Passage expansion for thematic queries + budgeted entry selection.
    is_lookup = _is_verse_lookup(user_message)
    verse_ids = [h.verse_id for h in reranked]

    if not is_lookup and passage_collection is not None:
        passages = _expand_to_passages(verse_ids, passage_collection)
    else:
        passages = {}

    entries: list[tuple[str, str]] = list(pinned)
    used = sum(len(vid) + len(text) + _ENTRY_OVERHEAD for vid, text in pinned)
    seen_passages: set[str] = set()
    dropped = 0
    for hit in reranked:
        if hit.verse_id in pinned_ids:
            continue
        if hit.verse_id in passages and passages[hit.verse_id] not in seen_passages:
            ref = f"{hit.verse_id} (passage)"
            text = passages[hit.verse_id]
            seen_passages.add(text)
        else:
            ref = hit.verse_id
            text = _clean_doc_text(hit.document, hit.verse_id)
        cost = len(ref) + len(text) + _ENTRY_OVERHEAD
        if used + cost > settings.context_max_chars:
            dropped += 1
            continue
        used += cost
        entries.append((ref, text))
    if dropped:
        logger.info(
            "Context budget %d chars reached: dropped %d lowest-ranked entries",
            settings.context_max_chars,
            dropped,
        )

    logger.debug("RAG retrieve total: %.3fs", time.monotonic() - _t0)
    return entries


def format_context_entry(ref: str, text: str) -> str:
    """Render one retrieved entry as a markdown bullet line."""
    return f"- **{ref}**: {text}"


async def _retrieve(
    user_message: str,
    top_k: int = 5,
    pin_refs: list[str] | None = None,
    search_query: str | None = None,
) -> str:
    """Backward-compatible wrapper: hybrid retrieval as a formatted context string."""
    entries = await _retrieve_entries(
        user_message, top_k=top_k, pin_refs=pin_refs, search_query=search_query
    )
    return "\n".join(format_context_entry(ref, text) for ref, text in entries)
