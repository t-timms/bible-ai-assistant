"""
FastAPI RAG middleware with hybrid retrieval (Dense + BM25 + RRF + Cross-Encoder Reranking).

Pipeline:
  1. Dense search (nomic-embed-text-v1.5 via ChromaDB) -> top 20
  2. BM25 sparse search (rank_bm25) -> top 20
  3. Reciprocal Rank Fusion to merge results
  4. Cross-encoder reranking (bge-reranker-v2-m3) -> top K
  5. Parent-child passage expansion for thematic questions

Run: uvicorn rag.rag_server:app --host 127.0.0.1 --port 8081
"""
import json as _json
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from rag.response_cleanup import strip_model_thinking

logger = logging.getLogger(__name__)

app = FastAPI(title="Bible AI RAG Server", version="1.0.0")

# Maximum request body size (bytes) to prevent DoS via oversized payloads
MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MB


@app.exception_handler(Exception)
async def _handle_unhandled(request: Request, exc: Exception):
    import traceback
    logger.error("Unhandled exception: %s", traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


def _validate_ollama_url(url: str) -> str:
    """Validate that OLLAMA_URL is a well-formed HTTP(S) URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"OLLAMA_URL must use http or https scheme, got: {url!r}")
    if not parsed.netloc:
        raise ValueError(f"OLLAMA_URL has no host: {url!r}")
    return url


OLLAMA_URL = _validate_ollama_url(os.getenv("OLLAMA_URL", "http://localhost:11434"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "20"))
# Reciprocal Rank Fusion smoothing constant (standard default from the RRF paper)
RRF_K = 60
VERSES_COLLECTION = "bible_verses"
PASSAGES_COLLECTION = "bible_passages"
QUERY_PREFIX = "search_query: "

# Topical questions: pin a few high-signal verses so hybrid retrieval + passage expansion
# cannot drown the topic (e.g. marriage → unrelated "love" parables).
_TOPICAL_PIN_TABLE: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (
        frozenset({
            "marriage", "married", "marry", "spouse", "husband", "wife",
            "wedding", "divorce", "remarry",
        }),
        ("Genesis 2:24", "Ephesians 5:31", "Matthew 19:5", "Mark 10:9"),
    ),
    (
        frozenset({"forgiveness", "forgive", "forgiving"}),
        ("Matthew 6:14", "Ephesians 4:32", "Colossians 3:13"),
    ),
    (
        frozenset({"money", "wealth", "rich", "greed", "steward"}),
        ("1 Timothy 6:10", "Matthew 6:24", "Proverbs 3:9"),
    ),
)

_COUNSELING_HINT = re.compile(
    r"\b("
    r"counseling|counsellor|counselor|counsel\s+me|\bcounsel\b|"
    r"therapy|therapist|psychiatr|"
    r"suicid|kill myself|end it all|self[- ]harm|"
    r"depress|anxiety|panic attack|ptsd|trauma|"
    r"marriage crisis|my marriage is|should i divorce|leaving my wife|leaving my husband|"
    r"abuse[sd]?\s+me|domestic violence|"
    r"pastoral care for me|pray\s+for\s+my\s+situation|need\s+someone\s+to\s+talk\s+to"
    r")\b",
    re.IGNORECASE,
)

_VERSE_REF_IN_QUESTION = re.compile(
    r"\b((?:[123]\s)?[A-Za-z][A-Za-z]+(?:\s[A-Za-z]+){0,3}\s\d{1,3}:\d{1,3})\b",
)

_COUNSELING_SYSTEM_GUARD = (
    "The user message may request personal counseling, therapy, crisis intervention, "
    "or intimate life direction (e.g. marriage crisis, mental health, abuse). "
    "You MUST NOT counsel, diagnose, or give tailored life advice. "
    "Respond briefly with kindness: you are a Scripture study aid, not a pastor or clinician; "
    "urge them to speak with a qualified pastor, licensed counselor, or appropriate crisis line. "
    "You may cite 1–2 broadly relevant verses only if they fit, without applying them to their private situation."
)

EMPTY_MODEL_REPLY = (
    "I didn't receive a complete reply from the model. "
    "Please try again or shorten your question."
)

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_chroma_client = None
_verse_collection = None
_passage_collection = None
_embedder = None
_bm25_data = None
_reranker = None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_rag():
    """Load ChromaDB collections and embedding model."""
    global _chroma_client, _verse_collection, _passage_collection, _embedder
    if _verse_collection is not None and _embedder is not None:
        return _verse_collection, _passage_collection, _embedder
    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "RAG requires chromadb and sentence-transformers."
        ) from e

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
    _embedder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    return _verse_collection, _passage_collection, _embedder


def _get_bm25():
    """Load BM25 index from JSON (preferred) or legacy pickle."""
    global _bm25_data
    if _bm25_data is not None:
        return _bm25_data
    db_dir = _get_project_root() / "rag" / "chroma_db"
    json_path = db_dir / "bm25_index.json"
    pkl_path = db_dir / "bm25_index.pkl"

    if json_path.exists():
        from rank_bm25 import BM25Okapi
        with open(json_path, encoding="utf-8") as f:
            data = _json.load(f)
        tokenized = [doc.lower().split() for doc in data["documents"]]
        bm25 = BM25Okapi(tokenized)
        _bm25_data = {"bm25": bm25, "ids": data["ids"], "documents": data["documents"]}
        logger.info("Loaded BM25 index from JSON (%d docs)", len(data["ids"]))
        return _bm25_data

    if pkl_path.exists():
        import pickle
        logger.warning("Loading BM25 from legacy pickle — rebuild index to use safer JSON format")
        with open(pkl_path, "rb") as f:
            _bm25_data = pickle.load(f)
        logger.info("Loaded BM25 index from pickle (%d docs)", len(_bm25_data["ids"]))
        return _bm25_data

    return None


def _get_reranker():
    """Load cross-encoder reranker model."""
    global _reranker
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


# ---------------------------------------------------------------------------
# Hybrid retrieval pipeline
# ---------------------------------------------------------------------------

def _dense_search(query: str, collection, embedder, n: int) -> list[tuple[str, str, float]]:
    """Dense vector search via ChromaDB. Returns [(id, doc_text, rank_score), ...]."""
    embedding = embedder.encode([QUERY_PREFIX + query], show_progress_bar=False)
    results = collection.query(
        query_embeddings=embedding.tolist(),
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    out: list[tuple[str, str, float]] = []
    if results and results["ids"] and results["ids"][0]:
        ids_list = results["ids"][0]
        docs_list = results["documents"][0]
        for i, (vid, doc) in enumerate(zip(ids_list, docs_list, strict=True)):
            out.append((vid, doc, float(i)))
    return out


def _bm25_search(query: str, n: int) -> list[tuple[str, str, float]]:
    """BM25 sparse search. Returns [(id, doc_text, rank_score), ...]."""
    bm25_data = _get_bm25()
    if bm25_data is None:
        return []
    bm25 = bm25_data["bm25"]
    ids = bm25_data["ids"]
    documents = bm25_data["documents"]
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:n]
    return [(ids[i], documents[i], float(rank)) for rank, i in enumerate(top_indices)
            if scores[i] > 0]


def _reciprocal_rank_fusion(
    *result_lists: list[tuple[str, str, float]], k: int = RRF_K
) -> list[tuple[str, str, float]]:
    """Merge multiple ranked lists using RRF. Returns sorted [(id, doc, rrf_score)]."""
    scores: dict[str, float] = {}
    docs: dict[str, str] = {}
    for results in result_lists:
        for vid, doc, rank in results:
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank + 1)
            docs[vid] = doc
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [(vid, docs[vid], scores[vid]) for vid in sorted_ids]


def _rerank(query: str, candidates: list[tuple[str, str, float]], top_k: int) -> list[tuple[str, str, float]]:
    """Cross-encoder reranking. Falls back to RRF order if reranker unavailable."""
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return candidates[:top_k]
    pairs = [(query, doc) for _, doc, _ in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [(vid, doc, float(score)) for (vid, doc, _), score in ranked[:top_k]]


def _clean_doc_text(doc: str, ref: str) -> str:
    """Strip embedding prefix and reference prefix from stored document text."""
    text = doc
    if text.startswith("search_document: "):
        text = text[len("search_document: "):]
    if ref and text.startswith(ref + ": "):
        text = text[len(ref) + 2:]
    return text.strip()


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


def _normalize_verse_id(ref: str) -> str:
    """Map common aliases to Chroma ids (e.g. Psalm 1:1 → Psalms 1:1)."""
    ref = re.sub(r"\s+", " ", (ref or "").strip())
    if not ref:
        return ref
    m = re.match(r"^(.+?)\s+(\d{1,3}:\d{1,3})$", ref)
    if not m:
        return ref
    book, cv = m.group(1).strip(), m.group(2)
    if book.lower() == "psalm":
        book = "Psalms"
    return f"{book} {cv}"


def _extract_verse_ref_from_lookup(question: str) -> str | None:
    """Book/chapter:verse named in a 'What does X say?' lookup, or None."""
    if not _is_verse_lookup(question):
        return None
    # Drop leading "What does/is …" so the verse regex cannot match "What does Hebrews…"
    t = question.strip()
    low = t.lower()
    for prefix in ("what does ", "what is ", "what says "):
        if low.startswith(prefix):
            t = t[len(prefix) :].strip()
            low = t.lower()
            break
    m = _VERSE_REF_IN_QUESTION.search(t)
    if not m:
        return None
    return _normalize_verse_id(m.group(1))


def _topical_anchor_refs(question: str) -> list[str]:
    """Extra verses to pin for broad topical questions (not verse lookups)."""
    if _is_verse_lookup(question):
        return []
    q = question.lower()
    for keywords, refs in _TOPICAL_PIN_TABLE:
        if any(kw in q for kw in keywords):
            return list(refs)
    return []


def _is_counseling_request(question: str) -> bool:
    """Personal counseling / crisis / intimate life-direction phrasing."""
    return bool(question and _COUNSELING_HINT.search(question))


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


def _merge_pin_order(pin_refs: list[str]) -> list[str]:
    """Dedupe while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for r in pin_refs:
        n = _normalize_verse_id(r)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _retrieve(user_message: str, top_k: int = RAG_TOP_K, pin_refs: list[str] | None = None) -> str:
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

    # Stage 1: Parallel dense + BM25 search
    dense_results = _dense_search(user_message, verse_collection, embedder, HYBRID_CANDIDATES)
    bm25_results = _bm25_search(user_message, HYBRID_CANDIDATES)

    # Stage 2: Reciprocal Rank Fusion
    fused = (
        _reciprocal_rank_fusion(dense_results, bm25_results) if bm25_results else dense_results
    )

    if not fused and not pinned:
        return ""

    # Stage 3: Cross-encoder reranking (skip ids we already pinned)
    if fused:
        fused_filtered = [(vid, doc, s) for vid, doc, s in fused if vid not in pinned_ids]
        reranked = _rerank(user_message, fused_filtered, top_k) if fused_filtered else []
    else:
        reranked = []

    # Stage 4: Format context (with passage expansion for thematic queries)
    is_lookup = _is_verse_lookup(user_message)
    verse_ids = [vid for vid, _, _ in reranked]

    if not is_lookup and passage_collection is not None:
        passages = _expand_to_passages(verse_ids, passage_collection)
    else:
        passages = {}

    lines = [f"- **{vid}**: {text}" for vid, text in pinned]
    seen_passages = set()
    for vid, doc, _ in reranked:
        if vid in pinned_ids:
            continue
        if vid in passages and passages[vid] not in seen_passages:
            seen_passages.add(passages[vid])
            lines.append(f"- **{vid} (passage)**: {passages[vid]}")
        else:
            text = _clean_doc_text(doc, vid)
            lines.append(f"- **{vid}**: {text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str | None) -> str:
    """Delegate to shared cleanup (Qwen `</think>` + plain 'Thinking Process:' blocks)."""
    return strip_model_thinking(text)


def _strip_repetition_and_meta(text: str) -> str:
    if not text:
        return text
    # Strip leading "? Answer:" etc. before length check (fixes short responses)
    text = re.sub(r"^\s*\??\s*Answer:\s*", "", text, flags=re.IGNORECASE)
    if len(text) < 30:
        return text.strip()
    text = re.sub(r"[═─━]{3,}", "", text)
    for cutoff in [
        "Meta-instruction", "TYPED RESPONSE", "Crucial:", "Violation",
        "You have followed", "The key is:", "No matter how many times",
        "No matter what format", "You are running a standalone",
        "You do not respond to", "You do not generate",
    ]:
        idx = text.find(cutoff)
        if idx > 0:
            text = text[:idx].rstrip()
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+", " ", text)).strip()


def _strip_thinking_from_stream(sse_text: str) -> bytes:
    full_content = []
    for line in sse_text.split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = _json.loads(payload)
            for choice in obj.get("choices", []):
                c = choice.get("delta", {}).get("content", "")
                if c:
                    full_content.append(c)
        except _json.JSONDecodeError:
            continue
    cleaned = _strip_thinking("".join(full_content))
    cleaned = _strip_repetition_and_meta(cleaned)
    if not cleaned.strip():
        cleaned = EMPTY_MODEL_REPLY
    out = 'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":' + _json.dumps(cleaned) + '},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    return out.encode("utf-8")


# ---------------------------------------------------------------------------
# Query classification helpers
# ---------------------------------------------------------------------------

def _is_verse_lookup(text: str) -> bool:
    """True if question asks for a specific verse (e.g. 'What does John 3:16 say?')."""
    t = text.lower().strip()
    if not re.search(r"what does .+ say\??", t):
        return False
    # Require a verse reference (Book 1:2) to distinguish from topical questions
    # e.g. "What does the Bible say about love?" is topical, not a verse lookup
    return bool(re.search(r"\d+:\d+", t))


def _is_meta_question(text: str) -> bool:
    t = text.lower().strip()
    patterns = (
        "what can you do", "what could you do", "what it could do",
        "what are you", "how can you help", "what are your capabilities",
        "what is your purpose", "who are you", "what do you do",
        "introduce yourself", "tell me about yourself",
    )
    return any(p in t for p in patterns) or t in ("help", "hi", "hello", "hey")


def _strip_openclaw_metadata(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    text = re.sub(
        r"Sender\s*\(untrusted\s*metadata\)\s*:\s*```json\s*\{[^}]{0,2000}\}\s*```\s*",
        "", text, flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"```json\s*\{[^}]{0,2000}\}\s*```\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+\w+\]\s*", "", text)
    if "```" in text and not text.strip().startswith("["):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[-1].strip()
    return text.strip() or text


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "rag", "version": "1.0.0"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Guard against oversized payloads
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    body = await request.json()
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise HTTPException(status_code=422, detail="'messages' must be an array")
    model = body.get("model", "bible-assistant")
    stream = body.get("stream", False)
    last_q_for_policy: str | None = None

    last_user_content = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_content = m.get("content")
            if isinstance(last_user_content, list):
                for part in last_user_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        last_user_content = part.get("text", "")
                        break
                else:
                    last_user_content = ""
            break

    _SUFFIXES = (
        "? answer in quotes, then add explanation.",
        "? answer in quotes, then add explanation",
        ". answer in quotes, then add explanation.",
        ". answer in quotes, then add explanation",
        " answer in quotes, then add explanation.",
        " answer in quotes, then add explanation",
        "answer in quotes, then add explanation.",
        "answer in quotes, then add explanation",
    )

    if last_user_content and last_user_content.strip():
        q = _strip_openclaw_metadata(last_user_content.strip())
        for suffix in _SUFFIXES:
            if q.lower().endswith(suffix.lower()):
                q = q[:-len(suffix)].strip()
                break
        last_q_for_policy = q
        if _is_meta_question(q):
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = {**messages[i], "content": q}
                    break
            body = {**body, "messages": messages}
        else:
            pin_refs: list[str] = []
            vr = _extract_verse_ref_from_lookup(q)
            if vr:
                pin_refs.append(vr)
            topical_pins = _topical_anchor_refs(q)
            pin_refs.extend(topical_pins)
            context = _retrieve(q, top_k=RAG_TOP_K, pin_refs=pin_refs or None)
            if context and context.strip():
                notes: list[str] = []
                if vr:
                    notes.append(
                        "The context includes the verse you were asked about; quote it exactly."
                    )
                if topical_pins and not _is_verse_lookup(q):
                    notes.append(
                        "Use only passages that fit the topic; do not substitute unrelated stories."
                    )
                note_block = ("\n\nNote: " + " ".join(notes)) if notes else ""
                augmented = "Context:\n" + context + "\n\nQ: " + q + note_block
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i] = {**messages[i], "content": augmented}
                        break
                body = {**body, "messages": messages}

    def _content_to_str(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
            return ""
        return str(content) if content is not None else ""

    normalized = []
    for m in body.get("messages", []):
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = _content_to_str(m.get("content"))
        content = content if isinstance(content, str) else str(content) if content is not None else ""
        normalized.append({"role": role, "content": content})
    messages = normalized

    if last_q_for_policy and _is_counseling_request(last_q_for_policy):
        messages.insert(0, {"role": "system", "content": _COUNSELING_SYSTEM_GUARD})

    ollama_payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": body.get("max_tokens") or 2048,
        # Qwen3+ in Ollama: suppress chain-of-thought (see ollama/ollama issues on `think`)
        "think": False,
    }
    # Allow clients to re-enable (e.g. debugging) with `"think": true` in request body
    if "think" in body:
        ollama_payload["think"] = bool(body["think"])

    url = OLLAMA_URL.rstrip("/") + "/v1/chat/completions"
    async with httpx.AsyncClient(timeout=120.0) as client:
        if stream:
            req = client.build_request("POST", url, json=ollama_payload)
            response = await client.send(req, stream=True)
            if response.status_code != 200:
                err_body = (await response.aread()).decode("utf-8", errors="replace")
                await response.aclose()
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama {response.status_code}: {err_body}",
                )
            raw_chunks = []
            async for chunk in response.aiter_bytes():
                raw_chunks.append(chunk)
            await response.aclose()

            full_sse = b"".join(raw_chunks).decode("utf-8", errors="replace")
            cleaned_bytes = _strip_thinking_from_stream(full_sse)

            async def stream_bytes():
                yield cleaned_bytes

            return StreamingResponse(
                stream_bytes(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        r = await client.post(url, json=ollama_payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text}")
        data = r.json()
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            content = _strip_thinking(raw) if raw else ""
            content = _strip_repetition_and_meta(content) if content else ""
            out = content.rstrip()
            if out:
                if out[-1] in ",;:":
                    out = out[:-1] + "."
                elif out[-1] not in ".?!\"'":
                    for end in (". ", "? ", "! "):
                        idx = out.rfind(end)
                        if idx != -1:
                            out = out[: idx + 1].rstrip()
                            break
                data["choices"][0]["message"]["content"] = out
            else:
                data["choices"][0]["message"]["content"] = EMPTY_MODEL_REPLY
        except (IndexError, KeyError, TypeError) as e:
            logger.debug("Post-processing skipped: %s", e)
        return data


@app.on_event("shutdown")
async def _cleanup():
    """Release heavy model objects on shutdown."""
    global _chroma_client, _verse_collection, _passage_collection, _embedder, _bm25_data, _reranker
    _chroma_client = None
    _verse_collection = None
    _passage_collection = None
    _embedder = None
    _bm25_data = None
    _reranker = None
    logger.info("RAG server resources released")
