"""
FastAPI RAG middleware with hybrid retrieval (Dense + BM25 + RRF + Cross-Encoder Reranking).

Pipeline:
  1. Dense search (nomic-embed-text-v1.5 via ChromaDB) -> top 20
  2. BM25 sparse search (rank_bm25) -> top 20
  3. Reciprocal Rank Fusion to merge results
  4. Cross-encoder reranking (bge-reranker-v2-m3) -> top K
  5. Parent-child passage expansion for thematic questions

Run: uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081
"""
import json as _json
import os
import pickle
import re
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from rag.response_cleanup import strip_model_thinking

app = FastAPI(title="Bible AI RAG Server", version="1.0.0")


@app.exception_handler(Exception)
async def _handle_unhandled(request: Request, exc: Exception):
    import traceback
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__,
                 "traceback": traceback.format_exc()},
    )


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "20"))
RRF_K = 60
VERSES_COLLECTION = "bible_verses"
PASSAGES_COLLECTION = "bible_passages"
QUERY_PREFIX = "search_query: "

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
    except Exception:
        _passage_collection = None
    _embedder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    return _verse_collection, _passage_collection, _embedder


def _get_bm25():
    """Load pickled BM25 index."""
    global _bm25_data
    if _bm25_data is not None:
        return _bm25_data
    bm25_path = _get_project_root() / "rag" / "chroma_db" / "bm25_index.pkl"
    if not bm25_path.exists():
        return None
    with open(bm25_path, "rb") as f:
        _bm25_data = pickle.load(f)
    print(f"[RAG] Loaded BM25 index ({len(_bm25_data['ids'])} docs)")
    return _bm25_data


def _get_reranker():
    """Load cross-encoder reranker model."""
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        print("[RAG] Loaded cross-encoder reranker (bge-reranker-v2-m3)")
        return _reranker
    except Exception as e:
        print(f"[RAG] Reranker unavailable: {e}")
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
    out = []
    if results and results["ids"] and results["ids"][0]:
        for i, (vid, doc) in enumerate(
            zip(results["ids"][0], results["documents"][0], strict=True)
        ):
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
        except Exception:
            pass
    return expanded


def _retrieve(user_message: str, top_k: int = RAG_TOP_K) -> str:
    """Hybrid retrieval: Dense + BM25 -> RRF -> Rerank -> format context string."""
    try:
        verse_collection, passage_collection, embedder = _get_rag()
    except FileNotFoundError:
        return ""

    # Stage 1: Parallel dense + BM25 search
    dense_results = _dense_search(user_message, verse_collection, embedder, HYBRID_CANDIDATES)
    bm25_results = _bm25_search(user_message, HYBRID_CANDIDATES)

    # Stage 2: Reciprocal Rank Fusion
    fused = (
        _reciprocal_rank_fusion(dense_results, bm25_results) if bm25_results else dense_results
    )

    if not fused:
        return ""

    # Stage 3: Cross-encoder reranking
    reranked = _rerank(user_message, fused, top_k)

    # Stage 4: Format context (with passage expansion for thematic queries)
    is_lookup = _is_verse_lookup(user_message)
    verse_ids = [vid for vid, _, _ in reranked]

    if not is_lookup and passage_collection is not None:
        passages = _expand_to_passages(verse_ids, passage_collection)
    else:
        passages = {}

    lines = []
    seen_passages = set()
    for vid, doc, _ in reranked:
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
    if not cleaned:
        return b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
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
        r"Sender\s*\(untrusted\s*metadata\)\s*:\s*```json\s*\{[^}]*\}\s*```\s*",
        "", text, flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"```json\s*\{[^}]*\}\s*```\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
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
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "bible-assistant")
    stream = body.get("stream", False)

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
        if _is_meta_question(q):
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = {**messages[i], "content": q}
                    break
            body = {**body, "messages": messages}
        else:
            context = _retrieve(q, top_k=RAG_TOP_K)
            if context:
                augmented = "Context:\n" + context + "\n\nQ: " + q
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

    url = f"{OLLAMA_URL.rstrip('/')}/v1/chat/completions"
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
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                content = _strip_thinking(content)
                content = _strip_repetition_and_meta(content)
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
                    # Always write back post-processed text (previously we skipped when
                    # the last char was already .?!\"' — leaving raw Ollama output in JSON).
                    data["choices"][0]["message"]["content"] = out
        except (IndexError, KeyError, TypeError):
            pass
        return data
