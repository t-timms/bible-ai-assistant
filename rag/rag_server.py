"""
FastAPI RAG middleware: retrieve verses, augment user message, forward to LLM.
Expects OpenAI-compatible POST /v1/chat/completions; forwards to OLLAMA_URL (default localhost:11434).
Run: uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081
"""
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Bible AI RAG Server", version="0.5.0")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
COLLECTION_NAME = "bible_verses"
QUERY_PREFIX = "search_query: "

# Lazy-loaded globals
_chroma_client = None
_collection = None
_embedder = None


def _get_rag():
    global _chroma_client, _collection, _embedder
    if _collection is not None and _embedder is not None:
        return _collection, _embedder
    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("RAG requires chromadb and sentence-transformers. pip install chromadb sentence-transformers.") from e

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "rag" / "chroma_db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChromaDB index not found at {db_path}. Run: python rag/build_index.py"
        )
    _chroma_client = chromadb.PersistentClient(path=str(db_path), settings=Settings(anonymized_telemetry=False))
    _collection = _chroma_client.get_collection(COLLECTION_NAME)
    _embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    return _collection, _embedder


def _retrieve(user_message: str, top_k: int = RAG_TOP_K) -> str:
    """Return a context string of relevant verses for the user message."""
    try:
        collection, embedder = _get_rag()
    except FileNotFoundError:
        return ""
    query_embedding = embedder.encode([QUERY_PREFIX + user_message], show_progress_bar=False)
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    if not results or not results["metadatas"] or not results["metadatas"][0]:
        return ""
    lines = []
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        ref = meta.get("reference", "?")
        # Document may be stored with "search_document: " prefix; show clean text
        text = doc.replace("search_document: ", "", 1) if doc.startswith("search_document:") else doc
        lines.append(f"- **{ref}**: {text}")
    return "\n".join(lines)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "rag"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible endpoint: augment last user message with RAG context, forward to Ollama."""
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "bible-assistant")
    stream = body.get("stream", False)

    # Find last user message
    last_user_content = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_content = m.get("content")
            if isinstance(last_user_content, list):
                # Multimodal: take first text part
                for part in last_user_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        last_user_content = part.get("text", "")
                        break
                else:
                    last_user_content = ""
            break

    if last_user_content and last_user_content.strip():
        context = _retrieve(last_user_content.strip(), top_k=RAG_TOP_K)
        if context:
            augmented = (
                "Relevant Scripture (use these to ground your answer):\n\n"
                + context
                + "\n\n---\n\nUser question: "
                + last_user_content
            )
            # Replace the last user message with augmented content
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = {**messages[i], "content": augmented}
                    break
            body = {**body, "messages": messages}

    # Forward to Ollama
    url = f"{OLLAMA_URL.rstrip('/')}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=120.0) as client:
        if stream:
            async with client.stream("POST", url, json=body) as response:
                response.raise_for_status()
                async def stream_bytes():
                    async for chunk in response.aiter_bytes():
                        yield chunk
                return StreamingResponse(
                    stream_bytes(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"},
                )
        r = await client.post(url, json=body)
        r.raise_for_status()
        return r.json()
