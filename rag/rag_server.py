"""
FastAPI RAG middleware: retrieve verses, augment user message, forward to LLM.
Expects OpenAI-compatible POST /v1/chat/completions; forwards to OLLAMA_URL (default localhost:11434).
Run: uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081
"""
from pathlib import Path

from fastapi import FastAPI

app = FastAPI(title="Bible AI RAG Server", version="0.5.0")

# TODO: Add PersistentClient(RAG_DB), get_collection('bible_verses'), load embedder nomic-embed-text-v1.5
# TODO: @app.post("/v1/chat/completions") -> get user message, query_embedding with "search_query: ...",
#       build context from top-k, augment last message, POST to OLLAMA_URL/v1/chat/completions, return response
# See guide Section 12 Step 2.


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "rag"}
