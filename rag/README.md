# RAG (Retrieval-Augmented Generation)

ChromaDB-based verse retrieval to reduce hallucination. Sits between OpenClaw/Ollama and the LLM as middleware.

## Layout

| Script | Purpose |
|--------|---------|
| `build_index.py` | Build ChromaDB vector index from Bible JSON (nomic-embed-text-v1.5). |
| `rag_server.py` | FastAPI server: accepts chat, retrieves verses, augments prompt, forwards to LLM. |
| `response_cleanup.py` | Strip Qwen `</think>` / plain “Thinking Process:” blocks from model text (shared with `training/evaluate.py`). Paired think blocks are removed before flex `<think>` peeling so content is not partially stripped. |
| `query_test.py` | Sanity-check retrieval quality. |
| `chroma_db/` | Persistent index (not committed). |

## Embedding Model

Use **nomic-ai/nomic-embed-text-v1.5** with task prefixes:
- Documents: `search_document: {text}`
- Queries: `search_query: {user question}`

## Run

```bash
# One-time: build index (requires data/raw/ Bible JSON)
python rag/build_index.py

# Start RAG server (forwards to Ollama at localhost:11434)
uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081
```

Any OpenAI-compatible client can point to `http://localhost:8081/v1` for chat completions. Checkpoint: **v0.5.0** when RAG is complete.
