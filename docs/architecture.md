# Architecture

## Core Stack

A fine-tuned Bible Q&A model with RAG for grounded answers. No agent orchestration — direct path from client to model.

```
User
  → Gradio Web UI / curl / any OpenAI-compatible client
    → RAG Server (port 8081)
      → ChromaDB retrieves relevant verses
      → Augmented prompt: "Context:\n[verses]\n\nQ: [question]"
      → Ollama bible-assistant (Qwen3.5-4B fine-tuned)
    → Response (post-processed)
```

Or use Ollama directly (no RAG) for quick local testing: `ollama run bible-assistant`.

## Two-Phase Design

### Phase 1: Development (Windows PC)

- **You** → Gradio Web UI (or curl) → **RAG Server** (port 8081) → Ollama (Qwen3.5-4B GGUF) + ChromaDB RAG → **You**
- Optional: Faster-Whisper (STT) + Kokoro TTS for voice.
- Cost: $0/month. Purpose: build, test, iterate.

### Phase 2: Production (Jetson)

- **You** → Client (e.g. Gradio, API) → Tailscale VPN → **RAG + llama.cpp** on Jetson Orin Nano Super (Qwen3.5-4B Q4 + ChromaDB RAG) → response.
- Voice: PC only. Jetson: text-only inference.

## Component Summary

| Component | Dev (PC) | Prod (Jetson) |
|----------|----------|---------------|
| LLM | Ollama (GGUF) | llama.cpp (GGUF) |
| RAG | FastAPI + ChromaDB | Same or on Jetson |
| Voice | Whisper + Kokoro | — |
| Interface | Gradio, curl, API | Same |

---

## Overfitting: Lessons Learned

Initial training on ~31,000 repetitive examples (same format: "What does X say?" → verse text) caused severe overfitting:

- **Verse fabrication** — Model invented or misquoted verses instead of using RAG context
- **Instruction leaking** — Model output its own system prompt instructions
- **Repetition loops** — Same phrase repeated 5-10 times in one response
- **Bible answers for everything** — Non-Bible questions still got Scripture responses

**Fix:** Reduced to ~1,800 diverse examples across 7 categories (verse lookups, RAG-grounded, thematic, general assistant, meta-questions, multi-turn, refusals). Shortened the system prompt from 157 lines to ~15 lines. Current config: LoRA rank 16, 3 epochs, LR 2e-4, dropout 0.1, 10% eval split to monitor train vs. eval loss.

**Key insight for small models (4B):** Less is more. Short system prompts, diverse training data, and minimal post-processing beat long instruction lists and massive repetitive datasets.

---

## Constitutional AI Approach

The model's behavior is aligned using a constitutional approach grounded in the Ten Commandments (see `CONSTITUTION.md`):

1. **Never fabricate Scripture** — The model must only quote verses it is certain are real. If unsure, it says so. RAG retrieval provides grounding.
2. **Never role-play as God or Jesus** — The model declines requests to speak in first person as divine figures.
3. **Never produce harmful content** — Explicit, violent, or degrading content is refused.
4. **Cite accurately** — When Bible verses are provided in context, the model quotes them exactly as given.

This constitution is embedded in three places:
- `prompts/system_prompt.txt` — baked into the Ollama Modelfile (not injected at runtime)
- Training data — refusal examples teach the model to decline constitutionally
- RAG server — minimal post-processing strips any leaked instructions as a safety net

---

## RAG Design Decisions

- **Embedding model:** nomic-embed-text-v1.5 with task prefixes (`search_document:` for indexing, `search_query:` for retrieval). v1.5 has better retrieval quality than v1.
- **Vector store:** ChromaDB with persistent storage in `rag/chroma_db/`. One-time index build; ~31,000 verses from the World English Bible.
- **Hybrid retrieval:** Dense search (ChromaDB) + BM25 sparse search, Reciprocal Rank Fusion, then cross-encoder reranking (bge-reranker-v2-m3). For thematic (non-verse-lookup) questions, passage expansion adds parent context.
- **Minimal augmentation:** The augmented prompt is `"Context:\n" + context + "\n\nQ: " + question` — no behavioral instructions. The Modelfile's system prompt handles all response formatting.
