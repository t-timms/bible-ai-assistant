# Architecture

## Two-Phase Design

### Phase 1: Development (Windows PC)

- **You** → Faster-Whisper (STT) → Gradio Web UI → OpenClaw (Gateway) → **RAG Server** (port 8081) → Ollama (Qwen3 4B GGUF) + ChromaDB RAG → Kokoro TTS → **You**
- Cost: $0/month. Purpose: build, test, iterate.

### Phase 2: Production (Jetson + VPS)

- **You** (Telegram) → Telegram cloud → **OpenClaw Gateway** (DigitalOcean VPS, ~$12/month) → Tailscale VPN → **llama.cpp** on Jetson Orin Nano Super (Qwen3 4B Q4 + ChromaDB RAG) → response back via Telegram.
- Voice: PC only. Jetson: text-only inference.

## Component Summary

| Component      | Dev (PC)              | Prod (Jetson/VPS)     |
|---------------|------------------------|------------------------|
| LLM           | Ollama (GGUF)         | llama.cpp (GGUF)       |
| RAG           | FastAPI + ChromaDB    | Same or on Jetson      |
| Agent         | OpenClaw (local)      | OpenClaw on VPS       |
| Voice         | Whisper + Kokoro      | —                      |
| Interface     | Gradio + Telegram     | Telegram               |

## Why This Split

- PC: full tooling, training, voice; no cloud GPU cost.
- Jetson: always-on edge inference; one-time hardware cost.
- VPS: stable gateway and Telegram connectivity; Tailscale keeps link to Jetson private.

---

## Router Pattern: Two-Model Architecture

The system uses two models in a router pattern:

1. **Agent Brain (Qwen3.5 27B)** — Runs in Ollama on port 11434. OpenClaw's primary model for tool calling, web search, session memory, and Telegram routing. We never modify this model.

2. **Bible Specialist (Qwen3 4B, fine-tuned)** — Runs in Ollama as `bible-assistant`. Accessed only through the RAG server on port 8081. Receives simple questions with verse context and returns clean answers. Never receives agent prompts or tool schemas.

```
User (Telegram / Gradio)
  → OpenClaw Gateway (27B agent brain)
    → Detects Bible question → calls bible-lookup skill
      → RAG Server (port 8081)
        → ChromaDB retrieves relevant verses
        → Augmented prompt: "Context:\n[verses]\n\nQ: [question]"
        → Ollama bible-assistant (4B)
      → Clean response back to user
    → Non-Bible question → 27B answers directly (web search, general knowledge)
```

The 4B model never sees OpenClaw metadata, tool schemas, or agent instructions. The RAG server's augmented prompt contains only retrieved context and the user's question — no behavioral instructions. The system prompt in the Ollama Modelfile handles all formatting and tone.

---

## Overfitting: Lessons Learned

Initial training on ~31,000 repetitive examples (same format: "What does X say?" → verse text) caused severe overfitting:

- **Verse fabrication** — Model invented or misquoted verses instead of using RAG context
- **Instruction leaking** — Model output its own system prompt instructions
- **Repetition loops** — Same phrase repeated 5-10 times in one response
- **Bible answers for everything** — Non-Bible questions still got Scripture responses

**Fix:** Reduced to ~1,600 diverse examples across 7 categories (verse lookups, RAG-grounded, thematic, general assistant, meta-questions, multi-turn, refusals). Shortened the system prompt from 157 lines to ~15 lines. Lowered LoRA rank from 16 to 8, reduced to 2 epochs with LR 1e-4 and 0.15 dropout. Added 10% eval split to monitor train vs. eval loss.

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
- **Exact fetch:** For "What does X say?" questions, the RAG server tries an exact ID lookup before semantic search, avoiding wrong-verse results from embedding similarity.
- **Minimal augmentation:** The augmented prompt is `"Context:\n" + context + "\n\nQ: " + question` — no behavioral instructions. The Modelfile's system prompt handles all response formatting.
