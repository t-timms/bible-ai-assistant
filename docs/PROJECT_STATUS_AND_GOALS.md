# Bible AI Assistant — Project Status & End Goal

**Last updated:** March 2026 (v1 docs + RAG response hardening)

---

## What We've Done So Far

### 1. Core Infrastructure ✓
- **RAG pipeline:** ChromaDB vector store + nomic-embed-text-v1.5 embeddings indexing 31,000+ Bible verses
- **RAG server:** FastAPI middleware on port 8081 that retrieves verses, augments prompts, and post-processes model output (including `rag/response_cleanup.py`: Qwen `</think>` + plain “Thinking Process:” blocks)
- **Gradio UI:** Text + voice chat (Faster-Whisper STT → RAG → Kokoro TTS)

### 2. Cutting-Edge Upgrades (March 2026) ✓
- **Upgrade 1:** Swapped base model to Qwen3.5-4B (newer, better performance)
- **Upgrade 5:** Context window 2048 (4096 OOMs on 16GB with bf16 LoRA; per config.yaml)
- **Upgrade 4:** Parent-child chunking — 5-verse passage windows alongside individual verses for thematic questions
- **Upgrade 3:** Hybrid RAG — Dense + BM25 + Reciprocal Rank Fusion + cross-encoder reranking (bge-reranker-v2-m3)
- **Upgrade 2:** ORPO preference alignment — built ~500 preference pairs and train_orpo.py script (teaches model what NOT to do)
- **Upgrade 6:** LLM-as-judge evaluation — default judge Ollama `qwen3.5:27b` (`--judge-model` to change) scores responses on 5 dimensions (faithfulness, citation accuracy, hallucination, helpfulness, conciseness)
- **W&B fixes:** UTF-8 console encoding + extended service timeout for Windows so training logs to Weights & Biases

### 3. Dataset & Training ✓
- **Dataset:** ~1,600 diverse Q&A examples across 7 categories (verse lookups, RAG-grounded, thematic, general, meta, multi-turn, refusals) — quality over quantity to avoid overfitting
- **SFT training:** bf16 LoRA fine-tuning with Unsloth on Qwen3.5-4B (LoRA r=16, 2048 context; config.yaml)
- **Preference data:** 500 chosen/rejected pairs for ORPO stage

### 4. Fixes Applied
- Training stack: see `docs/ENVIRONMENT_REQUIREMENTS.md` / `ORPO_TWO_ENV_SETUP.md` for transformers / Unsloth versions (SFT+ORPO vs RAG env).
- Fixed RAG passage index CUDA OOM by reducing batch size for long passage embeddings
- **Response cleanup:** shared `strip_model_thinking()` used by RAG server and `training/evaluate.py` so leaked chain-of-thought is stripped before users / eval see text
- **RAG JSON correctness (non-streaming):** cleaned assistant text is always written to `choices[0].message.content` (fixes an edge case where replies ending in `.` / `?` / `!` / quotes skipped the final assignment and left raw Ollama output)
- **Cleanup ordering:** paired `</think>`…`</think>` blocks are removed before flex `<think>`-style peeling so partial stripping cannot strand think bodies in the visible reply

---

## End Goal

**A production-ready Bible AI assistant** that:
1. Answers Bible questions with accurate citations and no fabricated verses
2. Uses hybrid RAG (dense + sparse + reranking) for better retrieval
3. Benefits from two-stage training (SFT + ORPO) for alignment
4. Is evaluated rigorously with LLM-as-judge across 5 quality dimensions
5. Runs locally (Ollama) with optional edge deployment (Jetson Orin Nano)
6. Supports voice via Gradio; can be used as a backend by other clients (e.g. agents) via the RAG API

### Remaining / polish (not blocking v1)
- **Optional:** Full LLM-as-judge benchmark (`scripts/run_benchmark.py --judge`) when you have time — see `docs/BENCHMARK_PROTOCOL.md`
- **Optional:** Keyword benchmark (`run_benchmark.py` without `--judge`) for fast regression
- **Optional:** Gradio/voice polish, edge (Jetson) deploy, API hardening
- **Ship checklist:** `docs/SHIP_v1_AND_POLISH_BACKLOG.md` — close v1, track polish

### Done for this training cycle (typical)
- ORPO preference training (when run) → merge → GGUF Q4 + F16 → Ollama models (e.g. `bible-assistant-orpo`, `bible-assistant-orpo-f16`) per `docs/training_results/POST_TRAINING_CHECKLIST.md`

---

## Architecture Snapshot

```
User (Gradio / API client)
  → RAG Server (8081): Hybrid retrieval (Dense + BM25 + RRF + reranker)
    → ChromaDB (verses + passages) + BM25 index
  → Ollama (e.g. `bible-assistant-orpo` — Qwen3.5-4B SFT+ORPO)
  → Response (post-processed)
```
