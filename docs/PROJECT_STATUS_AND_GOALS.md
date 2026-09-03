# Bible AI Assistant — Project Status & End Goal

**Last updated:** 2026-08-29 (v2-4b SFT trained, evaluated, and published — see below)

---

## V2 Rebuild — v2-4b SFT shipped 2026-08-29

Decision: the v1 model layer (1.8k-example 4B fine-tune) is the ceiling, not the system.
The validated RAG/eval/CI stack stays; the model layer is rebuilt at full-canon scale.

### Done
- **v2 dataset** (`training/build_dataset_v2.py`, branch `v2/dataset-full-upgrade-2026-08-28`):
  56,022 examples — 8 scripture-citation categories (~35.6k, capped from ~61k) + `grounded_exegesis`
  (Matthew Henry's Commentary, CC0), `pastoral_triage` (escalation / tradition-aware / calibrated
  abstention), and a ~24 % `general_blend` from smoltalk2 (Apache-2.0) as a catastrophic-forgetting
  guard. Provenance-tracked (per-source SHA + license); decontaminated vs. the eval suite (0 overlap).
- **v2-4b SFT** (`training/config.v2-4b.yaml`): Qwen3.5-4B bf16 LoRA (r=32), 1 epoch / 3,474 steps,
  eval loss 0.25 → 0.21, ~10.4 h on the 5070 Ti. Published:
  [`Ttimms/Bible-Assistant-Qwen3.5-4B-v2`](https://huggingface.co/Ttimms/Bible-Assistant-Qwen3.5-4B-v2)
  + [`…-v2-GGUF`](https://huggingface.co/Ttimms/Bible-Assistant-Qwen3.5-4B-v2-GGUF).
- **Protocol-v3 eval + v1 A/B** (`docs/MODEL_COMPARISON.md`): verse-lookup exact accuracy
  **58 % → 76.5 %**, citation **88 % → 98.9 %**, hallucination flat ~2 %; but overall fuzzy
  0.48 → 0.40 — the templated *answers* in the scripture categories regressed open-ended
  thematic synthesis. **The dataset's templated answers are the bottleneck.**
- `config.v2-9b.yaml` (9B QLoRA) and `config.v2.yaml` (27B stretch) exist for escalation.
  GRPO scaffold in `training/train_grpo.py`. Repo suite 439 tests.

### Next (the v3 SOTA push)
| Phase | Work |
|---|---|
| 1 | Judge re-score v2 + v1 (fair scoring on synthesis categories) |
| 2 | **Dataset v3 = teacher distillation** — regenerate all answers with a strong model, natural + grounded, non-templated; cut recall-drill volume |
| 3 | SFT on v3 (4B first) → **GRPO** with the verifiable citation reward — the stage meant to clear the ≥85 % verse-accuracy bar |
| 4 | Re-eval + FMG-Bench (`scripts/fmg_bench.py`); escalate to 9B only on a measured shortfall |
| 5 | Retrieval upgrade (embedder stronger than nomic-v1.5) + constrained verse-reference decoding |

Also open: `rag_server.py` commentary-retrieval path (so `grounded_exegesis` training matches
inference); Ollama GGUF support (pending Ollama's bundled-llama.cpp bump).

### Rules that outlive v1
- Eval/benchmark data is never training data (decontamination enforced).
- v1 lesson stands: diversity over repetition (the 31k repetitive dump caused overfit failures).

---
## What We've Done So Far

### 1. Core Infrastructure ✓
- **RAG pipeline:** ChromaDB vector store + nomic-embed-text-v1.5 embeddings indexing 31,000+ Bible verses
- **RAG server:** FastAPI middleware on port 8081 that retrieves verses, augments prompts, and post-processes model output (including `rag/response_cleanup.py`: Qwen `</think>` + plain “Thinking Process:” blocks)
- **Gradio UI (6.x):** Landing page, RAG/TTS health check, text + voice (Faster-Whisper → RAG → Kokoro); `docs/DEMO_LAUNCH.md`, `requirements-ui.txt`, auto port if 7860 is busy

### 2. Cutting-Edge Upgrades (March 2026) ✓
- **Upgrade 1:** Swapped base model to Qwen3.5-4B (newer, better performance)
- **Upgrade 5:** Context window 2048 (4096 OOMs on 16GB with bf16 LoRA; per config.yaml)
- **Upgrade 4:** Parent-child chunking — 5-verse passage windows alongside individual verses for thematic questions
- **Upgrade 3:** Hybrid RAG — Dense + BM25 + Reciprocal Rank Fusion + cross-encoder reranking (bge-reranker-v2-m3)
- **Upgrade 2:** ORPO preference alignment — built ~500 preference pairs and train_orpo.py script (teaches model what NOT to do)
- **Upgrade 6:** LLM-as-judge evaluation — scores responses on 5 dimensions (faithfulness, citation accuracy, hallucination, helpfulness, conciseness). Judge default is `qwen3:8b` since protocol v4 (`--judge-model` to change); `qwen3.5:27b` does not fit 16 GB VRAM (333 s/call — see `docs/BENCHMARK_PROTOCOL.md`), and since v4 the judge is calibration-only, not a gate
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
- **Cleanup ordering:** remove paired Qwen think-tag wrappers before flex `<think>`-style peeling so partial stripping cannot strand think bodies in the visible reply (`rag/response_cleanup.py`)
- **Pinned verses & anchors:** explicit verse lookups and selected topical queries prepend Chroma-fetched verses so hybrid search cannot drop the asked reference; counseling-like phrasing adds a system guardrail message

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
