# Development Workflow

This document maps the **Bible AI Assistant Guide v2** to an industry-standard, phase-gated workflow. Complete each phase and checkpoint before moving to the next. Use version tags and CHANGELOG for every milestone.

---

## Phase 0: Project bootstrap ✅

- [x] Repository scaffold (this repo).
- [ ] **v0.1.0:** Initial commit: scaffold, README, CONSTITUTION, prompts, .gitignore, requirements.
- [ ] Create GitHub repo `bible-ai-assistant`, add remote, push.

**Guide ref:** Section 3 (Repository Structure), Section 19 (GitHub Versioning).

---

## Phase 1: Environment and base model

- [ ] Install stack in order: Git, Miniconda, **CUDA 12.8+** (required for RTX 5070 Ti / Blackwell), Node.js 22, VS Code, Docker Desktop, Ollama.
- [ ] Create conda env `bible-ai` (Python 3.11). Install **PyTorch nightly** (CUDA 12.8+), not stable. Install `requirements.txt`.
- [ ] Verify: `torch.cuda.is_available()`, `get_device_name(0)`, `sm_120` in `arch_list`.
- [ ] `huggingface-cli login`, `wandb login`.
- [ ] Download base model: `huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/base_model --exclude "*.msgpack"`.

**Checkpoint:** v0.1.0 (or v0.1.1 after base model download). **Guide ref:** Section 5–7.

---

## Phase 2: Data and training

- [ ] Obtain raw Bible data (e.g. scrollmapper, World English Bible). Place in `data/raw/`.
- [ ] Implement `training/dataset_builder.py`: output `data/processed/train.json` in Qwen3 chat format with system prompt. Include verse lookup, theology, constitution-testing, and uncertainty examples.
- [ ] Target 30k–50k examples for first run. Commit `data/sample.json` as documentation.
- [ ] **v0.2.0:** Dataset builder complete, 50k Bible Q&A generated (or your actual count).

**Guide ref:** Section 8.

---

## Phase 3: Fine-tuning and evaluation

- [ ] Implement `training/train_unsloth.py` (QLoRA, **bf16=True**). Use `config.yaml` and W&B.
- [ ] Train; monitor loss in W&B. Save screenshots to `docs/training_results/`.
- [ ] Implement `training/merge_adapters.py`; produce `models/bible-qwen3-4b-merged`.
- [ ] Implement `training/evaluate.py` using `prompts/evaluation_questions.json`. Ensure zero fabricated verses and constitution pass.
- [ ] **v0.3.0:** Fine-tuning complete — Qwen3 4B trained on Bible Q&A. Optionally push merged model to Hugging Face.

**Guide ref:** Section 9–10.

---

## Phase 4: Quantization and local run

- [ ] Export to GGUF (Unsloth or llama.cpp). Recommended: Q4_K_M (or Q4_K_XL if using Unsloth dynamic).
- [ ] Create Ollama Modelfile with `prompts/system_prompt.txt`. `ollama create bible-assistant -f Modelfile`.
- [ ] Smoke test: `ollama run bible-assistant "What does John 3:16 say?"`
- [ ] **v0.4.0:** Model quantized to GGUF, tested in Ollama locally.

**Guide ref:** Section 11.

---

## Phase 5: RAG

- [ ] Implement `rag/build_index.py` (ChromaDB + nomic-embed-text-v1.5, task prefixes). Build `rag/chroma_db/`.
- [ ] Implement `rag/rag_server.py`: FastAPI, augment user message with retrieved verses, forward to Ollama.
- [ ] Run `uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081`. Test with curl or UI.
- [ ] **v0.5.0:** RAG layer complete — ChromaDB indexed with Bible.

**Guide ref:** Section 12.

---

## Phase 6: Development stack and Telegram

- [ ] Configure OpenClaw: SOUL.md (constitution), config pointing to `http://localhost:8081/v1`. Run OpenClaw + Ollama + RAG together.
- [ ] Create Telegram bot via @BotFather; add `TELEGRAM_BOT_TOKEN` to `.env`.
- [ ] **v0.6.0:** Full dev stack — Ollama + OpenClaw + RAG + Telegram.

**Guide ref:** Section 13, 17.

---

## Phase 7: Voice and Gradio

- [ ] Add Faster-Whisper (STT) and Kokoro TTS (Docker). Integrate into `ui/app.py`: voice tab → transcribe → RAG+LLM → synthesize → playback.
- [ ] **v0.6.5:** Voice mode added — Faster-Whisper STT + Kokoro TTS.
- [ ] **v0.9.0:** Gradio web UI with voice mode for demo.

**Guide ref:** Section 14, 18.

---

## Phase 8: Edge and production

- [ ] Jetson: transfer GGUF and chroma_db; install llama.cpp; systemd services; Tailscale.
- [ ] **v0.7.0:** Jetson deployment live — llama.cpp serving Qwen3 4B.
- [ ] VPS: DigitalOcean droplet; Node.js 22, OpenClaw, Tailscale; point to Jetson Tailscale IP.
- [ ] **v0.8.0:** Production deployment — OpenClaw on VPS, model on Jetson.

**Guide ref:** Section 15–16.

---

## Commit and tag discipline

- **Commit message rules:** Present tense, capital letter, &lt;72 chars first line. Include version for milestones (e.g. `v0.3.0: Fine-tuning complete`).
- **Tags:** `git tag -a v1.0.0 -m "Version 1.0.0 — Production deployment complete"` then `git push origin --tags`.
- **CHANGELOG:** Update on every release; use [Keep a Changelog](https://keepachangelog.com/) format.

---

## When to pause

- **Training:** If loss does not decrease or model fabricates verses → fix data and/or hyperparameters before deploying.
- **Evaluation:** Do not deploy if evaluation fails (fabrications or constitution violations). Retrain or add data.
- **Hardware:** If Jetson OOM, reduce context size or GPU layers before going to production.

This workflow keeps the project in a shippable state at each checkpoint and aligns with the guide’s intended order.
