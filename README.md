# Bible AI Assistant

A fine-tuned Bible Q&A assistant built on Qwen3.5-4B, with hybrid RAG (ChromaDB + dense + sparse retrieval), deployed locally via Ollama. Trained on ~1,800 diverse examples with bf16 LoRA and optional ORPO alignment.

## Demo

[Screenshot or GIF of Gradio UI with voice mode here]

## Architecture

- **Fine-tuning:** bf16 LoRA on Qwen3.5-4B (Unsloth, PEFT, TRL)
- **Base model:** Qwen/Qwen3.5-4B
- **RAG:** ChromaDB + nomic-embed-text-v1.5 + hybrid retrieval (dense + BM25 + reranker)
- **Inference:** Ollama (GGUF f16 or Q4_K_M), optional llama.cpp on Jetson
- **Voice:** Faster-Whisper (STT) + Kokoro TTS (optional Gradio UI)

## Skills Demonstrated

- LLM fine-tuning with bf16 LoRA (Unsloth, PEFT, TRL)
- Retrieval-Augmented Generation (ChromaDB, hybrid retrieval)
- Model quantization and deployment (GGUF, Ollama)
- MLOps: experiment tracking (W&B)

## Repository Structure

```
bible-ai-assistant/
├── data/           # Raw and processed Bible datasets
├── training/       # Fine-tuning scripts and config
├── rag/            # ChromaDB RAG server and index builder
├── scripts/        # Utility scripts (leaderboard, tests)
├── tests/          # Pytest unit tests
├── voice/          # STT (Faster-Whisper) and TTS (Kokoro) services
├── prompts/        # System prompt and evaluation questions
├── deployment/     # PC, Jetson, and VPS setup
├── ui/             # Gradio web interface with voice
└── docs/           # Guides, workflow, and reference
```

- **New to the project?** Start with **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)** for the step-by-step guide (Steps 1–12).
- **Doc index:** See **[docs/README.md](docs/README.md)** for a full list of documentation.
- **Architecture:** See [docs/architecture.md](docs/architecture.md) for diagrams and phase-by-phase deployment.

## Quick Start

1. **Environment:** Create conda env and install dependencies (see [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) Steps 4–5).
2. **Data:** Build dataset in `data/processed/` (see `data/README.md`).
3. **Train:** Run bf16 LoRA fine-tuning (see `training/README.md`).
4. **RAG:** Build ChromaDB index and start RAG server (see `rag/README.md`).
5. **Run locally:** Ollama + RAG server (see `deployment/pc/`).
6. **Gradio demo UI:** [docs/DEMO_LAUNCH.md](docs/DEMO_LAUNCH.md) — Ollama + RAG + `python ui/app.py` (landing page at http://127.0.0.1:7860). If you only installed RAG deps, run `pip install -r requirements-ui.txt` first.

**Wrapping up a training cycle (no new training):** [docs/SHIP_v1_AND_POLISH_BACKLOG.md](docs/SHIP_v1_AND_POLISH_BACKLOG.md) + [docs/training_results/POST_TRAINING_CHECKLIST.md](docs/training_results/POST_TRAINING_CHECKLIST.md).

## Changelog & quality

- **[CHANGELOG.md](CHANGELOG.md)** — notable code and behavior changes (RAG cleanup, eval, benchmarks).
- **Tests:** `pip install pytest` (or `pip install -e ".[dev]"`), then from the repo root: `python -m pytest tests/`. CI sets `PYTHONPATH=.` for the same layout (see `.github/workflows/ci.yml`).
- **CI:** GitHub Actions runs **ruff** on `training/`, `rag/`, `scripts/`, `ui/`, `voice/` and **pytest** on `tests/` (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Author

Tremayne Timms

## License

Model and code: MIT. Scripture sources: public domain (KJV, WEB, etc.).
