# Bible AI Assistant

A fine-tuned Bible Q&A assistant built on Qwen3 4B, deployed on the NVIDIA Jetson Orin Nano Super with a RAG layer, voice mode, OpenClaw agent integration, and a constitutional alignment system grounded in the Ten Commandments.

## Demo

[Screenshot or GIF of Gradio UI with voice mode here]

## Architecture

- **Fine-tuning:** QLoRA on RTX 5070 Ti (16GB VRAM)
- **Base model:** Qwen3-4B-Instruct-2507 (Alibaba)
- **RAG:** ChromaDB + nomic-embed-text-v1.5 (31,102 verses)
- **Inference:** llama.cpp (Q4_K_M) on Jetson Orin Nano Super
- **Voice:** Faster-Whisper (STT) + Kokoro 82M (TTS)
- **Agent:** OpenClaw via DigitalOcean VPS + Tailscale VPN
- **Interface:** Telegram + Gradio Web UI with Voice

## Skills Demonstrated

- LLM fine-tuning with QLoRA (Unsloth, PEFT, TRL)
- Retrieval-Augmented Generation (ChromaDB, nomic-embed)
- Voice pipeline (Faster-Whisper STT, Kokoro TTS)
- Model quantization and edge deployment (llama.cpp, GGUF)
- AI agent orchestration (OpenClaw, Ollama)
- Constitutional AI design and implementation
- MLOps: experiment tracking (W&B), model versioning (HF Hub)
- Production: Docker, systemd, Tailscale VPN, DigitalOcean

## Repository Structure

```
bible-ai-assistant/
├── data/           # Raw and processed Bible datasets
├── training/       # Fine-tuning scripts and config
├── rag/            # ChromaDB RAG server and index builder
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

1. **Environment:** Create conda env and install dependencies (see [Environment Setup](#environment-setup) in the guide).
2. **Data:** Build dataset in `data/processed/` (see `data/README.md`).
3. **Train:** Run QLoRA fine-tuning (see `training/README.md`).
4. **RAG:** Build ChromaDB index and start RAG server (see `rag/README.md`).
5. **Run locally:** Ollama + OpenClaw + RAG (see `deployment/pc/`).

## Author

Tremayne Timms

## License

Model and code: Apache 2.0 where applicable. Scripture sources: public domain (KJV, WEB, etc.).
