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
