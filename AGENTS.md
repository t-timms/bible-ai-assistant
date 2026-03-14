# Agent and Developer Guidance

This file orients AI agents and developers on how to work on the Bible AI Assistant repo.

## Project summary

- **Goal:** Bible-specialized AI assistant: Qwen3 4B fine-tuned on Scripture, RAG (ChromaDB), voice (STT/TTS), constitutional guardrails, deployment on PC (dev) and Jetson + VPS (production).
- **Guide:** *Bible AI Assistant Guide v2* (PDF). Follow section order and checkpoints.

## Key docs

| Doc | Purpose |
|-----|--------|
| `docs/DEVELOPMENT_WORKFLOW.md` | Phase-gated workflow with version checkpoints (v0.1.0 → v0.9.0). **Start here** for “what to do next.” |
| `CONSTITUTION.md` | Behavioral rules (Ten Commandments + Way of Jesus). All prompts and model behavior must align. |
| `prompts/system_prompt.txt` | Canonical system prompt; used in SOUL.md, Gradio, and training data. |
| `docs/architecture.md` | Two-phase architecture (PC vs Jetson+VPS). |

## Tech constraints

- **Training:** bf16 only; PyTorch nightly + CUDA 12.8+ for RTX 5070 Ti (Blackwell). No fp16.
- **Data:** Qwen3 chat format (`messages` with system/user/assistant). See `data/README.md` and `data/sample.json`.
- **RAG:** nomic-embed-text-v1.5 with `search_document:` / `search_query:` prefixes. ChromaDB in `rag/chroma_db/` (not committed).

## Cursor rules

Project-specific rules live in `.cursor/rules/` (e.g. `bible-ai-conventions.mdc`). They enforce constitution alignment, guide order, and versioning.

## Next steps (after scaffold)

1. Create GitHub repo, push scaffold, tag **v0.1.0**.
2. Follow `docs/DEVELOPMENT_WORKFLOW.md` Phase 1: environment and base model download.
