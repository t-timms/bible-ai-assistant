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

## V2 rebuild workstream (branch `v2`, in progress)

Full-canon retraining campaign; details in `docs/PROJECT_STATUS_AND_GOALS.md` (V2 section).

- **Dataset engine:** `python training/build_dataset_v2.py [--limit-per-cat N] [--offline-only]` ->
  ~62k examples -> `data/processed/train_v2.json` + sha-pinned manifest sidecar. Sources auto-download
  to `data/raw_v2/` (gitignored, re-downloadable).
- **Config:** `training/config.v2.yaml` (Qwen3.5-14B QLoRA + GRPO verifiable-reward scaffold).
- **New categories:** verse_recall, translation_specific, reverse_lookup, near_miss_guard,
  passage_recall, cross_reference_chains, topical_collections, chapter_context.
- **Hard rules:**
  - NEVER train on benchmark/eval data - decontamination runs automatically; CommonEval and
    `benchmarks/suites/*` are eval-only by design.
  - Cross-reference data is CC-BY (openbible.info) - training use only, attribution stays in manifest.
  - Test with `uv run --extra dev --extra rag pytest -q` (429 tests; API tests need the rag extra).

## Tech constraints

- **Training:** bf16 only; PyTorch nightly + CUDA 12.8+ for RTX 5070 Ti (Blackwell). No fp16.
- **Data:** Qwen3 chat format (`messages` with system/user/assistant). See `data/README.md` and `data/sample.json`.
- **RAG:** nomic-embed-text-v1.5 with `search_document:` / `search_query:` prefixes. ChromaDB in `rag/chroma_db/` (not committed).

## Cursor rules

Project-specific rules live in `.cursor/rules/` (e.g. `bible-ai-conventions.mdc`). They enforce constitution alignment, guide order, and versioning.

## Next steps (after scaffold)

1. Create GitHub repo, push scaffold, tag **v0.1.0**.
2. Follow `docs/DEVELOPMENT_WORKFLOW.md` Phase 1: environment and base model download.
