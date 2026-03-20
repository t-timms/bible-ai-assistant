# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for milestone releases.

## [Unreleased]

### Added

- `rag/response_cleanup.py` — shared `strip_model_thinking()` for Qwen/Ollama chain-of-thought and planning scaffolds
- `scripts/run_benchmark.py`, `scripts/compare_benchmark_runs.py`, `benchmarks/manifest.v1.yaml` — versioned keyword / judge benchmarks
- `tests/` — pytest coverage for RAG helpers, eval keywords, evaluation manifest, and benchmark manifest
- `.github/workflows/ci.yml` — CI running `ruff check` and `pytest`
- `pyproject.toml`, `requirements-rag.txt`, `prompts/README.md`
- Documentation: `docs/PROJECT_STATUS_AND_GOALS.md`, `docs/SHIP_v1_AND_POLISH_BACKLOG.md`, `docs/BENCHMARK_PROTOCOL.md`, `docs/training_results/POST_TRAINING_CHECKLIST.md`, `deployment/pc/README.md`, and related guides

### Changed

- `rag/rag_server.py` — Ollama requests default to `"think": false`; non-streaming responses always write post-processed `choices[0].message.content` (including after punctuation normalization)
- `training/evaluate.py` — strips model thinking on RAG replies; robust judge HTTP (`trust_env=False`, endpoint fallbacks); configurable `--judge-model` (default `qwen3.5:27b`)
- `prompts/system_prompt.txt` — discourages visible chain-of-thought; Modelfile regeneration via `deployment/pc/generate_modelfile.py`
- `README.md` and `docs/README.md` — pointers to ship checklist, post-training steps, and changelog

### Fixed

- `strip_model_thinking()` — remove paired `</think>`…`</think>` blocks *before* flex `<think>` peeling (avoids stripping only the opener and leaving leaked content); strip leading BOM when it remains after tag removal
- RAG OpenAI-compatible JSON — previously skipped assigning cleaned text when the reply already ended with `.`, `?`, `!`, `"`, or `'`, leaving raw model output in the payload
- **GitHub Actions (lint)** — Ruff clean on `training/`, `rag/`, `scripts/`, `ui/`, `voice/` (import order, `zip(strict=True)`, `raise … from`, `contextlib.suppress`, per-file ignores for `evaluate.py` path bootstrap and `dataset_builder` format branches)

## [0.1.0] - YYYY-MM-DD

### Added

- Project scaffold and repository structure
- Biblical Constitution (CONSTITUTION.md) and system prompt
- .gitignore, .env.example, requirements.txt
- README and docs/architecture.md
- Placeholder directories and READMEs for data, training, rag, voice, deployment, ui
- Development workflow guide (docs/DEVELOPMENT_WORKFLOW.md)

### Notes

- Base model download and environment setup are the next steps (Section 6–7 of the guide).
