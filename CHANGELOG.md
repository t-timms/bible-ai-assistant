# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for milestone releases.

## [Unreleased]

### Added

- `rag/response_cleanup.py` — shared `strip_model_thinking()` for Qwen/Ollama chain-of-thought and planning scaffolds
- **RAG retrieval hardening** — pin explicit verse refs for “What does X say?” lookups; topical anchor verses (marriage, forgiveness, money); Psalm→Psalms id normalization; counseling-pattern detection with an extra system guard in chat payloads
- `EMPTY_MODEL_REPLY` fallback when the model returns empty content (non-stream and stream paths)
- `scripts/run_benchmark.py`, `scripts/compare_benchmark_runs.py`, `benchmarks/manifest.v1.yaml` — versioned keyword / judge benchmarks
- `requirements-ui.txt` — Gradio + voice deps for envs that only installed `requirements-rag.txt`
- `docs/DEMO_LAUNCH.md`, `scripts/start_demo.ps1` — Ollama + RAG + Gradio launch checklist
- **Gradio 6 UI** — landing hero, stack health check, model override field, amber theme, `theme`/`css` on `launch()`, auto-pick free port if `7860` is busy (`GRADIO_PORT` / `GRADIO_SERVER_PORT`)
- `tests/` — RAG helpers (verse extraction, counseling detection, topical pins), eval keywords, manifests; CI runs `tests/` and `deployment/` with Ruff

### Changed

- `rag/rag_server.py` — Ollama `"think": false` by default; non-streaming always assigns cleaned `message.content`; hybrid retrieval merges pinned verses with reranked results
- `prompts/system_prompt.txt` — stronger topical relevance, counseling boundaries, verse-lookup accuracy
- `training/evaluate.py` — strips thinking on RAG replies; judge HTTP `trust_env=False` and endpoint fallbacks; `--judge-model` (default `qwen3.5:27b`)
- `README.md`, `docs/README.md`, `ui/README.md`, `requirements-rag.txt` — demo/UI install and env notes
- **CI** — Python 3.10–3.12 matrix; coverage report without a fail-under threshold (training scripts are mostly CLI)

### Fixed

- `strip_model_thinking()` — paired `</think>`…`</think>` before flex `think` peeling; leading BOM after tag removal
- RAG OpenAI JSON — always persist cleaned assistant text (punctuation edge case)
- **Ruff** — clean `training/`, `rag/`, `scripts/`, `ui/`, `tests/`, `deployment/` (per-file ignores where intentional)
- **Verse ref extraction** — avoid matching “What does Hebrews…” as the reference; strip lookup prefixes before regex
- **Gradio 6** — removed unsupported `Chatbot` `type=` / `show_copy_button`; moved `theme`/`css` to `launch()`

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
