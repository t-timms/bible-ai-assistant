# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for milestone releases.

## [Unreleased]

### Added

- **Makefile task runner** — `make demo`, `make demo-build`, `make down`, `make logs`, `make status`, `make ollama`, `make model`, `make index`, `make test`, `make lint`, `make security`, `make ci`; replaces manual command sequences with a single entry point
- **Single-command launch** — `make demo` auto-detects whether Ollama is running and starts it in the background if not; no second terminal required
- **Kokoro TTS service** — `docker-compose.yml` now includes a third service (`ghcr.io/remsky/kokoro-fastapi-cpu`) on port 8880 with a named volume for model caching and a healthcheck; `gradio-ui` waits for TTS to be healthy before starting; end-to-end voice pipeline (STT → RAG → TTS) now runs entirely in Docker
- `deployment/pc/Dockerfile.ui` — `HF_HOME=/app/.cache/huggingface` with correct ownership; created home directory for `appuser` so Faster-Whisper model cache writes succeed
- CI badge added to README

### Fixed

- `deployment/pc/Dockerfile.rag` — replaced editable install (`-e`) with two-step non-editable install: deps cached in one layer, package installed separately with `--no-deps`; resolves `ModuleNotFoundError: No module named 'rag.rag_server'` on container startup
- `deployment/pc/Dockerfile.ui` — same two-step install pattern for consistency; source copied in builder stage
- `pyproject.toml` — corrected author from placeholder `"John AI"` to `"Tremayne Timms"`

---

### Added

- **Production hardening** — API key enforcement (`X-API-Key`, 401), Content-Type enforcement (415), body-first 1 MB size guard (413), startup warning when `API_KEY` is unset on a non-localhost bind address
- **Observability** — `X-Request-ID` correlation header threaded into every log line via `contextvars.ContextVar`; structured JSON logging (`LOG_JSON=true`); `exc_info=True` on unhandled exceptions; Prometheus `/metrics` endpoint (graceful no-op if `prometheus-fastapi-instrumentator` not installed)
- **Module split** — `rag_server.py` refactored into four focused modules: `helpers.py` (pure functions, no I/O), `retrieval.py` (hybrid pipeline), `settings.py` (Pydantic-validated config), `rag_server.py` (HTTP layer only)
- `rag/settings.py` — `pydantic-settings` singleton; validates `ollama_url`, `rag_top_k`, `hybrid_candidates`, `log_level`; reads from env / `.env`
- `RetrievalHit` NamedTuple — replaces anonymous `tuple[str, str, float]` throughout the retrieval pipeline
- BM25 JSON schema validation — validates `ids`/`documents` keys, types, and length parity before trusting deserialized index content
- Multi-stage Docker builds — `Dockerfile.rag` and `Dockerfile.ui` pinned to `python:3.11.12-slim`, non-root user (uid 1001), `HEALTHCHECK`, no build tools in runtime image
- `.dockerignore` — excludes ML artefacts (checkpoints, GGUF, ChromaDB, wandb) from Docker build context
- **Property-based tests** — `tests/test_hypothesis.py` with 28 Hypothesis tests across 5 pure helpers (idempotency, type invariants, length bounds)
- **API tests expanded** — `test_rag_api.py` from 6 → 10 tests: API key auth (401/200), Content-Type (415), request correlation (`X-Request-ID` echo), Prometheus metrics endpoint

### Changed

- `rag_server.py` — request body is now the authoritative size check (removes reliance on forged `Content-Length` header); `_RequestIDMiddleware` uses `ContextVar.set()`/`reset()` token pattern for correct async isolation
- `training/merge_adapters.py` — all `print()` calls replaced with `logger.*`; proper `logging.getLogger(__name__)` setup
- `pyproject.toml` — upper version bounds added to all critical dependencies (`chromadb<1.0.0`, `sentence-transformers<4.0.0`, `gradio<6.15`, `transformers<5.0.0`, `prometheus-fastapi-instrumentator<8.0.0`)
- `.github/workflows/ci.yml` — fixed broken `pip-audit` command; now installs the project then scans the installed environment; tightened test dep install to `.[rag,dev]`
- Test suite: **183 tests, 55% line coverage** (was 175 tests, 54%)

### Fixed

- `tests/test_training_utils.py` — `test_default_lora_path_prints_warning_then_raises` updated from `capsys` to `caplog` after `merge_adapters.py` converted `print()` to `logger.warning()`

---

## [Unreleased — prior]

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

## [0.1.0] - 2026-01-15

### Added

- Project scaffold and repository structure
- Biblical Constitution (CONSTITUTION.md) and system prompt
- .gitignore, .env.example, requirements.txt
- README and docs/architecture.md
- Placeholder directories and READMEs for data, training, rag, voice, deployment, ui
- Development workflow guide (docs/DEVELOPMENT_WORKFLOW.md)

### Notes

- Base model download and environment setup are the next steps (Section 6–7 of the guide).
