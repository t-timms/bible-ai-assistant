# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for milestone releases.

## [Unreleased]

### Added
- **V2 dataset engine** — `training/build_dataset_v2.py`: 61,556 post-dedupe examples across
  8 categories (multi-translation verbatim recall, reverse lookup, off-by-one near-miss guards,
  passage spans, TSK cross-reference chains, anchored topical sets, unique-trigram chapter
  context); sha-pinned public-domain sources (6 translations) with manifest sidecar; 17 offline tests
- **V2 training config** — `training/config.v2.yaml`: Qwen3.5-14B QLoRA recipe + GRPO scaffold
  with fully programmatic verifiable rewards


- **Shared prompt format contract** — `rag/prompt_format.py` with `augment_question` / `extract_question` as the single source of truth for SFT/inference/evaluation prompt assembly; byte-exact smoke tests (`tests/test_prompt_format.py`)
- **Benchmark protocol v3** — `benchmarks/manifest.v3.yaml`: frozen suite snapshots (sha256-pinned), contamination disclosure, fuzzy threshold 0.85, min-n=30 policy, decoding-param recording
- **282-question evaluation suite** — `benchmarks/suites/evaluation_questions.v2.json` (expanded from 57); v1 snapshot preserved for reproducibility
- **Retrieval evaluation harness** — `scripts/build_qrels.py` + `scripts/retrieval_metrics.py`: recall@k / MRR / nDCG over dense/BM25/fused/fused+rerank variants; synthetic qrels tests
- **Benchmark statistics** — `scripts/benchmark_stats.py`: Wilson CIs, McNemar exact test, paired bootstrap delta (pure-python, no scipy); `tests/test_stats.py`
- **Train-eval overlap detector** — `scripts/check_train_eval_overlap.py` + pytest wrapper; enforces zero normalized-question overlap between training pools and eval suites
- **GGUF export in-repo** — `training/export_gguf.py` (was external); quantize + Modelfile generation pipeline
- **Decontamination filter** — training pipeline filters eval-overlapping questions from generated datasets using shared `normalize_question` contract
- **Prompt loss masking** — SFT training masks system prompt and padding tokens (was padding-only)
- **Cosine scheduler + warmup** — explicit `lr_scheduler_type: cosine` and `warmup_ratio: 0.05` in training config
- **`uv lock --check` gate** in CI — ensures lockfile freshness
- **15 new test modules** — 412 total tests (was 183)

### Changed

- **uv.lock regenerated** — gradio 6.9.0→6.26.0 (CVE-2026-1839 fix), starlette 1.6.0, transformers 5.3.0, torch 2.11.0, chromadb stays 1.5.5 per documented rationale
- **BM25 tokenizer** — punctuation-aware tokenization (`re.findall(r"[a-z0-9']+")`) replaces whitespace-only; identical in indexing and query paths; stale index detection via version marker
- **Embedding normalization** — collections now use cosine space; `normalize_embeddings=True` at encode time
- **Context budget** — multi-turn context accumulation eliminated (extract stale context from prior turns); `context_max_chars=3500` enforced after rerank; `num_ctx` raised to 4096
- **Serving hardening** — model allowlist, max_tokens clamp, body-size guard before buffering, X-Request-ID validation, rate-limit-on-auth-fail, warm-up on startup, dedicated dense-search thread pool
- **Citation verification** — exception-guarded (never destroys a generated response over infra failure); possessive fragments and connective words properly stripped from verse refs
- **ORPO training** — conversational format inputs, hard negatives, rebalanced pair budget (2080 total), revision pinning, weight hashing
- **docker-compose.yml** — security env passthrough (API_KEY, RATE_LIMIT, LOG_JSON, CORS_ORIGINS); latent healthcheck bug fixed (curl-less python-slim image → stdlib probe)

### Fixed

- **SFT/inference prompt-format skew** — bulk of SFT data trained a format that never occurred at inference; unified via shared `prompt_format.py`
- **Train/eval contamination** — ~100+ verbatim duplicates between training pools and eval suite identified and decontaminated
- **`verify_citations` crash** — unguarded call in response path could 500 after generation succeeded
- **Streaming bypass** — streaming requests now go through verification and output scrubbing
- **Benchmark default** — `run_benchmark.py` defaulted to manifest v1; now auto-resolves latest
- **Passage-expansion collision** — `"John 3:16"` no longer matches `"1 John 3:16"` via pipe-delimited child_ids
- **Citation aliases** — expanded beyond Psalm(s) to cover Song of Solomon, numeric-prefix abbreviations (~20 entries)
- **Dead knobs** — `settings.hybrid_candidates` now wired into retrieval; `settings.ollama_model` is the allowlist source
- **Dockerfile.rag healthcheck** — switched from curl (not in python-slim) to stdlib probe; added timeout
- **Stale lockfile** — `uv.lock` regenerated with CVE-fixed gradio; `uv lock --check` in CI prevents future drift

---

## [0.9.0] - 2026-05-07

### Added

- **Configurable title** — `TITLE` env var in RAG Settings, `GRADIO_TITLE` env var in Gradio UI (Block 1)
- **Remediation roadmap** — `ROADMAP.md` on desktop tracks all remediation blocks; all 12 blocks now complete
- **ORPO validation split** — `test_size=0.1` split, `eval_dataset` to ORPOTrainer, `eval_steps=20` (Block 5)
- **ORPO warmup fix** — `warmup_steps=20` → `warmup_steps=5` (~8% of total steps) (Block 6)
- **WANDB_PROJECT env var** — replaces hardcoded `"bible-ai"` with `os.getenv("WANDB_PROJECT", "bible-ai")` (Block 7)
- **LLM judge truncation** — removed `response[:1000]` truncation in evaluate.py (Block 8)
- **APP_ENV gating** — traceback details only when `APP_ENV=development` (Block 12 / O-2)
- **5-stage CI pipeline** — type-check, dependency gating, test artifacts
- CI badge added to README
- Why section added to README

### Changed

- README Quick Start — `requirements.txt` → `pip install -e ".[rag,ui,train,dev]"` (Block 2)
- Badges standardized to flat-square style
- Architecture diagram converted to Mermaid
- `.gitignore` — ignores checkpoint README stubs (Block 12 / O-4)
- Various dependency bumps (chromadb, transformers, trl, datasets, gradio, etc.)

### Fixed

- Multiple blocks already implemented in code but missing from roadmap — now tracked correctly

---

## [0.6.0] - 2026-03-24

### Added

- **Makefile task runner** — `make demo`, `make demo-build`, `make down`, `make logs`, `make status`, `make ollama`, `make model`, `make index`, `make test`, `make lint`, `make security`, `make ci`; replaces manual command sequences with a single entry point
- **Single-command launch** — `make demo` auto-detects whether Ollama is running and starts it in the background if not; no second terminal required
- **Kokoro TTS service** — `docker-compose.yml` now includes a third service (`ghcr.io/remsky/kokoro-fastapi-cpu`) on port 8880 with a named volume for model caching and a healthcheck; `gradio-ui` waits for TTS to be healthy before starting; end-to-end voice pipeline (STT → RAG → TTS) now runs entirely in Docker
- `deployment/pc/Dockerfile.ui` — `HF_HOME=/app/.cache/huggingface` with correct ownership; created home directory for `appuser` so Faster-Whisper model cache writes succeed
- Docker preflight check to demo targets
- MIT license
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/dependabot.yml` for automated dependency updates

### Changed

- `deployment/pc/Dockerfile.rag` — replaced editable install with two-step non-editable install
- `deployment/pc/Dockerfile.ui` — same two-step install pattern for consistency
- `pyproject.toml` — corrected author from `"John AI"` to `"Tremayne Timms"`
- CI — Python 3.10–3.12 matrix; coverage report without fail-under threshold
- `scripts/start_demo.ps1` — PowerShell for Ollama auto-start on Windows

---

## [0.5.0] - 2026-03-24

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

- `rag_server.py` — request body is now the authoritative size check (removes reliance on forged `Content-Length` header)
- `_RequestIDMiddleware` — uses `ContextVar.set()`/`reset()` token pattern for correct async isolation
- `training/merge_adapters.py` — all `print()` calls replaced with `logger.*`; proper `logging.getLogger(__name__)` setup
- `pyproject.toml` — upper version bounds added to all critical dependencies
- `.github/workflows/ci.yml` — fixed broken `pip-audit` command; now installs the project then scans the installed environment; tightened test dep install to `.[rag,dev]`
- Test suite: **183 tests, 55% line coverage** (was 175 tests, 54%)

### Fixed

- `tests/test_training_utils.py` — `test_default_lora_path_prints_warning_then_raises` updated from `capsys` to `caplog` after `merge_adapters.py` converted `print()` to `logger.warning()`

---

## [0.3.0] - 2026-03-14

### Added

- **SFT fine-tuning** — `training/train_unsloth.py`: bf16 LoRA on Qwen3.5-4B, config.yaml, W&B logging, Blackwell xformers workaround
- **ORPO preference alignment** — `training/train_orpo.py`: preference optimization with ORPOTrainer, `load_in_4bit=False`, `warmup_steps=5`, `eval_steps=20`, `test_size=0.1` validation split
- **Hybrid RAG server** — `rag/rag_server.py`: FastAPI middleware with ChromaDB dense + BM25 sparse + RRF + Cross-Encoder reranking (bge-reranker-v2-m3)
- **ChromaDB index builder** — `rag/build_index.py`: nomic-embed-text-v1.5 with `search_document:` / `search_query:` prefixes; chunks Bible into verses and passages
- **GGUF quantization** — Unsloth export to GGUF; llama.cpp q4_k_m and f16 variants
- **Ollama Modelfile generator** — `deployment/pc/generate_modelfile.py`: produces Modelfile from system prompt template
- **Evaluation pipeline** — `training/evaluate.py`: keyword-overlap scoring and LLM-as-judge (qwen3.5:27b) on 54 questions across 6 categories
- **Merge adapters** — `training/merge_adapters.py`: merges LoRA adapters with base model
- **RAG retrieval hardening** — verse reference pinning, topical anchor verses (marriage, forgiveness, money), counseling-pattern detection with system guard
- `rag/response_cleanup.py` — shared `strip_model_thinking()` for Qwen/Ollama chain-of-thought
- `EMPTY_MODEL_REPLY` fallback when model returns empty content
- `scripts/run_benchmark.py`, `scripts/compare_benchmark_runs.py`, `benchmarks/manifest.v1.yaml` — versioned keyword / judge benchmarks
- `requirements-ui.txt` — Gradio + voice deps for envs without full training stack
- `docs/DEMO_LAUNCH.md`, `docs/WALKTHROUGH.md` — launch checklist and comprehensive walkthrough
- **Gradio 6 UI** — `ui/app.py`: landing hero, stack health check, model override field, amber theme, voice tab, auto-pick free port
- `docs/MODEL_COMPARISON.md` — SFT vs SFT+ORPO head-to-head with counter-intuitive hallucination analysis
- `docs/ENVIRONMENT_REQUIREMENTS.md` — environment setup guide
- Tests for RAG helpers, eval keywords, manifests

### Changed

- `rag/rag_server.py` — Ollama `"think": false` by default; non-streaming always assigns cleaned `message.content`; hybrid retrieval merges pinned verses with reranked results; meta-question handling, OpenClaw metadata stripping
- `prompts/system_prompt.txt` — stronger topical relevance, counseling boundaries, verse-lookup accuracy, tone guidelines
- `training/evaluate.py` — strips thinking on RAG replies; judge HTTP `trust_env=False` with 3 endpoint fallbacks; `--judge-model` (default `qwen3.5:27b`)
- README, docs/README.md, ui/README.md, requirements-rag.txt — demo/UI install and env notes

### Fixed

- `strip_model_thinking()` — paired `</think>`…`</think>` before flex `think` peeling; leading BOM after tag removal
- RAG OpenAI JSON — always persist cleaned assistant text (punctuation edge case)
- Verse ref extraction — avoid matching "What does Hebrews…" as the reference; strip lookup prefixes before regex
- Gradio 6 — removed unsupported `Chatbot` `type=` / `show_copy_button`; moved `theme`/`css` to `launch()`
- Ollama response quality — concise verse answers, informative translation note

---

## [0.2.0] - 2026-01-15

### Added

- WEB dataset pipeline (`training/dataset_builder.py`)
- ~1,800 diverse Bible Q&A examples
- `data/sample.json` as documentation
- Initial training data generation and formatting

---

## [0.1.0] - 2026-01-15

### Added

- Project scaffold and repository structure
- Biblical Constitution (`CONSTITUTION.md`) and system prompt
- `.gitignore`, `.env.example`, `requirements.txt`
- README and `docs/architecture.md`
- Placeholder directories and READMEs for data, training, rag, voice, deployment, ui
- Development workflow guide (`docs/DEVELOPMENT_WORKFLOW.md`)
