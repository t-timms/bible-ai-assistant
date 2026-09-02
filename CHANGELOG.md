# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for milestone releases.

## [Unreleased]

### Added
- **Benchmark protocol v4 + SOTA evaluation track (2026-09-02)** —
  `benchmarks/manifest.v4.yaml` (`bible_assistant_baseline_v4`) splits the single
  `verse_lookup` category into `verse_quote` (66 Qs — "What does X say?", "Quote X") and
  `verse_exposition` (36 Qs — "What does X teach?", "What is X about?"); deterministic rule,
  same 282 questions, produced by `scripts/make_v4_suite.py` →
  `benchmarks/suites/evaluation_questions.v3.json` (sha-pinned). `verse_exposition`'s headline
  metric is fuzzy pass-rate, not exact-match (an explanation of a verse is a pass at
  exact-match 0 — same reasoning protocol v3 applied to `refusal`); overall fuzzy mean is
  reported **all-in and exposition-excluded**. `scripts/rescore_v4.py` moves an existing
  protocol-v3 keyword run to v4 with no re-generation (deterministic re-bucket, aggregation
  reused from `evaluate.py`); `scripts/exposition_sidebyside.py` dumps the 36 exposition items
  v2-vs-v3 for a manual read.
  - **SOTA track — `docs/SOTA_EVAL.md`**: `benchmarks/external_comparators.yaml` (8 open
    comparators — `sleepdeprived3/Christian-Bible-Expert` 8B/12B, `nbeerbower/llama-3-bible-dpo-8B`,
    `Phora68/bible-study-phi3-mini`, `rhemabible/BibleAI`, Qwen3-8B/14B/32B instruct) run through
    the **unchanged** RAG stack on the v4 suite via `scripts/run_external_baselines.sh` +
    `scripts/_run_ext_eval.sh`; `scripts/sota_scoreboard.py` builds the ranked head-to-head
    (Wilson CIs, paired McNemar vs. our best) and a **scoped verdict** — "best *open* model at
    RAG-grounded scripture Q&A, size-independent" + "SOTA for the 16 GB Blackwell class", never
    a frontier / unconstrained-hardware claim.
- **LLM-judge on 16 GB: `qwen3.5:27b` is infeasible (measured, 2026-09-02)** — the v3 default
  judge (Q4_K_M, ~17 GB) does not fit the 16 GB VRAM budget, CPU-offloads, and one rubric call
  measured **333.7 s** on an idle GPU — past `evaluate.py`'s 180 s HTTP timeout, which is why
  the 2026-09-01 and 2026-09-02 judge runs both failed on question 1. Protocol-v4 judge runs
  (if any) use `qwen3:8b` and are calibration-only, not a gate. Recorded in
  `benchmarks/manifest.v4.yaml` and `docs/BENCHMARK_PROTOCOL.md`.
- **FMG-Bench external-calibration adapter (2026-08-31)** — `scripts/fmg_bench.py` runs the
  open Faith & Moral Guidance Benchmark (`FideAI/fmg-bench`, CC-BY-4.0, 120 scenarios + 37
  perturbations, no hidden-test leaderboard). Fetch → generate → rubric LLM-judge → weighted
  per-dimension scores + escalation recall/false-escalation (Wilson CIs) + disallowed-failure
  rate, broken down by family/triage; run JSON records the dataset sha256. `--dry-run` (stub
  judge) validates the whole pipeline offline; a real run needs a served model + judge.
  Reported as honest calibration, **not** a pass/fail gate — it tests a different task than
  the protocol-v3 verse-citation suite. +8 tests (**464 total**; 7 skip in CI on the
  corpus-gated cases). Docs corrected: **FaithBench**
  (faithbench.com, the Christian-theology site) is *not* usable — leaderboard-only research
  preview, no public dataset, linked repo 404s — removed from the "wire in" plan across
  ROADMAP / MODEL_CARD / V2_EXECUTION_PLAN / V3_DATASET_PLAN / PROJECT_STATUS_AND_GOALS.
- **v3 dataset pipeline + `train_v3.json` (2026-08-31)** — teacher-distilled answers replace the
  templated ones that regressed v2's open-ended quality. `training/build_v3_inputs.py` emits
  16,995 `(context, question)` inputs from the four templated-answer generators;
  `training/distill_answers.py` (now with `--concurrency` and an `enable_thinking:false` /
  `<think>`-strip patch for the OpenAI-compat `vllm` backend) regenerates the answers against a
  local **Qwen3-14B Q5_K_M GGUF** served by `llama-server` (vLLM is unusable on this box —
  `UVA is not available` under WSL2), every answer citation-validated — **16,809 / 16,995 kept
  (98.9%)**; `training/assemble_v3.py` merges those with freshly-built keep-as-is categories
  (verse-drill cut ~60% to ~7k, `near_miss_guard`, `pastoral_triage`, `general_blend` ~11k) into
  **`data/processed/train_v3.json` — 39,463 examples**, general/reasoning share 27.9% (clears the
  catastrophic-forgetting floor), zero eval-suite overlap. `training/config.v3-4b.yaml` (fork of
  `config.v2-4b.yaml`, seed 20260830). `thematic_qa` deferred to a follow-up (needs the live RAG
  retriever). Status + next action: `docs/V3_STATUS.md`; plan: `docs/V3_DATASET_PLAN.md`. SFT not
  yet run. +8 tests.
- **v2-4b model + protocol-v3 evaluation (2026-08-29)** — Qwen3.5-4B bf16 LoRA SFT, 1 epoch on
  55,570 examples (`training/config.v2-4b.yaml`), eval_loss 0.25→0.21. First measurement under
  benchmark protocol v3: verse-lookup exact accuracy **58% → 76.5%** vs. the v1 model, citation
  rate **88% → 98.9%**, hallucination flat at 2.3%; overall fuzzy mean regressed 0.48 → 0.40
  because the dataset's templated *answers* made open-ended thematic responses worse. Full A/B
  in `docs/MODEL_COMPARISON.md`; raw results in `docs/benchmark_runs/2026-08-29_*`; card
  rewritten in `docs/MODEL_CARD.md`. Known: no GGUF/Ollama yet (Qwen3.5-4B hybrid arch
  unsupported by llama.cpp); served for eval via `scripts/_tf_openai_server.py`.
- **v2 dataset "full upgrade"** — capped the 8 scripture-citation categories (~61k → ~35.6k),
  added `grounded_exegesis` (Matthew Henry's Commentary, CC0, via
  `training/fetch_mhc_commentary.py`), `pastoral_triage` (escalation / tradition-aware /
  calibrated abstention), and `general_blend` (HuggingFaceTB/smoltalk2, Apache-2.0, `<think>`
  stripped, ~24% of the mix as a catastrophic-forgetting guard); probabilistic real-user
  framing prefixes on every generator; per-source SHA + license in the manifest. 56,022
  examples, 0 eval-suite overlap. `training/train_unsloth.py`: completion-mask length filter +
  fixed-padding revert (dynamic padding fragmented the CUDA allocator near the 16 GB ceiling).
- **V2 dataset engine** — `training/build_dataset_v2.py`: originally 61,556 post-dedupe examples
  across 8 categories (multi-translation verbatim recall, reverse lookup, off-by-one near-miss
  guards, passage spans, TSK cross-reference chains, anchored topical sets, unique-trigram
  chapter context); sha-pinned public-domain sources (6 translations) with manifest sidecar; 17 offline tests
- **V2 training configs** — `training/config.v2-4b.yaml` (4B bf16 LoRA, near-term target),
  `training/config.v2-9b.yaml` (9B QLoRA), `training/config.v2.yaml` (27B QLoRA stretch);
  GRPO scaffold with fully programmatic verifiable rewards


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

### Documentation

- **SOTA audit pass (2026-08-28)** — reconciled every self-reported number with a
  from-scratch CI-toolchain run. Fixed: test count `412 → 430` and `15 → 16` modules
  (README, CONTRIBUTING); eval-suite size (`54` historical vs `282` current / 8 categories)
  disambiguated with an explicit "not re-run under protocol v2" note; CI section rewritten
  as the real staged pipeline incl. the undocumented `Type Check` job; broken `pip install
  -e ".[dev]"` test command → `".[rag,dev]"`; broken `ollama pull bible-assistant-orpo`
  Quick Start step → local build path; `github.com/your-org/...` placeholders ×3,
  non-existent `docs/evaluation_results.json` / `training/{sft,orpo}_train.py` references,
  and the `TODO`/`2025` citation block fixed; HF badge repointed to the real adapter;
  `MODEL_COMPARISON.md` RAM `64 → 96 GB`. Full findings + the open non-doc gaps in
  `docs/CODEBASE_AUDIT.md` § "SOTA Audit — 2026-08-28".

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
