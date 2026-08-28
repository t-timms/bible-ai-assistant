# Bible AI Assistant — Remediation Roadmap

## Block 0 — V2 rebuild workstream 🚧 (branch `v2`)

Full-canon model-layer replacement; the RAG/eval/CI stack below stays as-is.
Status & schedule live in `docs/PROJECT_STATUS_AND_GOALS.md` → "V2 Rebuild".

- [x] Dataset engine: 8 categories, ~62k post-dedupe examples, sha-pinned sources (`cec12ce`)
- [x] V2 config **files**: `config.v2-9b.yaml` (near-term) + `config.v2.yaml` (27B stretch) + GRPO reward block
- [x] Config wiring: `train_unsloth.py --config/--data`; `_load_config_yaml` reads `data.train_file`
- [x] GRPO scaffold: `training/train_grpo.py` — verifiable reward (citation-exists + text-match + format) reusing `rag/verification.py`; `--dry-run` verified. Needs a GPU `--max-steps 2` smoke before any real run.
- [ ] `data/processed/train_v2.json` build + `check_train_eval_overlap.py` pass
- [ ] SFT launch (9B first via `config.v2-9b.yaml`; 27B only if 9B clears the v1 baseline)
- [ ] ORPO on the 9B SFT; then GRPO smoke → full GRPO
- [ ] Constrained decoding: verse-reference trie, applied to the citation span only (mind the "alignment tax" — arXiv 2604.06066)
- [ ] External-benchmark adapters — **now available** (2026-08-28 recheck): [FMG-Bench](https://github.com/FideAI/fmg-bench) (120 scenarios, has code), [FaithBench](https://faithbench.com/) (300+ cases, held-out). Use as honest external calibration, not win targets — both test theological *reasoning*/tradition-awareness, a harder and different task than RAG verse-citation.

---

**Purpose:** Reference for contributors. Completed blocks are marked ✅. Each block is a self-contained PR.

---

## Block 1 — Configurable title ✅

Add `TITLE` env var to RAG Settings, `GRADIO_TITLE` env var to Gradio UI.

- `rag/settings.py`: `title` field (default `"Bible AI RAG Server"`)
- `rag/rag_server.py`: uses `settings.title`
- `ui/app.py`: `GRADIO_TITLE` env var (default `"Bible AI Assistant"`)
- Tests: default, env override, FastAPI integration

---

## Block 2 — README Quick Start fix (C-1) ✅

Replace `requirements.txt` reference in README with `pip install -e ".[rag,ui,train,dev]"`.
Also updated `docs/WALKTHROUGH.md`, `docs/DEMO_LAUNCH.md`, `docs/DEVELOPMENT_WORKFLOW.md` to use the pyproject.toml install method.

---

## Block 3 — ReDoS vulnerability fix (C-2) 🆕

Replace `_strip_openclaw_metadata` regex with safe parser. The `{0,2000}` quantifier with `re.DOTALL` is a denial-of-service vector.

- `rag/helpers.py:293–299`: replace regex with string split / JSON detection
- Add test: `tests/test_rag_helpers.py`: property-based test with adversarial input

---

## Block 4 — Async event loop blocking fix (C-3) 🆕

Wrap `_rerank` CPU call in `asyncio.to_thread()` to prevent blocking the event loop under load.

- `rag/retrieval.py:230–241`: wrap `reranker.predict()` in thread pool
- Add performance test: benchmark 20-candidate rerank under concurrent load

---

## Block 5 — Enforce mypy in CI (C-4) 🆕

Remove `continue-on-error: true` from type-check job. Fix all existing mypy errors.

- `.github/workflows/ci.yml:63`: remove `continue-on-error`
- `pyproject.toml`: add `mypy` to dev dependencies
- Fix type errors in `rag/`, `training/`, `scripts/`

---

## Block 6 — ORPO precision fix (C-3 from previous audit)

Set `load_in_4bit=False` in `train_orpo.py` to match SFT precision. If VRAM constrained, document the tradeoff.

---

## Block 7 — Preference pair diversity (C-4 from previous audit)

Expand `_build_verbose_pairs` and `_build_bible_for_everything_pairs` in `build_preference_data.py` — currently 28% of pairs repeat only 12 unique prompts.

---

## Block 8 — Test coverage to 70% (H-1) 🆕

Raise coverage threshold and add missing tests.

- `pyproject.toml`: change `fail_under = 60` to `70`
- `.github/workflows/ci.yml`: change `--cov-fail-under=50` to `70`
- Add tests for `rag/retrieval.py` (mock ChromaDB)
- Add tests for `rag/build_index.py` (mock embeddings)
- Add tests for `rag/response_cleanup.py` (dedicated file)

---

## Block 9 — Integration / E2E tests (H-2) 🆕

Add tests that exercise the full request lifecycle.

- `tests/test_integration.py`: spin up in-memory ChromaDB, populate test verses, call `/v1/chat/completions`, verify response contains retrieved verses
- Use `pytest-recording` or VCR.py for HTTP mocking

---

## Block 10 — CORS middleware (H-3) 🆕

Add configurable CORS to RAG server.

- `rag/settings.py`: add `cors_origins: list[str] = []`
- `rag/rag_server.py`: add `CORSMiddleware`
- Tests: verify CORS headers on preflight request

---

## Block 11 — Configurable request body limit (H-4) 🆕

Move `MAX_REQUEST_BODY_BYTES` from hardcoded constant to settings.

- `rag/settings.py`: add `max_request_body_bytes: int = 1_048_576`
- `rag/rag_server.py`: use `settings.max_request_body_bytes`
- Tests: verify 413 still fires with default, verify override works

---

## Block 12 — Pin model revisions (H-5) 🆕

Eliminate `trust_remote_code=True` supply chain risk.

- `rag/retrieval.py`, `rag/build_index.py`: pin `nomic-ai/nomic-embed-text-v1.5` revision
- `training/train_unsloth.py`, `training/train_orpo.py`: pin `Qwen/Qwen3.5-4B` revision
- Document in `SECURITY.md`

---

## Block 13 — Input sanitization / prompt injection guard (H-6) 🆕

Prevent users from injecting system messages via API.

- `rag/rag_server.py`: strip or reject `role="system"` messages from user input
- Tests: verify system message injection returns 400 or is normalized to user

---

## Block 14 — Tighten dependency bounds (H-7) 🆕

Reduce version range width to prevent breaking changes.

- `pyproject.toml`: tighten upper bounds on all packages
- Add Dependabot auto-merge workflow after CI passes

---

## Block 15 — ChromaDB query timeouts (H-8) 🆕

Prevent hung requests when ChromaDB is unresponsive.

- `rag/retrieval.py`: wrap `collection.query()` in `asyncio.wait_for()` or process pool
- Tests: verify timeout returns 504 or empty context

---

## Block 16 — Refactor settings singleton (H-9) 🆕

Replace module-level singleton with dependency injection.

- `rag/settings.py`: add `get_settings()` factory
- `rag/rag_server.py`: use `Depends(get_settings)` instead of global `settings`
- Update all tests to use dependency override

---

## Block 17 — Fix project root resolution (H-10) 🆕

Make `_get_project_root()` robust to pip installs.

- `rag/retrieval.py`, `rag/build_index.py`: use `importlib.resources` or env var
- `rag/settings.py`: add `chroma_db_path: Path` setting
- Tests: verify works when imported from site-packages

---

## Block 18 — Fix `_content_to_str` (H-11) 🆕

Handle multiple text parts in content arrays.

- `rag/helpers.py:308–317`: concatenate all text parts
- Tests: verify list with 3 text parts returns concatenated string

---

## Block 19 — Add `__all__` exports (H-12) 🆕

Control public API surface.

- `rag/__init__.py`: add `__all__`
- `training/__init__.py`: add `__all__`

---

## Block 20 — Expand training dataset (H-13) 🆕

Grow from 1,800 SFT + 500 preference pairs to 10,000+ SFT + 2,000+ preference pairs.

- Generate from public domain commentaries (Matthew Henry, JFB)
- Add systematic theology Q&A (Grudem, Berkhof frameworks)
- Add cross-reference reasoning pairs
- Add refusal/boundary examples for constitutional behavior

---

## Block 21 — Health endpoint auth gate (H-14) 🆕

Prevent version leak to unauthenticated scanners.

- `rag/rag_server.py`: optionally require auth on `/health` or remove version from response
- Add setting: `health_auth_required: bool = False`

---

## Block 22 — Add CODEOWNERS (H-15) 🆕

Enable automatic PR reviewers.

- `.github/CODEOWNERS`: add `@t-timms`

---

## Block 23 — ORPO validation split (H-1 from previous audit)

Add `test_size=0.1` split and `eval_dataset` to ORPOTrainer. Add `eval_steps=20`.

---

## Block 24 — ORPO warmup fix (H-2 from previous audit)

Change `warmup_steps=20` to `warmup_steps=5` (~8% of 63 total steps).

---

## Block 25 — WANDB_PROJECT env var (H-3 from previous audit)

Replace hardcoded `"bible-ai"` with `os.getenv("WANDB_PROJECT", "bible-ai")` in `train_orpo.py`.

---

## Block 26 — LLM judge truncation (H-4 from previous audit)

Remove 1000-char truncation in `training/evaluate.py` or raise to 4000+.

---

## Block 27 — Remove time.sleep in eval (H-5 from previous audit)

Remove `time.sleep(0.5)` from `training/evaluate.py`. Use `asyncio.Semaphore` if throttling needed.

---

## Block 28 — Judge failure error (H-6 from previous audit)

Raise `RuntimeError` or emit `logging.error` + `"judge_available": false` when all judge endpoints fail.

---

## Block 29 — Local docker-compose (H-7 from previous audit)

Add `docker-compose.yml` for local dev (RAG server + Ollama). Add `start.sh` script.

---

## Block 30 — Quality polish (P2)

| ID | Action | File | Status |
|----|--------|------|--------|
| M-1 | Generalize `_strip_repetition_and_meta` | `rag/helpers.py` | 🆕 |
| M-2 | Simplify `_EVAL_SUFFIXES` to single regex | `rag/helpers.py` | ✅ |
| M-3 | Expand topical pin table to 20+ topics | `rag/helpers.py` | ✅ |
| M-4 | Add dedicated `response_cleanup.py` tests | `tests/test_response_cleanup.py` | 🆕 |
| M-5 | Extract shared Windows encoding fix | `training/_windows_fix.py` | 🆕 |
| M-6 | Implement or remove `voice/stt_server.py` | `voice/stt_server.py` | 🆕 |
| M-7 | Fix Docker healthchecks | `docker-compose.yml` | ✅ |
| M-8 | Align pyproject.toml version with releases | `pyproject.toml` | ✅ |
| M-9 | Remove redundant requirements files | Root directory | ✅ |
| M-10 | Remove response truncation logic | `rag/rag_server.py` | ✅ |
| M-11 | Fix `_is_verse_lookup` to require ref before "say" | `rag/helpers.py` | ✅ |
| M-2 | Move `random.seed(42)` into `main()` | `build_preference_data.py` | from prev |
| M-3 | Align MAX_SEQ_LENGTH (2048) between SFT and ORPO | `train_orpo.py` | from prev |
| M-4 | Print warning when using default adapter path | `merge_adapters.py` | from prev |
| M-5 | Explain counter-intuitive hallucination rate | `docs/MODEL_COMPARISON.md` | from prev |
| M-6 | Remove personal documents from public repo | `docs/`, root | from prev |
| O-2 | Gate traceback on `APP_ENV` | `rag/rag_server.py` | from prev |
| O-3 | Structured logging at RAG pipeline stages | `rag/rag_server.py` | from prev |
| O-4 | Gitignore checkpoint README stubs | `.gitignore` | from prev |

---

## Iteration loop (from OPTIMIZATION_PLAN.md)

```
1. Deploy current model (vN)
2. Run evaluate.py --judge --model-tag vN
3. Find worst category + worst questions
4. Add training examples for those failure modes
5. Rebuild data -> train vN+1
6. (Optional) Run ORPO on vN
7. Deploy vN+1 and vN-orpo, re-eval
8. Update leaderboard; compare
9. Repeat from step 2
```

---

## Block 13 — P3 Backlog (L items) ✅

| ID | Action | File | Status |
|----|--------|------|--------|
| L-1 | Rate limit: add settings validator + functional test | `rag/settings.py`, `tests/test_rag_api.py`, `tests/test_rag_pure_functions.py` | ✅ |
| L-2 | Manifest JSON schema: Pydantic model with full validation | `scripts/benchmark_schema.py`, `tests/test_benchmark_manifest.py` | ✅ |
| L-3 | Eval questions: 3 `topical_pin` questions for marriage, forgiveness, money | `prompts/evaluation_questions.json` | ✅ |
| L-4 | PS1 failure pattern in `build_preference_data.py` | — | ❌ (unable to define) |
| L-5 | Preference build audit script: structure, diversity, length checks | `scripts/audit_preference_data.py` | ✅ |

---

## See also

- `docs/CODEBASE_AUDIT.md` — full audit with severity ratings and line-level references
- `docs/OPTIMIZATION_PLAN.md` — strategies for maximizing domain scores
- `docs/SHIP_v1_AND_POLISH_BACKLOG.md` — v1 completion checklist
- `docs/DEVELOPMENT_WORKFLOW.md` — phase-gated workflow
