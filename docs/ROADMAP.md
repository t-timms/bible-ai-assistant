# Bible AI Assistant — Remediation Roadmap

## Block 0 — V2 rebuild workstream 🚧 (branch `v2/dataset-full-upgrade-2026-08-28`)

Full-canon model-layer replacement; the RAG/eval/CI stack below stays as-is.
**v2-4b SFT is trained, evaluated, and published** (2026-08-29) — see "▶ Resume here" below
for what's next. Narrative status: `docs/PROJECT_STATUS_AND_GOALS.md`; full detail:
`docs/V2_EXECUTION_PLAN.md`.

- [x] Dataset engine: 8 categories, ~62k post-dedupe examples, sha-pinned sources (`cec12ce`)
- [x] V2 config **files**: `config.v2-9b.yaml` (near-term) + `config.v2.yaml` (27B stretch) + GRPO reward block
- [x] Config wiring: `train_unsloth.py --config/--data`; `_load_config_yaml` reads `data.train_file`
- [x] GRPO scaffold: `training/train_grpo.py` — verifiable reward (citation-exists + text-match + format) reusing `rag/verification.py`; `--dry-run` verified. Needs a GPU `--max-steps 2` smoke before any real run.
- [x] **Dataset full upgrade** (2026-08-28, branch `v2/dataset-full-upgrade-2026-08-28`): scripture cats capped ~61k→~34k; new `grounded_exegesis` (Matthew Henry, CC0), `pastoral_triage` (escalation/tradition-aware/abstention), `general_blend` (smoltalk2, Apache-2.0, `<think>` stripped); persona/phrasing diversity. `train_v2.json` = 56,022 examples, **23.9 % general/reasoning** (clears the forgetting floor), 0 eval overlap, 439 tests pass. See `docs/V2_EXECUTION_PLAN.md`.
- [x] `data/processed/train_v2.json` build + `check_train_eval_overlap.py` pass
- [x] **Base-model decision (2026-08-28): 4B first.** `training/config.v2-4b.yaml` — `Qwen/Qwen3.5-4B` bf16 LoRA (fully unquantized). Escalate to 9B only on a measured shortfall.
- [x] **SFT run (v2-4b), 2026-08-29** — 1 epoch / 3,474 steps / ~10.4 h, eval_loss 0.25→0.21, no overfit. Merged → `models/qwen3.5-4b-bible-v2-merged`.
- [x] **Protocol-v3 eval + v1 A/B, 2026-08-29** — v2 **+18.5 pp verse-lookup (58→76.5%)**, **+11 pp citation (→98.9%)**, hallucination flat 2.3%; but **−0.09 fuzzy overall** — templated answers regressed thematic responses vs. the lightly-tuned v1. Bottleneck = the dataset's templated *answers*. `docs/MODEL_COMPARISON.md`, `docs/benchmark_runs/2026-08-29_*`.
- [x] **GGUF built + published, 2026-08-29** — `convert_hf_to_gguf.py --no-mtp` on current llama.cpp (the `~/wsl41361/llama.cpp` used at first was a 307-line stub). F16 + Q8_0/Q6_K/Q5_K_M/Q4_K_M, verified with `llama-server` (~85 tok/s). Ollama 0.33.x's bundled runtime is still too old for `qwen35` — LM Studio / current llama.cpp work now.
- [x] **Published to HF, 2026-08-29** — [`Ttimms/Bible-Assistant-Qwen3.5-4B-v2`](https://huggingface.co/Ttimms/Bible-Assistant-Qwen3.5-4B-v2) (safetensors + card) and [`…-v2-GGUF`](https://huggingface.co/Ttimms/Bible-Assistant-Qwen3.5-4B-v2-GGUF). Docs (README, MODEL_CARD, MODEL_COMPARISON, this file, PROJECT_STATUS_AND_GOALS, evaluation_results, CHANGELOG) updated on branch `v2/dataset-full-upgrade-2026-08-28`.

### ▶ Resume here (2026-08-29)

1. [ ] **Commit + merge** branch `v2/dataset-full-upgrade-2026-08-28` (11 modified + untracked: `config.v2-4b.yaml`, `fetch_mhc_commentary.py`, `scripts/run_v2_4b_sft.sh`, `scripts/_tf_openai_server.py`, `scripts/_run_v3_eval.sh`, `docs/benchmark_runs/20260829_*`).
2. [ ] **Judge re-score** v2 + v1 under protocol v3 with `--judge` (judge model `qwen3:8b` is pulled) — the keyword `verse_accuracy` scores 0 on synthesis questions with no canonical verse; the judge gives them a fair score.
3. [x] **Dataset v3 = teacher distillation (2026-08-31)** — `training/build_v3_inputs.py` +
   `training/distill_answers.py` (local Qwen3-14B Q5_K_M GGUF teacher via `llama-server`; vLLM
   dead on this box) + `training/assemble_v3.py` → **`data/processed/train_v3.json`, 39,463
   examples**, templated answers regenerated as synthesized cited prose, verse-drill cut ~60%,
   98.9% distillation keep-rate, zero eval overlap. `config.v3-4b.yaml` added. `thematic_qa`
   deferred (needs the RAG retriever). `docs/V3_STATUS.md` / `docs/V3_DATASET_PLAN.md`.
4. [ ] **SFT on v3** (4B first, `training/config.v3-4b.yaml`) → **GRPO** (`training/train_grpo.py`, verifiable citation reward) — the stage meant to clear the ≥85 % verse-accuracy bar and fix the thematic-synthesis regression. Needs a `--max-steps 2` GRPO smoke first.
5. [ ] **Re-eval** protocol v3 + FMG-Bench calibration (`scripts/fmg_bench.py`). Escalate to 9B (`config.v2-9b.yaml`) **only** on a measured shortfall.
6. [ ] `rag_server.py` **commentary-retrieval path** (so `grounded_exegesis` training matches inference — else it's the F-2/F-3 format mismatch).
7. [ ] **Retrieval upgrade** — embedder stronger than `nomic-embed-text-v1.5`; then **constrained verse-reference decoding** (trie on the citation span; mind the alignment tax, arXiv 2604.06066).
8. [ ] **Ornith GGUF backfill** — feasible: convert the *non-MTP-stripped* pruned bf16 (or the with-MTP variant); `unsloth/Qwen3.5-35B-A3B-GGUF` proves `qwen3_5_moe` GGUF works upstream.
9. [ ] *(optional)* `microsoft/WSL#41361` — the fresh llama.cpp build (`3173a56`) is the commit the maintainer asked for; do a deliberate long-run hang repro + call stack if reopening.

### Deferred / blocked
- [ ] **vLLM** — `Qwen3_5ForCausalLM` registered locally but `UVA is not available` under WSL2 (0.26.0 `GPUModelRunnerV2`). Eval ran through `scripts/_tf_openai_server.py` instead.
- External-benchmark adapter — [FMG-Bench](https://github.com/FideAI/fmg-bench) (120 scenarios + 37 perturbations, open, CC-BY-4.0): **done** — `scripts/fmg_bench.py` (`--dry-run` offline; real run needs a served model + judge). Honest calibration, not a win target — it tests theological triage/comparison/escalation, a harder/different task than RAG verse-citation. Run at step 5. [FaithBench](https://faithbench.com/) (the Christian-theology site) is **not usable** — leaderboard-only research preview, no public dataset, linked repo 404s (2026-08-31); watch for a data release.

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
