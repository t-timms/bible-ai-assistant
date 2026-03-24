# Professional Codebase Audit — Bible AI Assistant

**Audit date:** 2026-03-23
**Auditor:** Second-pass deep review (post-v1 hardening)
**Scope:** Full codebase — ML pipeline, training data, evaluation, RAG, API, UI, CI/CD, docs, deployment
**Baseline state:** Previous audit applied security hardening, streaming fix, Pydantic models, thread-safe loaders, CI improvements, and 56-test suite.

---

## Executive Summary

The project is a well-conceived, end-to-end ML system demonstrating SFT + ORPO fine-tuning, hybrid RAG, and voice UI.  The first-pass hardening resolved the most critical runtime and security issues. This audit goes deeper: into ML training correctness, data pipeline quality, evaluation validity, and deployment readiness. Seventeen distinct findings remain, ranging from training methodology flaws that affect model quality to a broken Quick Start that prevents new users from running the project at all.

| Severity | Count | Examples |
|----------|-------|---------|
| **Critical** | 4 | Broken Quick Start, monolithic deps, ORPO QLoRA precision mismatch, verbose-pair duplication |
| **High** | 7 | ORPO no eval split, WANDB hardcoded, judge truncation, warmup misconfigured, no deployment artifacts |
| **Medium** | 6 | Outdated audit doc, global seed side-effect, silent eval failures, seq-length inconsistency |

---

## Critical Findings (P0)

### C-1 — Quick Start is Broken (`README.md:106-108`)

The Quick Start instructs users to run `pip install -r requirements.txt`, but `requirements.txt` no longer exists — it was superseded by `pyproject.toml`. A first-time user following the README will hit an immediate `FileNotFoundError` before they can run anything.

Additionally, `python rag/build_index.py` (`README.md:113`) should be `build-index` (the installed entry point declared in `pyproject.toml:101`) or `python -m rag.build_index`. And `uvicorn rag.rag_server:app --port 8081` (`README.md:117`) should specify `--host 127.0.0.1` to match the secure default established in `rag_server.py`.

**Fix:**
```bash
# 2. Install the package (all components)
pip install -e ".[rag,ui,train,dev]"

# 3. Build RAG index
build-index

# 4. Start services
ollama run bible-assistant-orpo
rag-server --port 8081          # or: uvicorn rag.rag_server:app --host 127.0.0.1 --port 8081
python ui/app.py
```

---

### C-2 — Monolithic `[project.dependencies]` (`pyproject.toml:25-50`)

Every install target — `transformers`, `unsloth`, `bitsandbytes`, `chromadb`, `gradio`, `faster-whisper`, `wandb` — lives in `[project.dependencies]` (mandatory). The `[project.optional-dependencies]` groups (`rag`, `ui`, `train`) exist but are **additive** on top of the already-installed full stack. This means:

- `pip install -e .` pulls the entire ML stack including CUDA deps, Unsloth, and Gradio — even for a user who only wants the RAG server.
- `pip install -e .[rag]` still installs all mandatory deps first, then adds `[rag]`'s duplicates.
- CI spends minutes installing torch, unsloth, bitsandbytes that are never used during tests.

**Fix:** Move all deps to optional groups; keep `[project.dependencies]` minimal (e.g. only `pydantic`, `python-dotenv`, `PyYAML`). Update CI to use `pip install -e .[dev]`.

---

### C-3 — ORPO Trains with 4-bit Quantized Base (Precision Mismatch) (`training/train_orpo.py:173`)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    ...
    load_in_4bit=True,       # ← QLoRA
    ...
)
```

The SFT stage (`train_unsloth.py`) trains with `load_in_4bit=False` (full bf16 LoRA). ORPO then reloads the base in **NF4 4-bit** and overlays the SFT adapter — making ORPO a QLoRA stage on a LoRA checkpoint. This introduces two problems:

1. **Quantization noise contaminates the alignment signal.** The ORPO loss computes log-probability ratios between chosen/rejected under the quantized model, but the SFT checkpoint was calibrated under full precision. The reference policy and the training policy use different quantization states.

2. **The ORPO loss starts from a corrupted baseline.** The initial ORPO training loss of 1.19 (vs SFT loss of ~0.96) may partly reflect the quantization dequantization error rather than genuine preference ambiguity.

The observed ORPO reward accuracy of 100% with a modest loss reduction (1.19 → 0.69) is consistent with the model "learning" to exploit quantization artifacts rather than learning the intended preference signal.

**Fix:** Set `load_in_4bit=False` in `train_orpo.py` to match SFT precision. If VRAM is a constraint, document it explicitly and accept the tradeoff.

---

### C-4 — Verbose and Bible-for-Everything Preference Pairs Are Memorization Traps (`training/build_preference_data.py:170-287`)

Two of the seven rejection-pattern generators produce pairs from fixed, hard-coded prompts:

- `_build_verbose_pairs(n=70)`: Generates 70 pairs by randomly sampling from **5 fixed prompts** (`prompts_and_chosen` list, lines 173-202). With `n=70`, each of the 5 prompts appears an average of **14 times**. The `chosen` response is identical on every repetition — only the selection varies via `random.choice`.

- `_build_bible_for_everything_pairs(n=70)`: Generates 70 pairs from **7 fixed QA tuples** (lines 226-281). Each tuple repeats ~10 times.

Combined, these two categories contribute 140 out of ~500 preference pairs (28%) where the model sees the same question → same chosen response repeated 10-14 times. This teaches memorization of 12 specific responses, not a general "be concise" or "don't shoehorn Scripture" principle. Any held-out question that wasn't in these 12 prompts gets no coverage.

**Fix:** Expand `_build_verbose_pairs` to use diverse verse-drawn prompts (same pattern as other generators) with programmatically varied verbose tails. Expand `_build_bible_for_everything_pairs` with at least 30-40 unique factual QA pairs.

---

## High Findings (P1)

### H-1 — ORPO Training Has No Validation Split (`training/train_orpo.py:251-278`)

The ORPO dataset is loaded and immediately used for training with no held-out validation:

```python
dataset = load_dataset("json", data_files=str(pref_path), split="train")
# ...
trainer = ORPOTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=config)
```

With only ~500 pairs (28% of which are high-duplication — see C-4), there is no mechanism to detect preference overfitting. The training run reports only train loss. Given the repetitive data, the model could overfit the 12 memorized prompts while losing generalization — and there is no metric to catch it.

**Fix:** Add a `test_size=0.1` split before training. Pass `eval_dataset` to `ORPOTrainer`. Add `eval_steps=20` to `ORPOConfig`.

---

### H-2 — ORPO Warmup Misconfigured (`training/train_orpo.py:265`)

```python
config = ORPOConfig(
    ...
    warmup_steps=20,   # out of 63 total steps
    ...
)
```

With 63 total training steps (63 batches from ~500 pairs at batch size 2 × grad accum 4 = effective batch 8), 20 warmup steps = **32% of total training is warmup**. The learning rate never reaches its peak before the cosine decay begins. Standard practice is warmup = 5-10% of total steps = 3-6 steps for this run. Excess warmup wastes the limited preference signal.

**Fix:** Change `warmup_steps=20` to `warmup_steps=5` (approximately 8% of 63 steps).

---

### H-3 — W&B Project Hardcoded in `train_orpo.py` (`training/train_orpo.py:159,161`)

```python
wandb.init(project="bible-ai", name=run_name, mode="disabled")   # line 159
wandb.init(project="bible-ai", name=run_name, ...)                # line 161
```

`train_unsloth.py` was updated to use `os.getenv("WANDB_PROJECT", "bible-ai")` for the project name, but `train_orpo.py` still hardcodes `"bible-ai"`. Users who override the W&B project via environment variable will find ORPO runs land in the wrong project.

**Fix:** Replace both occurrences with `project=os.getenv("WANDB_PROJECT", "bible-ai")`.

---

### H-4 — LLM-as-Judge Truncates Response to 1000 Characters (`training/evaluate.py`)

The LLM-as-judge scoring constructs its prompt using `response[:1000]`, capping the judged content at the first thousand characters. For Bible verse answers, citations commonly appear after the explanatory text — which may place the `Book chapter:verse (WEB)` reference beyond the 1000-character boundary. This means:

- The judge scores citations as missing when they're present beyond the cutoff.
- The keyword-overlap metric scores the full response, making the two metrics non-comparable.
- The citation rate and hallucination rate in the README evaluation table may be measured against different response windows.

**Fix:** Increase truncation limit to at least 4000 characters, or remove it entirely. Document the truncation behavior in `BENCHMARK_PROTOCOL.md`.

---

### H-5 — Blocking `time.sleep(0.5)` in Evaluation Loop (`training/evaluate.py`)

Every LLM-as-judge call is followed by `time.sleep(0.5)`. For a 54-question benchmark this wastes 27+ seconds of wall-clock time per run. The sleep was likely added as a rate-limit workaround for a remote API, but the project uses local Ollama — which has no rate limit. If the sleep exists to avoid overloading Ollama, `asyncio`-based concurrency with a semaphore would be both faster and correct.

**Fix:** Remove `time.sleep(0.5)`. If Ollama stability requires throttling, use `asyncio.Semaphore` with async httpx calls.

---

### H-6 — Silent Zero-Score Fallback in LLM Judge (`training/evaluate.py`)

The LLM-as-judge function attempts three Ollama endpoints in sequence. If all three fail (e.g. Ollama is not running), it falls through silently and returns a score of `0` or equivalent without raising an exception or logging a warning. Benchmark runs with Ollama unavailable will produce plausible-looking metric tables where every judged score is 0 — without any indication that the judge never ran.

**Fix:** After exhausting all fallback endpoints, raise a `RuntimeError("LLM judge unavailable — all endpoints failed")` or at minimum emit a `logging.error` warning and include a `"judge_available": false` field in the output JSON.

---

### H-7 — No Actual Deployment Artifacts (`deployment/`)

The `deployment/` directory contains only markdown documentation files:
- `deployment/pc/` — `README.md`, `ollama_setup.md`, `voice_setup.md`, `generate_modelfile.py`
- `deployment/vps/` — `setup.md`
- `deployment/jetson/` — `deploy.md`

There are no Docker Compose files, systemd unit files, Nginx configs, environment templates, or health-check scripts. The architecture describes a two-phase deployment (PC dev + Jetson production) and a VPS path, but none of these are reproducible without manual steps. The gap between documentation and executable deployment is large.

**Minimum fix:** Add a `docker-compose.yml` for local PC development (RAG server + Ollama) so the stack can be started with a single command. Add a `deployment/pc/start.sh` script.

---

## Medium Findings (P2)

### M-1 — `docs/CODEBASE_AUDIT.md` Was Severely Outdated

The previous audit document (now replaced by this file) still claimed "No pyproject.toml", "No pytest", "No GitHub Actions" — all of which have been implemented. An outdated audit document is worse than no audit document: it misleads contributors about the project's actual state. *(This finding is self-resolving with this replacement.)*

---

### M-2 — `random.seed(42)` at Module Level (`training/build_preference_data.py:19`)

```python
import random
random.seed(42)       # ← module-level side effect
```

Seeding Python's `random` module at import time affects the global random state for any code running in the same process. If `build_preference_data` is imported as a library (not just run as a script), callers lose control of their own random state. This is a well-known anti-pattern.

**Fix:** Move `random.seed(42)` into `main()`, or use an isolated `random.Random(42)` instance and pass it to each generator.

---

### M-3 — MAX_SEQ_LENGTH Inconsistency Between SFT and ORPO

SFT (`train_unsloth.py`): `MAX_SEQ_LENGTH = 2048`
ORPO (`train_orpo.py`): `MAX_SEQ_LENGTH = 4096`, but `ORPOConfig(max_length=2048)`

The model is loaded with 4096 positional encoding capacity but trained on sequences capped at 2048. This means positional encodings for tokens 2049-4096 are initialized but never updated during ORPO training — a subtle capacity/training mismatch. Both stages should use the same `MAX_SEQ_LENGTH`.

**Fix:** Align to `MAX_SEQ_LENGTH = 2048` in both scripts, or deliberately expand to 4096 in both with a comment explaining the decision.

---

### M-4 — Hardcoded Default in `merge_adapters.py` (`training/merge_adapters.py:25`)

```python
DEFAULT_LORA_NAME = "qwen3.5-4b-bible-John-v8"
```

When a new model version (v9, v10) is trained, this default won't be updated. Users running `python training/merge_adapters.py` without `--lora-path` will silently merge the wrong adapter. The script has no warning that it's using a hardcoded default.

**Fix:** Change the default to `None` and require `--lora-path` explicitly, or print a prominent `WARNING: using default adapter path {lora_path}. Pass --lora-path to override.` before proceeding.

---

### M-5 — Hallucination Rate Counter-Intuitive Across Quantization (`README.md:80-81`)

The evaluation table shows:
- `SFT+ORPO (Q4_K_M)`: **20%** hallucination (11/54)
- `SFT+ORPO (F16)`: **26%** hallucination (14/54)

The quantized model hallucinates **less** than the full-precision model. This is counter-intuitive and unexplained. Possible causes: (a) Q4 quantization introduces response truncation that accidentally avoids hallucinating by being shorter, (b) the keyword-overlap "hallucination" metric counts different false positives at different precisions, (c) random variance at n=54 is high enough to flip the order. At n=54, the difference between 11 and 14 hits is within a 95% CI.

This result should be explained, not silently published. It either represents a real finding worth discussing or a metric artifact that should be corrected.

**Fix:** Add a paragraph in `docs/MODEL_COMPARISON.md` addressing this result. Consider running both models with a fixed seed and increasing n≥100 for more stable estimates.

---

### M-6 — Personal Documents Tracked in Repository

The following files contain personal/operational content that should not be in a professional public repository:

- `docs/interview_notes/INTERVIEW_PREP.md` — personal job interview preparation
- `docs/SERVING_GOD.md` — personal devotional content
- `CURSOR_CHECKLIST.md` — IDE-specific development checklist at repo root

These are visible to anyone cloning the repo and mix personal content with technical documentation. For a portfolio project shown to employers, this creates an unprofessional impression.

**Fix:** Add to `.gitignore` and remove from tracking, or move to a private branch / personal notes repo.

---

## Additional Observations

### O-1 — `trust_remote_code=True` is Intentional but Undocumented

Both `merge_adapters.py:118` and `train_orpo.py` load Qwen3.5-4B with `trust_remote_code=True`. This executes arbitrary Python code from the model's repository on HuggingFace. For Qwen3.5-4B from Alibaba's official repo, this is a known and necessary requirement. However, it is not documented as a conscious security decision anywhere in the codebase.

**Fix:** Add a comment at each `trust_remote_code=True` call: `# Qwen3.5-4B requires trust_remote_code for its custom architecture modules (gated DeltaNet etc.); only use with models from verified sources.`

---

### O-2 — Exception Handler Exposes Tracebacks in Production (`rag/rag_server.py`)

The global exception handler returns `repr(exc)` and the full traceback in the HTTP 500 JSON body. This is acceptable for local development but exposes internal implementation details (file paths, module names, stack frames) to any client in a networked deployment.

**Fix:** Gate on an `APP_ENV` environment variable: if `APP_ENV != "development"`, return only `{"detail": "Internal server error"}` and log the traceback server-side.

---

### O-3 — No Observability Infrastructure

The RAG pipeline touches five sequential stages: BM25 indexing, dense embedding, ChromaDB query, RRF merging, and cross-encoder reranking — each with distinct latency profiles. Currently there is no request-level timing, no structured logging with request IDs, and no metrics endpoint. Diagnosing why a particular query is slow requires adding `print` statements.

**Minimum fix:** Add a `logging.getLogger(__name__)` logger and emit structured log lines at each pipeline stage with elapsed time.

---

### O-4 — 17 Checkpoint `README.md` Files Pollute Git History

The `checkpoints/` directory contains 17 individual checkpoint folders each with its own `README.md` (HuggingFace auto-generated). These are tracked in git, inflating the repository and adding noise to `git log`. Binary model weights themselves are presumably gitignored, but the README stubs are not.

**Fix:** Add `checkpoints/*/README.md` to `.gitignore`. Consolidate checkpoint metadata into a single `checkpoints/README.md`.

---

## Test Coverage Analysis

| Module | Unit tests | Integration tests | Notes |
|--------|------------|-------------------|-------|
| `rag/rag_server.py` | ✅ 8 HTTP tests | ❌ No real ChromaDB | Body validation, streaming, 413, 422 covered |
| `rag/build_index.py` | ❌ | ❌ | No tests |
| `training/evaluate.py` | ✅ keyword scoring | ❌ No LLM judge test | Score calculation logic tested |
| `training/dataset_builder.py` | ❌ | ❌ | Format/output not tested |
| `training/build_preference_data.py` | ❌ | ❌ | Output counts/format not tested |
| `training/train_unsloth.py` | ❌ | ❌ | Import-only (torch unavailable in CI) |
| `training/train_orpo.py` | ❌ | ❌ | Same |
| `training/merge_adapters.py` | ❌ | ❌ | No key-remap tests |
| `scripts/*.py` | ✅ manifest YAML | ❌ | Manifest structure validated |

**Coverage gap:** `training/` coverage is reported by `--cov=training` in CI, but all training imports fail silently (no torch/unsloth in CI environment), so the actual measured training coverage is 0%. The `--cov-fail-under=70` threshold is met only because `rag/` coverage is high enough to carry the average.

**Fix:** Add `--cov-config` to exclude unreachable modules from the threshold calculation. Alternatively, add lightweight unit tests for pure-Python training utilities (`_remap_lora_state_dict`, `_build_hallucination_pairs`, `format_for_orpo`) that don't require torch.

---

## Priority Action Matrix

### P0 — Fix Before Public Share

| ID | Action | File | Effort |
|----|--------|------|--------|
| C-1 | Replace `requirements.txt` reference in README Quick Start | `README.md` | 15 min |
| C-2 | Move ML deps from `[project.dependencies]` to optional groups | `pyproject.toml` | 30 min |
| C-3 | Change `load_in_4bit=False` in ORPO to match SFT precision | `train_orpo.py:173` | 5 min |
| C-4 | Diversify `_build_verbose_pairs` and `_build_bible_for_everything_pairs` | `build_preference_data.py` | 2-3 hrs |

### P1 — Fix Before Next Training Run

| ID | Action | File | Effort |
|----|--------|------|--------|
| H-1 | Add validation split to ORPO training | `train_orpo.py` | 30 min |
| H-2 | Fix ORPO warmup: `warmup_steps=5` | `train_orpo.py:265` | 5 min |
| H-3 | `os.getenv("WANDB_PROJECT", "bible-ai")` in ORPO | `train_orpo.py:159,161` | 5 min |
| H-4 | Remove 1000-char truncation in LLM judge | `evaluate.py` | 10 min |
| H-5 | Remove `time.sleep(0.5)` | `evaluate.py` | 5 min |
| H-6 | Raise error when all judge endpoints fail | `evaluate.py` | 15 min |

### P2 — Quality Polish

| ID | Action | File | Effort |
|----|--------|------|--------|
| H-7 | Add `docker-compose.yml` for local dev | `deployment/` | 1 hr |
| M-2 | Move seed into `main()` | `build_preference_data.py:19` | 5 min |
| M-3 | Align MAX_SEQ_LENGTH between SFT and ORPO | `train_orpo.py:34` | 5 min |
| M-4 | Warn when using default adapter path in merge script | `merge_adapters.py:44` | 10 min |
| M-5 | Explain counter-intuitive hallucination metric result | `docs/MODEL_COMPARISON.md` | 20 min |
| M-6 | Remove personal documents from public repo | `docs/`, root | 10 min |
| O-2 | Gate traceback exposure on `APP_ENV` | `rag/rag_server.py` | 15 min |
| O-3 | Add structured logging at RAG pipeline stages | `rag/rag_server.py` | 30 min |
| O-4 | Gitignore checkpoint README stubs | `.gitignore` | 5 min |

---

## Strengths (Do Not Break)

These aspects of the codebase are genuinely strong and should be preserved:

- **Constitutional AI implementation** — behavioral guardrails embedded at three independent layers (system prompt, training data, post-processing) is the right architecture for safety-sensitive applications.
- **Hybrid RAG pipeline** — dense + BM25 + RRF + cross-encoder is state-of-the-art retrieval for a domain-specific corpus and is correctly implemented.
- **Thread-safe lazy loading** — double-checked locking on all three globals (`_rag_lock`, `_bm25_lock`, `_reranker_lock`) correctly handles concurrent requests without race conditions.
- **True async streaming** — branching on `think_enabled` to proxy chunks directly vs. buffering only when stripping `<think>` tags is the correct latency/correctness tradeoff.
- **Versioned benchmark protocol** — `benchmarks/manifest.v1.yaml` with schema validation tests is solid MLOps practice for reproducible evaluation.
- **Overfitting diagnosis and fix** — the architecture doc's honest account of 31K→1.8K dataset reduction with reasoning is valuable institutional knowledge.
- **Two-stage training motivation** — the SFT-only incoherence → ORPO recovery narrative is a legitimate and interesting ML finding worth highlighting.
- **Blackwell xformers workaround** — correctly handles sm_120 capability detection in both SFT and ORPO scripts, with graceful fallback.

---

*This document supersedes the previous `CODEBASE_AUDIT.md`. Re-audit recommended after P0/P1 fixes are applied.*
