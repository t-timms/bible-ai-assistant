# V2 Execution Plan

**Written:** 2026-08-28. Supersedes the scattered "next steps" in `ROADMAP.md` Block 0,
`PROJECT_STATUS_AND_GOALS.md` § V2, and `OPTIMIZATION_PLAN.md` — those stay as reference;
this is the ordered, audited path to a model that clears its own bar.

---

## Where the model stands

- Shipped model: v1 Qwen3.5-4B, SFT (~1.8k) + ORPO (500). Last measured (v1 protocol,
  historical): **verse accuracy 5.6–9.3%, hallucination 20–26%**. The project's own pass
  criteria (`docs/evaluation_results.md`): **≥85% verse accuracy, zero fabricated verses.**
  It fails its own bar and has **not** been re-scored under protocol v3.
- The RAG / eval / CI stack is strong and stays. The **model layer** is what's rebuilt.

## Gaps found in the pre-2026-08-28 plan

| Gap | Status now |
|-----|-----------|
| No runnable environment (no durable clone, conda env, corpus, or index) | Being stood up: WSL `~/bible-ai-assistant`, conda `bible-orpo` (torch 2.11+cu128, transformers 5.5, trl 0.24, unsloth 2026.8.22 — `sm_120` confirmed in torch arch list). |
| `config.v2.yaml` read by no code | **Fixed** — `train_unsloth.py --config/--data`; `_load_config_yaml` now takes a path and reads `data.train_file`. New `config.v2-9b.yaml`. |
| GRPO (Stage 3) had zero implementation | **Scaffolded** — `training/train_grpo.py` (Unsloth GRPO + verifiable reward from `rag/verification.py`). `--dry-run` verified (good→1.00, bad→0.075). Needs a GPU `--max-steps 2` smoke. |
| `train_v2.json` never built | Track A step 2 below (no GPU). |
| `BENCHMARK_PROTOCOL.md` stale (said "v2", mutable suite) | **Fixed** — now v3, frozen sha-pinned suites. |
| Benchmark contamination (~100/282 in the 2026-08-24 dataset) | Non-issue going forward: `dataset_builder.py` + `build_dataset_v2.py` decontaminate against `benchmarks/suites/*.json` by default; `check_train_eval_overlap.py` verifies in CI. The 2026-08-24 dataset was ephemeral and is gone — regenerate fresh. Do **not** cite any pre-v3 topical/character/cross-ref/verse-lookup number. |
| "No external Bible benchmark exists" (checked 2026-08-24) | **Outdated.** See Research below. |

## Research check — is the strategy SOTA? (2026-08-28)

Yes, with three adjustments. Sources in the PR description; key points:

1. **Pipeline** — SFT (QLoRA/Unsloth) → preference (ORPO) → **GRPO when the reward is
   verifiable** is the 2026-standard flow. This project's reward *is* fully verifiable
   (verse exists? quote matches?), so the RL stage is well-motivated.
2. **External benchmarks now exist** — [FMG-Bench](https://github.com/FideAI/fmg-bench)
   (120 scenarios, code, arXiv 2608.12324) and [FaithBench](https://faithbench.com/) (300+,
   held-out, version-controlled). Wire them in as **honest external calibration**, not win
   targets — they test theological *reasoning* / tradition-awareness, a harder and
   different task than RAG verse-citation. Expect modest scores; report them anyway.
3. **Guard against reasoning collapse** — Unsloth's own guidance: keep ≥20–25%
   general/reasoning/refusal data or Qwen3 loses reasoning. The v2 corpus is verse-recall
   heavy. Verify the mix ratio in `train_v2.manifest.json` before launching; lean on GRPO
   (not SFT) for the citation-shaped behaviour. "5k curated ≥ 50k noisy" is the repeated
   2026 refrain.
4. **GRPO variants** — DAPO (more stable), GSPO (Qwen-team, sequence-level), DUPO (faster).
   Baseline: GRPO via Unsloth; DAPO documented as the fallback if GRPO is unstable.
5. **FP8 GRPO** now runs on consumer GPUs; Blackwell sm_120 has native FP8. 9B QLoRA +
   FP8 GRPO on the 5070 Ti is realistic. **9B-first** stays right; 27B QLoRA is very tight on 16 GB
   per Unsloth (batch=1 + `gradient_checkpointing="unsloth"`) but is a stretch.
6. **Constrained decoding** (OUTLINES / grammar-FSM) is the standard "verse-reference
   trie" tool — apply it **narrowly** (citation span only); there's a documented
   "alignment tax" on reasoning (arXiv 2604.06066).
7. **RAG > model scaling for citation accuracy** — the highest-leverage work is
   GRPO-for-faithfulness + constrained citations, not parameter count.

## Track A — prep, no training GPU

| # | Task | Command / artifact | GPU |
|---|------|--------------------|-----|
| A1 | Durable clone + `bible-orpo` env | WSL `~/bible-ai-assistant`; `docs/ORPO_TWO_ENV_SETUP.md` recipe | none |
| A2 | Fetch `data/raw/` corpus + build ChromaDB index | `build-index` (force CPU while GPU is busy) | none |
| A3 | Build the SFT set | `python training/build_dataset_v2.py` → `data/processed/train_v2.json` + manifest | none |
| A4 | Contamination check | `python scripts/check_train_eval_overlap.py` (needs `data/raw/`) | none |
| A5 | Config wiring (done) + `config.v2-9b.yaml` (done) | — | none |
| A6 | `train_grpo.py` scaffold (done) | `--dry-run` passes | none |
| A7 | Doc truth-up (done) | `BENCHMARK_PROTOCOL.md`, this file, `ROADMAP.md` Block 0 | none |
| A8 | Install Ollama (WSL) + prepare eval stack | RAG server + Ollama + index | eval only |
| A9 | **v1 baseline under protocol v3** | `python scripts/run_benchmark.py --ollama-model bible-assistant-orpo` (+`--judge`), Q4 + F16 | light (inference) |
| A10 | Overnight runner | `scripts/overnight_v2.sh` — smoke-gate → full 9B SFT → merge → GGUF | queued |

## Track B — GPU sessions (gated: gaming / other jobs)

| # | Task | Gate to proceed |
|---|------|-----------------|
| B1 | 9B SFT (`--config training/config.v2-9b.yaml`) → merge → GGUF Q4+F16 → Ollama → eval v3 | clears the v1 baseline on verse accuracy **and** hallucination |
| B2 | ORPO on the 9B SFT (regenerate 2,080 pref pairs) → eval v3 A/B | ORPO improves judge scores without hurting verse accuracy |
| B3 | GRPO smoke (`--max-steps 2`) → full GRPO on 9B-ORPO → eval v3 | reward curve rises; no KL blowup; eval improves |
| B4 | Constrained-citation decoding (OUTLINES, citation span only) → eval v3 | hallucination ↓ with < 2 pt helpfulness cost |
| B5 | 27B repeat of B1-B3 | only if 9B clears convincingly |
| B6 | Publish | HF model card + GGUF, `scripts/leaderboard.py` table, FMG-Bench + FaithBench entries, README/MODEL_CARD updated with **measured** v3 numbers |

## Non-negotiables

- Eval/benchmark data is never training data (enforced; verified by `check_train_eval_overlap.py`).
- Every reported number states its `protocol_id`; v1/v2/v3 are not comparable.
- Measured vs. estimated is always explicit. No number ships without an artifact behind it.
- GPU work is gated around gaming / other GPU jobs — no contention.


---

## Where this stands (updated 2026-08-28)

**Environment** — a durable local setup exists and is not in git (rebuild if lost):
- WSL clone `~/bible-ai-assistant`; conda env `bible-orpo` (torch 2.11+cu128 sm_120, trl 0.24,
  unsloth 2026.8.22, transformers 5.5); venv `~/bible-ai-assistant/.venv-rag` (`.[rag,dev]`).
- `data/raw/bible_web.json` (31,103 verses) — `python training/convert_web_tehshrike.py ~/world-english-bible`.
- `data/processed/train_v2.json` (61,556 examples, contamination-clean) — `python training/build_dataset_v2.py`.
- ChromaDB index `rag/chroma_db/` (verses 31,103 / passages 10,771 / bm25) — `CUDA_VISIBLE_DEVICES="" build-index` (~1 h on CPU).
- Ollama 0.33.1 in WSL (`127.0.0.1:11434`), no model registered yet.

**Base model** — the Qwen3.5 dense line is **0.8 / 2 / 4 / 9 / 27B** (no 8B or 14B).
Near-term recipe: `training/config.v2-9b.yaml` (`Qwen/Qwen3.5-9B`, QLoRA 4-bit — 9B has no
bf16-LoRA path on 16 GB). `config.v2.yaml` is the 27B stretch.

**`train_unsloth.py`** has been made compatible with the pinned trl 0.24 / unsloth 2026.8 stack
(`max_length` not `max_seq_length`; `processing_class` not `tokenizer`; unwrap the Qwen3VLProcessor;
import SFTConfig/SFTTrainer *after* the model loads; explicit `eos_token="<|im_end|>"`).
It now takes `--config` / `--data` / `--max-steps`.

**The 9B probe ran and is verified working** — ~25 s/opt-step, VRAM ~12–15.5 / 16 GB (fits),
loss `0.22 → ~0.04` by step ~60. It was cancelled at step ~165/250 to free the GPU; partial
LoRA checkpoints are on disk at `checkpoints_v2_9b/checkpoint-{41,82,123,164}` (gitignored).
`train_unsloth.py` has no `--resume-from`, so the resume is a fresh 250-step run.

**Resume**: `tmux new-session -d -s v2 'scripts/overnight_v2.sh'` (GPU-gated; ~2.5 h incl.
merge + GGUF F16). Then the pending item is **A9 — the protocol-v3 baseline of the shipped v1
model** (merge `Ttimms/bible-ai-qwen3.5-4b-lora` → GGUF → `ollama create` → `run_benchmark.py`),
which is still the honest scoreboard's unmeasured zero point. Compare probe vs. baseline to
decide whether `train_v2.json` needs a general/reasoning-data blend before a full SFT
(current mix is 99.9 % verse-recall / 0.1 % general — early probe loss suggests trivial
memorisation, so reasoning retention is the open risk).
