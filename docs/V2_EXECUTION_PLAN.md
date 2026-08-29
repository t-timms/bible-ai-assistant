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
- `data/raw_v2/mhc_commentary.json` (4,244 passages, CC0) — `python training/fetch_mhc_commentary.py` (once).
- `data/processed/train_v2.json` — `python training/build_dataset_v2.py` (needs HF auth for the blend).
- ChromaDB index `rag/chroma_db/` (verses 31,103 / passages 10,771 / bm25) — `CUDA_VISIBLE_DEVICES="" build-index` (~1 h on CPU).
- Ollama 0.33.1 in WSL (`127.0.0.1:11434`), no model registered yet.

**Base model — decision 2026-08-28: smallest-that-wins, 4B first.** This is a RAG-faithfulness
task, and 2026 research is consistent that citation grounding + calibrated abstention are
learnable in small models. Train **`Qwen/Qwen3.5-4B` bf16 LoRA** (fully unquantized — ~9.3 GB
weights, ~10–13 GB peak train, F16 GGUF serves in ~8 GB), run the full v3 + FMG-Bench /
FaithBench set, and **escalate to 9B QLoRA only on a measured shortfall** on a metric that
matters (theological reasoning is the likely gap). The 9B probe stays validated as the fallback.
`config.v2.yaml` is the 27B stretch, only if 9B clears convincingly. The Qwen3.5 dense line is
**0.8 / 2 / 4 / 9 / 27B** (no 8B or 14B); Gemma 4 12B noted as a documented A/B alternative but
not chosen (Gemma license vs. the project's Apache-2.0 throughline; Qwen wins on knowledge/
factual tasks per Artificial Analysis).

**`train_unsloth.py`** has been made compatible with the pinned trl 0.24 / unsloth 2026.8 stack
(`max_length` not `max_seq_length`; `processing_class` not `tokenizer`; unwrap the Qwen3VLProcessor;
import SFTConfig/SFTTrainer *after* the model loads; explicit `eos_token="<|im_end|>"`).
It now takes `--config` / `--data` / `--max-steps`.

**The 9B probe ran and is verified working** — ~25 s/opt-step, VRAM ~12–15.5 / 16 GB (fits),
loss `0.22 → ~0.04` by step ~60. It was cancelled at step ~165/250 to free the GPU; partial
LoRA checkpoints are on disk at `checkpoints_v2_9b/checkpoint-{41,82,123,164}` (gitignored).
`train_unsloth.py` has no `--resume-from`, so the resume is a fresh 250-step run.

### Dataset "full upgrade" — DONE 2026-08-28 (branch `v2/dataset-full-upgrade-2026-08-28`)

The 99.9 %-verse-recall / 0.1 %-general mix that triggered the reasoning-collapse concern is
fixed. `train_v2.json` rebuilt: **56,022 examples**, contamination-clean (both
`scripts/check_train_eval_overlap.py` and a direct check vs. all 282 v3 suite questions →
0 overlap). All 439 tests pass; `ruff`/`mypy` clean.

| bucket | count | share |
|---|---|---|
| 8 scripture-citation categories (capped from ~61k) | 35,604 | 63.6 % |
| `grounded_exegesis` — Matthew Henry (CC0) commentary in context → grounded interpretation | 7,000 | 12.5 % |
| `general_blend` — HuggingFaceTB/smoltalk2 (Apache-2.0), 11 SFT splits, ~40/60 think/no-think, `<think>` stripped | 12,996 | 23.2 % |
| `pastoral_triage` — escalation + tradition-aware + calibrated abstention (FMG-Bench rubric dims, not its scenarios) | 352 | 0.6 % |
| inherited v1 general/meta/refusals | 70 | 0.1 % |

**Strict general/reasoning share = 23.9 %** (blend + triage + inherited) — clears Unsloth's
≥20–25 % catastrophic-forgetting floor; 36 % if `grounded_exegesis` is counted as reasoning.
Also added: probabilistic real-user framing prefixes on every generator, `_TOPICS` 10 → 22,
2–3× larger phrasing pools. Full provenance (per-source SHA + license, per-split kept counts)
in `train_v2.manifest.json`.

Known minor: two greeting-heavy smoltalk2 splits (`*everyday_conversations*`) yielded ~0 —
the first-user-text dedup collapses their formulaic openers against `smol_magpie_ultra`, which
runs first. Acceptable (a Bible assistant doesn't need 500 "hi"/"hello" pairs; the inherited
meta pool covers greetings). **Open dependency**: `grounded_exegesis` puts commentary in the
context block, so `rag_server.py` needs a commentary-retrieval path before a model trained on
it is served — otherwise it's the F-2/F-3 train/inference format mismatch again.

### v2-4b SFT + protocol-v3 eval — DONE 2026-08-29

**Trained**: `Qwen/Qwen3.5-4B` bf16 LoRA (`training/config.v2-4b.yaml`, r=32/α=64, seq 1280
fixed-pad, effective batch 16, lr 2e-4, **1 epoch** = 3,474 steps, ~10.4 h). eval_loss
0.2515 → **0.2138**, monotonic over all 70 evals, no overfit. Adapter
`models/qwen3.5-4b-bible-v2-sft`; merged bf16 `models/qwen3.5-4b-bible-v2-merged` (8.4 GB,
vision tower dropped). `train_unsloth.py` gained a length filter (drop examples whose
assistant turn doesn't survive truncation — 10/56k at 1280) and reverted to fixed
`padding="max_length"` after dynamic padding fragmented the CUDA allocator near the 16 GB
ceiling (step time climbed 6.5 s → 116 s; first launch aborted at step 57).

**Serving reality**:
- **GGUF WORKS** (corrected — an earlier note here wrongly said "blocked"). The
  `~/wsl41361/llama.cpp` used at first is a 307-line *stub* (Blackwell CUDA-hang repro
  harness), not real llama.cpp — its naive converter produced malformed GGUFs. With a
  fresh `llama.cpp` master (`~/llama.cpp-full`, commit `3173a56`) the fix is
  `convert_hf_to_gguf.py --no-mtp`: the base config's `mtp_num_hidden_layers: 1` otherwise
  makes the converter set `block_count = 33` and expect an MTP-head tensor
  (`blk.32.attn_norm.weight`) this SFT doesn't carry. F16 + Q8_0/Q6_K/Q5_K_M/Q4_K_M all
  built and verified with `llama-server` (~85 tok/s, coherent). Ollama 0.33.x's *bundled*
  llama.cpp is still too old for the `qwen35` arch — GGUFs staged in `models/gguf2/` with a
  card for a `-GGUF` repo. `qwen3_5_moe` (Ornith) GGUF also works upstream
  (`unsloth/Qwen3.5-35B-A3B-GGUF` exists); the MTP-strip is what trips the MoE converter path.
- **vLLM**: `Qwen3_5ForCausalLM` class exists but was unregistered (added the `registry.py`
  line); then hit `RuntimeError: UVA is not available` — the 0.26.0 `GPUModelRunnerV2`
  UVA-buffer path is broken under WSL2.
- **Eval workaround** used for the protocol-v3 run: `scripts/_tf_openai_server.py` (stdlib
  `http.server` OpenAI-compat wrapper over the merged HF model) behind the RAG server;
  `scripts/_run_v3_eval.sh` orchestrates it.

**Result** (`docs/benchmark_runs/20260829_v2-4b_keyword.json`, protocol v3, keyword/no-judge):

| | v1 (Qwen3-4B, ~1.8k SFT+500 ORPO) | v2-4b (Qwen3.5-4B, 56k SFT) | Δ |
|---|---|---|---|
| verse_lookup exact | 58 % | **76.5 %** | +18.5 pp |
| overall citation rate | 88 % | **98.9 %** | +11 pp |
| overall verse acc | 22 % | 29.3 % | +7.3 pp |
| overall hallucination | 2.0 % | 2.3 % | flat |
| overall fuzzy mean | **0.483** | 0.396 | **−0.087** |

**Diagnosis** (controlled A/B, `docs/MODEL_COMPARISON.md`): v2 is much better at the core
RAG task (verbatim recall, citation) and learned the new pastoral/exegesis/cross-ref
behaviours, but the eight scripture categories' rigid fill-in-the-blank **answer** templates
made it worse at open-ended thematic questions than the lightly-tuned v1. The dataset upgrade
diversified questions, not answers. The smoltalk2 forgetting-guard (24 %) slowed but didn't
stop the format overfit against 35k templated answers in 1 epoch.

**Note**: the shipped v1 adapter `Ttimms/bible-ai-qwen3.5-4b-lora` is mislabeled — its
`adapter_config.json` says `base_model_class: Qwen3ForCausalLM` (plain **Qwen3-4B**, r=16).
Merged onto `Qwen/Qwen3-4B` → `models/qwen3-4b-bible-v1-merged` for the baseline run.

### v3 plan (the SOTA push) — NOT STARTED

1. **Judge re-score** v2 + v1 (fair scoring on the synthesis categories) — pending a judge
   model (`qwen3:8b` pulled).
2. **Dataset v3 = teacher distillation**: regenerate answers for all categories with a strong
   model — natural, grounded, non-templated; cut recall-drill volume; keep the
   provenance-clean sources. This is the bottleneck fix. Teacher + scope TBD by the user
   (Claude API vs local 27–32B; ~18–50k).
3. **SFT on v3** (4B first), then **GRPO** (`training/train_grpo.py` scaffold exists) with the
   verifiable citation reward — the stage that should clear the ≥85 % verse-accuracy bar.
4. Re-eval + FMG-Bench / FaithBench; escalate to 9B only on a measured shortfall.
5. Retrieval upgrade (embedder stronger than `nomic-embed-text-v1.5`) + constrained
   verse-reference decoding.

Also open: `rag_server.py` commentary-retrieval path (so `grounded_exegesis` training matches
inference); GGUF publish once llama.cpp adds Qwen3.5-hybrid support.
