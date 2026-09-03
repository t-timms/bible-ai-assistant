# V3 status — resume here (2026-09-01)

Plan: `docs/V3_DATASET_PLAN.md`. This file = exact state + the next action.

---

## ►►►►► RUN OVERNIGHT (2026-09-03) — v3.1 pipeline is wired; fire it

The v3.1 dataset prep is done: `training/build_v3_thematic.py` +
`training/v3_thematic_questions.json` (103 stems) generated
**`data/raw_v3/thematic_inputs.jsonl` (2,395 rows)**. The one-command end-to-end
pipeline is `scripts/_run_v3.1_pipeline.sh` — distill (Qwen3-14B teacher) →
assemble (`train_v3.1.json`) → SFT (`config.v3.1-4b.yaml`, ~7 h) → merge →
coherence-gate → protocol-v4 eval → prints the ship/hold verdict. All outputs use
`*v3.1*` names; **no v3 artifact is overwritten**.

**Before firing:** reboot Windows (box up 2+ days; `.wslconfig memory=64GB` is the
mitigation but a 7 h SFT is the load that froze the VM before).

```bash
cd ~/bible-ai-assistant
git pull                                            # if #50/#51 merged
SMOKE=1 bash scripts/_run_v3.1_pipeline.sh           # ~10 min — validates stages 1-3, no SFT
tmux new-session -d -s v31 'bash ~/bible-ai-assistant/scripts/_run_v3.1_pipeline.sh'
tail -f ~/bible-ai-assistant/logs/v3.1_pipeline_*.log
```

**Morning check:** `grep -E 'EXIT_|VERDICT|GATE ' logs/v3.1_pipeline_*.log`.
`EXIT_0` = clean; `EXIT_1x` = the stage that failed (10 preflight · 11 teacher ·
12 distill · 13 assemble · 14 SFT · 15 merge · 16 coherence · 17 eval). The
VERDICT line says SHIP or HOLD; per-category fuzzy means + the two gates
(overall expo-excl ≥ 0.52, each synthesis category ≥ 0.50) print above it.

**If SHIP:** CPU-only — `convert_hf_to_gguf.py --no-mtp` → `llama-quantize` ladder
→ publish `Ttimms/Bible-Assistant-Qwen3.5-4B-v3.1` + `-v3.1-GGUF` (HF push needs
the owner's token) → cards + README/MODEL_CARD/MODEL_COMPARISON → external SOTA
board once (`scripts/run_external_baselines.sh`).
**If HOLD:** read the per-category table — which synthesis categories still lag,
and whether it's a data-coverage gap (add stems to `v3_thematic_questions.json`)
or a teacher-quality gap (inspect `thematic_out.jsonl` `status:dropped` rows).

---

## ►►►► RE-EVAL DONE (2026-09-03) — v3-SFT misses 0.52 through the fixed RAG; HOLD, v3.1 retargeted

Ran `scripts/_run_v3_eval_all.sh` on all three checkpoints through the **#46-fixed** RAG
stack. Merged models were gone (disk pressure) → rebuilt fresh from
`models/qwen3.5-4b-bible-{v2,v3}-sft/adapter_model.safetensors` via `training/merge_adapters.py`
(`--output …-{v2,v3}-merged`), both coherence-checked (verbatim John 3:16 recall, no LoRA-skip).
Artifacts: `docs/benchmark_runs/20260903_{v2-4b,v3-sft,v3-grpo}_keyword.json` + `…_v4keyword.json`.

| metric (protocol v4, **post-#46**) | v2-4b | **v3-sft** | v3-grpo | bar | vs pre-#46 (09-01) |
|---|--:|--:|--:|--:|--|
| verse_quote exact (n=66, real recall) | 78.8% | **77.3%** | 77.3% | ≥74% ✓ | 77.3 → 77.3 (held) |
| verse_quote McNemar vs v2 | — | p=0.50 (v3 +2 / v2 +0) | — | held | — |
| verse_exposition fuzzy mean (n=36) | 0.509 | **0.542** | 0.539 | — | **0.418 → 0.542** (#46 worked) |
| overall fuzzy mean, expo-excl (n=230) | 0.394 | **0.497** | 0.496 | ≥0.52 ✗ (−0.023) | 0.499 → 0.497 (flat) |
| overall fuzzy mean, all-in (n=266) | 0.409 | **0.503** | 0.501 | ≥0.52 ✗ (−0.017) | 0.488 → 0.503 (up) |
| hallucination_detected (corpus mode) | 9/282 | **4/282** | 5/282 | — | Gen 19:28 **now clean** |

**Decision gate = "clears 0.52 (expo-excl) AND the Genesis-19:28-class hallucination is gone."**
- Hallucination half — **MET.** Gen 19:28 is clean in all three runs (was a confident
  fabrication pre-#46); total flags down to 4 (fewest of the three, half of v2's). Of v3-sft's
  4: one real misquote (Song of Solomon 2:13 → the model returns 1:13's text); the other three
  read as corpus-checker false positives on exposition-style answers.
- 0.52 half — **NOT MET.** 0.497 expo-excl, short by 0.023, flat vs the 0.499 pre-#46.

**→ HOLD. Do not ship v3-SFT.** This reverses the "ship" recommendation in the RESULT
block below (that call was made *before* this re-eval).

**Where the gap is — v3.1 must retarget.** v3-sft per-category fuzzy mean: `verse_lookup 0.707`
carries the average; **every synthesis category is stuck at ~0.31–0.41** — cross_reference 0.396,
context 0.374, topical 0.380, character 0.365, theological_reliability 0.311. That ~163-question
block at ~0.37 is what pins the overall at 0.497. v3-SFT already ~doubled these vs v2-4b, but
they plateau well short of the ~0.52 that clears the bar. **Exposition is already fixed by #46**,
so the on-file v3.1 plan (quote-first exposition templates + hallucination hardening) aims at
the wrong target. **v3.1's real job: push the synthesis categories 0.37 → ~0.52** with
teacher-distilled *explanatory* answers for character / context / cross_reference / topical /
theological questions that match the reference style. ROADMAP item 7 rewritten.

**NEXT:** v3.1 dataset work — thematic-synthesis distillation (`training/distill_answers.py`
teacher path, targeting the five synthesis categories), then re-SFT → re-eval protocol v4 →
SOTA board once. Publish nothing until v3.1.

---

## ►►► RESULT (2026-09-03, earlier) — SUPERSEDED by the RE-EVAL DONE block above

*(The "ship v3-SFT as v3" recommendation here was made from the pre-#46 rescore. The
2026-09-03 re-eval through the fixed RAG stack — block above — shows v3-SFT still at 0.497
expo-excl, so the decision is HOLD, not ship. Kept for the `verse_lookup`-artifact analysis,
which still stands.)*

Ran the Path-D tooling (no GPU, no model re-run): `scripts/rescore_v4.py`,
`scripts/exposition_sidebyside.py`, `scripts/sota_scoreboard.py`. Artifacts:
`docs/benchmark_runs/20260902_{v2-4b,v3-sft,v3-grpo}_v4keyword.json`,
`docs/benchmark_runs/20260902_exposition_v2_vs_v3.md`, `docs/SOTA_EVAL.md`.

| metric (protocol v4) | v2-4b | v3-sft | v3-grpo | bar |
|---|--:|--:|--:|--:|
| **verse_quote** exact (n=66, the real recall metric) | 78.8% | **77.3%** | 77.3% | ≥74% ✓ |
| verse_quote vs v2: McNemar | — | p=0.50 (v3 +2 / v2 +0) | — | **held** |
| verse_exposition fuzzy mean (n=36) | 0.427 | 0.418 | 0.418 | — (tie) |
| overall fuzzy mean, exposition-excluded (n=230) | 0.391 | **0.499** | 0.498 | ≥0.52 ✗ (−0.021) |
| overall fuzzy mean, all-in (n=266) | 0.396 | 0.488 | 0.487 | ≥0.52 ✗ |
| citation rate | 98.9% | 97.7% | 98.1% | ≥97% ✓ |
| hallucination rate | 2.3% | 1.5% | 1.9% | ≤2.5% ✓ |

**The `verse_lookup` "regression" was an eval artifact — confirmed.** Splitting
`verse_lookup` into `verse_quote` / `verse_exposition` shows quote recall held
(77.3% vs 78.8%, not significant); the "50%" came entirely from 26/36
exposition-phrased questions where v2 "passed" exact-match by dumping the verbatim
verse and v3 answers with a prose explanation instead.

**Manual read of all 36 `verse_exposition` items** (`20260902_exposition_v2_vs_v3.md`):
v3-SFT is better-or-tie on **34/36**. v2's answers are verbatim-quote + a
"for comparison, here are passages on…" list that is almost always
**verse-number-coincidence matches, not thematic** (e.g. "1 Chronicles 9:17" →
"2 Chronicles 17:9 / 1 Chronicles 17:17"). v3's are real, accurate explanations.
Two real v3 issues: (1) **item 6, Genesis 19:28** — v3 hallucinated a "God
protects Lot" reading (the verse is Abraham seeing Sodom's smoke); one confident
factual error out of 36. (2) items 12, 30 — v3 honestly says "not in the provided
context" where v2 emitted number-matched junk (v3 wins on faithfulness).

**Surfaced (higher-leverage than model work): the RAG retriever returns
verse-*reference*-token matches for exposition questions instead of thematic
neighbours** — hits v2 AND v3, feeds both bad context. `rag/retrieval.py`, GPU-free
to investigate. This is the single biggest lever on the exposition category.

**GRPO still inert** — v3-grpo == v3-sft to 3 d.p. on every metric.

### Recommendation — ship v3-SFT as v3

- verse_quote recall **held** (the headline risk is disproven)
- the actual v2 regression — synthesis categories (character/context/topical) — is
  fixed: their fuzzy means are ~1.8× v2 (from the 2026-09-01 protocol-v3 run)
- exposition answers are genuinely better (34/36 manual read); citation + hallucination hold
- overall fuzzy expo-excl **0.499 beats v1 (0.48) and v2 (0.40)**; it misses the round
  0.52 target by 0.021, and `manifest.v4` itself notes the fuzzy mean "is NOT an accuracy"

**Ship steps (all CPU — no GPU):** `merge_adapters.py` (adapter →
`models/qwen3.5-4b-bible-v3-merged`, already built) → `convert_hf_to_gguf.py --no-mtp`
→ `llama-quantize` ladder (Q4_K_M / Q5_K_M / Q6_K / Q8_0 + imatrix) → publish
`Ttimms/Bible-Assistant-Qwen3.5-4B-v3` + `-v3-GGUF` (needs HF_TOKEN — user runs the
push) → add the `## Architecture` mermaid to the new cards → update README /
MODEL_CARD / MODEL_COMPARISON to v3.

**Then (GPU, ~3–4 h):** `run_external_baselines.sh` + `sota_scoreboard.py` — fills
in `docs/SOTA_EVAL.md`'s 8 pending comparators for the "best open at the task" claim.

**If instead going Path B (v3.1 retrain):** add quote-first exposition templates to
the verse-drill generators AND fix the `rag/retrieval.py` reference-token matching
first — the retrieval issue caps how much a retrain can help the exposition category.

---

## ►► UPDATE (2026-09-02) — judge is dead, Path D tooling + SOTA track staged

**The judge eval is abandoned.** Both attempts (2026-09-01 18:04 and 2026-09-02
05:54) died on question 1: `qwen3.5:27b` (Q4_K_M, ~17 GB) does not fit the 16 GB
VRAM budget, CPU-offloads, and one rubric call measured **333.7 s** on 2026-09-02
(idle GPU) — past `evaluate.py`'s 180 s timeout. Do not retry the 27B judge on
this box. `benchmarks/manifest.v4.yaml` records this; v4 judge (if ever) = `qwen3:8b`,
calibration-only.

**Path D (recommended) — decide from keyword + a manual read, no judge, no GPU.**
All tooling is staged and syntax-checked (nothing run yet — waiting on you):

| file | what |
|---|---|
| `benchmarks/suites/evaluation_questions.v3.json` | frozen v4 suite; `verse_lookup` 102 → `verse_quote` 66 + `verse_exposition` 36 (rule: ends "teach?"/"about?"). sha `f6640605…` |
| `benchmarks/manifest.v4.yaml` | protocol v4; exposition scored by fuzzy not exact; overall fuzzy reported all-in **and** exposition-excluded |
| `scripts/make_v4_suite.py` | the inert suite generator (already run) |
| `scripts/rescore_v4.py` | re-bucket the 3 `20260901_*_keyword.json` under v4, re-aggregate via `evaluate.py`'s own formulas → `20260902_*_v4keyword.json` + comparison table |
| `scripts/exposition_sidebyside.py` | the 36 exposition items, v2 vs v3-sft, → `docs/benchmark_runs/20260902_exposition_v2_vs_v3.md` for the manual read |

Run on return: `python scripts/rescore_v4.py` then `python scripts/exposition_sidebyside.py`.
Decision: verse_lookup "regression" should dissolve (quote recall shown holding,
exposition off exact-match). Open question stays the **overall fuzzy bar 0.52** —
v3-SFT all-in is 0.488; exposition-excluded should clear. If the side-by-side read
shows v3's explanations are accurate and better than v2's raw dumps → ship v3-SFT
as v3. Else → Path B retrain.

**SOTA track (new, 2026-09-02) — `docs/SOTA_EVAL.md`.** Establishes the
"best *open* model at RAG-grounded scripture Q&A, size-independent" claim by
measurement. Staged, GPU sweep not run:

| file | what |
|---|---|
| `benchmarks/external_comparators.yaml` | 8 comparators: `sleepdeprived3` Christian-Bible-Expert 8B/12B, `nbeerbower/llama-3-bible-dpo-8B`, `Phora68/bible-study-phi3-mini`, `rhemabible/BibleAI`, Qwen3-8B/14B/32B instruct |
| `scripts/run_external_baselines.sh` | promotes v4 suite (sha-checked, backs up `prompts/evaluation_questions.pre-v4.json`), pulls/creates each model, runs protocol-v4 keyword through the **unchanged** RAG stack. ETA ~3–4 h. `--only <key> --smoke-first` to validate one first |
| `scripts/_run_ext_eval.sh` | per-model RAG-server + `run_benchmark.py` helper (Ollama-served, no tf-server) |
| `scripts/sota_scoreboard.py` | reads ours (`rescore_v4`) + ext runs → ranked head-to-head + scoped verdict → rewrites `docs/SOTA_EVAL.md` |

Claim rules baked into `SOTA_EVAL.md`: quality = best open at the task; hardware =
16 GB Blackwell class; **not** a frontier / unconstrained-hardware claim.

---

## ► RESUME HERE (2026-09-01 evening)  — superseded by the 2026-09-02 update above

**State:** SFT + merge + f16 GGUF done. GRPO 150-step probe done and **inert**
(0/266 questions changed — do not ship, do not rerun without a new recipe).
Protocol-v3 **keyword** eval done, 3-way (`docs/benchmark_runs/20260901_*.json`).
**v3 does not clear the acceptance bar** — full analysis in the
`## RESULT (2026-09-01)` section below.

**Judge eval:** FAILED both times — see the 2026-09-02 update. The lines below are
the original (now-stale) plan that assumed the judge would work.

**Decision to make on return (pick one, then run overnight):**

**Path A — fix the eval, ship v3-SFT (no retrain).** The verse_lookup "regression"
is 25/26 exposition-phrased questions ("What does X *teach*?") where v3 explains
instead of quoting; v2 only "passed" by dumping raw text. Split `verse_lookup` in
`benchmarks/manifest.v4.yaml` into `verse_quote` (exact-match) + `verse_exposition`
(fuzzy/judge), re-score the 3 existing runs, and if the judge (running now) shows
v3-SFT ≥ v2 on faithfulness/helpfulness → publish v3-SFT. ~2 h, no GPU.
```
# after judge finishes:
python scripts/compare_benchmark_runs.py \
  docs/benchmark_runs/20260901_v2-4b_judge.json \
  docs/benchmark_runs/20260901_v3-sft_judge.json
# then author manifest.v4.yaml + re-score; then merge+GGUF+publish v3-sft as v3
```

**Path B — v3.1 retrain (7 h GPU, overnight).** Add exposition-phrased templates
(`"What does {ref} teach?"`, `"What is {ref} about?"`) to the verse-drill
generators in `training/build_dataset_v2.py` with **quote-first answers**
(`"{ref} reads: \"{verbatim}\". [1–2 sentence explanation]"`), ~600–1000 examples;
optionally bump `KEEP_BUDGETS` in `training/assemble_v3.py` (7,000 → ~11–13k).
Rebuild → `data/processed/train_v3.1.json`, `check_train_eval_overlap.py`, then:
```
# edit config.v3-4b.yaml: train_file -> data/processed/train_v3.1.json,
#   output_dir -> checkpoints_v3.1_4b, run-name -> qwen3.5-4b-bible-v3.1-sft
tmux new-session -d -s v31sft 'bash ~/bible-ai-assistant/scripts/run_v3_4b_sft.sh'
# ~7 h -> merge_adapters.py -> convert_hf_to_gguf.py --no-mtp -> re-eval keyword
```

**Recommendation:** wait for the judge result, then **Path A** unless the judge
says v3-SFT's exposition is actually *worse* than v2's raw-quote dumps (unlikely).
Path B only if A leaves a real, specific gap.

**Uncommitted working tree** (all this session; PR with the v3-model when it ships):
`docs/V3_STATUS.md`, `training/config.v2.yaml` (grpo `max_completion_length`
512→768), `training/train_grpo.py` (unsloth-before-trl import fix),
`scripts/{run_v3_4b_sft,run_v3_grpo,_run_v3_eval_all,_run_v3_judge}.sh`,
`docs/benchmark_runs/20260901_*.json`, `checkpoints_v3_4b/` (gitignored SFT ckpts).
Plus a site-packages patch: `~/miniforge3/envs/bible-orpo/.../trl/mergekit_utils.py`
`try/except` (lost on env rebuild; the import-order fix in `train_grpo.py` is the
real one). **Do NOT `pip install mergekit`** — it breaks the Unsloth stack.

Merged models on disk (gitignored): `models/qwen3.5-4b-bible-v3-{sft,merged,grpo-merged}`,
`models/qwen3.5-4b-bible-v3-grpo` (adapter), `models/gguf3/bible-v3-4b-f16.gguf`.

---

## Done

- **`data/raw_v3/distill_inputs.jsonl`** — **16,995** `(context, question)` inputs,
  built by `training/build_v3_inputs.py` from the four templated-answer generators
  (`topical_collections` 5,000, `cross_reference_chains` 3,495, `chapter_context`
  2,500, `grounded_exegesis` 6,000).

- **Distillation done** — teacher = **Qwen3-14B Q5_K_M GGUF** served by
  `llama-server` (`-ngl 99 --parallel 6 -fa on` on :8001). vLLM is dead on this box
  (`RuntimeError: UVA is not available`, vLLM 0.26.0 `GPUModelRunnerV2` / `UvaBuffer`,
  driver-level under WSL2 GPU passthrough). `training/distill_answers.py` got a
  `--concurrency` (ThreadPoolExecutor) flag and a `_vllm_teacher` `enable_thinking:false`
  + `<think>`-strip patch; both are in this branch.
  - **16,809 / 16,995 kept (98.9%)** → `data/raw_v3/distill_out.jsonl`. The 186 drops
    are mostly "Song of Solomon" → "Solomon" citation abbreviations from the teacher
    and a few `Romans 16:26/27` refs that don't resolve in the WEB corpus. Resumable:
    survived two interruptions (gaming + one WSL hang) and resumed clean each time.

- **`data/processed/train_v3.json` = 39,463 examples** (+ `.manifest.json`). Built by
  `training/assemble_v3.py` (distilled answers + freshly-built keep-as-is categories
  via `build_dataset_v2.finalize`). Mix:

  | category | count | % |
  |---|--:|--:|
  | general_blend | 10,996 | 27.9 |
  | grounded_exegesis_v3 | 5,955 | 15.1 |
  | topical_collections_v3 | 4,749 | 12.0 |
  | near_miss_guard | 4,480 | 11.4 |
  | cross_reference_chains_v3 | 3,449 | 8.7 |
  | chapter_context_v3 | 2,486 | 6.3 |
  | passage_recall | 1,998 | 5.1 |
  | verse_recall | 1,998 | 5.1 |
  | reverse_lookup | 1,500 | 3.8 |
  | translation_specific | 1,500 | 3.8 |
  | pastoral_triage | 352 | 0.9 |

  - **general/reasoning share = 27.9%** — clears Unsloth's ≥20–25% catastrophic-
    forgetting floor.
  - verse-drill (recall + passage + reverse + translation) = **6,996**, down ~60%
    from v2's ~18k, as planned.
  - Dedup + decontam via `build_dataset_v2.finalize` dropped **187** for
    contamination/dupes (`train_v3.manifest.json`: `topical_collections_v3` 170,
    `near_miss_guard` 11, `passage_recall` 2, `general_blend` 4).
    `check_train_eval_overlap.py`: **zero** normalized-question overlap vs. all
    `benchmarks/suites/*.json` (re-verified 2026-08-31).
  - Known cosmetic: `train_v3.manifest.json` carries `protocol_id:
    bible_assistant_v2_train` (inherited from `build_dataset_v2.finalize`); the
    data is v3.

- **Scripts, lint-clean** (this branch): `training/build_v3_inputs.py`,
  `training/assemble_v3.py`, `training/config.v3-4b.yaml`, the
  `training/distill_answers.py` patch. `ruff format`/`ruff check`/`mypy` clean;
  `tests/test_build_v3_inputs.py` + `tests/test_assemble_v3.py` added.

## Not built — deferred follow-up

- **`thematic_qa`** ("Who is Jesus?", "What is the gospel?" — the v3 eval's actual
  open-ended gap). Needs the live RAG retriever for context (`.venv-rag`,
  `rag/retrieval.py`) over `training/v3_thematic_questions.json`, then a teacher pass.
  ~+2,500 examples. If the post-SFT eval shows `thematic`/`theological` still weak,
  this is the v3.1 iteration.

## Done — SFT + merge + GGUF (2026-09-01)

- **SFT complete.** `scripts/run_v3_4b_sft.sh` → `training/train_unsloth.py`
  `--config training/config.v3-4b.yaml --run-name qwen3.5-4b-bible-v3-sft`.
  2,447/2,447 steps, 6 h 55 m, **rc=0**. final `train_loss 0.5501`,
  final `eval_loss 0.4923` (from 0.568 early — clean convergence, no overfit).
  Adapter → `models/qwen3.5-4b-bible-v3-sft/` (170 MB, r32). Ran on the
  stabilized box (`.wslconfig` memory 64 GB + reboot); no OOM over 7 h.
  Log: `logs/v3sft_20260831-221437.log`.
  - 83 train examples dropped as truncated at `max_seq_length=1280`
    (39,463 → 39,380 → 39,143 train / 237 eval after the 0.006 split).

- **Merge complete.** `training/merge_adapters.py --lora-path
  models/qwen3.5-4b-bible-v3-sft --output models/qwen3.5-4b-bible-v3-merged`.
  Base resolved to the cached pin `851bf6e8…` (only snapshot in the HF cache —
  matches training). bf16 `model.safetensors` 8.41 GB. LoRA delta verified
  present (q_proj / down_proj max|Δ| ~0.008 vs. base — not a silent skip).

- **f16 GGUF complete.** `~/llama.cpp-full/convert_hf_to_gguf.py
  models/qwen3.5-4b-bible-v3-merged --outfile models/gguf3/bible-v3-4b-f16.gguf
  --outtype f16 --no-mtp`. 426 tensors, 8.42 GB (byte-identical size to v2's
  f16). Arch `Qwen3_5ForCausalLM` → `conversion/qwen.py` (hybrid linear-attn,
  `supports_mtp_export`). **Not** `~/wsl41361/llama.cpp` — that checkout is the
  #41361 repro harness and is unbuilt; the built, current one is
  `~/llama.cpp-full` (`LLM_ARCH_QWEN35` runtime support).
  - Runtime smoke (`llama-cli -ngl 99 -st`, greedy): coherent, correctly cited
    — "Bethlehem … Micah 5:2 … Matthew 2:5-6", 83 t/s on GPU.

## GRPO — validated, ready for the full run (2026-09-01)

- **Reward dry-run** (`--dry-run`, no GPU): reward(good)=0.872, reward(bad)=0.075
  against the real `data/raw/bible_web.json` corpus + `data/processed/train_v3.json`
  prompts. Wiring sane.
- **2-step GPU smoke** (`--max-steps 2 --limit-prompts 32`): rc=0, 156 s, no OOM.
  `rewards/verifiable_bible_reward/mean 0.76` (std 0.10), `kl 0.215`. Adapter
  saved+reloaded clean. `completions/clipped_ratio 0.94` — the SFT model rarely
  emits EOS within 512 tok, so most completions train truncated (raise
  `max_completion_length` to ~768 for the full run, or accept it).
- **Two fixes made to get here** (uncommitted, in the working tree):
  - `training/train_grpo.py`: import `unsloth` **before** `trl` in the real-training
    path (line ~285). Without it, `trl.trainer.callbacks` eagerly imports optional
    deps (`mergekit`, `llm_blender`) that aren't installed and the import dies.
  - `~/miniforge3/envs/bible-orpo/.../trl/mergekit_utils.py`: wrapped the
    `from mergekit ...` import in `try/except` (site-packages patch, lost on env
    rebuild — belt-and-braces; the import-order fix is the real one).
  - **Do NOT `pip install mergekit`** into `bible-orpo` — it force-downgrades
    `accelerate` (1.14→1.6), `huggingface_hub`, `pydantic`, `safetensors` and
    breaks the Unsloth/transformers-5.5 stack. It was installed then fully
    reverted on 2026-09-01; env re-verified (`unsloth 2026.8.22`, `trl 0.24.0`,
    `accelerate 1.14.0`, `transformers 5.5.0`, torch 2.11.0+cu128).

### Full GRPO — command + the one open decision

```
python training/train_grpo.py \
    --policy-path models/qwen3.5-4b-bible-v3-sft \
    --config training/config.v2.yaml \
    --data data/processed/train_v3.json \
    --corpus data/raw/bible_web.json \
    --run-name qwen3.5-4b-bible-v3-grpo \
    --max-steps <N> --limit-prompts <M> --no-wandb
```

**Decision needed: `--max-steps` / `--limit-prompts`.** `--max-steps -1` (the
default) = 3 epochs over all 39,463 prompts ≈ 14.8k steps × ~78 s ≈ **prohibitive
(~300 h)**. GRPO reward-shaping for a citation objective is typically a few
hundred steps. Proposed default: **`--max-steps 400 --limit-prompts 4000`**
(≈ 8–9 h at the smoke's 78 s/step; revisit step time on a fresh 25-step probe
first). Bump `max_completion_length` 512→768 in `training/config.v2.yaml`'s
`grpo:` block, or pass it through, to cut the 0.94 clip ratio.

### After GRPO

```
# eval protocol-v3 (282-q suite, greedy, 3 seeds, vs. the v2-4b checkpoint)
#   + FMG-Bench (scripts/fmg_bench.py, calibration only)
# merge GRPO adapter -> bf16 -> GGUF (~/llama.cpp-full/convert_hf_to_gguf.py --no-mtp)
#   -> quant ladder Q4_K_M/Q5_K_M/Q6_K/Q8_0 [+imatrix], mirror models/gguf2/ naming
#   -> publish v3.
```

The `models/gguf3/bible-v3-4b-f16.gguf` already built is an **SFT-stage**
validation artifact — the published GGUFs come from the post-GRPO merge.

## RESULT (2026-09-01): protocol-v3 keyword eval, 3-way — v3 does not clear the bar, GRPO is a no-op

Ran `scripts/_run_v3_eval_all.sh` (tf-server :8001 -> RAG :8081 -> `run_benchmark.py`,
official protocol-v3 keyword scoring, suite sha256 verified) on all three merged
checkpoints. Runs in `docs/benchmark_runs/20260901_{v2-4b,v3-sft,v3-grpo}_keyword.json`.
Paired stats via `scripts/compare_benchmark_runs.py` (McNemar exact + 10k bootstrap).

| metric (n=266) | v2-4b | v3-SFT | v3-GRPO | bar |
|---|--:|--:|--:|--:|
| overall fuzzy mean | 0.396 | 0.488 | 0.487 | ≥0.52 ✗ |
| fuzzy pass-rate @0.85 | 16.3% | 15.6% | 15.6% | (McNemar p=0.69, flat) |
| verse_lookup exact | **76%** | 50% | 50% | ≥74% ✗ |
| verse_lookup fuzzy mean | 0.648 | 0.663 | 0.664 | — |
| character / context / topical fuzzy mean | .20/.23/.20 | **.37/.39/.38** | .37/.40/.37 | ≥v2+0.10 ✓ |
| citation rate | 99% | 98% | 98% | ≥97% ✓ |
| hallucination | 2% | 2% | 2% | ≤2.5% ✓ |

**GRPO changed nothing.** v3-GRPO vs v3-SFT: verse_accuracy delta +0.0pp, McNemar
b=0/c=0 — not one question flipped pass/fail; fuzzy mean identical to 3 d.p. The
150-step / lr 1e-6 / cosine-to-zero probe was confirmed inert (matches the training
log: reward bounced 0.33–0.70 with no trend, lr ~0 by step ~75). Do **not** ship
v3-GRPO and do **not** book a longer GRPO run without a changed recipe (constant or
higher lr, more steps, reshaped reward).

**The verse_lookup "regression" is a behaviour split on question phrasing, not a
recall loss** (diagnosed from the per-item results):

- QUOTE-style verse_lookup Qs ("What does X **say**?", "Quote X"): v3 held **51/52**.
- EXPOSITION-style Qs ("What does X **teach**?", "What is X **about**?"): **25 of 26**
  lost points. v2 answered these by dumping the verbatim WEB text (passing
  exact-match); v3-SFT answers with an accurate *explanation* (fails exact-match,
  fuzzy ~0.44). The teacher-distilled synthesis training generalised "explain in
  prose" onto exposition-phrased single-verse questions.
- `prompts/system_prompt.txt` already says *"For verse lookups: quote the verse,
  cite the reference, then explain"* — SFT overrode that instruction for the
  `teach`/`about` phrasings, so a prompt-only fix is unlikely to be sufficient.

### Verdict

v3-SFT: real, measurable synthesis gains (character/context/topical fuzzy means
~1.8x, clear ≥ v2+0.10); scripture recall intact on direct quote requests (98%);
citation and hallucination held. But it **misses the acceptance bar** — overall
fuzzy 0.488 < 0.52, and verse_lookup exact 50% < 74% because of the
exposition-phrasing behaviour split. **v3 does not ship as-is.** GRPO is not the
lever.

### Recommended next step: v3.1 — "quote-then-explain" for named references

Cheapest fix that keeps the synthesis gains and recovers verse_lookup:

1. In `training/build_dataset_v2.py`, add exposition-phrased templates
   (`"What does {ref} teach?"`, `"What is {ref} about?"`, `"What's the message of
   {ref}?"`) to the verse-drill generators, with answers in **quote-first form**:
   `"{ref} reads: “{verbatim}”. [1–2 sentence explanation]"`. ~600–1000
   examples across `verse_recall` / `passage_recall`.
2. Optionally bump `KEEP_BUDGETS` in `training/assemble_v3.py` (currently
   verse_recall 2000 / passage_recall 2000 / reverse_lookup 1500 /
   translation_specific 1500 = 7,000 vs v2's ~18,000) back toward ~11–13k total —
   partial restore, not full re-saturation.
3. Rebuild -> `data/processed/train_v3.1.json`; re-run `check_train_eval_overlap.py`.
4. SFT re-run (`config.v3-4b.yaml` with the new `train_file`), ~7 h — **user-gated**.
5. Re-eval keyword; if it clears the bar, ship v3.1 SFT (skip GRPO until the recipe
   question is worth revisiting).

Alternative if "distinguish explain-vs-quote" is the *desired* behaviour: fix the
**eval** instead — split `verse_lookup` into `verse_quote` (exact-match) and
`verse_exposition` (fuzzy/judge). Then v3-SFT's picture is "synthesis up, quote
recall held, no real loss." This is a `manifest.v4` change, not a retrain.

### Still open

- **Judge eval** (`_run_v3_judge.sh`, v2-4b + v3-sft, qwen3.5:27b via Ollama):
  queued. `ollama pull qwen3.5:27b` stalled on 2026-09-01 (registry throttling,
  dropped to ~26 KB/s); a detached retry loop is running. Judge would confirm
  whether v3-SFT's exposition answers score better on faithfulness/helpfulness —
  expected, and it informs the "fix eval vs retrain" choice above.
- **FMG-Bench** (`scripts/fmg_bench.py`) — not run; calibration only.

## Acceptance bar (from `docs/V3_DATASET_PLAN.md`)

Measured on the 282-question v3 suite, greedy, 3 seeds, vs. the v2-4b checkpoint:
`topical`/`context`/`character` fuzzy ≥ v1's numbers **and** ≥ v2 + 0.10;
`verse_lookup` exact ≥ 74%; citation ≥ 97%; hallucination ≤ 2.5%;
overall fuzzy mean ≥ 0.52 (beats v1's 0.48 and v2's 0.40).
Plus FMG-Bench (`scripts/fmg_bench.py`) reported as calibration — no regression
vs. a same-size baseline; not a pass/fail gate.
