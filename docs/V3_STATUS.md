# V3 status — resume here (2026-08-31)

Plan: `docs/V3_DATASET_PLAN.md`. This file = exact state + the next action.

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
  - Dedup + decontam via `build_dataset_v2.finalize` dropped 199
    (`check_train_eval_overlap.py`: **zero** normalized-question overlap vs. all
    `benchmarks/suites/*.json`, re-verified 2026-08-31).

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

## Next action

```
# 1. SFT — user triggers overnight (GPU-gated; box must be stabilized first:
#    .wslconfig memory 80GB -> 64GB + a Windows reboot, per the WSL2-load-hang notes).
python training/train_unsloth.py --config training/config.v3-4b.yaml \
    --run-name qwen3.5-4b-bible-v3-sft
# ~7 h: ~2,452 steps x ~8.9 s + eval (v3 is ~30% smaller than v2). eval_split 0.006
# for A/B comparability with v2-4b.

# 2. merge -> GGUF (convert_hf_to_gguf.py --no-mtp) -> GRPO (training/train_grpo.py,
#    citation reward, --max-steps 2 smoke first) -> eval protocol-v3 + FMG-Bench +
#    FaithBench -> publish v3.
```

## Acceptance bar (from `docs/V3_DATASET_PLAN.md`)

Measured on the 282-question v3 suite, greedy, 3 seeds, vs. the v2-4b checkpoint:
`topical`/`context`/`character` fuzzy ≥ v1's numbers **and** ≥ v2 + 0.10;
`verse_lookup` exact ≥ 74%; citation ≥ 97%; hallucination ≤ 2.5%;
overall fuzzy mean ≥ 0.52 (beats v1's 0.48 and v2's 0.40).
