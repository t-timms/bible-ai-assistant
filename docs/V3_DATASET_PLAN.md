# V3 dataset plan — kill the templated answers

> **Status (2026-08-31): `data/processed/train_v3.json` built — 39,463 examples.**
> Distillation ran against a local **Qwen3-14B Q5_K_M GGUF** (`llama-server`; vLLM is
> dead on this box), 16,809 / 16,995 answers kept (98.9%). The build came in smaller
> than the ~45.6k target below: `cross_reference_chains` / `chapter_context` landed
> under target after dedup, and **`thematic_qa` was not built** (it needs the live
> RAG retriever — deferred to a v3.1 pass). Category keys keep their v2 names with a
> `_v3` suffix rather than the renamed keys in the table below. Exact shipped mix +
> next action: `docs/V3_STATUS.md`. Everything else in this plan stands.

## Why v3

Protocol-v3 A/B (`docs/MODEL_COMPARISON.md`): v2-4b lifted verbatim verse recall
(58% → 76.5%) and citation rate (88% → 98.9%) but **regressed open-ended thematic
quality** vs. the lightly-tuned v1 (overall fuzzy 0.48 → 0.40; `character`,
`context`, `topical`, `theological_reliability` all down). Root cause, confirmed in
transcripts: the v2 dataset diversified *questions* but not *answers*. Several
categories ship a fixed answer template, so the model learned fill-in-the-blank
formats instead of judgment.

The offenders (from `training/build_dataset_v2.py`, seen in `train_v2.json`):

| category | count | answer shape today | verdict |
|---|---:|---|---|
| `topical_collections` | 4,358 | `"Here are five passages on X…"` + bullet list of quotes, **no synthesis** | **regenerate** |
| `cross_reference_chains` | 4,272 | `"X has strong scriptural echoes in:"` + bullet refs + one canned sentence | **regenerate** |
| `chapter_context` | 4,000 | despite the name, it's `"That's Psalm 84:4 (KJV): …"` — a verse lookup | **repurpose → real context/exposition** |
| `grounded_exegesis` | 7,000 | MHC commentary in context → condensed answer | **teacher-polish** for natural voice |
| `verse_recall` / `passage_recall` / `reverse_lookup` / `translation_specific` | ~18k | `"<ref> (<ver>) says: '<quote>'"` | **keep, cut volume ~60%** — this is correct behavior and the eval shows it's saturated |
| `near_miss_guard` | 4,491 | correction of a mis-attributed quote | **keep** |
| `pastoral_triage` | 352 | hand-authored escalation / abstention | **keep** |
| `general_blend` | 12,996 | smoltalk2, `<think>` stripped | **keep** (catastrophic-forgetting floor) |

## Target mix (~40–45k examples)

| category | v2 | v3 | change |
|---|---:|---:|---|
| verse-drill (recall / passage / reverse / translation) | ~18,000 | **7,000** | cut ~60%; keep a representative spread of books/translations |
| `near_miss_guard` | 4,491 | 4,491 | keep |
| `topical_synthesis` (was `topical_collections`) | 4,358 | **5,000** | **teacher-regenerated**: synthesized answer that explains the theme and weaves 3–5 cited verses in prose |
| `cross_reference_reasoning` (was `cross_reference_chains`) | 4,272 | **4,500** | **teacher-regenerated**: explain *why* the passages connect, not just list them |
| `passage_exposition` (was `chapter_context`) | 4,000 | **4,500** | **teacher-regenerated**: given a passage + optional MHC context, explain what it means in its context |
| `grounded_exegesis` | 7,000 | **6,000** | **teacher-polished** for voice; keep the MHC grounding |
| `thematic_qa` (**new**) | 0 | **2,500** | the gap the eval exposed — "Who is Jesus?", "What is the gospel?", "What does the Bible say about grace/suffering/forgiveness?" — teacher-written, grounded in retrieved verses, inline citations |
| `pastoral_triage` | 352 | **600** | +248 more hand-authored / teacher-drafted-then-reviewed |
| `general_blend` | 12,996 | 11,000 | keep the ≥20–25% general floor (lands ~27%) |
| **total** | 56,022 | **~45,600** | |

Regeneration/creation load: ~**22,500** teacher calls (`topical_synthesis` +
`cross_reference_reasoning` + `passage_exposition` + `grounded_exegesis` polish +
`thematic_qa`).

## The teacher

`training/distill_answers.py` is teacher-agnostic (`--backend`):
`anthropic` | `openai` | `hf` (HF Inference Providers) | `vllm` (local OpenAI-compat)
| `echo` (dry-run). **Decision needed: backend + budget.** Rough cost at ~320
output tokens/answer × 22.5k ≈ **7.2M output tokens** — a few $ (small open model via
deepinfra) to ~$30–60 (frontier). A local vLLM teacher (e.g. a 27B on the 5070 Ti,
or Qwen3.8-27B once quantized) is $0 but needs the GPU.

Every distilled answer is validated before it enters the set:
`rag.verification.verify_citations` — each cited `Book C:V` must resolve in the WEB
corpus, and each quoted span must match a real verse (normalized). Failures are
retried once with a stricter prompt, then dropped and logged. Target keep-rate ≥ 95%.

### Teacher prompt contract

System: *"You are a careful Bible study assistant. Answer using ONLY the verses in
CONTEXT for any scripture quotation or citation. Synthesize — explain the idea in
your own words and weave the cited verses in as support. Cite inline as `Book C:V`.
Never quote or cite a verse that is not in CONTEXT. If CONTEXT is insufficient, say
so briefly. 2–5 sentences unless the question genuinely needs more. No bullet-list
dumps. Pastoral, plain, non-sectarian."*

User: the same `Context:` block + `Q:` the dataset already builds, so train/inference
format is unchanged.

`thematic_qa` gets its context from the live RAG retriever (`.venv-rag`,
`rag/retrieval.py`) over the hand-authored question list in
`training/v3_thematic_questions.json`, so the grounding matches what the served model
will actually see.

## Pipeline

*(Implemented as two standalone scripts rather than `build_dataset_v2.py` flags.)*

1. `python training/build_v3_inputs.py --out data/raw_v3/distill_inputs.jsonl` — run
   the four templated-answer generators from `build_dataset_v2`, keep only
   `(context, question)`; the teacher regenerates the answer.
2. `python training/distill_answers.py --backend vllm --vllm-url http://127.0.0.1:8001/v1
   --model t --concurrency 6 --in data/raw_v3/distill_inputs.jsonl
   --out data/raw_v3/distill_out.jsonl` — resumable; every answer validated via
   `rag.verification` (bad ref → 1 stricter retry → drop + log). `--backend vllm`
   also drives a local `llama-server` (OpenAI-compat).
3. `python training/assemble_v3.py --distilled data/raw_v3/distill_out.jsonl
   --out data/processed/train_v3.json` — merge distilled answers + freshly-built
   keep-as-is categories (verse-drill at v3 budgets, `near_miss_guard`,
   `pastoral_triage`, `general_blend`) → dedup + decontaminate via
   `build_dataset_v2.finalize`; write `train_v3.manifest.json` (per-source SHA +
   license, per-category kept/dropped counts).
4. `python scripts/check_train_eval_overlap.py --train data/processed/train_v3.json`
   — enforces zero normalized-question overlap vs. all `benchmarks/suites/*.json`.
5. SFT: `training/config.v3-4b.yaml` (fork of `config.v2-4b.yaml`, `train_file:
   data/processed/train_v3.json`, same seq/padding).
6. **GRPO**: `training/train_grpo.py` (scaffold ready — reward = citation_exists +
   text_match + format) starting from the v3 SFT adapter. This is the stage that
   pushes past the 85% bar.
7. Eval: protocol v3 + **FMG-Bench** (`scripts/fmg_bench.py`; open, 120+37, self-scored)
   as calibration, not a win target. (FaithBench — the Christian-theology site — has no
   public dataset yet; not wired in.)

## Acceptance criteria (the bar)

Measured on the 282-question v3 suite, greedy, 3 seeds, vs. the v2-4b checkpoint on
the identical run:

- `topical` / `context` / `character` fuzzy: **v3 ≥ v1's** numbers (0.35 / 0.40 /
  0.36) — i.e. recover everything v2 lost — **and** ≥ v2 + 0.10.
- `verse_lookup` exact: **hold ≥ 74%** (don't trade recall away).
- overall citation rate: **hold ≥ 97%**; hallucination: **≤ 2.5%**.
- overall fuzzy mean: **≥ 0.52** (beat both v1's 0.48 and v2's 0.40).
- FMG-Bench: report the number; no regression vs. a same-size baseline. Not a pass/fail gate.

If 4B stalls on `thematic_qa` after GRPO → escalate to `config.v2-9b.yaml` (QLoRA),
per the base-model decision in `docs/V2_EXECUTION_PLAN.md`.

## Serving dependency (unchanged from v2 notes)

`rag_server.py` needs a commentary-retrieval path before a v3 model trained on
`grounded_exegesis` / `passage_exposition` is served, or it's the train/inference
context mismatch again. Build it against the format this plan freezes, not before.
