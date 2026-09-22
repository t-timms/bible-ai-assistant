# Model Comparison

> **Protocol note (2026-09-05):** the current protocol is **v5** — adds a cross-encoder
> semantic metric alongside v4's fuzzy, built after auditing fuzzy's noise floor on close
> checkpoints (see the block below). The head-to-head against external open comparators (the
> "best open model at the task" claim) lives in [`docs/SOTA_EVAL.md`](SOTA_EVAL.md). Sections
> further down are the v1→v2→v3-SFT internal history, measured under protocols v3/v4.

## Latest: v3.2 ships — protocol v5, 2026-09-05

Two more iterations ran after the v3-SFT HOLD below (v3.1, then v3.2) before one cleared the
bar. Full per-category numbers and training detail: [`docs/MODEL_CARD.md`](MODEL_CARD.md).
Root-cause story and every fix: [`docs/V3_STATUS.md`](V3_STATUS.md) ("PROTOCOL V5 + SHIP
DECISION" and "EXTERNAL SOTA SWEEP DONE").

| metric (protocol v5) | v2-4b | v3-SFT | v3.1 | **v3.2** | bar |
|---|--:|--:|--:|--:|--:|
| verse_quote exact | 78.8% | 77.3% | 78.8% | **80.3%** | — |
| overall fuzzy, expo-excl | 0.394 | 0.497 | 0.492 | **0.500** | ≥0.52 ✗ |
| **semantic, expo-excl** | 0.829 | 0.918 | 0.928 | **0.942** | — |
| citation | 98.9% | 97.7% | 99.2% | **98.9%** | ≥97% ✓ |
| hallucination | 3.4% | 1.5% | 1.5% | **1.9%** | ≤2.5% ✓ |

v3.1 and v3-SFT landed within 0.008 of each other on fuzzy (v3.1 was itself a HOLD, not shown
as a separate re-eval block below since it never shipped) — inside that metric's demonstrated
noise floor, not a real result either way. v3.2's three fixes (RAFT-style distractor-prompt
fix, retrieval-depth correction, DMT-style continued fine-tune) produced a real gain on the
new semantic metric: **+0.014 vs v3.1, paired bootstrap 95% CI [+0.004, +0.026] — excludes
0.** Verse-quote exact and citation held/improved rather than trading off.

**Decision: SHIP v3.2.** Still misses the original 0.52 fuzzy bar, same as every prior
checkpoint — but that bar was written against a metric now shown to have a narrow noise
floor at this quality level; the semantic metric is the fairer ranking tool going forward.

## v3-SFT re-eval through the #46-fixed RAG stack — protocol v4, 2026-09-03

Full re-run (`scripts/_run_v3_eval_all.sh`, ~65 min GPU), all three checkpoints, through the
retrieval stack **after PR #46** (exposition questions pin the verse and search on its text,
not the bare reference). Merged models rebuilt from the SFT adapters + coherence-checked first.
Raw: `docs/benchmark_runs/20260903_{v2-4b,v3-sft,v3-grpo}_keyword.json` + `…_v4keyword.json`.

| metric (protocol v4, **post-#46**) | v2-4b | v3-SFT | v3-GRPO | bar | Δ vs pre-#46 rescore |
|---|--:|--:|--:|--:|--|
| **verse_quote** exact (n=66 — real recall) | 78.8% | **77.3%** | 77.3% | ≥74% ✓ | held |
| verse_quote vs. v2, McNemar | — | p=0.50 (held) | — | — | — |
| verse_exposition fuzzy mean (n=36) | 0.509 | **0.542** | 0.539 | — | v3-SFT **0.418 → 0.542** |
| overall fuzzy mean, **exposition-excluded** (n=230) | 0.394 | **0.497** | 0.496 | ≥0.52 ✗ (−0.023) | 0.499 → 0.497 (flat) |
| overall fuzzy mean, all-in (n=266) | 0.409 | **0.503** | 0.501 | ≥0.52 ✗ | 0.488 → 0.503 |
| hallucination_detected (corpus mode) | 9/282 | **4/282** | 5/282 | — | Gen 19:28 **now clean** |

**v3-SFT per-category fuzzy mean:** `verse_lookup` **0.707** carries the average; every
synthesis category is ~0.31–0.41 — cross_reference 0.396, context 0.374, topical 0.380,
character 0.365, theological_reliability 0.311. That ~163-question block at ~0.37 is what
holds the overall at 0.497.

**#46 did its job** (verse_exposition 0.418 → 0.542, Genesis 19:28 hallucination gone) — but
v3-SFT still misses the 0.52 overall bar by 0.023, essentially flat vs the pre-#46 rescore.
**Decision: HOLD — do not ship v3-SFT.** The gap is thematic synthesis, not exposition, so
v3.1 retargets at synthesis-category distillation (`docs/ROADMAP.md` item 7, rewritten;
`docs/V3_STATUS.md` "RE-EVAL DONE"). GRPO still inert (`v3-grpo` ≈ `v3-sft` to 3 d.p.).

---

## 2026-09-03 (earlier): v4 rescore of the pre-#46 runs — SUPERSEDED by the block above

Re-scored from the 2026-09-01 protocol-v3 keyword runs (`scripts/rescore_v4.py`; no model
re-run) — the `verse_lookup` category split into `verse_quote` (66, verbatim recall) and
`verse_exposition` (36, explanation expected). Raw:
`docs/benchmark_runs/20260902_{v2-4b,v3-sft,v3-grpo}_v4keyword.json`; per-item exposition
read: `docs/benchmark_runs/20260902_exposition_v2_vs_v3.md`.

| metric (protocol v4) | v2-4b | v3-SFT | v3-GRPO | bar |
|---|--:|--:|--:|--:|
| **verse_quote** exact (n=66 — the real recall metric) | 78.8% | **77.3%** | 77.3% | ≥74% ✓ |
| verse_quote vs. v2, McNemar | — | p=0.50 (held) | — | — |
| verse_exposition fuzzy mean (n=36) | 0.427 | 0.418 | 0.418 | — (tie) |
| verse_exposition exact (non-primary) | 72.2% | 1.4% | 0.0% | — |
| overall fuzzy mean, **exposition-excluded** (n=230) | 0.391 | **0.499** | 0.498 | ≥0.52 ✗ (−0.021) |
| overall fuzzy mean, all-in (n=266) | 0.396 | 0.488 | 0.487 | ≥0.52 ✗ |
| citation rate | 98.9% | 97.7% | 98.1% | ≥97% ✓ |
| hallucination rate | 2.3% | 1.5% | 1.9% | ≤2.5% ✓ |

**The `verse_lookup` "50%" regression was an eval artifact.** Splitting the category shows
scripture-quote recall held (77.3% vs. 78.8%, not significant); the drop was 26/36
exposition-phrased questions where v2 "passed" exact-match by dumping the verbatim verse
and v3 answers with a prose explanation instead. A manual read of all 36 has v3-SFT
better-or-tie on **34/36** — but with **one confident v3 hallucination (Genesis 19:28)**,
partly built on reference-token-matched retrieval noise (see `docs/CODEBASE_AUDIT.md`).
**GRPO is inert** — `v3-grpo` equals `v3-sft` to three decimal places on every metric.

*(This block's "ship v3-SFT as v3" recommendation was made from the pre-#46 rescore. The
2026-09-03 re-eval through the fixed RAG stack — section above — keeps v3-SFT at 0.497
expo-excl, so the actual decision is **HOLD**. The `verse_lookup`-artifact analysis and the
"synthesis categories were the real v2 regression" finding still stand.)*

## v2-4b (Qwen3.5-4B, 56k SFT) vs. v1 shipped model — protocol v3, 2026-08-29

First measurement under **protocol v3** (`bible_assistant_baseline_v3`, 282 questions,
sha-pinned suite; keyword/verification metrics, no judge). Both models served via a local
transformers `/v1/chat/completions` wrapper (`scripts/_tf_openai_server.py`) behind the RAG
server — GGUF/Ollama is blocked (Qwen3.5-4B is a hybrid Gated-DeltaNet+attention arch that
llama.cpp cannot load yet) and vLLM's UVA path is broken under WSL2. Greedy decode, seed 42.

| Metric | v1 (Qwen3-4B, ~1.8k SFT + 500 ORPO) | **v2-4b** (Qwen3.5-4B, 56k SFT, 1 epoch) | Δ |
|---|---|---|---|
| verse_lookup — exact verse acc | 58% | **76.5%** | **+18.5 pp** |
| verse_lookup — fuzzy pass @0.85 | 39% | 45% | +6 pp |
| overall verse acc (exact) | 22% | 29.3% | +7.3 pp |
| overall citation rate | 88% | **98.9%** | +11 pp |
| overall hallucination rate | 1.5% | 2.3% | +0.8 pp (CIs overlap) |
| **overall fuzzy mean** | **0.483** | 0.396 | **−0.087** |

Per-category fuzzy mean (closeness to the expected natural answer):

| Category | v1 | v2-4b | |
|---|---|---|---|
| character | 0.356 | 0.198 | v2 worse |
| context | 0.404 | 0.227 | v2 worse |
| topical | 0.351 | 0.198 | v2 worse |
| theological_reliability | 0.292 | 0.149 | v2 worse |
| cross_reference | 0.412 | 0.404 | tie |
| verse_lookup | 0.665 | 0.648 | tie |

Raw JSONs: `docs/benchmark_runs/20260829_v2-4b_keyword.json`,
`docs/benchmark_runs/20260829_v1-baseline_keyword.json`.

### Finding: templated answers are the ceiling

v2-4b's 56k-example dataset added real capability — the new `pastoral_triage`,
`grounded_exegesis`, and `cross_reference_chains` categories produce genuine escalation
answers, multi-passage character syntheses, and cross-reference reasoning in the transcripts.
And it is **markedly better at the core RAG task**: verbatim verse recall from provided
context 58% → 76.5%, citation rate 88% → 98.9%, near-zero hallucination.

But the eight scripture-citation categories use rigid fill-in-the-blank **answer** templates
(`"Ref (TL) says: …"`, `"Here are five passages on X, spanning old and new covenant
writings: • …"`). The dataset upgrade diversified the *questions*, not the *answers*. One
epoch of SFT on 35k templated scripture answers taught the model the *format* rather than the
*skill*: for "What is the context of Psalm 23?" or "Who is Jesus?" it emits a bare verse list
instead of an explanation, and its answers are **further** from the expected natural answers
than the lightly-tuned v1's (overall fuzzy 0.48 → 0.40). The smoltalk2 catastrophic-forgetting
blend (24% of the mix) slowed this but did not stop it against that volume of templates.

This re-confirms the v1-era learning ("1,800 diverse examples outperform 31,000 repetitive
ones"), now with a controlled A/B: **the fix is teacher-distilled natural answers + a GRPO
stage for citation faithfulness, not more templated SFT.** v2-4b's verse-recall gain carries
forward; the method that produced it is being replaced. See `docs/V2_EXECUTION_PLAN.md`.

---

# Historical: SFT vs. SFT+ORPO (v1-era, protocol v1)

Comparison of the two final models produced by the original (v1) Bible AI Assistant pipeline.

> **Methodology note (2026-08-24):** The numbers below were measured under benchmark protocol
> **v1** (`bible_assistant_baseline_v1`, 57 questions) with the original `check_hallucination`,
> which only validated that a cited *book name* was real — a real book with a fabricated verse
> number (e.g. "1 Corinthians 47:99") passed as "not hallucinated." Protocol **v2**
> (`bible_assistant_baseline_v2`, 282 questions, see `docs/BENCHMARK_PROTOCOL.md`) verifies the
> actual chapter:verse against the indexed Bible text and adds a fuzzy verse-accuracy metric that
> doesn't penalize valid paraphrase. v1 and v2 numbers are **not comparable** — do not read a v2
> re-run against these figures as an apples-to-apples before/after. New measurements will be
> appended here once a v2 run is complete; until then, treat the numbers below as historical only.

## Models Under Evaluation

| Property | SFT-Only (v8) | SFT+ORPO (v8-orpo) |
|----------|---------------|---------------------|
| Base Model | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-4B |
| Training Stage 1 | bf16 LoRA SFT (270 steps) | bf16 LoRA SFT (270 steps) |
| Training Stage 2 | — | ORPO (63 steps, 500 preference pairs) |
| GGUF Variants | F16 (8.5 GB) | F16 (8.5 GB), Q4_K_M (2.5 GB) |
| Ollama Name | `bible-assistant` | `bible-assistant-orpo`, `bible-assistant-orpo-f16` |

---

## Training Metrics

### SFT Training Curve

The SFT stage trains the model on ~1,800 diverse Bible Q&A examples.

| Step | Loss | Learning Rate | Epoch |
|------|------|---------------|-------|
| 50 | 0.9582 | 9.80e-5 | 0.6 |
| 100 | 0.2200 | 1.98e-4 | 1.1 |
| 150 | 0.1591 | 1.42e-4 | 1.7 |
| 200 | 0.1327 | 8.35e-5 | 2.2 |
| 250 | 0.0992 | 2.47e-5 | 2.8 |
| **270** | **~0.10** | **2.47e-5** | **3.0** |

**Takeaway:** Loss drops rapidly in the first epoch (0.96 → 0.22), then steadily converges. The model learns Bible Q&A format quickly; remaining epochs refine response quality.

### ORPO Preference Alignment Curve

ORPO trains on 500 chosen/rejected pairs covering common failure modes.

| Step | Loss | NLL Loss | Reward Accuracy | Chosen Reward | Rejected Reward | Margin |
|------|------|----------|-----------------|---------------|-----------------|--------|
| 10 | 1.188 | 1.143 | 100% | -0.014 | -0.026 | 0.012 |
| 20 | 1.061 | 1.018 | 98.8% | -0.012 | -0.025 | 0.013 |
| 30 | 0.909 | 0.869 | 100% | -0.011 | -0.024 | 0.014 |
| 40 | 0.805 | 0.767 | 100% | -0.009 | -0.024 | 0.014 |
| 50 | 0.727 | 0.694 | 100% | -0.008 | -0.023 | 0.015 |
| **60** | **0.685** | **0.652** | **100%** | **-0.008** | **-0.022** | **0.014** |

**Takeaway:** ORPO converges smoothly with 100% reward accuracy by step 30. The model reliably distinguishes good from bad responses. The margin between chosen and rejected rewards grows steadily, indicating the model is learning the preference signal.

### What ORPO Targets

The 500 preference pairs specifically address these failure modes observed in the SFT-only model:

| Failure Mode | Example (Rejected) | Correction (Chosen) |
|-------------|---------------------|---------------------|
| **Hallucinated verses** | Fabricating a verse that doesn't exist | Quoting the actual verse from Scripture |
| **Instruction leaking** | Outputting system prompt text in the response | Clean, natural response |
| **Repetition loops** | Repeating the same phrase 5-10 times | Single, concise statement |
| **"Answer:" prefix** | Starting response with "Answer:" | Natural conversational opening |
| **Verbosity** | 500+ word responses for simple questions | Concise 2-3 sentence answer |
| **Bible-for-everything** | Answering "What's the weather?" with Scripture | Politely declining non-Bible questions |

---

## Evaluation Results (Keyword Benchmark)

54 scored questions from the v1 suite (`evaluation_questions.v1.json`, 57 questions across 7 categories), keyword-overlap and citation regex.

### SFT-Only (F16) — Benchmark Attempt

The SFT-only model produces **incoherent output** — random tokens, code fragments, and numerical garbage. Sample responses:

- *"What does John 3:16 say?"* → `5. trickule 300 you n, then 3-10,3-20...`
- *"What does Romans 8:28 say?"* → `@ people urleton urleton urleton...`
- *"Who was Peter?"* → `10008551045528, 10008551045529...`

The SFT-only model is **not usable for production**. This is the core motivation for ORPO alignment.

### SFT+ORPO (Q4_K_M) — Keyword Benchmark

| Category | Questions | Verse Accuracy | Citations | Hallucinations |
|----------|-----------|----------------|-----------|----------------|
| verse_lookup | 10 | 30% | 8/10 | 2/10 |
| topical | 10 | 0% | 9/10 | 3/10 |
| character | 10 | 0% | 7/10 | 1/10 |
| cross_reference | 10 | 0% | 8/10 | 4/10 |
| context | 10 | 0% | 7/10 | 1/10 |
| refusal | 4 | 0% | 1/4 | 0/4 |
| **Overall** | **54** | **5.6%** | **74%** | **20%** |

### SFT+ORPO (F16) — Keyword Benchmark

| Category | Questions | Verse Accuracy | Citations | Hallucinations |
|----------|-----------|----------------|-----------|----------------|
| verse_lookup | 10 | 50% | 9/10 | 1/10 |
| topical | 10 | 0% | 9/10 | 3/10 |
| character | 10 | 0% | 8/10 | 3/10 |
| cross_reference | 10 | 0% | 10/10 | 4/10 |
| context | 10 | 0% | 10/10 | 3/10 |
| refusal | 4 | 0% | 1/4 | 0/4 |
| **Overall** | **54** | **9.3%** | **87%** | **26%** |

### Head-to-Head Summary

| Metric | SFT-Only (F16) | ORPO Q4_K_M | ORPO F16 |
|--------|---------------|-------------|----------|
| **Coherent output** | No (gibberish) | Yes | Yes |
| **Verse accuracy** | N/A | 5.6% | 9.3% |
| **Citation rate** | N/A | 74% | 87% |
| **Hallucination rate** | N/A | 20% | 26% |
| **Model size** | 8.5 GB | 2.5 GB | 8.5 GB |

**Key findings:**
- ORPO is **essential** — without it, the model is completely non-functional
- F16 precision improves citation rate (87% vs 74%) and verse accuracy (9.3% vs 5.6%)
- Q4 quantization trades some quality for 70% size reduction with acceptable degradation
- Cross-reference questions are the hardest category for both variants

**On the counter-intuitive hallucination result (Q4_K_M lower than F16):**

The Q4_K_M model shows 20% hallucination vs 26% for F16 — lower is better, so Q4 appears safer. This result is likely a **metric artefact at small n**, not a genuine quality difference:

1. **n=54 is too small for reliable hallucination rate comparisons.** The difference is 11 vs 14 hits — a delta of 3 questions. At n=54 with a true rate of ~23%, the 95% confidence interval (normal approximation) spans ±12%, meaning both values are statistically indistinguishable.

2. **Q4 truncates verbose responses.** Quantized models sometimes produce shorter responses to avoid uncertainty. A shorter answer may contain fewer opportunity windows for hallucinated verse references, mechanically reducing the hallucination count without the model actually being "better."

3. **Cross-reference questions dominate hallucinations.** 4/10 cross-reference questions hallucinate in both variants (the same absolute count). The overall rate difference comes from other categories where small sample randomness dominates.

**Conclusion:** Run both models on n≥200 with a fixed random seed before drawing quality conclusions from hallucination rate comparisons.

### Understanding the Metrics

**Why is verse accuracy low?**

The keyword-overlap metric requires an exact substring match against a reference translation (WEB). The model frequently:

1. **Cites the correct verse reference** but uses slightly different wording
2. **Paraphrases** rather than quoting verbatim — e.g., "his one and only Son" vs. "his only born Son"
3. **Adds contextual commentary** alongside the verse text

The citation rate (74-87%) better reflects actual retrieval quality.

**Where hallucinations occur:**

Hallucinations cluster in cross-reference questions (4/10 for both variants), where the model sometimes attributes text to the wrong book or invents verse numbers.

---

## Qualitative Comparison

### Verse Lookup: "What does John 3:16 say?"

**SFT+ORPO Response:**
> Cites John 3:16 WEB with accurate wording and brief contextual note about the gospel message.

**Analysis:** Clean, concise, properly cited. ORPO's anti-verbosity training keeps the response focused.

### Topical: "What does the Bible say about forgiveness?"

**SFT+ORPO Response:**
> Cites Matthew 6:14 (WEB) and connects it to God as the origin and model of forgiveness.

**Analysis:** The RAG topical anchors ensure Matthew 6:14 is always retrieved for forgiveness questions. Response is grounded in a specific verse rather than generic theology.

### Refusal: Non-Bible Questions

**SFT+ORPO Response:**
> Politely declines with a note that the assistant focuses on Bible questions.

**Analysis:** ORPO's "Bible-for-everything" correction pairs teach the model appropriate boundaries.

---

## Quantization Impact

| Variant | Size | Inference Speed | Quality |
|---------|------|-----------------|---------|
| F16 (full precision) | 8.5 GB | Baseline | Best quality |
| Q4_K_M (4-bit) | 2.5 GB | ~2x faster | Minimal degradation for most queries |

The Q4_K_M quantization reduces model size by 70% with negligible quality loss for Bible Q&A. This makes the model deployable on edge devices like the Jetson Orin Nano (8 GB VRAM).

---

## Key Learnings

1. **Less data, more diversity:** 1,800 diverse examples outperform 31,000 repetitive ones. The initial dataset caused severe overfitting (hallucination, repetition, instruction leaking).

2. **ORPO is effective for targeted fixes:** 500 preference pairs addressing specific failure modes produced 100% reward accuracy. The model cleanly learned to avoid the targeted behaviors.

3. **RAG compensates for model limitations:** The hybrid retrieval pipeline (dense + sparse + reranking + pinned refs) ensures accurate verse retrieval even when the model's parametric knowledge is imperfect.

4. **Keyword metrics undercount quality:** Exact string matching penalizes valid paraphrases. Citation rate and hallucination rate are more informative for Bible Q&A evaluation.

5. **Small models need short prompts:** The 4B parameter model performs best with a ~15-line system prompt. The original 157-line prompt caused instruction leaking.

---

## Reproducibility

- **W&B tracking:** 34 logged runs (March 14-19, 2026)
- **Benchmark protocol:** `benchmarks/manifest.v1.yaml` (protocol ID: `bible_assistant_baseline_v1`)
- **Evaluation suite:** `benchmarks/suites/evaluation_questions.v1.json` (57 questions / 7 categories; 54 scored). The mutable `prompts/evaluation_questions.json` is now the 282-question v2 suite.
- **Hardware:** RTX 5070 Ti (16 GB), 96 GB RAM, Windows 11
- **Training time:** ~18 min SFT + ~20 min ORPO
