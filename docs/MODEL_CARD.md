# Model Card: Bible AI Assistant — v3.2 (Qwen3.5-4B)

> **Status: shipped (2026-09-05).** This is the fourth checkpoint in the project's
> iteration line and the first to clear every acceptance gate. Two prior
> iterations (v3-SFT, v3.1) were built and evaluated but held back — see
> *Version history* — before this one's fixes produced a statistically real
> improvement. HF repos:
> [`Ttimms/Bible-Assistant-Qwen3.5-4B-v3.2`](https://huggingface.co/Ttimms/Bible-Assistant-Qwen3.5-4B-v3.2) ·
> [`…-v3.2-GGUF`](https://huggingface.co/Ttimms/Bible-Assistant-Qwen3.5-4B-v3.2-GGUF).

---

## Version history

This model went through three real iterations before shipping — the earlier two
were built, evaluated, and **held back** because they didn't clear the bar. Kept
here rather than quietly skipped, because how a result was reached matters as
much as the result:

| Version | Change | Outcome |
|---|---|---|
| v2 | Initial SFT, 56k examples | Shipped. Strong verse recall, weak on open-ended synthesis (character/context/topical questions). |
| v3-SFT | Teacher-distilled synthesis answers | **Held.** Synthesis categories improved ~2x but still plateaued around 0.37 fuzzy mean — short of the bar. |
| v3.1 | Expanded thematic-synthesis dataset (103 question shapes, live-RAG-retrieved context) | **Held.** Flat vs v3-SFT (within 0.008) — inside the eval metric's own noise floor, not a real result either way. |
| **v3.2** | RAFT-style distractor-discrimination prompt + retrieval-depth fix (top_k 5→8) + a short DMT-style continued fine-tune from the v3.1 adapter | **Shipped.** Statistically real improvement over v3.1 (paired bootstrap +0.014, 95% CI excludes 0), on a metric built to actually be able to tell. |

The v3.1 plateau turned out to be a diagnosis problem, not a data problem: the
metric used to judge it (best-matching-sentence overlap) couldn't reliably rank
candidates within ~0.01 of each other. v3.2's three fixes were built after
root-causing that, then validated with a new cross-encoder metric built for
exactly this comparison. Full detail: [V3_STATUS.md](V3_STATUS.md).

## Model Summary

| Field | Value |
|-------|-------|
| Base model | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (rev `851bf6e8`) — hybrid Gated-DeltaNet + attention, 32 layers, ~4.2 B params |
| Fine-tuning | Continued LoRA fine-tune (r=32/α=64) from the v3.1 adapter, bf16, 1 epoch, 4,785 examples (50% new thematic data, 50% rehearsal), seq 1280, lr 5e-5 |
| Preference / RL stage | none (GRPO probe on this line was inert; not pursued further) |
| Language | English |
| License (weights) | Apache-2.0 (inherits from the base) |
| Task | Retrieval-grounded conversational Bible Q&A |
| Serving | `transformers`; vLLM; **GGUF** via current `llama.cpp` (see the [GGUF repo](https://huggingface.co/Ttimms/Bible-Assistant-Qwen3.5-4B-v3.2-GGUF)) |

## What it's for

A conversational model for answering Bible questions **as part of a RAG pipeline**: a
retriever fetches relevant passages and passes them as context; the model answers from that
context. It is not designed to be used context-free.

**Appropriate:** personal Bible study, verse lookup, sermon-prep passage finding, devotional
Q&A, educational exploration — with retrieval running.

**Not for:** medical / legal / financial advice; counselling or pastoral care (the model is
trained to *redirect* these to a pastor or crisis line); authoritative theological decisions;
unsupervised or at-scale deployment. Not adversarially red-teamed.

## Training

**Stage 1 (inherited from v3.1):** teacher-distilled (Qwen3-14B) synthesis answers over
character / context / cross-reference / topical / theological questions, live-RAG-retrieved
context, full SFT from the Qwen3.5-4B base.

**Stage 2 (this checkpoint, DMT-style continued fine-tune):** rather than mix a new data
slice into a fresh full epoch — which the literature on multi-task SFT flags as liable to
inject conflicting gradient signal when generation modes diverge (long-form synthesis vs.
short citation-drill) — v3.2 continues training *from the v3.1 adapter* on a short, targeted
second stage: 4,785 examples, ~50% newly regenerated `thematic_qa` (RAFT-style prompt fix
applied) and ~50% a stratified rehearsal slice of every other v3.1 category, 1 epoch, LoRA
r=32/α=64, lr 5e-5 (vs. the original SFT's 2e-4 — a nudge, not a retrain), bf16, on a single
RTX 5070 Ti (16 GB).

The two upstream fixes that made this stage worth running:

- **RAFT-style distractor discrimination** — the retrieved context for synthesis questions
  often contains a verse sharing vocabulary with the question without addressing it. Added an
  explicit note to the distillation prompt instructing the teacher to read every verse before
  writing, rather than default to the first vocabulary match. Validated on the exact failure
  case before regenerating the full set.
- **Retrieval-depth fix** — measured a real train/serve mismatch (training retrieved at
  top_k 7-9; serving used top_k 5) and a real recall gap between depth 5 and 8 for exactly the
  weakest generation categories. `rag_top_k` raised 5 → 8.

## Evaluation

Two protocols, reported together — neither replaces the other:

- **Fuzzy** (protocol v4): best-matching-sentence character overlap against the expected
  answer. The project's original acceptance-gate metric.
- **Semantic** (protocol v5, new for this release): a cross-encoder (`bge-reranker-v2-m3`)
  score over the *full* response vs. the *full* expected answer. Built after auditing fuzzy on
  three close candidates (v3-SFT/v3.1/v3.2 landed within 0.008 of each other on it) and finding
  it rewards sentence-bundling luck over content correctness — it could not rank those three
  fairly. Semantic can.

282-question suite, sha-pinned, greedy decode, seed 42, RAG context enabled.

| Category | N | Verse acc (exact) | Fuzzy | Semantic | Citation | Hallucination |
|---|--:|--:|--:|--:|--:|--:|
| verse_quote | 66 | **80.3 %** | 0.788 | 0.875 | 100 % | 1.5 % |
| verse_exposition | 36 | 47.2 %\* | 0.518 | 0.959 | 100 % | 0 % |
| cross_reference | 30 | 0 %\* | 0.429 | 0.997 | 100 % | 3.3 % |
| context | 30 | 0 %\* | 0.378 | 0.989 | 93.3 % | 3.3 % |
| character | 35 | 0 %\* | 0.382 | 0.971 | 100 % | 0 % |
| topical | 58 | 0 %\* | 0.371 | 0.941 | 98.3 % | 3.4 % |
| theological_reliability | 8 | 0 %\* | 0.310 | 0.991 | 100 % | 0 % |
| **Overall (expo-excl.)** | 230 | — | **0.500** | **0.942** | **98.9 %** | **1.9 %** |

\* `verse_accuracy` (exact) scores "quoted *the one expected verse* verbatim." Character /
topical / context / theological questions have no single canonical verse answer, so a
correct synthesised answer scores 0 on this column by design — semantic and fuzzy score
them fairly.

**Versus the prior checkpoint (v3.1, same protocol, same suite):** semantic 0.928 → **0.942**
(paired bootstrap +0.014, 95% CI [+0.004, +0.026] — excludes 0, a real improvement, not noise).
Verse-quote exact recall held/improved (78.8% → 80.3%) rather than trading off against the
synthesis-category gains.

**Versus external models, same suite / same RAG stack (12 models tested — the closest thing
to a "Bible LLM benchmark" that currently exists; none was found):**

- **Within its size class (≤4.5B), v3.2 leads every model tested on every metric** — semantic,
  fuzzy, verse-quote exactness, citation, and hallucination, against the two other
  bible-tuned ~4B models found (`rhemabible/GemmaBible`, `rhemabible/BibleAI`) and a
  base-Phi-3-mini repo mislabeled as bible-tuned.
- **Against larger models** (a 12B dedicated-bible fine-tune, a 14B general-instruct model):
  v3.2 ranks #1 of 12 on fuzzy but #3 of 12 on the semantic metric — both larger models score
  marginally higher there. But semantic alone rewards topical correctness, not verbatim
  accuracy or citation grounding; on those three metrics — the ones this task actually depends
  on — v3.2 beats both larger models decisively (80.3% vs. 55.3%/72.0% quote-exact; 98.9% vs.
  96.2%/98.1% citation; 1.9% vs. 4.5%/4.9% hallucination). Full table and methodology:
  [SOTA_EVAL.md](SOTA_EVAL.md).

**Claim, stated precisely:** best open model at RAG-grounded scripture Q&A *at its size*
(clean, no caveats) and *at the task* against larger models too, on the metrics the task
depends on — not on a generic semantic-similarity score alone. Both halves are reported;
neither is hidden to make a cleaner headline.

## Limitations

- **`refusal` category is weak on the semantic metric** (0.504) — the cross-encoder compares
  full-text similarity, and refusal responses are short/templated in ways that don't compare
  well against a reference refusal even when the behavior itself is correct. Refusal is scored
  by presence of the correct redirect behavior in practice, not this metric; it's shown here
  for completeness, not as a defect.
- **RAG dependency** — reliable verse accuracy needs the retriever + index; without context the
  model falls back to parametric memory. Always verify cited verses against a Bible.
- **Sequence length** — trained at 1280 tokens; longer inputs truncate.
- **English only.**
- **No RL/preference stage** — a GRPO probe on this line produced no measurable change and was
  not pursued further.

## Bias

Protestant canon across public-domain English translations (no Deuterocanonical books). The
`pastoral_triage` data deliberately models tradition-aware framing ("faithful Christians
differ on …") for disputed questions, but the sources still lean evangelical / Reformed
Protestant. Inherits any biases in Qwen3.5-4B. Apply critical judgment, especially on
contested theological questions.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

m = "Ttimms/Bible-Assistant-Qwen3.5-4B-v3.2"
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)

# The model expects a retrieval-augmented prompt: verses in a Context block, then the question.
user = (
    "Context:\n- **John 3:16**: For God so loved the world, that he gave his only begotten "
    "Son, that whosoever believeth in him should not perish, but have everlasting life.\n\n"
    "Q: What does John 3:16 say?"
)
msgs = [
    {"role": "system", "content": "You are a Bible AI assistant. Answer questions about Scripture accurately and conversationally."},
    {"role": "user", "content": user},
]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
out = model.generate(ids, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

The full RAG server (retriever + reranker + citation verification) is in the
[project repo](https://github.com/t-timms/bible-ai-assistant).

## License

- Weights: Apache-2.0 (from Qwen3.5-4B).
- Code: MIT (project repo).
- Bible translations: public domain.

## Citation

```bibtex
@misc{bible-ai-assistant-2026,
  title        = {Bible AI Assistant: A RAG-Grounded Bible Q\&A Model Fine-tuned on Qwen3.5-4B},
  author       = {Tremayne Timms},
  year         = {2026},
  howpublished = {GitHub},
  url          = {https://github.com/t-timms/bible-ai-assistant}
}
```
