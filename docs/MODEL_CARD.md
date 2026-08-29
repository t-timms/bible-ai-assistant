# Model Card: Bible AI Assistant — v2-4b (Qwen3.5-4B SFT)

> **Status: interim checkpoint (2026-08-29).** This is the SFT-only stage of a
> larger pipeline. It is a real improvement over the shipped v1 model on the core
> retrieval-grounded task, but it has a known regression on open-ended thematic
> answers (see *Evaluation* and *Limitations*). A v3 with teacher-distilled
> answers and a GRPO faithfulness stage is planned. Published now for
> transparency and reproducibility, not as a finished release.

---

## Model Summary

| Field | Value |
|-------|-------|
| **Model name** | Bible AI Assistant v2-4b |
| **Base model** | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (rev `851bf6e8`) — hybrid Gated-DeltaNet + attention, 32 layers, ~4.2 B params |
| **Fine-tuning** | Supervised fine-tuning only (LoRA, bf16), 1 epoch on 55,570 examples |
| **Preference / RL stage** | none yet (planned for v3) |
| **Primary language** | English |
| **License (code)** | MIT |
| **License (weights)** | Apache-2.0 (inherits from the Qwen3.5 base) |
| **Task** | Retrieval-grounded conversational Bible Q&A |
| **Serving** | `transformers`; vLLM (with a one-line registry entry); **GGUF via current llama.cpp** (`convert_hf_to_gguf.py --no-mtp`, verified with `llama-server`). Ollama 0.33.x's *bundled* llama.cpp is still too old for the `qwen35` arch — works once it updates, or run llama.cpp directly. |

---

## Model Description

A conversational model fine-tuned on Qwen3.5-4B for answering Bible questions as
part of a Retrieval-Augmented Generation (RAG) pipeline: a local hybrid retriever
(dense + BM25 + reranker + pinned references) fetches relevant passages and passes
them to the model as context; the model answers from that context.

**What v2-4b adds over the v1 shipped model:**

- Verbatim verse recall from provided context: **58 % → 76.5 %** (protocol v3).
- Citation rate: **88 % → 98.9 %** — it almost always cites a real reference.
- Hallucination stays near-zero (**2.3 %**).
- New behaviours from the v2 dataset: pastoral-triage / crisis escalation
  (points to a pastor or a crisis line rather than counselling), calibrated
  abstention on non-biblical sayings ("*'God helps those who help themselves' is
  not a Bible verse*"), multi-passage character syntheses, and cross-reference
  reasoning.

**What regressed** (see *Evaluation*): open-ended thematic questions ("What is
the context of Psalm 23?", "Who is Jesus?") get a templated verse list instead of
an explanation. One epoch of SFT on a template-heavy dataset taught the answer
*format* rather than the *skill*.

---

## Training Data

The v2 dataset engine (`training/build_dataset_v2.py`) produced **56,022 examples**
(55,570 after a length filter), fully provenance-tracked in the sidecar manifest
(per-source SHA + license). It is **decontaminated** against every question in the
frozen v3 evaluation suite (`scripts/check_train_eval_overlap.py` — zero overlap).

| Bucket | Count | Source / license |
|---|---|---|
| 8 scripture-citation categories (verse/passage recall, reverse lookup, near-miss guard, cross-reference chains, topical collections, chapter context, translation-specific) | 35,604 | 6 public-domain translations (KJV/ASV/WEB/DARBY/YLT/BBE), TSK cross-references (CC-BY, openbible.info) |
| `grounded_exegesis` — verse + commentary in context → grounded interpretation | 7,000 | Matthew Henry's Commentary on the Whole Bible (CC0 / public domain) |
| `general_blend` — general instruction / reasoning replay (catastrophic-forgetting guard) | 12,996 | [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) (Apache-2.0), `<think>` traces stripped |
| `pastoral_triage` — escalation, tradition-aware framing, calibrated abstention | 352 | hand-authored (aligned to FMG-Bench's rubric dimensions, not its held-out scenarios) |
| inherited v1 general / meta / refusal pools | 70 | project |

No proprietary, personal, or commercially licensed data was used.

---

## Training Procedure

| Parameter | Value |
|-----------|-------|
| Method | LoRA (r=32, α=64, dropout 0.05), bf16 — fully unquantized |
| Target modules | q/k/v/o/gate/up/down proj |
| Epochs | 1 |
| Effective batch | 16 (per-device 2 × grad-accum 8) |
| Sequence length | 1280 (fixed pad; data token p99 ≈ 1560) |
| LR / schedule | 2e-4, cosine, 3 % warmup |
| Loss masking | completion-only (mask through the assistant-start marker) |
| Hardware | 1 × RTX 5070 Ti (16 GB), ~10.4 h |
| Final eval loss | 0.2515 → **0.2138** (monotonic over all 70 evals, no overfit) |

Config: `training/config.v2-4b.yaml`. Script: `training/train_unsloth.py`.

---

## Evaluation

**Protocol v3** (`bible_assistant_baseline_v3`, 282 questions, sha-pinned suite),
keyword/verification metrics, greedy decode, seed 42. Served via a local
transformers `/v1/chat/completions` wrapper behind the RAG server.

| Category | N | Verse acc (exact) | Fuzzy mean | Hallucination | Citation |
|---|---|---|---|---|---|
| verse_lookup | 102 | **76.5 %** | 0.65 | 2.9 % | 100 % |
| cross_reference | 30 | 0 %\* | 0.40 | 3.3 % | 100 % |
| context | 30 | 0 %\* | 0.23 | 0 % | 93 % |
| character | 35 | 0 %\* | 0.20 | 2.9 % | 97 % |
| topical | 58 | 0 %\* | 0.20 | 1.7 % | 100 % |
| theological_reliability | 8 | 0 %\* | 0.15 | 0 % | 100 % |
| **Overall** | 266 | **29.3 %** | **0.40** | **2.3 %** | **98.9 %** |

\* Protocol v3 `verse_accuracy` scores "quoted *the one expected verse* verbatim."
Character / topical / context / theological questions have no single canonical
verse answer, so a good synthesised answer scores 0 on this metric. The fuzzy
column and a judge pass score them fairly; the judge run is pending.

**Head-to-head vs. v1** (same protocol, same day): v2 is **+18.5 pp** on
verse-lookup exact and **+11 pp** on citation rate, but **−0.087** on overall
fuzzy mean — the lightly-tuned v1's thematic answers are closer to the expected
natural answers than v2's templated ones. Full breakdown and diagnosis in
[`docs/MODEL_COMPARISON.md`](MODEL_COMPARISON.md). Raw JSONs under
`docs/benchmark_runs/`.

External benchmarks (FMG-Bench, FaithBench) are planned as honest calibration for
v3, not as win targets — they test theological reasoning, a harder and different
task than RAG verse-citation.

---

## Intended Use

**Appropriate:** personal Bible study, verse lookup, sermon-prep passage finding,
devotional Q&A, educational exploration of biblical narrative and history — always
with the RAG pipeline running.

**Not intended for:** medical / legal / financial advice; counselling or pastoral
care (the model is trained to *redirect* these to a pastor or crisis line, not to
handle them); authoritative theological decisions; unsupervised or at-scale
deployment. It has not been adversarially red-teamed.

---

## Limitations

- **Thematic-answer regression.** Open-ended "explain / who is / what is the
  context of" questions currently get a templated verse list rather than a
  synthesised answer. This is a dataset issue (template-heavy answers), targeted
  for v3.
- **RAG dependency.** Reliable verse accuracy requires the retriever + index
  running. Without context the model falls back to parametric memory, which is
  less reliable for exact citation. Always verify cited verses against a Bible.
- **Ollama not yet.** GGUF quants exist (F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M) and work in
  current `llama.cpp` / recent LM Studio, but Ollama 0.33.x's bundled llama.cpp is
  too old for the `qwen35` arch — use once Ollama updates, or run llama.cpp
  directly. Conversion requires `--no-mtp` (the base config's
  `mtp_num_hidden_layers: 1` otherwise makes the converter expect an MTP head this
  fine-tune doesn't carry).
- **Sequence length.** Trained at 1280 tokens; longer inputs are truncated.
- **English only.**
- **Not a finished release.** SFT-only; the preference and RL stages that the
  pipeline is designed around have not run.

---

## Bias and Fairness

- **Canon / translation.** Training scripture is the Protestant canon across six
  public-domain English translations; no Deuterocanonical books. Matthew Henry's
  commentary reflects an 18th-century Reformed Protestant perspective.
- **Interpretive lean.** The `pastoral_triage` and `grounded_exegesis` data
  deliberately model *tradition-aware* framing ("faithful Christians differ on
  …") for disputed questions, but the underlying sources still lean evangelical /
  Reformed Protestant.
- **Base-model bias.** Inherits any biases in Qwen3.5-4B; not audited for a Bible
  study context.
- **Hand-authored data.** The `pastoral_triage` pairs reflect the developer's
  judgment about safe escalation and neutral framing.

Apply critical judgment to outputs, especially on contested theological questions.

---

## License

- **Code:** [MIT](../LICENSE).
- **Weights:** Apache-2.0, inherited from Qwen3.5-4B. Review before commercial use.
- **Bible translations:** public domain worldwide.
- **Matthew Henry's Commentary:** CC0 / public domain.
- **smoltalk2:** Apache-2.0 for its new subsets; inherited subsets keep upstream
  licenses (see the smoltalk2 dataset card).

---

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

---

## Additional Resources

- [`docs/MODEL_COMPARISON.md`](MODEL_COMPARISON.md) — v2-4b vs. v1, protocol v3, full breakdown
- [`docs/V2_EXECUTION_PLAN.md`](V2_EXECUTION_PLAN.md) — the pipeline and the v3 plan
- [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) — protocol v3 definition
- `docs/benchmark_runs/` — machine-readable eval results
- `training/build_dataset_v2.py` — the v2 dataset engine
- `training/config.v2-4b.yaml`, `training/train_unsloth.py` — SFT config + script
