# SOTA evaluation — RAG-grounded scripture Q&A (protocol v4)

> **Status: SCAFFOLD — no external comparator scored yet.**
> `scripts/sota_scoreboard.py` overwrites this file with the real ranked table
> once `scripts/rescore_v4.py` (ours) and `scripts/run_external_baselines.sh`
> (comparators) have run. Until then this documents the method and the claim rules.

## What "SOTA" means for this project

From `CLAUDE.md`: `Bible-Assistant`'s task (RAG-grounded scripture Q&A with verified
citations) is niche enough to aim for **best open model *at the task*,
size-independent** — nobody large optimizes for it. Two separable claims:

| Claim | Backed by |
|---|---|
| Best **open** model at RAG-grounded scripture Q&A | Beating the dedicated open bible fine-tunes **and** holding vs. larger general instruct models, on the same suite + same RAG stack, on closeness-to-expected while meeting the citation and hallucination gates. |
| SOTA for the **16 GB consumer-Blackwell class** | Runs in ≤16 GB VRAM (NVFP4 / GGUF), Blackwell-native, at the quality above. |

**Not claimed:** beating frontier models on unconstrained hardware. A frontier-API
row, if ever added, is a labelled ceiling — never counted as a peer.

## Method (identical for every row)

- **Suite:** `benchmarks/suites/evaluation_questions.v3.json` (282 Qs, sha-pinned in
  `benchmarks/manifest.v4.yaml`), promoted into `prompts/evaluation_questions.json`
  by the runner with a hard normalized-sha256 check.
- **Stack:** unchanged hybrid RAG — dense + BM25 + RRF + `bge-reranker-v2-m3` +
  per-verse citation verification (`rag/verification.py`). Same commit, same Chroma index.
- **Decoding:** greedy, seed 42.
- **Metrics (keyword, no judge — the 27B judge is infeasible on 16 GB, 333 s/call):**
  `verse_quote` exact-match · `verse_exposition` fuzzy pass@0.85 · overall fuzzy mean
  (reported all-in **and** exposition-excluded) · fuzzy pass-rate · citation rate ·
  hallucination rate. Rates carry Wilson 95% CIs; our-best vs each comparator gets a
  paired McNemar on the verse-accuracy outcome.
- **Provenance note:** our models' v4 numbers come from `scripts/rescore_v4.py`
  (re-bucketing the 2026-09-01 protocol-v3 keyword runs — same 282 Qs, same responses,
  deterministic split), comparators are fresh v4 runs. The numbers are identical to
  what a fresh v4 run of our models would produce, so the board is apples-to-apples.

## Comparator set (`benchmarks/external_comparators.yaml`)

| key | HF | B | group | license |
|---|---|--:|---|---|
| christian-bible-expert-12b | sleepdeprived3/Christian-Bible-Expert-v2.0-12B | 12 | dedicated_bible | apache-2.0 |
| christian-bible-expert-8b | sleepdeprived3/Baptist-Christian-Bible-Expert-v2.0-8B | 8 | dedicated_bible | other |
| llama3-bible-dpo-8b | nbeerbower/llama-3-bible-dpo-8B | 8 | dedicated_bible | llama-3 |
| bible-study-phi3-mini | Phora68/bible-study-phi3-mini | 3.8 | size_matched | mit |
| rhema-bibleai-gemma | rhemabible/BibleAI | ~4 | size_matched | apache-2.0 |
| qwen3-8b-instruct | Qwen/Qwen3-8B | 8 | general_instruct | apache-2.0 |
| qwen3-14b-instruct | Qwen/Qwen3-14B | 14 | general_instruct | apache-2.0 |
| qwen3-32b-instruct | Qwen/Qwen3-32B | 32 | general_instruct_large | apache-2.0 |

Rationale: the `sleepdeprived3` family is the most-downloaded dedicated bible-model
line on HF; `bible-study-phi3-mini` and `rhema BibleAI` are the closest in size to our
4B (the fair size-for-size rows); the Qwen3 instruct ladder is the "does niche tuning
even beat a good general model, and how far up the size ladder does our 4B hold?"
control. No rigorous RAG-grounded eval with citation verification + CIs is published
for any of them — that gap is the differentiator.

## Run order

```
# 1. ours, under v4 (no GPU) — also produces the Path-D decision numbers
python scripts/rescore_v4.py

# 2. comparators (GPU, ~3–4 h; validate one first)
bash scripts/run_external_baselines.sh --only bible-study-phi3-mini --smoke-first
bash scripts/run_external_baselines.sh          # full sweep, in tmux

# 3. the board
python scripts/sota_scoreboard.py               # rewrites this file
```

## Results

_pending — see Status at top._
