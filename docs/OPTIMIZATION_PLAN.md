# Optimization Plan: Maximizing Domain Scores

This document outlines strategies to maximize evaluation scores across **faithfulness**, **citation**, **hallucination**, **helpfulness**, and **conciseness**. Apply iteratively after each evaluation run.

---

## 1. Training Data

| Lever | Current | Action | Impact |
|-------|---------|--------|--------|
| **Verse lookup coverage** | ~600 verses sampled | Increase `max_verse_examples`; ensure eval questions' verses appear in train data | ↑ faithfulness, citation |
| **RAG-style examples** | ~200 | Double; mirror exact "Context:\n... Q: ..." format from rag_server.py | ↑ faithfulness when context provided |
| **Weak category over-sampling** | None | After eval, add 2–3x examples for lowest-scoring categories | ↑ per-category scores |
| **Citation format** | Varied | Standardize: "**Reference**: quote" or "As X:Y says: …" in all examples | ↑ citation consistency |
| **Hallucination negative examples** | Few | Add explicit "I'm not sure" / "I don't have that verse" examples when context lacks answer | ↓ hallucination |
| **Conciseness** | Some long explanations | Trim verbose templates; add examples with 2–3 sentence explanations only | ↑ conciseness |
| **Deduplication** | Basic | Remove near-duplicate Q&A pairs; avoid overfitting on repeated verses | ↑ generalization |

**Concrete steps:**
- Run eval → identify worst categories (e.g. `cross_reference`, `context`)
- Add targeted examples in `dataset_builder.py` for those categories
- Rebuild: `python training/dataset_builder.py`
- Optionally: `--max-examples 3000` if data quality is high

---

## 2. Model & Training

| Lever | Current | Action | Impact |
|-------|---------|--------|--------|
| **LoRA rank** | r=16 | Try r=32 (more capacity) if not overfitting | ↑ capacity |
| **LoRA alpha** | 32 | Keep 2×r; if r=32 → alpha=64 | Scaling |
| **Epochs** | 2–3 | Monitor eval_loss; stop when it plateaus or rises | Avoid overfitting |
| **Learning rate** | 2e-4 | Try 1e-4 for stability; 3e-4 for faster convergence | Stability vs speed |
| **Max sequence length** | 2048 | Use 4096 if VRAM allows (less truncation) | ↑ long-context quality |
| **Batch size** | 2×8=16 | Increase grad_accum if OOM; effective batch 16–32 typical | ↑ stability |
| **Dropout** | 0.1–0.15 | Slightly higher (0.15) if overfitting | Regularization |
| **Config source** | config.yaml overrides | Ensure config.yaml is authoritative; remove script drift | Consistency |

**Concrete steps:**
- Edit `training/config.yaml` for hyperparameter changes
- Log to W&B; compare eval_loss across runs
- Use `load_best_model_at_end=True` (already set)

---

## 3. ORPO (Preference Alignment)

| Lever | Action | Impact |
|-------|--------|--------|
| **Preference data** | Build from SFT outputs: chosen = correct verse/citation, rejected = hallucination or wrong ref | ↑ faithfulness, ↓ hallucination |
| **ORPO beta** | Default 0.1; try 0.05 (gentler) or 0.2 (stronger) | Tuning |
| **Chunk size** | Match SFT max_seq_length | Consistency |

**Concrete steps:**
- Run `build_preference_data.py` with quality filters
- `train_orpo.py --sft-path models/...v8` after SFT stabilizes
- Eval v8 vs v8-orpo; keep if ORPO improves judge scores

---

## 4. RAG Pipeline

| Lever | Current | Action | Impact |
|-------|---------|--------|--------|
| **Top-K** | 5 | Try 7–10 for thematic questions; keep 5 for verse lookups | ↑ recall for topical |
| **Reranker** | bge-reranker-v2-m3 | Already strong; ensure it's loaded | Quality |
| **Hybrid candidates** | 20 | Increase to 30 for harder topical queries | ↑ recall |
| **Context format** | "Context:\n- **ref**: text" | Match training format exactly | ↑ faithfulness |
| **Verse vs passage** | Passage expansion for thematic | Verify passage collection built; expand for topical only | ↑ contextual answers |
| **Query prefix** | search_query: | Already correct for nomic-embed | - |

**Concrete steps:**
- `RAG_TOP_K=7` or `8` via env for thematic-heavy evals
- Ensure `rag/build_index.py` builds both `bible_verses` and `bible_passages` if using passages
- Log retrieval quality: are expected verses in top-K for eval questions?

---

## 5. System Prompt

| Lever | Current | Action | Impact |
|-------|---------|--------|--------|
| **Length** | Short, ~15 lines | Keep short; add one line: "When Context is provided, use it. Do not claim you lack data." | ↑ faithfulness |
| **Citation rule** | "cite the reference" | Make explicit: "Quote verse text, then cite reference (e.g. John 3:16)." | ↑ citation |
| **Conciseness** | "Say your piece once" | Reinforce: "No repetition. No filler. 2–4 sentences for verse lookups." | ↑ conciseness |
| **Hallucination** | "Never fabricate" | Add: "If Context does not contain the answer, say so. Do not invent verses." | ↓ hallucination |

**Concrete steps:**
- Edit `prompts/system_prompt.txt`
- Regenerate Modelfile: `python deployment/pc/generate_modelfile.py`
- `ollama create bible-assistant -f deployment/pc/Modelfile`

---

## 6. Inference (Ollama)

| Lever | Action | Impact |
|-------|--------|--------|
| **Temperature** | 0.1–0.3 for eval; 0.7 for chat | Low temp → more deterministic, fewer hallucinations |
| **Max tokens** | 2048 | Reduce to 512 for verse lookups (forces brevity) |
| **Top-p** | 0.9 default | Try 0.95 for diversity or 0.8 for focus |

**Concrete steps:**
- Set `temperature=0.2` in Modelfile PARAMETER for eval consistency
- RAG server passes through; Ollama uses Modelfile defaults

---

## 7. Evaluation

| Lever | Action | Impact |
|-------|--------|--------|
| **Eval set coverage** | 50+ questions across categories | Add questions for weak categories; balance verse_lookup, topical, cross_reference, context | Better signal |
| **Expected answers** | WEB translation | Ensure expected_answer matches your Bible source (bible_web.json) | Accurate keyword scoring |
| **Judge model** | Qwen3-Coder 32B | Use same judge for all runs; document judge version | Comparability |
| **Keyword vs judge** | Both | Use keyword for fast iteration; judge for final comparison | Efficiency |

**Concrete steps:**
- Audit `prompts/evaluation_questions.json`: do expected_answer strings match WEB?
- Add 5–10 questions for lowest-scoring categories
- Run judge eval for v8, v9, v8-orpo; record in `docs/evaluation_results_*.json`
- Use `scripts/leaderboard.py` to compare

---

## 8. Iteration Loop

```
1. Deploy current model (vN)
2. Run evaluate.py --judge --model-tag vN
3. Open evaluation_results_vN.json → find worst category + worst questions
4. Add training examples for those failure modes
5. Rebuild data → train vN+1
6. (Optional) Run ORPO on vN
7. Deploy vN+1 and vN-orpo, re-eval
8. Update leaderboard; compare
9. Repeat from step 2
```

---

## 9. External Benchmarks (Optional)

- **Great Commission Benchmark** — Ministry tasks, Gospel fidelity
- **Bible Trivia** — 1,290 questions; compare to ChatGPT/Gemini baselines
- **FaithJudge** — RAG hallucination benchmark

Run your model on these for external validation and portfolio differentiation.
