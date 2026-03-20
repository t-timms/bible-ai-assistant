# Evaluation & Score-Maximizing Checklist

Concise checklist for improving domain scores (faithfulness, citation, hallucination, helpfulness, conciseness).

## Phase 1: Deploy

- [ ] SFT training completes
- [ ] `python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v8`
- [ ] Convert to GGUF (f16 + optional q4_k_m) → `python deployment/pc/generate_modelfile.py [--gguf ...]` → `ollama create` (e.g. `bible-assistant-orpo` vs `bible-assistant-orpo-f16` for A/B — see `deployment/pc/README.md`)
- [ ] Start RAG: `uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081`

## Phase 2: Baseline eval

- [ ] `python training/evaluate.py --model-tag v8` (keyword)
- [ ] `python training/evaluate.py --judge --model-tag v8` (LLM-as-judge, if available)
- [ ] Save outputs to `docs/evaluation_results_v8.json`

## Phase 3: Analyze & improve

- [ ] Open `docs/evaluation_results_v8.json` → find weak categories and failure modes
- [ ] Add training examples for weak areas in `dataset_builder.py` / `train.json`
- [ ] Rebuild data: `python training/dataset_builder.py`
- [ ] Train v9: `python training/train_unsloth.py --run-name qwen3.5-4b-bible-John-v9`
- [ ] Optional ORPO: `python training/train_orpo.py --sft-path models/qwen3.5-4b-bible-John-v8`
- [ ] Merge/deploy v8-orpo and v9, re-run evaluation

## Phase 4: Compare

- [ ] Run evaluation for each variant with `--model-tag v8`, `--model-tag v9`, etc.
- [ ] Run leaderboard: `python scripts/leaderboard.py`
- [ ] **Versioned A/B:** `python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo --judge` (repeat for second variant) → `python scripts/compare_benchmark_runs.py <a.json> <b.json>` — see **[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)**

## Leaderboard

To see ranked scores across all model versions:

```bash
python scripts/leaderboard.py
```

Reads `docs/evaluation_results_*.json` and prints a ranked table (verse accuracy, citations, hallucinations for keyword mode; overall scores for llm-as-judge).
