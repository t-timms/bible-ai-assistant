# Benchmark & A/B protocol (Bible Assistant)

There is **no single global standard** for Bible RAG chatbots. This project uses a **versioned internal protocol** that follows **common product ML practice**:

1. **Fixed suite** — `prompts/evaluation_questions.json` (schema checked in `tests/test_evaluation_questions.py`).
2. **Version tag** — `benchmarks/manifest.v1.yaml` defines `protocol_id: bible_assistant_baseline_v1`. When you change questions, judge rubric, or metric meaning, **bump the manifest** (e.g. `manifest.v2.yaml` + new `protocol_id`) so scores are comparable across time.
3. **Two tiers** — **keyword** (fast CI / iteration) and **judge** (heavier, closer to human rubric).
4. **Artifacts** — JSON with `ollama_model`, `benchmark_protocol_id`, and per-item results for diffing.

## Quick start

**Prerequisites:** RAG server up, Ollama running, target model created (`ollama list`).

```powershell
cd bible-ai-assistant

# Fast pass (keyword metrics)
python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo

# Thorough pass (requires judge in Ollama; default qwen3.5:27b — `ollama pull qwen3.5:27b` if missing)
python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo --judge
```

A/B (e.g. Q4 vs F16):

```powershell
python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo --judge
python scripts/run_benchmark.py --label orpo-f16 --ollama-model bible-assistant-orpo-f16 --judge
python scripts/compare_benchmark_runs.py docs/benchmark_runs/<file_a>.json docs/benchmark_runs/<file_b>.json
```

## Manual `evaluate.py` (same protocol)

```powershell
python training/evaluate.py --protocol-id bible_assistant_baseline_v1 --ollama-model bible-assistant-orpo
python training/evaluate.py --judge --protocol-id bible_assistant_baseline_v1 --ollama-model bible-assistant-orpo --model-tag orpo-q4
```

## Evolving the benchmark (as you improve)

| Change | Action |
|--------|--------|
| Add/edit questions | Edit `evaluation_questions.json`; consider **new** `protocol_id` if scores are not comparable |
| Change judge prompt | New `protocol_id`; document in manifest |
| Change RAG retrieval | Document in run notes; major pipeline changes → new protocol or disclaimer |
| New metric | Extend `evaluate.py` + manifest; version bump |

## Troubleshooting

### Judge scores all zero (`F=0 C=0 ...`) in JSON `judge_scores.error`

- **Default judge URL** is `http://127.0.0.1:11434/v1/chat/completions` (avoids Windows `localhost` → IPv6 `::1` when Ollama listens on IPv4 only). Override with `--judge-url` if your Ollama uses another host/port.
- **Fallback chain:** OpenAI-compatible `POST`, then **`/api/chat`**, then **`/api/generate`** on the same origin. **HTTP proxy bypass:** judge calls use `trust_env=False` so `HTTP_PROXY` cannot steal `localhost` requests (a common cause of bogus **404**).
- **Verify Ollama:** `ollama list` includes the judge model (repo default **`qwen3.5:27b`**). Override with `evaluate.py --judge-model <name>`. Then `curl http://127.0.0.1:11434/api/tags`.
- Re-run judge eval after updating `training/evaluate.py`.

### `compare_benchmark_runs` “Invalid argument”

Use **real filenames**, not placeholders:

```powershell
python scripts/compare_benchmark_runs.py docs/benchmark_runs/20260320_orpo-q4_judge.json docs/benchmark_runs/20260320_orpo-f16_judge.json
```

## Files

| Path | Role |
|------|------|
| `benchmarks/manifest.v1.yaml` | Protocol metadata and checklist |
| `scripts/run_benchmark.py` | Writes timestamped JSON under `docs/benchmark_runs/` |
| `scripts/compare_benchmark_runs.py` | Side-by-side A/B summary |
| `training/evaluate.py` | Core runner (`--ollama-model`, `--protocol-id`) |
