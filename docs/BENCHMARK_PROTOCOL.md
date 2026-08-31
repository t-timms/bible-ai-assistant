# Benchmark & A/B protocol (Bible Assistant)

There is **no single global standard** for Bible RAG chatbots, and — checked directly, not
assumed — no reproducible, submittable external benchmark exists for this domain at all. The
closest real thing is The Gospel Coalition's "AI Christian Benchmark": 7 hand-graded theological
questions scored against frontier general-purpose chatbots (DeepSeek R1, GPT-4o, etc.), with no
published rubric and no way for a third party to submit or self-score a model against it. It
isn't a target this project (or any locally-served fine-tuned model) can be benchmarked on. This
project's own versioned internal protocol, below, is the benchmark:

1. **Frozen suite** — `benchmarks/suites/evaluation_questions.v2.json` (sha256-pinned in the manifest; `run_benchmark.py` fails fast on mismatch). 282 questions across 8 categories, including a small `theological_reliability` category *inspired by* (not affiliated with, not scored against) TGC's question framing. The mutable `prompts/evaluation_questions.json` is the editing surface and is **no longer referenced by any manifest** — historical results measured against it are not byte-reproducible.
2. **Version tag** — `benchmarks/manifest.vN.yaml` defines `protocol_id`. When you change questions, judge rubric, or metric meaning, **bump the manifest** (new `manifest.vN.yaml` + new `protocol_id`) so scores are comparable across time. Current: **v3** (`bible_assistant_baseline_v3`) — see `manifest.v3.yaml`'s `changes_from_v2` (frozen sha-pinned suites, pinned metric constants, a disclosed ~100/282 train/eval contamination in the pre-v3 expanded dataset, recorded decoding params). v1/v2/v3 scores are **not** comparable. Older manifests are kept, not edited in place.
3. **Two tiers** — **keyword** (fast CI / iteration) and **judge** (heavier, closer to human rubric). Keyword mode reports both the original exact-substring `verse_accuracy` and a `verse_accuracy_fuzzy` metric — the exact metric penalizes valid paraphrase (e.g. "his one and only Son" vs. "his only born Son" scores 0 despite both being faithful), which is a known, previously undiagnosed weakness; report both, don't drop the old one, since some run history only has it.
4. **Real citation verification** — `check_hallucination` (via `rag/verification.py`) now checks that a cited chapter:verse actually exists in the indexed Bible text, not just that the book name is real. A real book with a fabricated verse number (e.g. "1 Corinthians 47:99") now counts as a hallucination; it silently passed before.
5. **Artifacts** — JSON with `ollama_model`, `benchmark_protocol_id`, and per-item results for diffing.

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
python training/evaluate.py --protocol-id bible_assistant_baseline_v3 --ollama-model bible-assistant-orpo
python training/evaluate.py --judge --protocol-id bible_assistant_baseline_v3 --ollama-model bible-assistant-orpo --model-tag orpo-q4
```

## External calibration — FMG-Bench

`scripts/fmg_bench.py` runs the **Faith & Moral Guidance Benchmark** (`FideAI/fmg-bench`,
CC-BY-4.0, arXiv 2608.12324) — 120 base scenarios + 37 perturbations, rubric + per-scenario
dimension weights, **no hidden-test leaderboard** (fully self-scorable). It measures a
*different* task than this project's suite: theological triage, tradition-aware comparison,
preference fidelity, grounding discipline, and escalation boundaries. **Report it as honest
calibration, never as a pass/fail gate** — scores are not comparable to the protocol-v3
verse-citation numbers.

```bash
# offline: validates the whole pipeline with a stub judge, no model calls
python scripts/fmg_bench.py --dry-run --with-perturbations --out /tmp/fmg_dry.json

# real run (needs a served model + a judge, i.e. GPU)
python scripts/fmg_bench.py --with-perturbations --label bible-v3 \
    --model-url http://localhost:8081/v1/chat/completions --model bible-v3 \
    --judge-url http://127.0.0.1:11434/v1/chat/completions --judge-model qwen3:8b
```

Output JSON (in `docs/benchmark_runs/`) records the dataset sha256 + revision and reports:
weighted overall mean, per-dimension means, escalation recall + false-escalation rate
(Wilson 95% CIs), disallowed-failure-mode rate, and breakdowns by `family` / `triage_level`.

FaithBench (faithbench.com, the Christian-theology site — not the Vectara summarization
benchmark) is **not wired in**: research-preview leaderboard only, no public dataset, linked
repo 404s as of 2026-08-31. Revisit if a dataset is released.

## Evolving the benchmark (as you improve)

| Change | Action |
|--------|--------|
| Add/edit questions | Edit `prompts/evaluation_questions.json`, then re-snapshot to `benchmarks/suites/evaluation_questions.vN.json` and pin its sha256 in a **new** `manifest.vN.yaml` with a new `protocol_id` |
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
| `benchmarks/manifest.v3.yaml` | Current protocol metadata, sha-pinned suites, checklist (v1/v2 kept for history) |
| `scripts/run_benchmark.py` | Writes timestamped JSON under `docs/benchmark_runs/` |
| `scripts/compare_benchmark_runs.py` | Side-by-side A/B summary |
| `training/evaluate.py` | Core runner (`--ollama-model`, `--protocol-id`) |
| `scripts/fmg_bench.py` | FMG-Bench external-calibration adapter (`--dry-run` offline; real run needs a served model + judge) |
