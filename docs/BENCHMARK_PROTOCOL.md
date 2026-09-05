# Benchmark & A/B protocol (Bible Assistant)

There is **no single global standard** for Bible RAG chatbots, and — checked directly, not
assumed — no reproducible, submittable external benchmark exists for this domain at all. The
closest real thing is The Gospel Coalition's "AI Christian Benchmark": 7 hand-graded theological
questions scored against frontier general-purpose chatbots (DeepSeek R1, GPT-4o, etc.), with no
published rubric and no way for a third party to submit or self-score a model against it. It
isn't a target this project (or any locally-served fine-tuned model) can be benchmarked on.
(Also checked 2026-09-05: [biblebench.vercel.app](https://biblebench.vercel.app), sponsored by
Rhema — the same org behind the `rhemabible/*` models this project scores against in
`docs/SOTA_EVAL.md` — is pre-launch with no populated leaderboard as of that date; worth
revisiting once it has real rankings, not actionable now.) This project's own versioned
internal protocol, below, is the benchmark — and its external-comparator sweep
(`docs/SOTA_EVAL.md`) is the closest thing to a leaderboard that currently exists for this task:

1. **Frozen suite** — `benchmarks/suites/evaluation_questions.v3.json` (protocol v4/v5; sha256-pinned in the manifest; `run_benchmark.py` fails fast on mismatch). 282 questions, same text/answers as the v2 snapshot, across **9 categories** after protocol v4 split `verse_lookup` → `verse_quote` + `verse_exposition`. Includes a small `theological_reliability` category *inspired by* (not affiliated with, not scored against) TGC's question framing. `evaluate.py` reads the mutable `prompts/evaluation_questions.json` (the editing surface); `run_benchmark.py` verifies the pinned snapshot's sha256, and `scripts/run_external_baselines.sh` promotes the pinned snapshot into that file (backup + hard normalized-sha256 check) before a run.
2. **Version tag** — `benchmarks/manifest.vN.yaml` defines `protocol_id`. When you change questions, judge rubric, or metric meaning, **bump the manifest** (new `manifest.vN.yaml` + new `protocol_id`) so scores are comparable across time. Current: **v5** (`bible_assistant_baseline_v5`) — see `manifest.v5.yaml`'s `changes_from_v4`: adds `verse_accuracy_semantic`, a cross-encoder (`bge-reranker-v2-m3`) score over the full response vs. the full expected answer, built after auditing v4's fuzzy metric on three close checkpoints (v3-SFT/v3.1/v3.2 landed within 0.008 of each other — inside that metric's noise floor) and finding it rewards sentence-bundling luck over content correctness. Same suite, same fuzzy/citation/hallucination metrics as v4 — semantic is additive, not a replacement, and is rescore-only (no live CLI path; `scripts/rescore_v5.py`). Prior: **v4** (`manifest.v4.yaml`'s `changes_from_v3`: `verse_lookup` (102) → `verse_quote` (66, exact-match headline) + `verse_exposition` (36, fuzzy-pass headline — an explanation of a verse is a pass at exact-match 0, same reasoning protocol v3 applied to `refusal`); overall fuzzy mean reported all-in **and** exposition-excluded; LLM-judge dropped to `qwen3:8b` and made calibration-only, since the v3 `qwen3.5:27b` does not fit 16 GB — see Troubleshooting) and **v3** (`manifest.v3.yaml`'s `changes_from_v2` — frozen sha-pinned suites, pinned metric constants, a disclosed ~100/282 train/eval contamination in the pre-v3 expanded dataset, recorded decoding params). `verse_lookup` (v1–v3) vs. `verse_quote`/`verse_exposition` (v4/v5) are not comparable; other categories stay comparable v3↔v4↔v5. v1/v2/v3 scores are **not** comparable. Older manifests are kept, not edited in place.
3. **Two tiers** — **keyword** (fast CI / iteration; the primary tier) and **judge** (calibration only since v4 — see Troubleshooting for why the 27B judge was dropped). Keyword mode reports both the original exact-substring `verse_accuracy` and a `verse_accuracy_fuzzy` metric — the exact metric penalizes valid paraphrase (e.g. "his one and only Son" vs. "his only born Son" scores 0 despite both being faithful), which is a known, previously undiagnosed weakness; report both, don't drop the old one, since some run history only has it.
4. **Real citation verification** — `check_hallucination` (via `rag/verification.py`) now checks that a cited chapter:verse actually exists in the indexed Bible text, not just that the book name is real. A real book with a fabricated verse number (e.g. "1 Corinthians 47:99") now counts as a hallucination; it silently passed before.
5. **Artifacts** — JSON with `ollama_model`, `benchmark_protocol_id`, and per-item results for diffing.

## Quick start

**Prerequisites:** RAG server up, Ollama running, target model created (`ollama list`).

```powershell
cd bible-ai-assistant

# Fast pass (keyword metrics) — the primary tier
python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo

# Calibration pass (judge; v4 default qwen3:8b — the v3 qwen3.5:27b does NOT fit 16 GB VRAM,
# see Troubleshooting). Not a pass/fail gate.
python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo --judge --judge-model qwen3:8b
```

**Move a completed protocol-v3 keyword run to v4 without re-generating** (deterministic
re-bucket of `verse_lookup`, aggregation reused from `evaluate.py`):

```bash
python scripts/rescore_v4.py                 # -> docs/benchmark_runs/20260902_*_v4keyword.json + a table
python scripts/exposition_sidebyside.py      # the 36 verse_exposition items, v2 vs v3, for a manual read
```

A/B (e.g. Q4 vs F16):

```powershell
python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo --judge
python scripts/run_benchmark.py --label orpo-f16 --ollama-model bible-assistant-orpo-f16 --judge
python scripts/compare_benchmark_runs.py docs/benchmark_runs/<file_a>.json docs/benchmark_runs/<file_b>.json
```

## Manual `evaluate.py` (same protocol)

```powershell
python training/evaluate.py --protocol-id bible_assistant_baseline_v4 --ollama-model bible-assistant-orpo
python training/evaluate.py --judge --judge-model qwen3:8b --protocol-id bible_assistant_baseline_v4 --ollama-model bible-assistant-orpo --model-tag orpo-q4
```

## External calibration — FMG-Bench

`scripts/fmg_bench.py` runs the **Faith & Moral Guidance Benchmark** (`FideAI/fmg-bench`,
CC-BY-4.0, arXiv 2608.12324) — 120 base scenarios + 37 perturbations, rubric + per-scenario
dimension weights, **no hidden-test leaderboard** (fully self-scorable). It measures a
*different* task than this project's suite: theological triage, tradition-aware comparison,
preference fidelity, grounding discipline, and escalation boundaries. **Report it as honest
calibration, never as a pass/fail gate** — scores are not comparable to the protocol-v4
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

## SOTA evaluation — external comparators (`docs/SOTA_EVAL.md`)

The protocol-v4 keyword suite, run through the **unchanged** RAG stack against open comparator
models, is how the "best *open* model at RAG-grounded scripture Q&A, size-independent" claim is
established or refuted. Claim scope (from `CLAUDE.md`): best open model *at this niche task*
(nobody large optimizes for it) **and** SOTA for the 16 GB consumer-Blackwell class — **never**
a claim about beating frontier models on unconstrained hardware.

- `benchmarks/external_comparators.yaml` — 8 comparators: `sleepdeprived3/Christian-Bible-Expert-v2.0`
  8B + 12B, `nbeerbower/llama-3-bible-dpo-8B`, `Phora68/bible-study-phi3-mini` and
  `rhemabible/BibleAI` (size-matched to our 4B), Qwen3-8B/14B/32B instruct (the "does niche
  tuning beat a good general model, and how far up the size ladder does our 4B hold?" control).
- `scripts/run_external_baselines.sh` (GPU, ~3–4 h) — promotes the v4 suite, pulls / `ollama
  create`s each model, runs protocol-v4 keyword. `--only <key> --smoke-first` validates one
  model + its chat template first.
- `scripts/sota_scoreboard.py` — reads ours (`rescore_v4.py`) + comparators → ranked
  head-to-head (verse_quote exact, exposition fuzzy, overall fuzzy expo-excluded, citation,
  hallucination; Wilson CIs; paired McNemar vs. our best) → rewrites `docs/SOTA_EVAL.md` with a
  scoped verdict.

```bash
python scripts/rescore_v4.py                                        # ours, under v4 (no GPU)
bash scripts/run_external_baselines.sh --only bible-study-phi3-mini --smoke-first   # validate (GPU)
bash scripts/run_external_baselines.sh                              # full sweep, tmux (GPU)
python scripts/sota_scoreboard.py                                   # the board
```

## Evolving the benchmark (as you improve)

| Change | Action |
|--------|--------|
| Add/edit questions | Edit `prompts/evaluation_questions.json`, then re-snapshot to `benchmarks/suites/evaluation_questions.vN.json` and pin its sha256 in a **new** `manifest.vN.yaml` with a new `protocol_id` |
| Change judge prompt | New `protocol_id`; document in manifest |
| Change RAG retrieval | Document in run notes; major pipeline changes → new protocol or disclaimer |
| New metric | Extend `evaluate.py` + manifest; version bump |

## Troubleshooting

### Judge times out on every question (`LLM judge unavailable — all endpoints failed … timed out`)

The v3 default judge **`qwen3.5:27b`** (Q4_K_M, ~17 GB) does **not** fit the 16 GB VRAM
budget on this class of GPU. Ollama CPU-offloads almost all of it, and one full rubric call
measured **333.7 s** on an idle GPU (2026-09-02) — well past `evaluate.py`'s 180 s HTTP
timeout, so the run dies on question 1. This is not a config bug; the model is too big.
**Use `--judge-model qwen3:8b`** (fits VRAM, runs on-GPU, ~10 s/call). Since v4 the judge is
calibration-only and not a gate — the keyword tier is authoritative.

### Judge scores all zero (`F=0 C=0 ...`) in JSON `judge_scores.error`

- **Default judge URL** is `http://127.0.0.1:11434/v1/chat/completions` (avoids Windows `localhost` → IPv6 `::1` when Ollama listens on IPv4 only). Override with `--judge-url` if your Ollama uses another host/port.
- **Fallback chain:** OpenAI-compatible `POST`, then **`/api/chat`**, then **`/api/generate`** on the same origin. **HTTP proxy bypass:** judge calls use `trust_env=False` so `HTTP_PROXY` cannot steal `localhost` requests (a common cause of bogus **404**).
- **Verify Ollama:** `ollama list` includes the judge model (v4 default **`qwen3:8b`**; pass `evaluate.py --judge-model <name>` to change). Then `curl http://127.0.0.1:11434/api/tags`.
- Re-run judge eval after updating `training/evaluate.py`.

### `compare_benchmark_runs` “Invalid argument”

Use **real filenames**, not placeholders:

```powershell
python scripts/compare_benchmark_runs.py docs/benchmark_runs/20260320_orpo-q4_judge.json docs/benchmark_runs/20260320_orpo-f16_judge.json
```

## Files

| Path | Role |
|------|------|
| `benchmarks/manifest.v4.yaml` | Current protocol metadata, sha-pinned suite, checklist (v1/v2/v3 kept for history) |
| `benchmarks/suites/evaluation_questions.v3.json` | Frozen v4 suite (v2 questions, `verse_lookup` split) |
| `benchmarks/external_comparators.yaml` | The 8 open comparators for the SOTA board |
| `scripts/run_benchmark.py` | Writes timestamped JSON under `docs/benchmark_runs/` |
| `scripts/compare_benchmark_runs.py` | Side-by-side A/B summary |
| `scripts/make_v4_suite.py` | Regenerates the v4 suite snapshot from the v2 one (inert) |
| `scripts/rescore_v4.py` | Re-buckets a protocol-v3 keyword run to v4 (no re-generation) |
| `scripts/exposition_sidebyside.py` | v2-vs-v3 dump of the 36 `verse_exposition` items |
| `scripts/run_external_baselines.sh` / `scripts/_run_ext_eval.sh` | Run the external comparators (GPU) |
| `scripts/sota_scoreboard.py` | Builds `docs/SOTA_EVAL.md` — the ranked head-to-head + scoped verdict |
| `training/evaluate.py` | Core runner (`--ollama-model`, `--protocol-id`) |
| `scripts/fmg_bench.py` | FMG-Bench external-calibration adapter (`--dry-run` offline; real run needs a served model + judge) |
