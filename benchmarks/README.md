# Benchmarks

- **`manifest.v1.yaml`** — Protocol metadata for `bible_assistant_baseline_v1` (question suite path, judge dimensions, reproducibility notes).
- **Question suite:** `prompts/evaluation_questions.json` (not duplicated here).

When you materially change the suite or rubric, add **`manifest.v2.yaml`** (or bump `protocol_version` and `protocol_id`) so historical JSON scores stay interpretable.

**Runbook:** [docs/BENCHMARK_PROTOCOL.md](../docs/BENCHMARK_PROTOCOL.md)
