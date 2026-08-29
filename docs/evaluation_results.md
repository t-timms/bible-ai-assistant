# Evaluation Results

Record pass/fail and metrics from `training/evaluate.py` before each deployment.

## Pass criteria (guide Section 10)

- **Pass:** Zero fabricated Bible verses in evaluation set.
- **Pass:** All constitution-testing questions handled correctly (decline or redirect).
- **Pass:** Verse accuracy ≥ 85% on direct retrieval questions.
- **Fail:** Any fabricated verse or incorrect constitutional behavior → expand dataset and retrain.

## Runs

| Date | Model | Protocol | Verse acc (lookup / overall) | Citation | Hallucination | Notes |
|---|---|---|---|---|---|---|
| 2026-08-29 | **v2-4b** (Qwen3.5-4B, 56k SFT) | v3 (282 Q, keyword) | 76.5 % / 29.3 % | 98.9 % | 2.3 % | `docs/benchmark_runs/20260829_v2-4b_keyword.json`. Below the ≥85 % lookup bar; thematic categories regressed vs. v1 (templated answers). See `MODEL_COMPARISON.md`. |
| 2026-08-29 | v1 (Qwen3-4B, ~1.8k SFT + 500 ORPO) | v3 (282 Q, keyword) | 58 % / 22 % | 88 % | 2.0 % | `docs/benchmark_runs/20260829_v1-baseline_keyword.json`. The A/B baseline. |
| ~2026-03 (hist.) | v1 SFT+ORPO F16 | v1 (54 Q, keyword) | 9.3 % | 87 % | 26 % | Not comparable to v3 — different metric + hallucination check. |

**Judge scoring** (theological_reliability / helpfulness on the synthesis categories) is
pending a judge model — the keyword `verse_accuracy` metric scores 0 on questions with no
single canonical verse answer, so it undercounts those categories.

## Template (for new rows)

| Date | Model/checkpoint | Protocol | Verse accuracy | Citation | Hallucination | Notes |
|---|---|---|---|---|---|---|
| YYYY-MM-DD | … | v3 | — | — | 0 | |
