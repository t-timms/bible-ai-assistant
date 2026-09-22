#!/usr/bin/env python3
"""Build docs/SOTA_EVAL.md — the head-to-head that a "best open model at
RAG-grounded scripture Q&A" claim stands or falls on.

Reads (protocol v4/v5, keyword; all through the identical RAG stack):
  ours  : docs/benchmark_runs/20260904_{v2-4b,v3-sft,v3.1,v3.2}_v5semantic.json
          (produced by scripts/rescore_v5.py -- same 282 Qs / same responses /
           deterministic v4 re-bucket + protocol-v5 semantic score added, no
           model re-run)
  ext   : docs/benchmark_runs/*_ext-*_keyword.json
          (produced by scripts/run_external_baselines.sh) plus, once backfilled,
          a companion docs/benchmark_runs/*_ext-*_v5semantic.json
          (produced by `scripts/rescore_v5.py --file ... --out ...`)

Emits a ranked table + a scoped verdict. Runs fine before the external sweep
(ours-only, comparators marked pending). No model calls.

2026-09-05: rewritten to compute every statistic directly from each file's
`results` array (Wilson CI, category means) instead of trusting a precomputed
top-level aggregate blob -- v5semantic.json carries fuzzy fields through but
does not duplicate `overall_citation_rate` / `overall_hallucination_rate`, and
the old code silently read those as 0.0 when absent (a `.get(key, {})` on a
missing key), which would have shown every "ours" row at 0% citation / 0%
hallucination -- caught before this ever ran against real ours data. Also adds
a semantic (protocol v5) column and a second verdict computed on semantic once
every row in the table has a semantic score (true once the external sweep is
backfilled with `rescore_v5.py --file`).

Claim scope enforced in the output text (see CLAUDE.md):
  * quality  = best OPEN model at this niche task, size-independent.
  * hardware = SOTA for the 16 GB consumer-Blackwell class.
  * NOT "beats frontier models on unconstrained hardware".
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_stats import (  # noqa: E402
    mcnemar_pvalue,
    normalize_question,
    wilson_interval,
)
from scripts.rescore_v4 import COUNT_ONLY, rebucket  # noqa: E402

BENCH = PROJECT_ROOT / "docs/benchmark_runs"
OUT = PROJECT_ROOT / "docs/SOTA_EVAL.md"
COMPARATORS_YAML = PROJECT_ROOT / "benchmarks/external_comparators.yaml"

# task-claim gates (from docs/V3_DATASET_PLAN.md acceptance bar)
CITATION_GATE = 0.97
HALLUCINATION_GATE = 0.025
OVERALL_FUZZY_BAR = 0.52

# 2026-09-04: v3.2 shipped (protocol v5 semantic re-score) -- see docs/V3_STATUS.md
# "PROTOCOL V5 + SHIP DECISION". These 4 files already carry rebucketed categories
# (verse_quote/verse_exposition) and both fuzzy and semantic scores per item.
OURS = {
    "v2-4b": "20260904_v2-4b_v5semantic.json",
    "v3-sft": "20260904_v3-sft_v5semantic.json",
    "v3.1": "20260904_v3.1_v5semantic.json",
    "v3.2": "20260904_v3.2_v5semantic.json",
}
OUR_PARAMS_B = 4.0
EXPOSITION_CATEGORIES = {"verse_exposition"}


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _meta_from_yaml() -> dict[str, dict]:
    try:
        import yaml

        d = yaml.safe_load(COMPARATORS_YAML.read_text(encoding="utf-8"))
        return {c["key"]: c for c in d.get("comparators", [])}
    except Exception:
        return {}


def _normalized_results(d: dict) -> list[dict]:
    """Every result with its category re-bucketed (idempotent -- a no-op on
    already-split categories), so old (unsplit verse_lookup) and new (split)
    source files score identically."""
    out = []
    for r in d.get("results", []):
        nr = dict(r)
        nr["category"] = rebucket(r.get("category", ""), r.get("question", ""))
        out.append(nr)
    return out


def _rate_with_ci(results: list[dict], flag_key: str) -> tuple[float, float, float, int]:
    scored = [r for r in results if r["category"] not in COUNT_ONLY]
    n = len(scored)
    successes = sum(1 for r in scored if r.get(flag_key))
    lo, hi = wilson_interval(successes, n)
    v = successes / n if n else 0.0
    return v, lo, hi, n


def _mean(results: list[dict], field: str, *, exclude: set[str]) -> tuple[float, int]:
    vals = [
        float(r[field])
        for r in results
        if r["category"] not in COUNT_ONLY
        and r["category"] not in exclude
        and r.get(field) is not None
    ]
    n = len(vals)
    return (sum(vals) / n if n else 0.0, n)


def _cat_mean(results: list[dict], cat: str, field: str) -> float:
    vals = [float(r[field]) for r in results if r["category"] == cat and r.get(field) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def row_metrics(d: dict) -> dict:
    results = _normalized_results(d)
    of_allin, _ = _mean(results, "verse_accuracy_fuzzy", exclude=set())
    of_excl, n_excl = _mean(results, "verse_accuracy_fuzzy", exclude=EXPOSITION_CATEGORIES)
    fpass_v, *_ = _rate_with_ci(results, "fuzzy_pass")
    cite_v, cite_lo, cite_hi, _ = _rate_with_ci(results, "citation_present")
    hall_v, hall_lo, hall_hi, _ = _rate_with_ci(results, "hallucination_detected")

    has_semantic = any(r.get("verse_accuracy_semantic") is not None for r in results)
    sem_allin, _ = _mean(results, "verse_accuracy_semantic", exclude=set())
    sem_excl, sem_excl_n = _mean(results, "verse_accuracy_semantic", exclude=EXPOSITION_CATEGORIES)

    return {
        "results": results,
        "vq_exact": _cat_mean(results, "verse_quote", "verse_accuracy"),
        "ve_fuzzy": _cat_mean(results, "verse_exposition", "verse_accuracy_fuzzy"),
        "of_allin": of_allin,
        "of_excl": of_excl,
        "of_excl_n": n_excl,
        "fpass": fpass_v,
        "cite": cite_v,
        "cite_ci": (cite_lo, cite_hi),
        "hall": hall_v,
        "hall_ci": (hall_lo, hall_hi),
        "has_semantic": has_semantic,
        "sem_allin": sem_allin,
        "sem_excl": sem_excl,
        "sem_excl_n": sem_excl_n,
    }


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ext_semantic_companion(keyword_path: Path) -> Path:
    return keyword_path.with_name(keyword_path.name.replace("_keyword.json", "_v5semantic.json"))


def main() -> None:
    meta = _meta_from_yaml()

    entries: list[dict] = []  # {name, params, group, provenance, data, m}
    for name, fn in OURS.items():
        d = load(BENCH / fn)
        if d:
            entries.append(
                {
                    "name": name,
                    "params": OUR_PARAMS_B,
                    "group": "ours",
                    "provenance": "rescore_v5 (protocol v4 rebucket + semantic, no re-run)",
                    "data": d,
                    "m": row_metrics(d),
                }
            )

    ext_missing = []
    ext_paths = sorted(glob.glob(str(BENCH / "*_ext-*_keyword.json")))
    for path in ext_paths:
        p = Path(path)
        key = re.sub(r".*_ext-(.+?)_keyword\.json$", r"\1", p.name)
        sem_path = _ext_semantic_companion(p)
        d = load(sem_path) if sem_path.exists() else load(p)
        if not d:
            continue
        provenance = (
            "fresh v4 run + v5 semantic backfill"
            if sem_path.exists()
            else "fresh v4 run (semantic pending)"
        )
        mc = meta.get(key, {})
        entries.append(
            {
                "name": key,
                "params": float(mc.get("params_b", 0.0)),
                "group": mc.get("group", "external"),
                "provenance": provenance,
                "data": d,
                "m": row_metrics(d),
            }
        )
    ran_keys = {re.sub(r".*_ext-(.+?)_keyword\.json$", r"\1", Path(p).name) for p in ext_paths}
    for key, mc in meta.items():
        if key not in ran_keys:
            ext_missing.append((key, mc))

    # rank by exposition-excluded overall fuzzy mean (the v4 headline; kept as the
    # primary sort for continuity -- the semantic table below is the fairer one
    # once every row has a semantic score)
    entries.sort(key=lambda e: e["m"]["of_excl"], reverse=True)

    our_best = max(
        (e for e in entries if e["group"] == "ours"),
        key=lambda e: e["m"]["of_excl"],
        default=None,
    )

    L: list[str] = []
    L.append("# SOTA evaluation — RAG-grounded scripture Q&A (protocol v4/v5)\n")
    L.append(
        "_Generated by `scripts/sota_scoreboard.py`. All rows: the same 282-question "
        "v4 suite, the same hybrid-RAG stack (dense+BM25+RRF+reranker+citation "
        "verification), greedy decode, seed 42 — the only variable is the generator model._\n"
    )
    L.append("## Claim scope\n")
    L.append(
        "- **Quality:** best *open* model *at this task* — size-independent, because no "
        "large lab optimizes for RAG-grounded scripture Q&A with verified citations. "
        "Backed by beating the dedicated open bible fine-tunes **and** holding vs. larger "
        "general instruct models on citation + hallucination + closeness-to-expected.\n"
        "- **Hardware:** SOTA for the 16 GB consumer-Blackwell class (RTX 5070 Ti).\n"
        "- **Not** a claim about beating frontier models on unconstrained hardware. A "
        "frontier API row, if present, is a labelled ceiling, never a peer.\n"
    )

    L.append("## Scoreboard (ranked by fuzzy, expo-excl — see semantic table below)\n")
    L.append(
        "| rank | model | B | group | verse_quote exact | verse_expo fuzzy | "
        "overall fuzzy (expo-excl) | semantic (expo-excl) | citation | hallucination | provenance |"
    )
    L.append("|--:|---|--:|---|--:|--:|--:|--:|--:|--:|---|")
    for i, e in enumerate(entries, 1):
        m = e["m"]
        star = " ⬅" if e is our_best else ""
        sem_cell = f"{m['sem_excl']:.3f}" if m["has_semantic"] else "pending"
        L.append(
            f"| {i} | **{e['name']}**{star} | {e['params'] or '?'} | {e['group']} | "
            f"{pct(m['vq_exact'])} | {m['ve_fuzzy']:.3f} | "
            f"**{m['of_excl']:.3f}** (n={m['of_excl_n']}) | {sem_cell} | "
            f"{pct(m['cite'])} | {pct(m['hall'])} | {e['provenance']} |"
        )
    L.append("")
    L.append(
        f"Bars: overall fuzzy (expo-excl) ≥ {OVERALL_FUZZY_BAR:.2f} · "
        f"citation ≥ {pct(CITATION_GATE)} · hallucination ≤ {pct(HALLUCINATION_GATE)}\n"
    )

    if ext_missing:
        L.append("### Comparators not yet run\n")
        for key, mc in ext_missing:
            L.append(
                f"- `{key}` — {mc.get('hf', '')} ({mc.get('params_b', '?')} B, "
                f"{mc.get('group', '')}, license {mc.get('license', '?')})"
            )
        L.append("\nRun `scripts/run_external_baselines.sh` (GPU) to fill these in.\n")

    pending_semantic = [e["name"] for e in entries if not e["m"]["has_semantic"]]
    if entries and not pending_semantic:
        L.append("## Scoreboard (ranked by semantic, protocol v5, expo-excl)\n")
        L.append(
            "_check_verse_accuracy_semantic (cross-encoder, bge-reranker-v2-m3) — see "
            "benchmarks/manifest.v5.yaml. Built after auditing the fuzzy metric's noise "
            'floor on close internal candidates (docs/V3_STATUS.md "PROTOCOL V5 + SHIP '
            'DECISION"); shown here as the primary ranking once every row has a score._\n'
        )
        sem_ranked = sorted(entries, key=lambda e: e["m"]["sem_excl"], reverse=True)
        L.append("| rank | model | B | group | semantic (expo-excl) | fuzzy (expo-excl) |")
        L.append("|--:|---|--:|---|--:|--:|")
        sem_best = max(
            (e for e in sem_ranked if e["group"] == "ours"),
            key=lambda e: e["m"]["sem_excl"],
            default=None,
        )
        for i, e in enumerate(sem_ranked, 1):
            m = e["m"]
            star = " ⬅" if e is sem_best else ""
            L.append(
                f"| {i} | **{e['name']}**{star} | {e['params'] or '?'} | {e['group']} | "
                f"**{m['sem_excl']:.3f}** (n={m['sem_excl_n']}) | {m['of_excl']:.3f} |"
            )
        L.append("")
    elif entries:
        L.append(
            f"_Semantic ranking withheld: {len(pending_semantic)} row(s) not yet scored "
            f"under protocol v5 ({', '.join(pending_semantic)}). Backfill with "
            "`scripts/rescore_v5.py --file <ext keyword.json> --out <ext v5semantic.json>` "
            "per comparator once run._\n"
        )

    L.append("## Head-to-head vs. our best on verse-accuracy (McNemar, paired)\n")
    if our_best:
        L.append(f"Reference: **{our_best['name']}**\n")

        def verse_binary(item: dict) -> int:
            if "fuzzy_pass" in item:
                return 1 if item["fuzzy_pass"] else 0
            return 1 if float(item.get("verse_accuracy", 0.0)) >= 1.0 else 0

        def mcnemar_vs(best_results: list[dict], other_results: list[dict]) -> str:
            ia = {normalize_question(r["question"]): r for r in best_results}
            ib = {normalize_question(r["question"]): r for r in other_results}
            keys = sorted(set(ia) & set(ib))
            if not keys:
                return "no overlap"
            b = sum(1 for k in keys if verse_binary(ia[k]) and not verse_binary(ib[k]))
            c = sum(1 for k in keys if not verse_binary(ia[k]) and verse_binary(ib[k]))
            return f"best+{b} / other+{c} / p={mcnemar_pvalue(b, c):.3f} (n={len(keys)})"

        for e in entries:
            if e is our_best:
                continue
            L.append(
                f"- vs `{e['name']}`: {mcnemar_vs(our_best['m']['results'], e['m']['results'])}"
            )
    L.append("")

    L.append("## Verdict\n")
    if our_best is None:
        L.append("_No 'ours' runs found — run `scripts/rescore_v5.py` first._\n")
    else:
        m = our_best["m"]
        open_entries = [e for e in entries if e["group"] != "frontier"]
        rank = open_entries.index(our_best) + 1
        passes_cite = m["cite"] >= CITATION_GATE
        passes_hall = m["hall"] <= HALLUCINATION_GATE
        passes_bar = m["of_excl"] >= OVERALL_FUZZY_BAR
        ncomp = len([e for e in entries if e["group"] not in ("ours",)])

        L.append(
            f"- **{our_best['name']}**: overall fuzzy (expo-excl) `{m['of_excl']:.3f}`, "
            f"citation `{pct(m['cite'])}`, hallucination `{pct(m['hall'])}`, "
            f"verse_quote exact `{pct(m['vq_exact'])}`."
        )
        L.append(
            f"- Rank among open models on the fuzzy headline metric: **#{rank}** "
            f"of {len(open_entries)} ({ncomp} external comparators scored)."
        )
        L.append(
            f"- Acceptance bar (self, fuzzy): overall-fuzzy {'PASS' if passes_bar else 'MISS'} · "
            f"citation {'PASS' if passes_cite else 'MISS'} · "
            f"hallucination {'PASS' if passes_hall else 'MISS'}."
        )
        L.append("")
        if ncomp == 0:
            L.append(
                "> **Claim: not yet supported — no external comparator has been scored.** "
                "The ranking above is ours-only. Run the external sweep."
            )
        elif pending_semantic:
            L.append(
                "> **Claim: not yet supported — external comparator(s) scored on fuzzy only, "
                "not yet on protocol v5 semantic.** The fuzzy metric is known to have a narrow "
                "noise floor at this quality level (docs/V3_STATUS.md); do not use the fuzzy-only "
                "ranking above to make the SOTA claim. Backfill semantic scores for: "
                + ", ".join(pending_semantic)
                + "."
            )
        else:
            sem_rank = (
                sorted(entries, key=lambda e: e["m"]["sem_excl"], reverse=True).index(our_best) + 1
            )
            if sem_rank == 1 and passes_cite and passes_hall:
                L.append(
                    "> **Claim supported (scoped):** best *open* model at RAG-grounded "
                    "scripture Q&A on this suite — leads on protocol-v5 semantic "
                    "closeness-to-expected while meeting the citation and hallucination "
                    "gates, against dedicated bible fine-tunes and larger general instruct "
                    "models. State it with the size and hardware scope above; do not extend "
                    "it to frontier or unconstrained-hardware comparisons."
                )
            else:
                gap = []
                if sem_rank != 1:
                    lead = sorted(entries, key=lambda e: e["m"]["sem_excl"], reverse=True)[0]
                    gap.append(
                        f"trails `{lead['name']}` on semantic (expo-excl) "
                        f"({lead['m']['sem_excl']:.3f} vs {m['sem_excl']:.3f})"
                    )
                if not passes_cite:
                    gap.append(f"citation {pct(m['cite'])} < gate")
                if not passes_hall:
                    gap.append(f"hallucination {pct(m['hall'])} > gate")
                L.append("> **Claim not supported as-is.** Gap: " + "; ".join(gap) + ".")
    L.append("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(PROJECT_ROOT)}  ({len(entries)} models, "
        f"{len(ext_missing)} comparators pending, {len(pending_semantic) if entries else 0} pending semantic)"
    )


if __name__ == "__main__":
    main()
