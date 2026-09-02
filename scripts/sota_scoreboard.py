#!/usr/bin/env python3
"""Build docs/SOTA_EVAL.md — the head-to-head that a "best open model at
RAG-grounded scripture Q&A" claim stands or falls on.

Reads (protocol v4, keyword; all through the identical RAG stack):
  ours  : docs/benchmark_runs/20260902_{v2-4b,v3-sft,v3-grpo}_v4keyword.json
          (produced by scripts/rescore_v4.py — same 282 Qs / same responses /
           deterministic re-bucket as a fresh v4 run)
  ext   : docs/benchmark_runs/*_ext-*_keyword.json
          (produced by scripts/run_external_baselines.sh)

Emits a ranked table + a scoped verdict. Runs fine before the external sweep
(ours-only, comparators marked pending). No model calls.

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

from scripts.benchmark_stats import mcnemar_pvalue, normalize_question  # noqa: E402

BENCH = PROJECT_ROOT / "docs/benchmark_runs"
OUT = PROJECT_ROOT / "docs/SOTA_EVAL.md"
COMPARATORS_YAML = PROJECT_ROOT / "benchmarks/external_comparators.yaml"

# task-claim gates (from docs/V3_DATASET_PLAN.md acceptance bar)
CITATION_GATE = 0.97
HALLUCINATION_GATE = 0.025
OVERALL_FUZZY_BAR = 0.52

OURS = {
    "v2-4b": "20260902_v2-4b_v4keyword.json",
    "v3-sft": "20260902_v3-sft_v4keyword.json",
    "v3-grpo": "20260902_v3-grpo_v4keyword.json",
}
OUR_PARAMS_B = 4.0


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


def overall_fuzzy_excl_exposition(results: list[dict]) -> tuple[float, int]:
    vals = [
        float(r.get("verse_accuracy_fuzzy", 0.0))
        for r in results
        if r.get("category") not in {"refusal", "verse_exposition"}
    ]
    return (sum(vals) / len(vals) if vals else 0.0, len(vals))


def cat(d: dict, name: str, key: str, default=0.0):
    return d.get("category_summary", {}).get(name, {}).get(key, default)


def rate(d: dict, key: str) -> tuple[float, float, float, int]:
    r = d.get(key, {})
    v = float(r.get("value", 0.0))
    w = r.get("wilson95", {})
    n = int(r.get("n", 0))
    return v, float(w.get("lo", v)), float(w.get("hi", v)), n


def verse_binary(item: dict) -> int:
    if "fuzzy_pass" in item:
        return 1 if item["fuzzy_pass"] else 0
    return 1 if float(item.get("verse_accuracy", 0.0)) >= 1.0 else 0


def mcnemar_vs(best: dict, other: dict) -> str:
    ia = {normalize_question(r["question"]): r for r in best.get("results", [])}
    ib = {normalize_question(r["question"]): r for r in other.get("results", [])}
    keys = sorted(set(ia) & set(ib))
    if not keys:
        return "no overlap"
    b = sum(1 for k in keys if verse_binary(ia[k]) and not verse_binary(ib[k]))
    c = sum(1 for k in keys if not verse_binary(ia[k]) and verse_binary(ib[k]))
    return f"best+{b} / other+{c} / p={mcnemar_pvalue(b, c):.3f} (n={len(keys)})"


def row_metrics(d: dict) -> dict:
    res = d.get("results", [])
    of_excl, n_excl = overall_fuzzy_excl_exposition(res)
    cite_v, cite_lo, cite_hi, _ = rate(d, "overall_citation_rate")
    hall_v, hall_lo, hall_hi, _ = rate(d, "overall_hallucination_rate")
    fp_v = rate(d, "overall_fuzzy_pass_rate")[0]
    return {
        "vq_exact": float(cat(d, "verse_quote", "avg_verse_accuracy")),
        "ve_fuzzy": float(cat(d, "verse_exposition", "avg_verse_accuracy_fuzzy")),
        "of_allin": float(d.get("overall_verse_accuracy_fuzzy", 0.0)),
        "of_excl": of_excl,
        "of_excl_n": n_excl,
        "fpass": fp_v,
        "cite": cite_v,
        "cite_ci": (cite_lo, cite_hi),
        "hall": hall_v,
        "hall_ci": (hall_lo, hall_hi),
    }


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


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
                    "provenance": "rescore_v4 (from v3 keyword)",
                    "data": d,
                    "m": row_metrics(d),
                }
            )
    ext_missing = []
    for path in sorted(glob.glob(str(BENCH / "*_ext-*_keyword.json"))):
        d = load(Path(path))
        if not d:
            continue
        key = re.sub(r".*_ext-(.+?)_keyword\.json$", r"\1", Path(path).name)
        mc = meta.get(key, {})
        entries.append(
            {
                "name": key,
                "params": float(mc.get("params_b", 0.0)),
                "group": mc.get("group", "external"),
                "provenance": "fresh v4 run",
                "data": d,
                "m": row_metrics(d),
            }
        )
    ran_keys = {
        re.sub(r".*_ext-(.+?)_keyword\.json$", r"\1", Path(p).name)
        for p in glob.glob(str(BENCH / "*_ext-*_keyword.json"))
    }
    for key, mc in meta.items():
        if key not in ran_keys:
            ext_missing.append((key, mc))

    # rank by exposition-excluded overall fuzzy mean (the v4 headline)
    entries.sort(key=lambda e: e["m"]["of_excl"], reverse=True)

    our_best = max(
        (e for e in entries if e["group"] == "ours"),
        key=lambda e: e["m"]["of_excl"],
        default=None,
    )

    L: list[str] = []
    L.append("# SOTA evaluation — RAG-grounded scripture Q&A (protocol v4)\n")
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

    L.append("## Scoreboard\n")
    L.append(
        "| rank | model | B | group | verse_quote exact | verse_expo fuzzy | "
        "overall fuzzy (expo-excl) | fuzzy pass@.85 | citation | hallucination | provenance |"
    )
    L.append("|--:|---|--:|---|--:|--:|--:|--:|--:|--:|---|")
    for i, e in enumerate(entries, 1):
        m = e["m"]
        star = " ⬅" if e is our_best else ""
        L.append(
            f"| {i} | **{e['name']}**{star} | {e['params'] or '?'} | {e['group']} | "
            f"{pct(m['vq_exact'])} | {m['ve_fuzzy']:.3f} | "
            f"**{m['of_excl']:.3f}** (n={m['of_excl_n']}) | {pct(m['fpass'])} | "
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

    L.append("## Head-to-head vs. our best on verse-accuracy (McNemar, paired)\n")
    if our_best:
        L.append(f"Reference: **{our_best['name']}**\n")
        for e in entries:
            if e is our_best:
                continue
            L.append(f"- vs `{e['name']}`: {mcnemar_vs(our_best['data'], e['data'])}")
    L.append("")

    L.append("## Verdict\n")
    if our_best is None:
        L.append("_No re-scored 'ours' runs found — run `scripts/rescore_v4.py` first._\n")
    else:
        m = our_best["m"]
        open_entries = [e for e in entries if e["group"] != "frontier"]
        rank = open_entries.index(our_best) + 1
        leads_quality = rank == 1
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
            f"- Rank among open models on the headline metric: **#{rank}** "
            f"of {len(open_entries)} ({ncomp} external comparators scored)."
        )
        L.append(
            f"- Acceptance bar (self): overall-fuzzy {'PASS' if passes_bar else 'MISS'} · "
            f"citation {'PASS' if passes_cite else 'MISS'} · "
            f"hallucination {'PASS' if passes_hall else 'MISS'}."
        )
        L.append("")
        if ncomp == 0:
            L.append(
                "> **Claim: not yet supported — no external comparator has been scored.** "
                "The ranking above is ours-only. Run the external sweep."
            )
        elif leads_quality and passes_cite and passes_hall:
            L.append(
                "> **Claim supported (scoped):** best *open* model at RAG-grounded "
                "scripture Q&A on this suite — leads on closeness-to-expected while "
                "meeting the citation and hallucination gates, against dedicated bible "
                "fine-tunes and larger general instruct models. State it with the size "
                "and hardware scope above; do not extend it to frontier or "
                "unconstrained-hardware comparisons."
            )
        else:
            gap = []
            if not leads_quality:
                lead = open_entries[0]
                gap.append(
                    f"trails `{lead['name']}` on the headline metric "
                    f"({lead['m']['of_excl']:.3f} vs {m['of_excl']:.3f})"
                )
            if not passes_cite:
                gap.append(f"citation {pct(m['cite'])} < gate")
            if not passes_hall:
                gap.append(f"hallucination {pct(m['hall'])} > gate")
            L.append(
                "> **Claim not supported as-is.** Gap: "
                + "; ".join(gap)
                + ". Close it (v3.1 retrain / recipe change) before making the claim."
            )
    L.append("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(PROJECT_ROOT)}  ({len(entries)} models, "
        f"{len(ext_missing)} comparators pending)"
    )
    print("\n".join(L[: L.index("## Scoreboard\n") + 16]) if "## Scoreboard\n" in L else "")


if __name__ == "__main__":
    main()
