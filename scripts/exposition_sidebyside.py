#!/usr/bin/env python3
"""Side-by-side of the 36 verse_exposition items: v2-4b vs v3-sft.

For the manual read that replaces the (infeasible 27B) judge on the one
question that matters for the ship/retrain call: on exposition-phrased
verse questions ("What does X teach?", "What is X about?"), is v3-SFT's
prose explanation actually a better answer than v2's raw verbatim dump —
or worse?

Reads the existing protocol-v3 keyword artifacts (no model run). Writes a
markdown table to docs/benchmark_runs/20260902_exposition_v2_vs_v3.md and
prints the same to stdout.

Usage:
  python scripts/exposition_sidebyside.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_stats import normalize_question  # noqa: E402

BENCH = PROJECT_ROOT / "docs/benchmark_runs"
V2 = BENCH / "20260901_v2-4b_keyword.json"
V3 = BENCH / "20260901_v3-sft_keyword.json"
OUT = BENCH / "20260902_exposition_v2_vs_v3.md"

EXPOSITION_RE = re.compile(r"(teach|about)\?\s*$", re.IGNORECASE)


def load(p: Path) -> dict[str, dict]:
    d = json.loads(p.read_text(encoding="utf-8"))
    return {normalize_question(r["question"]): r for r in d["results"]}


def is_expo(q: str) -> bool:
    return bool(EXPOSITION_RE.search(q.strip()))


def main() -> None:
    v2, v3 = load(V2), load(V3)
    keys = [k for k in v2 if k in v3 and is_expo(v2[k]["question"])]

    lines: list[str] = []
    lines.append("# verse_exposition — v2-4b vs v3-sft (manual read)\n")
    lines.append(
        "36 exposition-phrased verse questions. Exact-match is the wrong metric "
        "here; the question is which answer a reader is better served by.\n"
    )
    lines.append(
        "| # | question | expected (verbatim) | v2-4b response | v2 exact/fuzzy | "
        "v3-sft response | v3 exact/fuzzy | better? |"
    )
    lines.append("|--:|---|---|---|--:|---|--:|:--:|")

    n_v2_exact = n_v3_exact = 0
    for i, k in enumerate(sorted(keys), 1):
        a, b = v2[k], v3[k]
        n_v2_exact += 1 if float(a.get("verse_accuracy", 0)) >= 1.0 else 0
        n_v3_exact += 1 if float(b.get("verse_accuracy", 0)) >= 1.0 else 0

        def cell(t: str) -> str:
            return t.replace("\n", " ").replace("|", "\\|").strip()

        lines.append(
            f"| {i} | {cell(a['question'])} | {cell(a['expected_answer'])} | "
            f"{cell(a['response'])} | {a.get('verse_accuracy', 0):.2f} / {a.get('verse_accuracy_fuzzy', 0):.2f} | "
            f"{cell(b['response'])} | {b.get('verse_accuracy', 0):.2f} / {b.get('verse_accuracy_fuzzy', 0):.2f} | |"
        )

    lines.append("")
    lines.append(f"- items: {len(keys)}")
    lines.append(f"- v2-4b exact-match passes: {n_v2_exact}/{len(keys)}")
    lines.append(f"- v3-sft exact-match passes: {n_v3_exact}/{len(keys)}")
    lines.append(
        "- fill the **better?** column by hand: `v3` / `v2` / `tie`. "
        "If v3 wins or ties the clear majority and none are factually wrong, "
        "the verse_lookup 'regression' is an eval artifact -> ship v3-SFT. "
        "If v3 has real errors or is consistently less useful -> Path B retrain."
    )

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
