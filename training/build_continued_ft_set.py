#!/usr/bin/env python3
"""Assemble the v3.2 DMT-style continued-fine-tune set: the regenerated (RAFT-
distractor-fixed) thematic_qa as the primary signal, plus a rehearsal slice
sampled from every other v3.1 category so the short stage-2 pass doesn't erode
what stage 1 already got right.

Unlike v3.1's full-mix approach (thematic_qa was 5.7% of one big epoch, diluted
by 41,830 examples of everything else), this stage is small and thematic_qa-
heavy on purpose — it's a short nudge starting FROM the v3.1 adapter
(training/train_unsloth.py --model-path models/qwen3.5-4b-bible-v3.1-sft), not
a fresh SFT. See docs/V3_STATUS.md for why: naive upsampling inside one mixed
epoch risks gradient interference between synthesis-style and citation-drill
examples; a separate short stage with the bulk of stage-1 data left out (only a
rehearsal slice kept) is the literature's answer to exactly this.

Usage:
  python training/build_continued_ft_set.py \
      --thematic data/raw_v3/thematic_out.jsonl \
      --base data/processed/train_v3.1.json \
      --out data/processed/train_v3.2-continued.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.assemble_v3 import load_distilled  # noqa: E402


def build_rehearsal(base_examples: list[dict], target_total: int, seed: int) -> list[dict]:
    """Stratified sample across every non-thematic_qa category in `base_examples`,
    proportional to each category's share, capped at `target_total` overall."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for ex in base_examples:
        if ex.get("category") == "thematic_qa":
            continue
        by_cat.setdefault(ex.get("category", "?"), []).append(ex)
    total_non_thematic = sum(len(v) for v in by_cat.values())
    if total_non_thematic == 0:
        return []
    out: list[dict] = []
    for _cat, items in by_cat.items():
        share = round(target_total * len(items) / total_non_thematic)
        share = min(share, len(items))
        out.extend(rng.sample(items, share) if share else [])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--thematic", required=True, type=Path)
    ap.add_argument(
        "--base",
        required=True,
        type=Path,
        help="a fully-assembled prior dataset, e.g. train_v3.1.json",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--rehearsal-ratio",
        type=float,
        default=1.0,
        help="rehearsal set size as a multiple of the thematic_qa count (default 1.0 -> ~50/50 mix)",
    )
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    thematic = load_distilled(args.thematic)
    for ex in thematic:
        ex["category"] = "thematic_qa"
    print(f"[thematic_qa] {len(thematic)}", flush=True)

    base = json.loads(args.base.read_text(encoding="utf-8"))
    rehearsal = build_rehearsal(base, round(len(thematic) * args.rehearsal_ratio), args.seed)
    print(
        f"[rehearsal] {len(rehearsal)} sampled across "
        f"{len({e.get('category') for e in rehearsal})} categories",
        flush=True,
    )

    combined = [{"messages": ex["messages"], "category": "thematic_qa"} for ex in thematic] + [
        {"messages": ex["messages"], "category": ex.get("category", "?")} for ex in rehearsal
    ]
    rng = random.Random(args.seed)
    rng.shuffle(combined)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(combined)} examples -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
