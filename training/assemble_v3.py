#!/usr/bin/env python3
"""Assemble train_v3.json: teacher-distilled answers for the regenerated
categories + freshly-built keep-as-is categories (verse-drill trimmed ~60%,
near_miss_guard, pastoral_triage, general_blend). Dedup + decontaminate via
build_dataset_v2.finalize.

  python training/assemble_v3.py \
      --distilled data/raw_v3/distill_out.jsonl \
      --thematic  data/raw_v3/thematic_out.jsonl \
      --out data/processed/train_v3.json

`--thematic` is optional (thematic_qa can be added in a later pass).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.build_dataset_v2 as b2  # noqa: E402
from training.dataset_builder import _msg  # noqa: E402

# keep-as-is category budgets for v3 (verse-drill cut ~60% vs v2)
KEEP_BUDGETS = {
    "verse_recall": 2000,
    "translation_specific": 1500,
    "reverse_lookup": 1500,
    "passage_recall": 2000,
    "near_miss_guard": 4491,
}
TRIAGE_N = 600
BLEND_N = 11000


def reuse_blend(prior_dataset: Path, cap: int) -> list[dict]:
    """Pull the already-cleaned `general_blend` examples from a prior assembled
    dataset JSON (records `{messages, category}`), capped at `cap`. Lets v3.1 skip
    re-streaming smoltalk2 (~2 h, HF-rate-limit-fragile) — the v3 blend is already
    `<think>`-stripped and decontaminated."""
    prior = json.loads(prior_dataset.read_text(encoding="utf-8"))
    return [
        {"messages": ex["messages"]}
        for ex in prior
        if ex.get("category") == "general_blend" and isinstance(ex.get("messages"), list)
    ][:cap]


def load_distilled(path: Path) -> list[dict]:
    """distill_out.jsonl (status ok only) -> [{messages}] examples."""
    sp = b2.load_system_prompt(PROJECT_ROOT, for_training=True)
    out: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rec = json.loads(ln)
        if rec.get("status") != "ok":
            continue
        user = f"Context:\n{rec['context']}\n\nQ: {rec['question']}"
        ex = _msg(sp, user, rec["answer"])
        ex["category"] = rec.get("category", "distilled")
        out.append(ex)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--distilled", required=True, type=Path)
    ap.add_argument("--thematic", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/train_v3.json")
    ap.add_argument("--offline-only", action="store_true")
    ap.add_argument(
        "--blend-from",
        type=Path,
        default=None,
        help=(
            "Reuse the already-cleaned general_blend from a prior assembled dataset "
            "(e.g. data/processed/train_v3.json) instead of re-streaming smoltalk2 "
            "(~2 h, rate-limit-fragile). Takes records with category=='general_blend', "
            "capped at BLEND_N."
        ),
    )
    args = ap.parse_args()

    cache_dir = PROJECT_ROOT / "data" / "raw_v2"
    corpus = b2.load_translations(cache_dir, args.offline_only)
    if not corpus:
        raise SystemExit("no scripture sources")
    xrefs = b2.load_crossrefs(cache_dir, args.offline_only)
    sp = b2.load_system_prompt(PROJECT_ROOT, for_training=True)

    examples: dict[str, list] = {}

    # 1. distilled (regenerated) categories
    dist = load_distilled(args.distilled)
    by_cat: dict[str, list] = {}
    for ex in dist:
        by_cat.setdefault(ex.pop("category"), []).append(ex)
    for cat, items in by_cat.items():
        examples[f"{cat}_v3"] = items
        print(f"[{cat}_v3 distilled] {len(items)}", flush=True)

    if args.thematic and args.thematic.exists():
        th = load_distilled(args.thematic)
        for ex in th:
            ex.pop("category", None)
        examples["thematic_qa"] = th
        print(f"[thematic_qa] {len(th)}", flush=True)

    # 2. keep-as-is categories, freshly built at v3 budgets
    for cat, budget in KEEP_BUDGETS.items():
        fn = getattr(b2, f"gen_{cat}")
        made = (
            fn(xrefs, corpus, sp, budget)
            if fn is b2.gen_cross_reference_chains
            else fn(corpus, sp, budget)
        )
        examples[cat] = made
        print(f"[{cat}] {len(made)}", flush=True)

    examples["pastoral_triage"] = b2.gen_pastoral_triage(sp, TRIAGE_N)
    print(f"[pastoral_triage] {len(examples['pastoral_triage'])}", flush=True)
    if args.blend_from and args.blend_from.exists():
        blend = reuse_blend(args.blend_from, BLEND_N)
        examples["general_blend"] = blend
        print(f"[general_blend] {len(blend)} (reused from {args.blend_from})", flush=True)
    else:
        examples["general_blend"] = b2.load_smoltalk2_blend(sp, BLEND_N, args.offline_only)
        print(f"[general_blend] {len(examples['general_blend'])}", flush=True)

    manifest = b2.finalize(examples, args.out)
    total = sum(len(v) for v in examples.values())
    print(f"train_v3: {total} examples -> {args.out}", flush=True)
    (args.out.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
