#!/usr/bin/env python3
"""Emit v3 distillation inputs: run the templated-answer generators from
build_dataset_v2, but keep only (context, question) — the answer is what the
teacher (training/distill_answers.py) regenerates.

Categories regenerated:
  topical_collections     -> synthesized thematic answer (was a bullet dump)
  cross_reference_chains  -> explain why the passages connect (was a ref list)
  chapter_context         -> keep as-is volume but re-voiced (it's verse-lookup;
                             kept small here, the real exposition gain is thematic_qa)
  grounded_exegesis       -> teacher-polish for natural voice (MHC grounding stays)

thematic_qa (the eval's real gap) needs the live RAG retriever for context and is
built separately by build_v3_thematic.py (runs in .venv-rag).

Usage:
  python training/build_v3_inputs.py --out data/raw_v3/distill_inputs.jsonl
  python training/build_v3_inputs.py --out /tmp/smoke.jsonl --limit 20
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
from rag.prompt_format import QUESTION_SEPARATOR  # noqa: E402

# category -> (generator, v3 target count)
REGEN = {
    "topical_collections": 5000,
    "cross_reference_chains": 4500,
    "chapter_context": 2500,
    "grounded_exegesis": 6000,
}


def split_context_question(user_msg: str) -> tuple[str, str]:
    """'Context:\\n- **ref**: t\\n...\\n\\nQ: <q>' -> (context_without_header, q)."""
    idx = user_msg.rfind(QUESTION_SEPARATOR)
    if idx == -1:
        return "", user_msg.strip()
    block = user_msg[:idx]
    question = user_msg[idx + len(QUESTION_SEPARATOR) :].strip()
    # drop the leading "Context:\n" header so distill_answers can re-wrap cleanly
    for header in ("Context:\n", "Context:\r\n", "Context: "):
        if block.startswith(header):
            block = block[len(header) :]
            break
    return block.strip(), question


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 = full v3 targets; else N per category")
    ap.add_argument("--offline-only", action="store_true")
    args = ap.parse_args()

    cache_dir = PROJECT_ROOT / "data" / "raw_v2"
    print("== loading corpus ==", flush=True)
    corpus = b2.load_translations(cache_dir, args.offline_only)
    if not corpus:
        raise SystemExit("no scripture sources — run once with network access")
    xrefs = b2.load_crossrefs(cache_dir, args.offline_only)
    sp = b2.load_system_prompt(PROJECT_ROOT, for_training=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    per_cat: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as sink:
        for cat, target in REGEN.items():
            n = args.limit or target
            fn = getattr(b2, f"gen_{cat}")
            made = (
                fn(xrefs, corpus, sp, n)
                if fn is b2.gen_cross_reference_chains
                else fn(corpus, sp, n)
            )
            for i, ex in enumerate(made):
                user = ex["messages"][1]["content"]
                orig_answer = ex["messages"][2]["content"]
                context, question = split_context_question(user)
                if not context or not question:
                    continue
                sink.write(
                    json.dumps(
                        {
                            "id": f"{cat}-{i}",
                            "category": cat,
                            "context": context,
                            "question": question,
                            "orig_answer": orig_answer,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                per_cat[cat] = per_cat.get(cat, 0) + 1
            print(f"[{cat}] {per_cat.get(cat, 0)}", flush=True)

    print(f"wrote {written} inputs -> {args.out}", flush=True)
    print("per-category:", per_cat, flush=True)


if __name__ == "__main__":
    main()
