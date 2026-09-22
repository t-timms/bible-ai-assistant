#!/usr/bin/env python3
"""Sanity-gate a freshly merged model before a benchmark run.

``training/merge_adapters.py`` has a documented silent-failure mode (Unsloth /
PEFT key-layout mismatch -> all LoRA weights skipped -> fluent-but-untuned or
garbage output). This loads the merged model, generates a few Bible-domain
completions greedily, and fails nonzero on empty / degenerate / non-text output.

    python scripts/_coherence_check.py models/qwen3.5-4b-bible-v3.1-merged
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = (
    "Quote Genesis 1:1 exactly.",
    "What does John 3:16 say, and what does it mean?",
    "What is the main theme of Psalm 23?",
)


def main() -> int:
    model_path = sys.argv[1]
    print(f"loading {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    bad = 0
    for prompt in PROMPTS:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=160,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        txt = tok.decode(out[0][plen:], skip_special_tokens=True).strip()
        print("\n" + "=" * 70 + f"\nPROMPT: {prompt}\n" + "-" * 70 + f"\n{txt}\n", flush=True)

        words = txt.split()
        if len(words) < 5:
            print("  !! too short", flush=True)
            bad += 1
        elif len(set(words)) / max(len(words), 1) < 0.35:
            print("  !! low lexical diversity (repetition)", flush=True)
            bad += 1
        elif not any(c.isascii() and c.isalpha() for c in txt):
            print("  !! non-text output", flush=True)
            bad += 1

    print("\n" + "=" * 70)
    verdict = "PASS" if bad == 0 else "FAIL"
    print(f"COHERENCE_RESULT: {verdict} ({bad}/{len(PROMPTS)} bad)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
