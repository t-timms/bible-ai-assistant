"""Audit built preference data for ORPO training.

Checks structure, diversity, and quality of generated preference pairs.

Usage:
  python scripts/audit_preference_data.py
  python scripts/audit_preference_data.py --input data/processed/preferences.json
  python scripts/audit_preference_data.py --input data/processed/preferences.json --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT_ROOT / "data" / "processed" / "preferences.json"


def _fmt(n: int) -> str:
    return f"{n:>6}"


def audit_preference_data(path: Path, verbose: bool = False) -> int:
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return 1

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print(f"ERROR: Expected JSON array, got {type(raw).__name__}")
        return 1

    total = len(raw)
    print(f"Total preference pairs: {total}")
    print()

    if total == 0:
        print("WARNING: No preference pairs to audit.")
        return 0

    issues = 0
    prompt_lens: list[int] = []
    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    user_prompts: list[str] = []
    missing_keys = 0
    bad_message_format = 0
    empty_chosen = 0
    empty_rejected = 0
    chosen_eq_rejected = 0
    chosen_shorter_or_eq = 0

    for i, pair in enumerate(raw):
        if not isinstance(pair, dict):
            print(f"  ERROR: Pair {i} is not a dict")
            issues += 1
            continue

        for key in ("prompt", "chosen", "rejected"):
            if key not in pair:
                print(f"  ERROR: Pair {i} missing key: {key}")
                missing_keys += 1
                issues += 1

        prompt = pair.get("prompt", [])
        chosen = pair.get("chosen", [])
        rejected = pair.get("rejected", [])

        if (
            not isinstance(prompt, list)
            or not isinstance(chosen, list)
            or not isinstance(rejected, list)
        ):
            print(f"  ERROR: Pair {i} has non-list message field")
            bad_message_format += 1
            issues += 1
            continue

        for field_name, field in [("prompt", prompt), ("chosen", chosen), ("rejected", rejected)]:
            if not field:
                continue
            for j, msg in enumerate(field):
                if not isinstance(msg, dict):
                    print(f"  ERROR: Pair {i} {field_name}[{j}] is not a dict")
                    bad_message_format += 1
                    issues += 1
                elif "role" not in msg or "content" not in msg:
                    print(f"  ERROR: Pair {i} {field_name}[{j}] missing role/content")
                    bad_message_format += 1
                    issues += 1

        user_msg = None
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        if user_msg is None or not user_msg.strip():
            if verbose:
                print(f"  WARNING: Pair {i} has no user message in prompt")
            issues += 1
        elif user_msg not in user_prompts:
            user_prompts.append(user_msg)

        c_text = chosen[0].get("content", "") if chosen else ""
        r_text = rejected[0].get("content", "") if rejected else ""

        prompt_lens.append(len(user_msg or ""))
        chosen_lens.append(len(c_text))
        rejected_lens.append(len(r_text))

        if not c_text.strip():
            empty_chosen += 1
            if verbose:
                print(f"  WARNING: Pair {i} has empty chosen content")
        if not r_text.strip():
            empty_rejected += 1
            if verbose:
                print(f"  WARNING: Pair {i} has empty rejected content")
        if c_text.strip() and r_text.strip() and c_text.strip() == r_text.strip():
            chosen_eq_rejected += 1
            if verbose:
                print(f"  WARNING: Pair {i} chosen equals rejected (identical content)")
        if len(c_text) <= len(r_text) * 0.05:
            chosen_shorter_or_eq += 1

    print("--- Structure ---")
    print(f"  Missing keys:           {_fmt(missing_keys)}")
    print(f"  Bad message format:     {_fmt(bad_message_format)}")
    print(f"  Empty chosen:           {_fmt(empty_chosen)}")
    print(f"  Empty rejected:         {_fmt(empty_rejected)}")
    print(f"  Chosen == rejected:     {_fmt(chosen_eq_rejected)}")
    print()

    print("--- Diversity ---")
    unique_prompts = len(set(user_prompts))
    print(f"  Unique user prompts:    {_fmt(unique_prompts)} / {total}")
    dup_prompts = total - unique_prompts
    pct_dup = 100.0 * dup_prompts / total if total else 0
    print(f"  Duplicate prompts:      {_fmt(dup_prompts)} ({pct_dup:.1f}%)")
    if verbose and dup_prompts > 0:
        prompt_counts = Counter(user_prompts)
        for prompt, count in prompt_counts.most_common(10):
            if count > 1:
                preview = prompt[:80].replace("\n", "\\n")
                print(f"    x{count}: {preview}")
    print()

    print("--- Lengths ---")
    if prompt_lens:
        avg_prompt = sum(prompt_lens) / len(prompt_lens)
        print(f"  Avg prompt length:      {avg_prompt:.1f} chars")
    if chosen_lens:
        avg_chosen = sum(chosen_lens) / len(chosen_lens)
        print(f"  Avg chosen length:      {avg_chosen:.1f} chars")
    if rejected_lens:
        avg_rejected = sum(rejected_lens) / len(rejected_lens)
        print(f"  Avg rejected length:    {avg_rejected:.1f} chars")
    if chosen_lens and rejected_lens:
        longer = sum(1 for c, r in zip(chosen_lens, rejected_lens, strict=False) if c < r)
        pct_longer = 100.0 * longer / total
        print(f"  Rejected longer:        {_fmt(longer)} / {_fmt(total)} ({pct_longer:.1f}%)")
    print(
        f"  Chosen much shorter:    {_fmt(chosen_shorter_or_eq)} ({100.0 * chosen_shorter_or_eq / total:.1f}%)"
    )
    print()

    print(f"--- Issues Found: {issues} ---")

    if issues == 0:
        print("  No issues detected.")
    return 0 if issues == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ORPO preference data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PATH, help="Path to preferences JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-pair warnings")
    args = parser.parse_args()

    sys.exit(audit_preference_data(args.input, verbose=args.verbose))


if __name__ == "__main__":
    main()
