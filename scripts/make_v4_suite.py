#!/usr/bin/env python3
"""Build the protocol-v4 frozen suite from the v3 (=v2-file) snapshot.

Inert transform: reads benchmarks/suites/evaluation_questions.v2.json and writes
benchmarks/suites/evaluation_questions.v3.json with the `verse_lookup` category
split into two:

  verse_quote       — "What does X say?", "Can you tell me what X says?",
                      "Quote X for me."   (verbatim recall; exact-match is valid)
  verse_exposition  — "What does X teach?", "What is X about?"
                      (explanation expected; exact-match is the wrong metric,
                       score by fuzzy pass-rate / judge / manual read)

No other question, answer, or category is touched. Every non-verse_lookup item
passes through byte-identical.

Rule: exposition iff the question (stripped) ends with 'teach?' or 'about?'
(case-insensitive). Everything else in verse_lookup is quote.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path.home() / "bible-ai-assistant"
SRC = REPO / "benchmarks/suites/evaluation_questions.v2.json"
DST = REPO / "benchmarks/suites/evaluation_questions.v3.json"

EXPOSITION_RE = re.compile(r"(teach|about)\?\s*$", re.IGNORECASE)


def classify(question: str) -> str:
    return "verse_exposition" if EXPOSITION_RE.search(question.strip()) else "verse_quote"


def main() -> None:
    items = json.loads(SRC.read_text(encoding="utf-8"))
    assert isinstance(items, list), "expected a JSON list"

    before = Counter(i["category"] for i in items)
    out: list[dict] = []
    moved = Counter()
    for it in items:
        new = dict(it)
        if it.get("category") == "verse_lookup":
            new["category"] = classify(it["question"])
            moved[new["category"]] += 1
        out.append(new)
    after = Counter(i["category"] for i in out)

    # 2-space indent, ensure_ascii=False, trailing newline — matches the v2 file.
    DST.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sha = hashlib.sha256(DST.read_bytes()).hexdigest()

    print(f"src : {SRC.relative_to(REPO)}  ({len(items)} items)")
    print(f"dst : {DST.relative_to(REPO)}  ({len(out)} items)")
    print()
    print("verse_lookup split:")
    print(f"  verse_quote      : {moved['verse_quote']}")
    print(f"  verse_exposition : {moved['verse_exposition']}")
    print(f"  (was verse_lookup: {before['verse_lookup']})")
    assert moved["verse_quote"] + moved["verse_exposition"] == before["verse_lookup"]
    print()
    print("category totals  (before -> after):")
    for cat in sorted(set(before) | set(after)):
        b, a = before.get(cat, 0), after.get(cat, 0)
        flag = "" if b == a else "   <-- changed"
        print(f"  {cat:<24} {b:>3} -> {a:>3}{flag}")
    print()
    print(f"suite_sha256: {sha}")
    print()
    print("exposition questions moved:")
    for it in out:
        if it["category"] == "verse_exposition":
            print(f"  {it['question']}")


if __name__ == "__main__":
    main()
