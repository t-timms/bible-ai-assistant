#!/usr/bin/env python3
"""Build the ``thematic_qa`` distillation inputs — the piece deferred from v3
(ROADMAP item 3) and retargeted for v3.1 after the 2026-09-03 re-eval pinned the
v3-SFT gap to the synthesis categories (character / context / cross_reference /
topical / theological_reliability, all ~0.37 fuzzy mean; ``verse_lookup`` 0.707
carries the average). Exposition is already fixed by #46 — this is the real lever.

For each hand-curated stem in ``training/v3_thematic_questions.json`` we generate
persona/paraphrase variants and, **for every variant**, pull a fresh Context block
from the live RAG retriever (``rag.retrieval._retrieve``). Output is the standard
``distill_answers.py`` input schema:

    {"id": str, "category": "thematic_qa", "context": str, "question": str}

Then (GPU): ``training/distill_answers.py`` writes the teacher answers, and
``training/assemble_v3.py --thematic <out>`` folds them into the v3.1 dataset.

Runs in ``.venv-rag`` (needs chromadb + sentence-transformers + rank_bm25 + the
built index at ``rag/chroma_db/``).

    # smoke — 20 inputs, still hits the real retriever
    python training/build_v3_thematic.py --out /tmp/thematic_smoke.jsonl --limit 20
    # full
    python training/build_v3_thematic.py --out data/raw_v3/thematic_inputs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval import _retrieve, verse_text_lookup  # noqa: E402

STEMS_PATH = PROJECT_ROOT / "training" / "v3_thematic_questions.json"

# Kept in sync with training/build_dataset_v2.py::_PERSONA_PREFIXES — inlined to
# avoid importing that module's dataset-fetch side effects here.
_PERSONA_PREFIXES = [
    "",
    "",
    "",
    "I'm preparing a Bible study and ",
    "Quick question for my devotions: ",
    "For a sermon I'm writing, ",
    "My kid asked me this and I want to get it right — ",
    "I'm double-checking something. ",
    "Teaching a class tonight. ",
    "New to the Bible here. ",
]


def _persona(q: str, rng: random.Random) -> str:
    p = rng.choice(_PERSONA_PREFIXES)
    if not p:
        return q
    if p.endswith((" ", "— ")) and q[:1].isupper() and not q.startswith("I "):
        q = q[0].lower() + q[1:]
    return p + q


# Per-shape paraphrase templates. {name} / {ref} / {a} / {b} filled from the stem.
_PARAPHRASE = {
    "character": [
        "Who was {name}?",
        "Tell me about {name} in the Bible.",
        "What does the Bible say about {name}?",
        "Give me a short summary of {name}'s life and role in Scripture.",
        "What is {name} known for in the Bible?",
        "Walk me through who {name} was and why they matter.",
    ],
    "context": [
        "What is the context of {ref}?",
        "What's the background of {ref}?",
        "Where does {ref} sit in Scripture, and what is it about?",
        "Explain the setting and situation of {ref}.",
        "What was going on around {ref} when it was written?",
        "Give me the historical and literary context of {ref}.",
    ],
    "cross_reference": [
        "How does {a} relate to {b}?",
        "What is the connection between {a} and {b}?",
        "How do {a} and {b} fit together?",
        "Explain how {a} and {b} are linked in Scripture.",
        "Why are {a} and {b} often read together?",
    ],
    # topical / theological stems: keep the wording, vary the framing only
    "topical": [
        "{q}",
        "{q}",
        "According to Scripture, {ql}",
        "In the Bible, {ql}",
    ],
}

_RE_CHAR = re.compile(r"who was\s+(.+?)\s*\??$", re.IGNORECASE)
_RE_CTX = re.compile(r"context of\s+(.+?)\s*\??$", re.IGNORECASE)
_RE_XREF = re.compile(r"how does\s+(.+?)\s+relate to\s+(.+?)\s*\??$", re.IGNORECASE)

# "Psalm 23" -> "Psalms 23:1"; "Romans 8:28" -> "Romans 8:28"; "Genesis 1" -> "Genesis 1:1";
# "Ephesians 2:8-9" -> "Ephesians 2:8". Named sections ("the Sermon on the Mount", "the book
# of Job") -> None, and those keep broad retrieval (surrounding context is what you want there).
_RE_PASSAGE = re.compile(
    r"^(?:the\s+)?((?:[1-3]\s+)?[A-Z][a-z]+(?:\s+of\s+[A-Z][a-z]+)?)\s+(\d+)(?::(\d+))?",
)
_BOOK_FIX = {"Psalm": "Psalms", "Canticles": "Song of Solomon"}


def _passage_ref(chunk: str) -> str | None:
    m = _RE_PASSAGE.match(chunk.strip())
    if not m:
        return None
    book, chap, verse = m.group(1).strip(), m.group(2), m.group(3) or "1"
    book = _BOOK_FIX.get(book, book)
    return f"{book} {chap}:{verse}"


# target variants per stem, by shape (synthesis shapes get more — they are the gap)
_TARGET = {"character": 25, "context": 25, "cross_reference": 25, "topical": 22}
# retrieval breadth by shape
# Matches rag_top_k=8 (rag/settings.py, bumped 2026-09-04 on measured recall
# gains) so training-time and serving-time retrieval depth stay in sync.
_TOPK = {"character": 8, "context": 8, "cross_reference": 8, "topical": 9}


def _shape_of(stem: dict) -> str:
    return stem.get("shape") or "topical"


def _fill(template: str, stem: dict) -> str | None:
    q = stem["q"]
    shape = _shape_of(stem)
    if shape == "character":
        m = _RE_CHAR.search(q)
        if not m:
            return None
        return template.format(name=m.group(1))
    if shape == "context":
        m = _RE_CTX.search(q)
        if not m:
            return None
        return template.format(ref=m.group(1))
    if shape == "cross_reference":
        m = _RE_XREF.search(q)
        if not m:
            return None
        return template.format(a=m.group(1), b=m.group(2))
    # topical / theological
    ql = q[0].lower() + q[1:] if q[:1].isupper() else q
    return template.format(q=q, ql=ql)


def _variants(stem: dict, rng: random.Random) -> list[str]:
    shape = _shape_of(stem)
    templates = _PARAPHRASE["topical" if shape not in _PARAPHRASE else shape]
    want = _TARGET.get(shape, 20)
    seen: set[str] = set()
    out: list[str] = []
    tries = 0
    while len(out) < want and tries < want * 8:
        tries += 1
        t = rng.choice(templates)
        base = _fill(t, stem)
        if not base:
            continue
        v = _persona(base, rng)
        key = re.sub(r"\s+", " ", v.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


async def _build(stems: list[dict], limit: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    n_stems = 0
    for stem in stems:
        shape = _shape_of(stem)
        topk = _TOPK.get(shape, 8)
        variants = _variants(stem, rng)
        n_stems += 1

        # For "what is the context of <passage>?" pin the named passage and search on its
        # text (the #46 pattern) — otherwise dense/BM25 drifts to neighbouring chapters and
        # the passage itself never lands in context. Named sections that don't parse
        # ("the Sermon on the Mount") keep broad retrieval, which is right for them.
        pin_refs: list[str] | None = None
        search_q: str | None = None
        if shape == "context":
            m = _RE_CTX.search(stem["q"])
            ref = _passage_ref(m.group(1)) if m else None
            if ref:
                pin_refs = [ref]
                search_q = verse_text_lookup(ref) or None

        for v in variants:
            try:
                ctx = await _retrieve(v, top_k=topk, pin_refs=pin_refs, search_query=search_q)
            except Exception as e:  # keep going; a dead stem shouldn't kill the run
                print(f"  ! retrieve failed for {v!r}: {e}", file=sys.stderr, flush=True)
                continue
            if not ctx or not ctx.strip():
                continue
            rows.append(
                {
                    "id": f"thematic_qa-{shape}-{len(rows):05d}",
                    "category": "thematic_qa",
                    "context": ctx.strip(),
                    "question": v,
                }
            )
            if limit and len(rows) >= limit:
                print(f"[limit {limit}] stopping after {n_stems} stems", flush=True)
                return rows
        if n_stems % 10 == 0:
            print(f"  {n_stems}/{len(stems)} stems -> {len(rows)} inputs", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="stop after N inputs (smoke)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = json.loads(STEMS_PATH.read_text(encoding="utf-8"))
    stems = meta["questions"]
    print(
        f"stems: {len(stems)}  (out={args.out}, limit={args.limit}, seed={args.seed})", flush=True
    )

    rows = asyncio.run(_build(stems, args.limit, args.seed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_shape: dict[str, int] = {}
    for r in rows:
        parts = r["id"].split("-")
        s = parts[1] if len(parts) >= 3 else "?"
        by_shape[s] = by_shape.get(s, 0) + 1
    print(f"wrote {len(rows)} inputs -> {args.out}", flush=True)
    for s, c in sorted(by_shape.items()):
        print(f"  {s}: {c}", flush=True)


if __name__ == "__main__":
    main()
