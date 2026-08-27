#!/usr/bin/env python3
"""
V2 dataset engine: full-canon, multi-translation training data at scale.

Targets ~50k examples across recall- and reasoning-heavy categories:

  1. verse_recall            ref -> verbatim text (per public-domain translation)
  2. translation_specific    "Quote X in KJV/ASV/WEB/DARBY/YLT/BBE"
  3. reverse_lookup          distinctive text snippet -> citation
  4. near_miss_guard         subtly-wrong quote -> correction (anti-hallucination)
  5. passage_recall          contiguous verse spans with range citations
  6. cross_reference_chains  TSK cross-references (openbible.info, CC-BY)
  7. topical_collections     theme -> canon-wide verse lists
  + small inherited pools of general / meta / refusal examples from v1 builders

All scripture sources are public domain (nigelmsipa/public-domain-bibles,
EyasuTew/bible_databases cross-references). Cross-reference data is CC-BY
(openbible.info) — used for training pairs only, attribution recorded in the
manifest sidecar.

Usage:
    python training/build_dataset_v2.py                      # full build (downloads+cache)
    python training/build_dataset_v2.py --offline-only       # cache/no-network smoke build
    python training/build_dataset_v2.py --limit-per-cat 200  # small QA build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.prompt_format import augment_question  # noqa: E402
from training.dataset_builder import (  # noqa: E402
    _msg,
    build_general_assistant,
    build_meta_questions,
    build_refusals,
    dedupe_by_normalized_question,
    filter_contaminated,
    load_contamination_questions,
    load_system_prompt,
)

RANDOM_SEED = 20260827
random.seed(RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════════════
# Public-domain source registry (URLs verified live 2026-08-27)
# ═══════════════════════════════════════════════════════════════════════

RAW_BASE = "https://raw.githubusercontent.com/nigelmsipa/public-domain-bibles/master"
XREF_URL = "https://raw.githubusercontent.com/EyasuTew/bible_databases/master/cross_references.txt"
TRANSLATIONS = ("KJV", "ASV", "WEB", "DARBY", "YLT", "BBE")

_loaded_sources: dict[str, dict] = {}
_xref_sha: str | None = None

_REF_LINE = re.compile(
    r"^(?P<book>[1-3]?\s?[A-Za-z ]+?)\s+(?P<ch>\d+)[:.](?P<v>\d+)\t(?P<text>.+)$"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_or_cache(name: str, url: str, cache_dir: Path, offline_only: bool) -> bytes | None:
    """Download into cache dir; reuse if present; return None when offline & missing."""
    path = cache_dir / name
    if path.exists():
        return path.read_bytes()
    if offline_only:
        print(f"  [skip] {name}: not cached and --offline-only")
        return None
    print(f"  [fetch] {name} <- {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bible-ai-assistant-v2"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (pinned https hosts)
            data = resp.read()
    except Exception as exc:  # pragma: no cover - network variance
        print(f"  [warn] {name} download failed: {exc}")
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def parse_translation_text(raw: str) -> dict[tuple[str, int, int], str]:
    """Parse 'Book C:V\\ttext' TSV (2 header lines) into canonical verses."""
    verses: dict[tuple[str, int, int], str] = {}
    for line in raw.splitlines()[2:]:
        m = _REF_LINE.match(line.strip())
        if not m:
            continue
        book = re.sub(r"\s+", " ", m.group("book")).strip()
        key = (book.title().replace("Of ", "of "), int(m.group("ch")), int(m.group("v")))
        verses[key] = m.group("text").strip()
    return verses


def load_translations(cache_dir: Path, offline_only: bool) -> dict[str, dict]:
    corpus: dict[str, dict] = {}
    for t in TRANSLATIONS:
        data = fetch_or_cache(f"{t}.txt", f"{RAW_BASE}/{t}.txt", cache_dir, offline_only)
        if not data:
            continue
        parsed = parse_translation_text(data.decode("utf-8", errors="replace"))
        if len(parsed) < 10_000:
            print(f"  [warn] {t}: only {len(parsed)} verses parsed — excluding")
            continue
        _loaded_sources[t] = {
            "url": f"{RAW_BASE}/{t}.txt",
            "sha256": sha256_bytes(data),
            "verses": len(parsed),
        }
        corpus[t] = {"verses": parsed}
        print(f"  [ok] {t}: {len(parsed)} verses")
    return corpus


def load_crossrefs(cache_dir: Path, offline_only: bool) -> list[tuple[str, str, int]]:
    """TSK cross-refs: rows of 'Gen.1.1<TAB>Isa.65.17<TAB>votes'."""
    data = fetch_or_cache("cross_references.txt", XREF_URL, cache_dir, offline_only)
    if not data:
        return []
    refs: list[tuple[str, str, int]] = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or parts[0].startswith("#"):
            continue
        try:
            refs.append((parts[0].strip(), parts[1].strip(), int(parts[2])))
        except ValueError:
            continue
    global _xref_sha
    _xref_sha = sha256_bytes(data)
    print(f"  [ok] cross_references: {len(refs)} edges")
    return refs


# ═══════════════════════════════════════════════════════════════════════
# OSIS short-code <-> display-book mapping for cross-refs
# ═══════════════════════════════════════════════════════════════════════

_OSIS_TO_BOOK = {
    "Gen": "Genesis",
    "Exod": "Exodus",
    "Lev": "Leviticus",
    "Num": "Numbers",
    "Deut": "Deuteronomy",
    "Josh": "Joshua",
    "Judg": "Judges",
    "Ruth": "Ruth",
    "Sam": "Samuel",
    "Kgs": "Kings",
    "Chr": "Chronicles",
    "Ezra": "Ezra",
    "Neh": "Nehemiah",
    "Esth": "Esther",
    "Job": "Job",
    "Ps": "Psalms",
    "Prov": "Proverbs",
    "Eccl": "Ecclesiastes",
    "Song": "Song of Solomon",
    "Isa": "Isaiah",
    "Jer": "Jeremiah",
    "Lam": "Lamentations",
    "Ezek": "Ezekiel",
    "Dan": "Daniel",
    "Hos": "Hosea",
    "Joel": "Joel",
    "Amos": "Amos",
    "Obad": "Obadiah",
    "Jonah": "Jonah",
    "Mic": "Micah",
    "Nah": "Nahum",
    "Hab": "Habakkuk",
    "Zeph": "Zephaniah",
    "Hag": "Haggai",
    "Zech": "Zechariah",
    "Mal": "Malachi",
    "Matt": "Matthew",
    "Mark": "Mark",
    "Luke": "Luke",
    "John": "John",
    "Acts": "Acts",
    "Rom": "Romans",
    "Cor": "Corinthians",
    "Gal": "Galatians",
    "Eph": "Ephesians",
    "Phil": "Philippians",
    "Col": "Colossians",
    "Thess": "Thessalonians",
    "Tim": "Timothy",
    "Titus": "Titus",
    "Phlm": "Philemon",
    "Heb": "Hebrews",
    "Jas": "James",
    "Pet": "Peter",
    "Jude": "Jude",
    "Rev": "Revelation",
}


def osis_to_ref(code: str) -> tuple[str, int, int] | None:
    """'Gen.1.1' or '1Sam.15.22' or '2Cor.5.17' -> canonical triple."""
    m = re.match(r"^([12]?)([A-Za-z]+)\.(\d+)\.(\d+)$", code.strip())
    if not m:
        return None
    num, book, ch, v = m.groups()
    base = _OSIS_TO_BOOK.get(book)
    if base is None:
        return None
    prefix = f"{num} {base}" if num else base
    return (prefix, int(ch), int(v))


_ORDINALS = {"first": "1 ", "second": "2 ", "third": "3 "}


def book_key(triple: tuple[str, int, int]) -> tuple[str, int, int]:
    """Canonical form shared by both parsers ('First Samuel' vs '1 Samuel')."""
    b, c, v = triple
    b2 = re.sub(r"^(first|second|third)\s+", lambda m: _ORDINALS[m.group(1)], b.lower())
    return (b2, c, v)


def norm_index(verses: dict) -> dict:
    """One-time map: normalized triple -> original corpus key."""
    return {book_key(k): k for k in verses}


# ═══════════════════════════════════════════════════════════════════════
# Question/response phrasing pools
# ═══════════════════════════════════════════════════════════════════════

_RECALL_Q = [
    "What does {ref} say?",
    "Quote {ref}.",
    "Please show me {ref}.",
    "What is written in {ref}?",
    "Give me the text of {ref}.",
    "Recite {ref} exactly.",
]
_TRANSLATION_Q = [
    "What does {ref} say in the {tl}?",
    "Quote {ref} from the {tl}.",
    "Show me {ref} in {tl}.",
]
_REVERSE_Q = [
    'Where does Scripture say "{snip}"?',
    'Which verse contains "{snip}"?',
    'Find the reference that reads "{snip}".',
]
_NEARMISS_Q = [
    "Is this correct: {bad}? If not, fix it.",
    'Someone quoted {ref} as "{bad}". Is that right?',
    "Check this quotation against your index: {bad}",
]


def clip_snippet(text: str, max_chars: int = 60) -> str:
    """Take a clean opening chunk of a verse for reverse-lookup prompts."""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    return cut[: cut.rfind(" ") if " " in cut else max_chars].rstrip(",.;:") + "..."


# ═══════════════════════════════════════════════════════════════════════
# Category generators
# ═══════════════════════════════════════════════════════════════════════


def gen_verse_recall(corpus, sp, n):
    out = []
    per_tl = max(1, n // max(len(corpus), 1))
    for tl, src in corpus.items():
        keys = sorted(src["verses"])
        picks = random.sample(keys, min(per_tl, len(keys)))
        for key in picks:
            q = random.choice(_RECALL_Q).format(ref=_display(key))
            ctx = [(_display(key), src["verses"][key])]
            a = f"{_display(key)} ({tl}) says: \u201c{src['verses'][key]}\u201d"
            out.append(_msg(sp, augment_question(q, ctx), a))
    return out


def gen_translation_specific(corpus, sp, n):
    out = []
    tls = list(corpus)
    if len(tls) < 2:
        return out
    for _i in range(n):
        tl = random.choice(tls)
        other = random.choice([t for t in tls if t != tl])
        key = random.choice(sorted(corpus[tl]["verses"]))
        if key not in corpus[other]["verses"]:
            continue
        q = random.choice(_TRANSLATION_Q).format(ref=_display(key), tl=tl)
        ctx = [(_display(key), corpus[tl]["verses"][key])]
        a = (
            f"{_display(key)} ({tl}) says: \u201c{corpus[tl]['verses'][key]}\u201d\n\n"
            f"For comparison, {other} renders it: \u201c{corpus[other]['verses'][key]}\u201d"
        )
        out.append(_msg(sp, augment_question(q, ctx), a))
    return out


_SALIENT_MIN, _SALIENT_MAX = 40, 140


def gen_reverse_lookup(corpus, sp, n):
    out = []
    kjar = corpus.get("KJV") or next(iter(corpus.values()))
    candidates = [
        k
        for k, txt in kjar["verses"].items()
        if _SALIENT_MIN <= len(txt) <= _SALIENT_MAX
        and "'" not in txt[:20]
        and len(txt.split()) >= 7
    ]
    if not candidates:
        return out
    for key in random.sample(candidates, min(n, len(candidates))):
        snip = clip_snippet(kjar["verses"][key])
        q = random.choice(_REVERSE_Q).format(snip=snip)
        ctx = [(_display(key), kjar["verses"][key])]
        a = f"That wording comes from {_display(key)} (KJV): \u201c{kjar['verses'][key]}\u201d"
        out.append(_msg(sp, augment_question(q, ctx), a))
    return out


def gen_near_miss_guard(corpus, sp, n):
    """Teach discrimination between adjacent/easily-confused verses.

    Each example quotes verse X's text but labels it as a neighbouring ref
    (same chapter, classic off-by-one); the assistant corrects attribution.
    """
    out = []
    kjv = corpus.get("KJV")
    if not kjv:
        return out
    # group by (normalized book, chapter), but keep ORIGINAL keys for text lookup
    chapters: dict[tuple[str, int], list] = {}
    for key in kjv["verses"]:
        b, c, v = book_key(key)
        chapters.setdefault((b, c), []).append(key)
    trap_chapters = [
        (bc, sorted(ks, key=lambda k: k[2])) for bc, ks in chapters.items() if len(ks) >= 6
    ]
    if not trap_chapters:
        return out
    attempts = 0
    while len(out) < n and attempts < n * 10:
        attempts += 1
        bc, keys = random.choice(trap_chapters)
        i = random.randrange(0, len(keys) - 1)
        cur_key, nxt_key = keys[i], keys[i + 1]
        target = kjv["verses"][cur_key]
        next_text = kjv["verses"][nxt_key]
        ref = _display(cur_key)
        bad_quote = f"{ref} \u2014 {next_text}"
        ctx = [(ref, target), (_display(nxt_key), next_text)]
        q = random.choice(_NEARMISS_Q).format(ref=ref, bad=bad_quote)
        correct = (
            f"Not quite \u2014 those words are actually the next verse.\n"
            f"{ref} reads: \u201c{target}\u201d\n"
            f"{_display(nxt_key)} is: \u201c{next_text}\u201d"
        )
        out.append(_msg(sp, augment_question(q, ctx), correct))
    return out


def gen_passage_recall(corpus, sp, n):
    out = []
    kjv = corpus.get("KJV") or next(iter(corpus.values()))
    chapters: dict[tuple[str, int], list] = {}
    for key in kjv["verses"]:
        b, c, v = book_key(key)
        chapters.setdefault((b, c), []).append(key)
    long_chapters = [(bc, ks) for bc, ks in chapters.items() if len(ks) >= 12]
    for _ in range(min(n * 3, max(1, len(long_chapters)))):
        bc, keys = random.choice(long_chapters) if long_chapters else ((None, None), [])
        if not keys:
            break
        keys.sort(key=lambda k: k[2])
        start_i = random.randrange(0, len(keys) - 4)
        span = keys[start_i : start_i + random.choice((2, 3, 4))]
        texts = [kjv["verses"][k] for k in span]
        book_disp, ch = span[0][0], span[0][1]
        q = f"What do these verses say together: {book_disp} {ch}:{span[0][2]}-{span[-1][2]}?"
        body = "\n".join(f"{book_disp} {ch}:{k[2]} \u2014 {t}" for k, t in zip(span, texts, strict=True))
        ctx = [(f"{book_disp} {ch}:{k[2]}", t) for k, t in zip(span, texts, strict=True)]
        a = f"{book_disp} {ch}:{span[0][2]}-{span[-1][2]} (KJV):\n{body}"
        out.append(_msg(sp, augment_question(q, ctx), a))
    return out[:n]


def gen_cross_reference_chains(xrefs, corpus, sp, n):
    out = []
    strong = [r for r in xrefs if r[2] >= 4]
    if not strong or not corpus:
        return out
    primary_name, primary_src = next(iter(corpus.items()))
    primary = primary_src["verses"]
    idx = norm_index(primary)  # O(1) lookups instead of per-edge scan
    primary_idx = norm_index(primary)
    anchor_pool: dict[tuple[str, int, int], list] = {}
    for frm, to, votes in strong:
        f = osis_to_ref(frm)
        if f is None:
            continue
        hit = idx.get(book_key(f))
        if hit is None:
            continue
        anchor_pool.setdefault(hit, []).append((osis_to_ref(to), votes))
    usable = [k for k in anchor_pool if anchor_pool[k]]
    if not usable:
        return out
    for key in random.sample(usable, min(n, len(usable))):
        disp = _display(key)
        links = sorted(anchor_pool[key], key=lambda x: -(x[1] or 0))[:3]
        hits = []
        for to, _votes in links:
            if to is None:
                continue
            t_hit = primary_idx.get(book_key(to))
            if t_hit is not None:
                hits.append(t_hit)
        hits = list(dict.fromkeys(hits))
        if len(hits) < 2:
            continue
        disp_hits = [_display(h) for h in hits]
        q = f"What passages connect with {disp}?"
        a = (
            f"{disp} has strong scriptural echoes in:\n"
            + "\n".join(f"\u2022 {r}" for r in disp_hits)
            + "\n\nReading them together shows how the theme develops across the canon."
        )
        ctx = list(zip(disp_hits, [primary[h] for h in hits], strict=True))
        out.append(_msg(sp, augment_question(q, ctx), a))
    return out


_TOPICS = {
    "love": ("love", "lovest", "beloved"),
    "faith": ("faith", "believeth", "belief"),
    "grace": ("grace",),
    "hope": ("hope",),
    "forgiveness": ("forgive", "forgiveness", "forgave"),
    "peace": ("peace", "peaceable"),
    "wisdom": ("wisdom", "wise"),
    "prayer": ("pray", "prayer"),
    "light": ("light", "lighteth"),
    "justice": ("justice", "judgment", "righteous judgment"),
}


def gen_topical_collections(corpus, sp, n):
    out = []
    kjv = corpus.get("KJV") or next(iter(corpus.values()))
    buckets = {}
    for key, txt in kjv["verses"].items():
        low = txt.lower()
        for topic, terms in _TOPICS.items():
            if any(term in low for term in terms):
                buckets.setdefault(topic, []).append(key)
    topics = [t for t, ks in buckets.items() if len(ks) >= 5]
    if not topics:
        return out
    for _ in range(n):
        topic = random.choice(topics)
        picks = random.sample(buckets[topic], 5)
        lines = [
            f"\u2022 {_display(k)} \u2014 \u201c{clip_snippet(kjv['verses'][k], 90)}\u201d"
            for k in picks
        ]
        q = f"What verses would you point me to about {topic}?"
        ctx = [(_display(k), kjv["verses"][k]) for k in picks]
        a = (
            f"Here are five passages on {topic}, spanning old and new covenant writings:\n"
            + "\n".join(lines)
        )
        out.append(_msg(sp, augment_question(q, ctx), a))
    return out


def _display(key: tuple[str, int, int]) -> str:
    book, ch, v = key
    pretty = {
        "Psalms": "Psalm",
        "Song Of Solomon": "Song of Solomon",
    }.get(book, book)
    return f"{pretty} {ch}:{v}"


# ═══════════════════════════════════════════════════════════════════════
# Assembly: contamination filter + dedupe + manifest sidecar
# ═══════════════════════════════════════════════════════════════════════

# Inherited v1 behavior pools stay proportionally small; recall/reasoning dominate.
_INHERITED_BUDGETS = {"general": 400, "meta": 200, "refusals": 300}


def build_all(limit_per_cat: int | None, offline_only: bool = False) -> dict:
    cache_dir = PROJECT_ROOT / "data" / "raw_v2"
    print("== loading translations ==")
    corpus = load_translations(cache_dir, offline_only)
    if not corpus:
        raise SystemExit(
            "No scripture sources available. Run once with network access "
            "(or pre-seed data/raw_v2/KJV.txt etc.) so verse-grounded data can be built."
        )
    xrefs = load_crossrefs(cache_dir, offline_only)
    sp = load_system_prompt(PROJECT_ROOT, for_training=True)

    n = limit_per_cat or 6000  # per-category default toward ~50k total
    budgets = {
        "verse_recall": (gen_verse_recall, n),
        "translation_specific": (gen_translation_specific, min(n, 4000)),
        "reverse_lookup": (gen_reverse_lookup, min(n, 6000)),
        "near_miss_guard": (gen_near_miss_guard, min(n, 5000)),
        "passage_recall": (gen_passage_recall, min(n, 6000)),
        "cross_reference_chains": (gen_cross_reference_chains, min(n, 8000)),
        "topical_collections": (gen_topical_collections, min(n, 4000)),
    }

    examples: dict[str, list] = {}
    for name, (fn, budget) in budgets.items():
        made = (
            fn(corpus, sp, budget)
            if fn is not gen_cross_reference_chains
            else fn(xrefs, corpus, sp, budget)
        )
        examples[name] = made
        print(f"[{name}] {len(made)}")

    for pool, count in _INHERITED_BUDGETS.items():
        builder = {
            "general": build_general_assistant,
            "meta": build_meta_questions,
            "refusals": build_refusals,
        }[pool]
        got = builder(sp)
        if len(got) > count:
            random.shuffle(got)
            got = got[:count]
        examples[pool] = got
        print(f"[{pool}:inherited] {len(got)}")

    return examples


def finalize(examples: dict, output_path: Path) -> dict:
    contaminated = load_contamination_questions(PROJECT_ROOT)
    flat, dropped_by_cat = [], {}
    for cat, items in examples.items():
        clean, removed = filter_contaminated(items, contaminated)
        deduped, dupes = dedupe_by_normalized_question(clean)
        dropped_by_cat[cat] = removed + dupes
        for ex in deduped:
            ex["category"] = cat
        flat.extend(deduped)
    random.shuffle(flat)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_id": "bible_assistant_v2_train",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": random.getstate()[1][0],
        "counts_dropped_contamination_or_dupes": dropped_by_cat,
        "examples": flat,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    counter = Counter(ex["category"] for ex in flat)
    manifest = {
        "protocol_id": "bible_assistant_v2_train",
        "total": len(flat),
        "per_category": dict(counter),
        "sources": dict(_loaded_sources),
        "cross_references": {"sha256": _xref_sha, "license": "CC-BY openbible.info"},
        "note": "Eval-only suites excluded via dataset_builder decontamination.",
    }
    out_manifest = output_path.with_suffix(".manifest.json")
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {len(flat)} examples -> {output_path}")
    print(json.dumps(counter, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/train_v2.json")
    ap.add_argument("--limit-per-cat", type=int, default=None)
    ap.add_argument("--offline-only", action="store_true", help="use only cached sources")
    ns = ap.parse_args()
    examples = build_all(ns.limit_per_cat, ns.offline_only)
    finalize(examples, ns.output)


if __name__ == "__main__":
    main()
