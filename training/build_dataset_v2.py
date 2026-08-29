#!/usr/bin/env python3
"""
V2 dataset engine: full-canon, multi-translation training data at scale.

Scripture-citation categories (capped down from the v1-era ~61k so the
general/reasoning blend can reach the catastrophic-forgetting floor):

  1. verse_recall            ref -> verbatim text (per public-domain translation)
  2. translation_specific    "Quote X in KJV/ASV/WEB/DARBY/YLT/BBE"
  3. reverse_lookup          distinctive text snippet -> citation
  4. near_miss_guard         subtly-wrong quote -> correction (anti-hallucination)
  5. passage_recall          contiguous verse spans with range citations
  6. cross_reference_chains  TSK cross-references (openbible.info, CC-BY)
  7. topical_collections     theme -> canon-wide verse lists
  8. chapter_context         unique-trigram opener -> reference

Reasoning / behaviour categories (added 2026-08-28, the "full upgrade"):

  9. grounded_exegesis       verse text + Matthew Henry commentary (CC0) in
                             context -> interpretive answer that rests on the
                             provided exposition, not free recall
 10. pastoral_triage         human-escalation, tradition-aware framing, and
                             calibrated abstention (aligned to FMG-Bench's
                             rubric dimensions, not its held-out scenarios)
 11. general_blend           HuggingFaceTB/smoltalk2 (Apache-2.0) replay —
                             ~25% of the mix so Qwen3.5 keeps its instruction-
                             following and reasoning; <think> traces stripped
  + small inherited pools of general / meta / refusal examples from v1 builders

All scripture sources are public domain (nigelmsipa/public-domain-bibles,
EyasuTew/bible_databases cross-references, Matthew Henry's Commentary via
codeberg.org/revisedcommonversion). Cross-reference data is CC-BY
(openbible.info). Every source's SHA + license is recorded in the manifest
sidecar. Persona/framing prefixes are applied probabilistically so the
fine-tune sees each task phrased the way real users ask it.

Usage:
    python training/fetch_mhc_commentary.py                  # once: build the commentary cache
    python training/build_dataset_v2.py                      # full build (downloads+cache)
    python training/build_dataset_v2.py --offline-only       # cache/no-network smoke build (no blend)
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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.prompt_format import augment_question  # noqa: E402
from training.dataset_builder import (  # noqa: E402
    _msg,
    _msg_multi,
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

_ALLOWED_SCHEME, _ALLOWED_HOST = "https", "raw.githubusercontent.com"
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
    if not url.startswith(f"{_ALLOWED_SCHEME}://{_ALLOWED_HOST}/"):
        raise ValueError(f"Refusing non-allowlisted source URL: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bible-ai-assistant-v2"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310 - scheme+host validated above
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
# General / reasoning replay blend — HuggingFaceTB/smoltalk2 (Apache-2.0)
# ═══════════════════════════════════════════════════════════════════════
# Catastrophic-forgetting guard: the eight scripture-citation categories are
# ~100% of the raw corpus, and Unsloth's Qwen3.5 guidance wants >=20-25%
# general/reasoning data or the base loses instruction-following and reasoning.
# smoltalk2 is the SFT mixture behind SmolLM3 (current SOTA open small model):
# benchmark-decontaminated, explicit think/no_think split, permissive.
# `<think>...</think>` traces are STRIPPED from the blended examples so the
# reasoning-shaped *prompts and answers* transfer without teaching a thinking
# format that would fight the grounded, no-think Bible answers.

_SMOLTALK2_REPO = "HuggingFaceTB/smoltalk2"
_SMOLTALK2_MIX = {  # split -> relative weight; ~40% think / ~60% no_think
    # no_think: general instruction, chat, rewrite/summarize, tool, system-following
    "smoltalk_smollm3_smol_magpie_ultra_no_think": 24,
    "OpenHermes_2.5_no_think": 13,
    "tulu_3_sft_personas_instruction_following_no_think": 7,
    "smoltalk_smollm3_explore_instruct_rewriting_no_think": 4,
    "smoltalk_smollm3_smol_rewrite_no_think": 3,
    "smoltalk_smollm3_smol_summarize_no_think": 3,
    "smoltalk_smollm3_everyday_conversations_no_think": 2,
    "smoltalk_smollm3_systemchats_30k_no_think": 3,
    # think (reasoning) — traces stripped, hard prompts + final answers kept
    "OpenThoughts3_1.2M_think": 22,
    "Mixture_of_Thoughts_science_no_think": 6,
    "multi_turn_reasoning_if_think": 5,
    "s1k_1.1_think": 2,
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think": 3,
}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_smoltalk2_meta: dict[str, dict] = {}


def _clean_turns(messages: list[dict]) -> list[tuple[str, str]] | None:
    """Return [(user, assistant), ...] with think-blocks stripped; None if unusable."""
    turns, pending_user = [], None
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "system":
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            ans = _THINK_RE.sub("", content).strip()
            if pending_user and ans:
                turns.append((pending_user, ans))
            pending_user = None
    return turns or None


def load_smoltalk2_blend(sp: str, total: int, offline_only: bool) -> list[dict]:
    if offline_only or total <= 0:
        print("  [skip] smoltalk2 blend (offline-only or zero budget)")
        return []
    try:
        import time

        from datasets import DownloadConfig, load_dataset  # type: ignore[import]  # lazy
    except ImportError:
        print("  [warn] smoltalk2 blend: `datasets` not installed — skipping")
        return []

    # authenticated streaming + generous retries: HF rate-limits (429) anonymous
    # parquet range-reads hard, and the retry ceiling in streaming mode is low.
    dl = DownloadConfig(token=True, max_retries=8)

    wsum = sum(_SMOLTALK2_MIX.values())
    out: list[dict] = []
    seen: set[str] = set()  # normalized first-user text — pre-empt finalize()'s dedupe
    for split, weight in _SMOLTALK2_MIX.items():
        # over-fetch ~1.3x: cross-split near-duplicates and finalize()'s
        # normalized-question dedupe otherwise pull the kept total under target.
        quota = max(1, round(total * weight / wsum))
        target = round(quota * 1.3)
        got = 0
        try:
            ds = load_dataset(
                _SMOLTALK2_REPO,
                "SFT",
                split=split,
                streaming=True,
                token=True,
                download_config=dl,
            )
            for row in ds:
                if got >= target:
                    break
                msgs = row.get("messages") or row.get("conversations")
                if not isinstance(msgs, list):
                    continue
                turns = _clean_turns(msgs)
                if not turns or sum(len(u) + len(a) for u, a in turns) > 6000:
                    continue
                key = re.sub(r"\s+", " ", turns[0][0].lower()).strip()[:200]
                if key in seen:
                    continue
                seen.add(key)
                out.append(_msg_multi(sp, turns))
                got += 1
        except Exception as exc:  # pragma: no cover - network/version variance
            print(f"  [warn] smoltalk2 {split}: {exc!r} — keeping {got} so far")
        _smoltalk2_meta[split] = {"requested": quota, "kept": got}
        print(f"  [ok] smoltalk2 {split}: {got} (target {target})")
        time.sleep(1.0)  # be gentle with the hub between splits
    random.shuffle(out)
    return out[:total]


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
    "How does {ref} read?",
    "Can you pull up {ref} for me?",
    "I'd like to see the wording of {ref}.",
    "Read me {ref}.",
    "What are the exact words of {ref}?",
    "Show {ref} word for word.",
    "Put {ref} in front of me.",
    "Give the verbatim text of {ref}.",
]
_TRANSLATION_Q = [
    "What does {ref} say in the {tl}?",
    "Quote {ref} from the {tl}.",
    "Show me {ref} in {tl}.",
    "How does the {tl} render {ref}?",
    "I'm reading the {tl} — what does it have for {ref}?",
    "Give me {ref} as the {tl} translates it.",
    "What's the {tl} wording of {ref}?",
]
_REVERSE_Q = [
    'Where does Scripture say "{snip}"?',
    'Which verse contains "{snip}"?',
    'Find the reference that reads "{snip}".',
    'I remember a verse that goes "{snip}" — what is it?',
    'What passage says "{snip}"?',
    'Can you place this line: "{snip}"?',
    'Which verse is "{snip}" from?',
    'Help me locate "{snip}" in the Bible.',
]
_NEARMISS_Q = [
    "Is this correct: {bad}? If not, fix it.",
    'Someone quoted {ref} as "{bad}". Is that right?',
    "Check this quotation against your index: {bad}",
    'A friend told me {ref} says "{bad}" — can you verify?',
    "I saw {ref} written as “{bad}”. Accurate?",
    "Double-check this for me: {bad}",
    'Is "{bad}" really how {ref} goes?',
]

# Persona/framing prefixes — applied probabilistically so the fine-tune sees the
# same underlying task asked the way real users ask it (study prep, teaching,
# devotion, skeptical checking), not one flat template voice. Kept short so they
# don't dominate the wrapped-context format the RAG server actually emits.
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


def _persona(q: str) -> str:
    """Prepend a light framing prefix ~40% of the time; lowercase the question's
    first letter when the prefix ends mid-sentence."""
    p = random.choice(_PERSONA_PREFIXES)
    if not p:
        return q
    if p.endswith((" ", "— ")) and q[:1].isupper() and not q.startswith("I "):
        q = q[0].lower() + q[1:]
    return p + q


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
            q = _persona(random.choice(_RECALL_Q).format(ref=_display(key)))
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
        q = _persona(random.choice(_TRANSLATION_Q).format(ref=_display(key), tl=tl))
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
        q = _persona(random.choice(_REVERSE_Q).format(snip=snip))
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
        q = _persona(random.choice(_NEARMISS_Q).format(ref=ref, bad=bad_quote))
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
    total_attempts = min(n * 6, 60_000)
    for _attempt in range(total_attempts):
        bc, keys = random.choice(long_chapters) if long_chapters else ((None, None), [])
        if not keys:
            break
        keys.sort(key=lambda k: k[2])
        start_i = random.randrange(0, len(keys) - 4)
        span = keys[start_i : start_i + random.choice((2, 3, 4))]
        texts = [kjv["verses"][k] for k in span]
        book_disp, ch = span[0][0], span[0][1]
        _span_ref = f"{book_disp} {ch}:{span[0][2]}-{span[-1][2]}"
        q = _persona(
            random.choice(
                [
                    f"What do these verses say together: {_span_ref}?",
                    f"Read me {_span_ref}.",
                    f"Show the full passage {_span_ref}.",
                    f"Quote {_span_ref} in order.",
                    f"I want {_span_ref} as one block.",
                ]
            )
        )
        body = "\n".join(
            f"{book_disp} {ch}:{k[2]} \u2014 {t}" for k, t in zip(span, texts, strict=True)
        )
        ctx = [(f"{book_disp} {ch}:{k[2]}", t) for k, t in zip(span, texts, strict=True)]
        a = f"{book_disp} {ch}:{span[0][2]}-{span[-1][2]} (KJV):\n{body}"
        out.append(_msg(sp, augment_question(q, ctx), a))
    return out[:n]


def gen_cross_reference_chains(xrefs, corpus, sp, n):
    out = []
    strong = [r for r in xrefs if r[2] >= 1]
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
        q = _persona(
            random.choice(
                [
                    f"What passages connect with {disp}?",
                    f"Which other verses echo {disp}?",
                    f"Give me cross-references for {disp}.",
                    f"Where else in Scripture does the theme of {disp} show up?",
                    f"I'm studying {disp} \u2014 what should I read alongside it?",
                ]
            )
        )
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
    "mercy": ("mercy", "merciful", "compassion"),
    "fear of the Lord": ("fear of the Lord", "fear the Lord", "reverence"),
    "repentance": ("repent", "repentance", "turn from"),
    "humility": ("humble", "humility", "lowly"),
    "patience": ("patience", "patient", "longsuffering"),
    "joy": ("joy", "rejoice", "joyful"),
    "trust": ("trust", "refuge", "strong tower"),
    "obedience": ("obey", "obedience", "keep my commandments"),
    "salvation": ("salvation", "saved", "redeemer"),
    "the kingdom of God": ("kingdom of God", "kingdom of heaven"),
    "suffering": ("affliction", "suffering", "tribulation"),
    "thankfulness": ("thanks", "thanksgiving", "give thanks"),
}


def gen_topical_collections(corpus, sp, n):
    """Anchored topical sets: the anchor reference varies, so every question
    survives dedupe while staying thematically real."""
    out = []
    kjv = corpus.get("KJV") or next(iter(corpus.values()))
    buckets = {}
    for key, txt in kjv["verses"].items():
        low = txt.lower()
        for topic, terms in _TOPICS.items():
            if any(term in low for term in terms):
                buckets.setdefault(topic, []).append(key)
    topics = [t for t, ks in buckets.items() if len(ks) >= 8]
    if not topics:
        return out
    facets = [
        ("anchored", "{topic} study plan built around {ref}?"),
        ("list", "Beyond {ref}, what else speaks about {topic}?"),
        ("chain", "Give me a {topic} chain starting at {ref}."),
        ("survey", "Survey Scripture on {topic}; begin from {ref}."),
        ("five", "Five verses about {topic} — include {ref} among them."),
    ]
    for _ in range(n):
        topic = random.choice(topics)
        anchor = random.choice(buckets[topic])
        picks_pool = [k for k in buckets[topic] if k != anchor]
        if len(picks_pool) < 4:
            continue
        picks = [anchor, *random.sample(picks_pool, 4)]
        verb = random.choice(facets)[1]
        q = _persona(verb.format(topic=topic, ref=_display(anchor)))
        lines = [f"• {_display(k)} — “{clip_snippet(kjv['verses'][k], 90)}”" for k in picks]
        a = (
            f"Here are five passages on {topic}, spanning old and new covenant writings:\n"
            + "\n".join(lines)
        )
        ctx = [(_display(k), kjv["verses"][k]) for k in picks]
        out.append(_msg(sp, augment_question(q, ctx), a))
    return out


# ═══════════════════════════════════════════════════════════════════════
# Chapter-context recognition (leading-trigram uniqueness guarantees that
# every question is unambiguous corpus-wide and survives dedupe)
# ═══════════════════════════════════════════════════════════════════════

_TRIGRAMS = re.compile(r"[a-z']{4,}")


def gen_chapter_context(corpus, sp, n):
    """'Where does it say "<distinctive opening>"?' -> book/chapter + verse."""
    out = []
    kjv = corpus.get("KJV") or next(iter(corpus.values()))
    counts: dict[str, int] = {}
    for txt in kjv["verses"].values():
        toks = _TRIGRAMS.findall(txt.lower())
        seen = set()
        for i in range(len(toks) - 2):
            gram = " ".join(toks[i : i + 3])
            if gram not in seen:
                seen.add(gram)
                counts[gram] = counts.get(gram, 0) + 1
    unique_openers = []
    for key, txt in kjv["verses"].items():
        if not (40 <= len(txt) <= 160) or len(txt.split()) < 7:
            continue
        low = txt.lower()
        toks = _TRIGRAMS.findall(low)
        for i in range(min(6, len(toks) - 2)):
            gram_tokens = toks[i : i + 3]
            if counts.get(" ".join(gram_tokens), 0) != 1:
                continue
            # tokens may sit apart due to punctuation -> tolerant separators
            pat = re.compile(r"[^a-z']{0,4}".join(map(re.escape, gram_tokens)), re.I)
            m = pat.search(txt)
            if not m:
                continue
            snippet = txt[max(0, m.start() - 8) : m.end() + 12].strip()
            unique_openers.append((key, clip_snippet(snippet, 70)))
            break
    random.shuffle(unique_openers)
    qs = [
        'Where does the Bible say "{snip}"?',
        'Locate this line in Scripture: "{snip}".',
        'I remember a passage going "{snip}" — what is its reference?',
        'What verse opens with "{snip}"?',
        'A line stuck with me: "{snip}". Where is it?',
        'Can you find "{snip}" for me?',
    ]
    for key, snip in unique_openers[:n]:
        q = _persona(random.choice(qs).format(snip=snip))
        text = kjv["verses"][key]
        ctx = [(_display(key), text)]
        a = f"That\u2019s {_display(key)} (KJV): \u201c{text}\u201d"
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
# Category: grounded exegesis (Matthew Henry's Commentary, CC0 / public domain)
# ═══════════════════════════════════════════════════════════════════════
# Teaches interpretive answers that stay GROUNDED IN PROVIDED COMMENTARY TEXT
# rather than free-recalled exposition (the fabrication risk this project
# exists to reduce). The commentary excerpt is injected as an explicit context
# entry alongside the verse text, matching the wrapped-context format the RAG
# server emits — with the caveat that rag_server.py must gain a commentary
# retrieval path before a model trained on this is served (tracked in
# docs/V2_EXECUTION_PLAN.md). Source built by training/fetch_mhc_commentary.py.

_MHC_CACHE = PROJECT_ROOT / "data" / "raw_v2" / "mhc_commentary.json"
_mhc_meta: dict[str, object] = {}

_EXEGESIS_Q = [
    "What does {ref} mean?",
    "Help me understand {ref}.",
    "Explain the significance of {ref}.",
    "What's the point of {ref}?",
    "How should I read {ref}?",
    "Walk me through {ref}.",
    "What is {ref} teaching?",
    "I'm stuck on {ref} — what's going on here?",
]


def _condense(text: str, max_chars: int = 900) -> str:
    """Trim a commentary block to a coherent opening chunk, cut on a sentence."""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(": "))
    return (cut[: end + 1] if end > max_chars * 0.5 else cut).rstrip() + " …"


def gen_grounded_exegesis(corpus, sp, n):
    """Verse text + Matthew Henry commentary excerpt in context -> grounded
    interpretive answer that explicitly rests on the provided exposition."""
    if not _MHC_CACHE.exists():
        print(
            f"  [skip] grounded_exegesis: {_MHC_CACHE.name} not built "
            f"(run training/fetch_mhc_commentary.py)"
        )
        return []
    blob = json.loads(_MHC_CACHE.read_text(encoding="utf-8"))
    records = blob["records"] if isinstance(blob, dict) else blob
    if isinstance(blob, dict):
        _mhc_meta.update(blob.get("meta", {}))
    kjv = corpus.get("KJV") or next(iter(corpus.values()))
    kjv_idx = norm_index(kjv["verses"])
    out = []
    random.shuffle(records)
    # if the budget exceeds the record count, emit two distinctly-phrased
    # questions per passage (same grounded answer) to reach it.
    variants = 2 if n > len(records) else 1
    for rec in records:
        if len(out) >= n:
            break
        book, ch = rec["book"], int(rec["chapter"])
        v_start, v_end = int(rec["verse_start"]), int(rec["verse_end"])
        commentary = _condense(rec["text"])
        if len(commentary) < 200:
            continue
        # gather the actual verse text for the range (KJV) as primary context
        verse_ctx = []
        for vnum in range(v_start, min(v_end, v_start + 6) + 1):
            hit = kjv_idx.get(book_key((book, ch, vnum)))
            if hit is not None:
                verse_ctx.append((f"{_display((book, ch, vnum))}", kjv["verses"][hit]))
        if not verse_ctx:
            continue
        span_ref = (
            f"{_display((book, ch, v_start))}"
            if v_start == v_end
            else f"{_display((book, ch, v_start))}-{v_end}"
        )
        ctx = [*verse_ctx, (f"Matthew Henry's Commentary on {span_ref}", commentary)]
        a = (
            f"Working from Matthew Henry's commentary on {span_ref}:\n\n"
            f"{commentary}\n\n"
            f"(Matthew Henry's Commentary on the Whole Bible, public domain. "
            f"Christians in different traditions read some details of this passage "
            f"differently; the verse text above is the fixed point.)"
        )
        for qt in random.sample(_EXEGESIS_Q, k=min(variants, len(_EXEGESIS_Q))):
            if len(out) >= n:
                break
            q = _persona(qt.format(ref=span_ref))
            out.append(_msg(sp, augment_question(q, ctx), a))
    return out


# ═══════════════════════════════════════════════════════════════════════
# Category: pastoral triage & calibrated abstention
# ═══════════════════════════════════════════════════════════════════════
# Aligned to the *rubric dimensions* FMG-Bench scores (human-escalation
# recognition, tradition-aware representation, pastoral safety) — NOT its
# scenarios, which are held out. Also teaches the model to abstain cleanly
# when the retrieved context does not contain what's being asked.

_PASTORAL_TRIAGE_QA: list[tuple[str, str]] = [
    (
        "I think I want to end my life. What does the Bible say?",
        "I'm really glad you told someone. This is more important than a Bible "
        "study question right now — please reach out to a crisis line today "
        "(in the US and Canada, call or text 988) and to someone who can be "
        "with you in person. If you're in immediate danger, call emergency "
        "services. Scripture speaks of God being near to the brokenhearted "
        "(Psalm 34:18), but you also need real people around you right now: a "
        "pastor, a counselor, a trusted friend. Please don't carry this alone.",
    ),
    (
        "My husband hits me. Doesn't the Bible say I have to submit and stay?",
        "Your safety comes first. Nothing in Scripture requires you to stay in "
        "a place where you are being harmed — protecting yourself and any "
        "children is right, not a failure of faith. Please contact a domestic "
        "violence hotline (in the US, 1-800-799-7233) and talk to a pastor or "
        "counselor who takes abuse seriously. This is something to work "
        "through with people who can help in person, not something to settle "
        "from a verse.",
    ),
    (
        "Should I divorce my spouse?",
        "That's a weighty, personal decision, and it's not one I should try to "
        "make for you from a distance. It deserves unhurried conversation with "
        "a pastor who knows your situation, and often a licensed counselor as "
        "well. Christians hold differing convictions about the biblical grounds "
        "for divorce (Matthew 19, 1 Corinthians 7 are the usual passages "
        "discussed). If there is abuse or danger involved, please prioritize "
        "safety and seek help now.",
    ),
    (
        "Is my dead grandmother in heaven?",
        "I can't tell you that — I don't know her heart or her life, and that "
        "judgment belongs to God alone, who is both perfectly just and rich in "
        "mercy. What Scripture does offer is comfort in grief (2 Corinthians "
        "1:3-4) and the assurance that God does what is right. A pastor who can "
        "sit with you would be a good person to talk to about your grief.",
    ),
    (
        "What medication should I take for my depression?",
        "I'm not able to give medical advice — that's a conversation for a "
        "doctor or psychiatrist, and seeing one is a wise and appropriate step, "
        "not a lack of faith. Alongside that, a pastor or Christian counselor "
        "can walk with you spiritually. Scripture doesn't treat caring for your "
        "body and mind as opposed to trusting God.",
    ),
    (
        "My teenager just told me they're gay. What do I do?",
        "First: keep loving them and keep the relationship open — that matters "
        "more than getting a perfect first response. This is something to "
        "process with your pastor and, if it helps, a counselor, rather than "
        "from a single answer. Christians and churches differ significantly on "
        "the underlying questions here; whatever your convictions, your child "
        "needs to know they're still your child and still loved.",
    ),
    (
        "Does God still love me after what I've done?",
        "Yes. Scripture is direct about this: nothing in all creation can "
        "separate you from the love of God in Christ (Romans 8:38-39), and "
        "if we confess our sins, he is faithful and just to forgive (1 John "
        "1:9). If the weight of it is heavy, please talk to a pastor — that's "
        "exactly what they're there for, and saying it out loud to another "
        "person often helps.",
    ),
    (
        "Is it a sin to be cremated?",
        "Christians disagree on this, and Scripture doesn't give a direct "
        "command either way. Some point to burial as the consistent biblical "
        "pattern and a picture of resurrection hope; others note that God is "
        "not limited by how a body is handled. It's a matter of conscience and "
        "family conviction — a good thing to talk through with your pastor.",
    ),
    (
        "Which denomination is the true church?",
        "I'm not going to rank traditions for you — faithful Christians land in "
        "different places on church government, the sacraments, and worship, "
        "and that's a question to work through with real teachers in a real "
        "congregation. What the historic creeds hold in common (the Trinity, "
        "the person and work of Christ, the resurrection) is the shared core.",
    ),
    (
        "Can you pray for me?",
        "I can't actually pray — I'm a tool, not a person before God. But I can "
        "help you find words if you'd like, and I'd encourage you to ask "
        "someone in your life to pray with you, and to bring it to God "
        "yourself; he invites exactly that (Philippians 4:6-7).",
    ),
    (
        "I'm being pressured to have an abortion. What should I do?",
        "That's a heavy situation and you shouldn't have to face it from a text "
        "box. Please talk with people who can actually be present — a pastor, "
        "and a pregnancy support center or counselor who can walk through your "
        "real options and any pressure or coercion you're under. If you feel "
        "unsafe, prioritize your safety first. Christians care deeply about "
        "both you and the life you're carrying; you deserve unhurried, "
        "in-person support.",
    ),
    (
        "My pastor did something that felt abusive. Am I sinning by leaving?",
        "No. Leaving a harmful situation is not sin. If a leader has abused "
        "their position, that is serious, and it may need to be reported — to "
        "the church's elders or governing body, and, where a crime may have "
        "occurred, to the authorities. Please talk to a trusted mature "
        "Christian outside that church, and consider a counselor. Your safety "
        "and integrity come first.",
    ),
    (
        "Is it okay for a Christian to drink alcohol?",
        "Christians land in different places here. Some hold that Scripture "
        "permits moderate drinking while strongly warning against drunkenness "
        "(Ephesians 5:18); others practice or teach total abstinence out of "
        "wisdom, conscience, or care for others (Romans 14). If alcohol has "
        "been a struggle for you or your family, that's worth weighing "
        "seriously and talking through with your pastor.",
    ),
    (
        "Will I lose my salvation if I keep sinning?",
        "This touches a real, long-standing difference between Christian "
        "traditions — some hold that a genuine believer is eternally secure, "
        "others that salvation can be forfeited by persistent, unrepentant "
        "rejection of Christ. Both sides agree that ongoing willful sin is "
        "serious and that genuine faith bears fruit. If you're anxious about "
        "where you stand, that's exactly the conversation to have with a "
        "pastor.",
    ),
    (
        "What does the Bible say about tattoos?",
        "Leviticus 19:28 is the verse usually raised. Christians read it "
        "differently: some see a timeless prohibition, many see a command "
        "tied to pagan mourning practices of that culture that doesn't carry "
        "over directly. It's widely treated as a matter of conscience "
        "(Romans 14). Your pastor can help you think it through in your own "
        "context.",
    ),
    (
        "Should women be allowed to be pastors?",
        "This is one of the most debated questions among faithful Christians. "
        "Complementarian churches reserve the office of pastor/elder for men, "
        "citing passages like 1 Timothy 2 and 3; egalitarian churches "
        "affirm women in every office, citing Galatians 3:28, Romans 16, and "
        "the ministry of women throughout Scripture. Denominations differ, and "
        "this is a question to work out within a specific church tradition.",
    ),
    (
        "Is the earth young or old? What does Genesis require?",
        "Bible-believing Christians hold several views: young-earth creationism, "
        "old-earth creationism, and evolutionary creationism, plus differing "
        "readings of the Genesis 1 'days'. They share the core confession that "
        "God is the Creator and humanity is made in his image. The age of the "
        "earth itself is not something the text settles for everyone, and "
        "sincere believers disagree.",
    ),
    (
        "My spouse had an affair. Does the Bible say I have to forgive and stay married?",
        "Forgiveness and reconciliation are not the same thing, and neither is "
        "instant. Many Christians understand Scripture to permit divorce in "
        "the case of adultery (Matthew 19:9) without requiring it. What to do "
        "here is a decision to make slowly, with a pastor and ideally a "
        "counselor who can hear the whole situation. If there's any abuse "
        "involved, safety comes first.",
    ),
    (
        "I don't feel God anymore and I'm not sure I believe. Am I lost?",
        "Doubt and dryness are part of many sincere believers' walk — you're "
        "not the first, and feelings aren't the measure of where you stand "
        "with God. This is worth talking through with a patient pastor or "
        "mature Christian friend rather than carrying alone. The Psalms give a "
        "lot of honest language for exactly this (Psalm 42, Psalm 88).",
    ),
    (
        "Someone I love just died. Why did God let this happen?",
        "I'm so sorry. That question has been cried out by faithful people "
        "throughout Scripture, and it doesn't have a tidy answer I can hand "
        "you. What I'd urge is not to grieve alone — a pastor, and people who "
        "can sit with you, matter more right now than an explanation. "
        "Scripture says God is near to the brokenhearted (Psalm 34:18) and "
        "that grief is real, not a lack of faith.",
    ),
    (
        "Can you tell me if my marriage is God's will?",
        "I can't discern that for you — I don't know you, your spouse, or your "
        "history, and that's not a call to make from a chat. A pastor who can "
        "meet with you both, and often a counselor, is the right place to work "
        "through it. Scripture's teaching on love, faithfulness, and "
        "reconciliation applies, but the specific decision needs real people "
        "who know you.",
    ),
    (
        "Is speaking in tongues for today, or did it stop with the apostles?",
        "Christians genuinely differ. Continuationists believe the miraculous "
        "gifts, including tongues, continue now; cessationists believe they "
        "ceased with the apostolic age or the completion of Scripture. Whole "
        "denominations line up on each side. Both affirm the Spirit's work in "
        "the church today — the disagreement is about form.",
    ),
    (
        "My child is being bullied and I want to hurt the other kid's parents. What does the Bible say?",
        "It makes sense to feel fierce about protecting your child. Scripture "
        "does call us away from personal vengeance (Romans 12:19) — but that's "
        "not the same as doing nothing. Work the real channels: the school, "
        "and if there's a threat of harm, the authorities. A pastor can help "
        "you carry the anger without it turning into something you'd regret.",
    ),
    (
        "Am I going to hell for being divorced?",
        "No — divorce is not an unforgivable sin, and Scripture doesn't teach "
        "that divorced people are beyond God's grace. Christians differ on "
        "when remarriage is permitted, and that's worth talking through with a "
        "pastor, but the fear you're describing isn't where the Bible leaves "
        "someone who has been through this.",
    ),
    (
        "Should I give money to this ministry that promises God will bless me back?",
        "Be careful with anyone who frames giving as a guaranteed transaction "
        "for personal wealth — that 'sow a seed and God must pay you back' "
        "teaching is widely regarded as a distortion of Scripture. Genuine "
        "biblical generosity is free, cheerful, and not a scheme (2 "
        "Corinthians 9:7). A trusted pastor can help you evaluate a specific "
        "ministry.",
    ),
    (
        "I'm struggling with an addiction and I'm ashamed to tell anyone at church.",
        "Shame thrives in secret, and it's worth breaking that even though "
        "it's hard. A pastor, a Christian counselor, or a recovery group "
        "(many churches host them) can meet you without condemnation. James "
        "5:16 ties confession and healing together for a reason. You don't "
        "have to have this beaten before you ask for help.",
    ),
    (
        "What do I do if I think a child at my church is being abused?",
        "If you suspect a child is being harmed, contact the authorities — in "
        "the US, child protective services or the police. Reporting is the "
        "protective, right thing to do; don't try to investigate it yourself "
        "or route it only through church leadership. A child's safety comes "
        "first, always.",
    ),
    (
        "Does God hate me? I feel like everything I do is wrong.",
        "No — that feeling is real, but it isn't a reliable read on how God "
        "sees you. Scripture describes God as slow to anger and abounding in "
        "steadfast love (Psalm 103:8). When the sense of being wrong is this "
        "heavy and constant, please tell a pastor and consider a counselor; "
        "that weight is worth having help to carry.",
    ),
    (
        "Is it a sin to take antidepressants or see a therapist?",
        "No. Caring for your mental health through a doctor or therapist is "
        "wise stewardship, not a failure of faith — the same way you'd treat "
        "any other illness. Many Christians do both: professional care and "
        "the support of their church. If a leader has told you otherwise, "
        "that's worth a second opinion.",
    ),
]

# Questions where the honest answer is "faithful Christians differ" — teaches
# tradition-aware framing rather than picking a side.
_TRADITION_AWARE_Q = [
    ("Do infants need to be baptized?", "infant vs. believer's baptism"),
    ("What happens to the bread and wine in communion?", "the nature of the Lord's Supper"),
    ("Are the gifts of prophecy and healing for today?", "continuationism vs. cessationism"),
    ("Is there a rapture, and when?", "views on the end times"),
    ("Can a Christian lose their salvation?", "eternal security vs. conditional perseverance"),
    ("How were the 'days' of Genesis 1 meant to be read?", "the age of creation"),
    ("Should the church have bishops, elders, or congregational rule?", "church government"),
    ("Is predestination taught in the Bible?", "Calvinist vs. Arminian readings"),
    ("Should Christians keep the Sabbath on Saturday or Sunday?", "the Christian Sabbath"),
    ("Is it okay to drink alcohol in moderation?", "conscience and Christian liberty"),
]

_ABSTENTION_Q = [
    'Where does the Bible say "{fabricated}"?',
    'Isn\'t there a verse that goes "{fabricated}"?',
    'Quote the verse about "{fabricated}".',
    'What\'s the reference for "{fabricated}"?',
    'Can you give me chapter and verse for "{fabricated}"?',
    'My friend says "{fabricated}" is in the Bible. Is it?',
]

# Sayings that are simply not in Scripture at all.
_NOT_IN_BIBLE = [
    "God helps those who help themselves",
    "cleanliness is next to godliness",
    "God will not give you more than you can handle",
    "God will never give you more than you can bear",
    "the eye is the window to the soul",
    "this too shall pass",
    "hate the sin, love the sinner",
    "charity begins at home",
    "God works in mysterious ways",
    "the Lord works in mysterious ways",
    "everything happens for a reason",
    "the family that prays together stays together",
    "to thine own self be true",
    "moderation in all things",
    "a penny saved is a penny earned",
    "God said it, I believe it, that settles it",
    "when God closes a door he opens a window",
    "the road to hell is paved with good intentions",
    "let go and let God",
    "an eye for an eye makes the whole world blind",
    "we accept the love we think we deserve",
    "honesty is the best policy",
    "patience is a virtue",
    "idle hands are the devil's workshop",
]

# Popular near-misquotes: the terse form isn't in the text, but a real verse
# is close — teach the correction, not a flat "not in the Bible".
_MISQUOTES = {
    "money is the root of all evil": (
        "That's a near-miss. 1 Timothy 6:10 says “the love of money is a root "
        "of all kinds of evil” (KJV: “the love of money is the root of all "
        "evil”) — it's the love of money, not money itself, and the wording "
        "is often shortened in a way that changes the point."
    ),
    "spare the rod, spoil the child": (
        "That exact phrase isn't in the Bible — it comes from a 17th-century "
        "poem. The verse people have in mind is Proverbs 13:24, “He who "
        "spares the rod hates his son, but he who loves him is diligent to "
        "discipline him.”"
    ),
    "pride comes before the fall": (
        "Close, but the wording is different. Proverbs 16:18 reads, “Pride "
        "goes before destruction, and a haughty spirit before a fall.”"
    ),
    "this is the day the Lord has made": (
        "That one IS in Scripture — Psalm 118:24: “This is the day that the "
        "Lord has made; let us rejoice and be glad in it.” (I only flag it "
        "because it's often quoted without the reference.)"
    ),
}


def _abstain_answer(saying: str) -> str:
    return (
        f"“{saying}” is a common saying, but it is not a Bible verse — it "
        f"doesn't appear in Scripture. I won't invent a reference for it. If "
        f"you tell me the idea you're after, I can point you to what the "
        f"Bible actually does say near it."
    )


def _tradition_answer(topic: str) -> str:
    return (
        f"Faithful, Bible-believing Christians genuinely differ on {topic}, "
        f"and whole denominations line up on each side. I can lay out the main "
        f"views and the passages each leans on, but I'm not going to hand you "
        f"one answer as the answer — that's something to work through within a "
        f"church tradition, with teachers who know you."
    )


def gen_pastoral_triage(sp, n):
    """Escalation / tradition-aware / calibrated-abstention behaviour. A
    hand-authored core, expanded only by question framing so the modelled
    answer stays fixed and correct. Interleaved so a small ``n`` still gets a
    mix of all three behaviours."""
    frames = [
        "{q}",
        "Someone asked me this and I didn't know what to say: {q}",
        "Be honest with me — {q_lower}",
        "{q} Please don't just give me a verse.",
        "I keep going back and forth on this. {q}",
        "{q} What would you actually tell someone?",
    ]
    triage_pairs = [
        (fr.format(q=q, q_lower=q[0].lower() + q[1:]), a)
        for q, a in _PASTORAL_TRIAGE_QA
        for fr in frames
    ]
    tradition_pairs = [(_persona(q), _tradition_answer(topic)) for q, topic in _TRADITION_AWARE_Q]
    abstain_pairs = [
        (_persona(qt.format(fabricated=s)), _abstain_answer(s))
        for s in _NOT_IN_BIBLE
        for qt in _ABSTENTION_Q
    ] + [
        (_persona(qt.format(fabricated=s)), corr)
        for s, corr in _MISQUOTES.items()
        for qt in _ABSTENTION_Q
    ]
    random.shuffle(triage_pairs)
    random.shuffle(tradition_pairs)
    random.shuffle(abstain_pairs)

    out: list[dict] = []
    pools = [iter(triage_pairs), iter(tradition_pairs), iter(abstain_pairs)]
    exhausted = [False, False, False]
    while len(out) < n and not all(exhausted):
        for i, it in enumerate(pools):
            if exhausted[i]:
                continue
            nxt = next(it, None)
            if nxt is None:
                exhausted[i] = True
                continue
            out.append(_msg(sp, nxt[0], nxt[1]))
            if len(out) >= n:
                break
    return out


# ═══════════════════════════════════════════════════════════════════════
# Assembly: contamination filter + dedupe + manifest sidecar
# ═══════════════════════════════════════════════════════════════════════

# Inherited v1 behavior pools stay proportionally small; recall/reasoning dominate.
_INHERITED_BUDGETS = {"general": 400, "meta": 200, "refusals": 300}


def build_all(
    limit_per_cat: int | None,
    offline_only: bool = False,
    blend_total: int = 13000,
    exegesis_n: int = 7000,
    triage_n: int = 400,
) -> dict:
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

    n = limit_per_cat or 12000
    smoke = limit_per_cat is not None and limit_per_cat < 500
    if smoke:
        blend_total = min(blend_total, n * 3)
        exegesis_n = min(exegesis_n, n)
        triage_n = min(triage_n, n)

    # Scripture-citation categories, capped down from the v1-era ~61k so the
    # general/reasoning blend can reach the >=20-25% catastrophic-forgetting
    # floor without an unwieldy total. Anti-hallucination (near_miss_guard) and
    # the two reasoning-leaning categories are trimmed least.
    budgets: dict[str, tuple[Callable[..., list], int]] = {
        "verse_recall": (gen_verse_recall, min(n, 5000)),
        "translation_specific": (gen_translation_specific, min(n, 4500)),
        "reverse_lookup": (gen_reverse_lookup, min(n, 4000)),
        "near_miss_guard": (gen_near_miss_guard, min(n, 4500)),
        "passage_recall": (gen_passage_recall, min(n, 5000)),
        "cross_reference_chains": (gen_cross_reference_chains, min(n, 5500)),
        "topical_collections": (gen_topical_collections, min(n, 4500)),
        "chapter_context": (gen_chapter_context, min(n, 4000)),
        "grounded_exegesis": (gen_grounded_exegesis, min(n, exegesis_n)),
    }

    examples: dict[str, list] = {}
    for name, (fn, budget) in budgets.items():
        made = (
            fn(xrefs, corpus, sp, budget)
            if fn is gen_cross_reference_chains
            else fn(corpus, sp, budget)
        )
        examples[name] = made
        print(f"[{name}] {len(made)}")

    # Pastoral triage / escalation / calibrated abstention (hand-authored core).
    examples["pastoral_triage"] = gen_pastoral_triage(sp, triage_n)
    print(f"[pastoral_triage] {len(examples['pastoral_triage'])}")

    # General / reasoning replay blend (smoltalk2) — the forgetting guard.
    examples["general_blend"] = load_smoltalk2_blend(sp, blend_total, offline_only)
    print(f"[general_blend] {len(examples['general_blend'])}")

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
    # Flat JSON array of chat examples — same shape as the v1 dataset_builder.py
    # output, so `datasets.load_dataset("json", ...)` in train_unsloth.py /
    # train_grpo.py consumes it directly. All provenance lives in the sidecar
    # manifest below (nothing is lost by not wrapping the array).
    output_path.write_text(json.dumps(flat, ensure_ascii=False, indent=1), encoding="utf-8")

    counter = Counter(ex["category"] for ex in flat)
    manifest = {
        "protocol_id": "bible_assistant_v2_train",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": random.getstate()[1][0],
        "total": len(flat),
        "per_category": dict(counter),
        "counts_dropped_contamination_or_dupes": dropped_by_cat,
        "sources": dict(_loaded_sources),
        "cross_references": {"sha256": _xref_sha, "license": "CC-BY openbible.info"},
        "commentary": {
            "source": "Matthew Henry's Commentary on the Whole Bible",
            "license": "CC0 / public domain",
            "via": "codeberg.org/revisedcommonversion/matthew-henry-commentary",
            **_mhc_meta,
        },
        "general_blend": {
            "repo": _SMOLTALK2_REPO,
            "license": "Apache-2.0 for new subsets; inherited subsets keep their "
            "upstream licenses (see repo card)",
            "think_blocks": "stripped from blended examples",
            "splits": dict(_smoltalk2_meta),
        },
        "note": "Eval-only suites excluded via dataset_builder decontamination.",
    }
    out_manifest = output_path.with_suffix(".manifest.json")
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {len(flat)} examples -> {output_path}")
    print(json.dumps(counter, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1])
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/train_v2.json")
    ap.add_argument("--limit-per-cat", type=int, default=None)
    ap.add_argument("--offline-only", action="store_true", help="use only cached sources")
    ap.add_argument(
        "--blend-total",
        type=int,
        default=13000,
        help="smoltalk2 general/reasoning replay examples (~40%% think)",
    )
    ap.add_argument(
        "--exegesis", type=int, default=7000, help="grounded-exegesis (Matthew Henry) examples"
    )
    ap.add_argument("--triage", type=int, default=400, help="pastoral-triage / abstention examples")
    ns = ap.parse_args()
    examples = build_all(
        ns.limit_per_cat,
        ns.offline_only,
        blend_total=ns.blend_total,
        exegesis_n=ns.exegesis,
        triage_n=ns.triage,
    )
    finalize(examples, ns.output)


if __name__ == "__main__":
    main()
