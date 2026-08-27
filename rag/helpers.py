"""Pure helper functions for the Bible AI RAG server.

No ChromaDB, sentence-transformers, or external I/O — safe to import in tests.
"""

from __future__ import annotations

import json as _json
import re
from typing import Any

from rag.response_cleanup import strip_model_thinking

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reciprocal Rank Fusion smoothing constant (standard default from the RRF paper)
RRF_K = 60

VERSES_COLLECTION = "bible_verses"
PASSAGES_COLLECTION = "bible_passages"
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "

# Schema version of the persisted RAG artifacts (Chroma collections, BM25 JSON,
# index_meta.json marker). Bump whenever build_index.py output changes shape or
# semantics — e.g. BM25 tokenization, HNSW distance space, embedding
# normalization, child_ids encoding. Loaders compare this against what is on
# disk and refuse stale artifacts loudly (log + skip/error) instead of serving
# silently wrong results.
INDEX_VERSION = 3

# Topical questions: pin a few high-signal verses so hybrid retrieval + passage expansion
# cannot drown the topic (e.g. marriage → unrelated "love" parables).
_TOPICAL_PIN_TABLE: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    # --- Relationships & Family ---
    (
        frozenset(
            {
                "marriage",
                "married",
                "marry",
                "spouse",
                "husband",
                "wife",
                "wedding",
                "divorce",
                "remarry",
            }
        ),
        ("Genesis 2:24", "Ephesians 5:31", "Matthew 19:5", "Mark 10:9"),
    ),
    (
        frozenset({"children", "parent", "father", "mother", "son", "daughter", "family"}),
        ("Ephesians 6:1", "Proverbs 22:6", "Colossians 3:20"),
    ),
    # --- Virtues & Character ---
    (
        frozenset({"forgiveness", "forgive", "forgiving", "pardon"}),
        ("Matthew 6:14", "Ephesians 4:32", "Colossians 3:13"),
    ),
    (
        frozenset({"love", "charity", "agape", "loving", "beloved"}),
        ("1 Corinthians 13:4", "John 3:16", "1 John 4:8"),
    ),
    (
        frozenset({"faith", "believe", "trust", "faithful"}),
        ("Hebrews 11:1", "Romans 10:17", "Ephesians 2:8"),
    ),
    (
        frozenset({"hope", "hoping", "future", "expectation"}),
        ("Romans 15:13", "Jeremiah 29:11", "Hebrews 6:19"),
    ),
    (
        frozenset({"joy", "rejoice", "gladness", "happiness"}),
        ("Philippians 4:4", "Psalms 16:11", "Galatians 5:22"),
    ),
    (
        frozenset({"peace", "reconcile", "harmony", "shalom"}),
        ("Philippians 4:7", "Matthew 5:9", "Romans 5:1"),
    ),
    (
        frozenset({"patience", "endure", "perseverance", "longsuffering"}),
        ("Romans 12:12", "James 1:3", "Galatians 5:22"),
    ),
    (
        frozenset({"kindness", "gentle", "goodness", "compassion"}),
        ("Ephesians 4:32", "Galatians 5:22", "Colossians 3:12"),
    ),
    (
        frozenset({"humility", "humble", "meek", "lowly"}),
        ("Philippians 2:3", "James 4:6", "Matthew 5:5"),
    ),
    # --- Sin & Temptation ---
    (
        frozenset({"sin", "sinner", "iniquity", "transgression", "wicked"}),
        ("Romans 3:23", "1 John 1:9", "Romans 6:23"),
    ),
    (
        frozenset({"temptation", "tempt", "trial", "testing"}),
        ("1 Corinthians 10:13", "James 1:12", "Matthew 4:1"),
    ),
    (
        frozenset({"repent", "repentance", "turn away", "confession"}),
        ("Acts 3:19", "2 Chronicles 7:14", "1 John 1:9"),
    ),
    # --- Salvation & Gospel ---
    (
        frozenset({"salvation", "saved", "save", "redeem", "saviour"}),
        ("Ephesians 2:8", "John 3:16", "Romans 10:9"),
    ),
    (
        frozenset({"grace", "mercy", "favour", "undeserved"}),
        ("Ephesians 2:8", "Titus 2:11", "Romans 5:8"),
    ),
    (
        frozenset({"gospel", "good news", "evangel", "preach"}),
        ("Romans 1:16", "Mark 16:15", "1 Corinthians 15:1"),
    ),
    # --- Holy Spirit ---
    (
        frozenset({"holy spirit", "spirit of god", "comforter", "counsellor", "paraclete"}),
        ("John 14:26", "Acts 1:8", "Galatians 5:22"),
    ),
    # --- Prayer & Worship ---
    (
        frozenset({"prayer", "pray", "supplication", "intercession"}),
        ("Philippians 4:6", "1 Thessalonians 5:17", "Matthew 6:6"),
    ),
    (
        frozenset({"worship", "praise", "adore", "glorify", "honour"}),
        ("Psalms 95:6", "John 4:24", "Romans 12:1"),
    ),
    # --- Scripture & Word ---
    (
        frozenset({"bible", "scripture", "word of god", "law", "torah", "prophets"}),
        ("2 Timothy 3:16", "Psalms 119:105", "Matthew 4:4"),
    ),
    # --- Life After Death ---
    (
        frozenset({"heaven", "eternal life", "kingdom of god", "paradise"}),
        ("John 14:2", "Matthew 6:33", "Revelation 21:4"),
    ),
    (
        frozenset({"hell", "damnation", "eternal punishment", "lake of fire"}),
        ("Matthew 25:46", "Revelation 20:15", "Romans 6:23"),
    ),
    (
        frozenset({"resurrection", "raised from the dead", "new life", "reborn"}),
        ("1 Corinthians 15:20", "John 11:25", "Romans 6:4"),
    ),
    # --- Practical Living ---
    (
        frozenset({"money", "wealth", "rich", "greed", "steward", "possessions"}),
        ("1 Timothy 6:10", "Matthew 6:24", "Proverbs 3:9"),
    ),
    (
        frozenset({"work", "labour", "toil", "employ", "job", "vocation"}),
        ("Colossians 3:23", "Proverbs 14:23", "Ecclesiastes 9:10"),
    ),
    (
        frozenset({"wisdom", "wise", "understanding", "knowledge", "discernment"}),
        ("Proverbs 9:10", "James 1:5", "Proverbs 3:13"),
    ),
    (
        frozenset({"fear", "afraid", "anxiety", "worry", "concern"}),
        ("Philippians 4:6", "Isaiah 41:10", "2 Timothy 1:7"),
    ),
    (
        frozenset({"anger", "wrath", "rage", "furious", "mad"}),
        ("Ephesians 4:26", "James 1:19", "Proverbs 15:1"),
    ),
    (
        frozenset({"suffering", "pain", "affliction", "persecution", "tribulation"}),
        ("Romans 5:3", "James 1:2", "1 Peter 4:12"),
    ),
    (
        frozenset({"death", "die", "dying", "mortality", "grief", "mourning"}),
        ("Psalms 23:4", "John 11:25", "Revelation 21:4"),
    ),
    # --- Church & Community ---
    (
        frozenset({"church", "assembly", "congregation", "body of christ", "fellowship"}),
        ("Matthew 16:18", "Hebrews 10:25", "1 Corinthians 12:27"),
    ),
    (
        frozenset({"evangelism", "witness", "mission", "disciple", "great commission"}),
        ("Matthew 28:19", "Acts 1:8", "Mark 16:15"),
    ),
    (
        frozenset({"baptism", "baptise", "baptized", "immersed", "water"}),
        ("Matthew 28:19", "Romans 6:4", "Acts 2:38"),
    ),
    # --- Covenant & Promise ---
    (
        frozenset({"covenant", "promise", "oath", "vow", "pledge", "agreement"}),
        ("Hebrews 8:6", "Genesis 12:1", "2 Peter 1:4"),
    ),
    (
        frozenset({"second coming", "return of christ", "parousia", "end times", "last days"}),
        ("Acts 1:11", "Matthew 24:42", "Revelation 22:20"),
    ),
)

_COUNSELING_HINT = re.compile(
    r"\b("
    r"counseling|counsellor|counselor|counsel\s+me|\bcounsel\b|"
    r"therapy|therapist|psychiatr|"
    r"suicid|kill myself|end it all|self[- ]harm|"
    r"depress|anxiety|panic attack|ptsd|trauma|"
    r"marriage crisis|my marriage is|should i divorce|leaving my wife|leaving my husband|"
    r"abuse[sd]?\s+me|domestic violence|"
    r"pastoral care for me|pray\s+for\s+my\s+situation|need\s+someone\s+to\s+talk\s+to"
    r")\b",
    re.IGNORECASE,
)

# Per-word {2,20}/{1,20} caps bound backtracking on adversarial long inputs
# (defense-in-depth; no real book name exceeds 20 characters).
_VERSE_REF_IN_QUESTION = re.compile(
    r"\b((?:[123]\s)?[A-Za-z]{2,20}(?:\s[A-Za-z]{1,20}){0,3}\s\d{1,3}:\d{1,3})\b",
)

_COUNSELING_SYSTEM_GUARD = (
    "The user message may request personal counseling, therapy, crisis intervention, "
    "or intimate life direction (e.g. marriage crisis, mental health, abuse). "
    "You MUST NOT counsel, diagnose, or give tailored life advice. "
    "Respond briefly with kindness: you are a Scripture study aid, not a pastor or clinician; "
    "urge them to speak with a qualified pastor, licensed counselor, or appropriate crisis line. "
    "You may cite 1–2 broadly relevant verses only if they fit, without applying them to their private situation."
)

EMPTY_MODEL_REPLY = (
    "I didn't receive a complete reply from the model. Please try again or shorten your question."
)

# Suffixes injected by some benchmark/eval clients — strip before RAG retrieval
_EVAL_SUFFIX_PATTERN = re.compile(
    r"[?.]?\s*answer in quotes, then add explanation\.?$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Text / document helpers
# ---------------------------------------------------------------------------


def _clean_doc_text(doc: str, ref: str) -> str:
    """Strip embedding prefix and reference prefix from stored document text."""
    text = doc
    if text.startswith("search_document: "):
        text = text[len("search_document: ") :]
    if ref and text.startswith(ref + ": "):
        text = text[len(ref) + 2 :]
    return text.strip()


_BM25_TOKEN_RE = re.compile(r"[a-z0-9']+")


def strip_document_prefix(text: str) -> str:
    """Drop the ``search_document: `` embedding prefix before BM25 tokenization."""
    if text.startswith(DOCUMENT_PREFIX):
        return text[len(DOCUMENT_PREFIX) :]
    return text


def tokenize_for_bm25(text: str) -> list[str]:
    """Punctuation-aware BM25 tokenization shared by indexing and query paths.

    Lowercases and splits on non-alphanumeric characters (apostrophes kept so
    ``god's`` stays one token), so document tokens like ``3:16:`` / ``love,``
    match query tokens ``3:16`` / ``love``. Must be applied identically when
    building the index (rag/build_index.py, after stripping the
    ``search_document:`` prefix) and at query time (rag/retrieval.py).
    """
    return _BM25_TOKEN_RE.findall(text.lower())


# Canonical book-name aliases (lowercase key -> canonical Chroma id book name).
# Covers Psalm(s), the Song of Solomon family, and common abbreviations incl.
# numeric-prefix forms ("1 Cor" -> "1 Corinthians"). Applied only when a ref
# parses as "<book> <chapter>:<verse>", so plain prose is never rewritten.
_BOOK_ALIASES: dict[str, str] = {
    "psalm": "Psalms",
    "psalms": "Psalms",
    "ps": "Psalms",
    "song of songs": "Song of Solomon",
    "canticles": "Song of Solomon",
    "song of solomon": "Song of Solomon",
    "gen": "Genesis",
    "exod": "Exodus",
    "lev": "Leviticus",
    "deut": "Deuteronomy",
    "josh": "Joshua",
    "judg": "Judges",
    "1 sam": "1 Samuel",
    "2 sam": "2 Samuel",
    "1 chr": "1 Chronicles",
    "2 chr": "2 Chronicles",
    "prov": "Proverbs",
    "eccles": "Ecclesiastes",
    "eccl": "Ecclesiastes",
    "isa": "Isaiah",
    "jer": "Jeremiah",
    "lam": "Lamentations",
    "ezek": "Ezekiel",
    "dan": "Daniel",
    "hos": "Hosea",
    "matt": "Matthew",
    "rom": "Romans",
    "1 cor": "1 Corinthians",
    "2 cor": "2 Corinthians",
    "gal": "Galatians",
    "eph": "Ephesians",
    "phil": "Philippians",
    "col": "Colossians",
    "1 thess": "1 Thessalonians",
    "2 thess": "2 Thessalonians",
    "1 tim": "1 Timothy",
    "2 tim": "2 Timothy",
    "philem": "Philemon",
    "heb": "Hebrews",
    "jas": "James",
    "1 pet": "1 Peter",
    "2 pet": "2 Peter",
    "1 john": "1 John",
    "2 john": "2 John",
    "3 john": "3 John",
    "rev": "Revelation",
}


def _normalize_verse_id(ref: str) -> str:
    """Map common aliases to Chroma ids (e.g. Psalm 1:1 → Psalms 1:1,
    Song of Songs 1:1 → Song of Solomon 1:1, 1 Cor 13:4 → 1 Corinthians 13:4)."""
    ref = re.sub(r"\s+", " ", (ref or "").strip())
    if not ref:
        return ref
    m = re.match(r"^(.+?)\s+(\d{1,3}:\d{1,3})$", ref)
    if not m:
        return ref
    book, cv = m.group(1).strip(), m.group(2)
    canonical = _BOOK_ALIASES.get(book.lower())
    if canonical:
        book = canonical
    return f"{book} {cv}"


def _extract_verse_ref_from_lookup(question: str) -> str | None:
    """Book/chapter:verse named in a 'What does X say?' lookup, or None."""
    if not _is_verse_lookup(question):
        return None
    # Drop leading "What does/is …" so the verse regex cannot match "What does Hebrews…"
    t = question.strip()
    low = t.lower()
    for prefix in ("what does ", "what is ", "what says "):
        if low.startswith(prefix):
            t = t[len(prefix) :].strip()
            low = t.lower()
            break
    m = _VERSE_REF_IN_QUESTION.search(t)
    if not m:
        return None
    return _normalize_verse_id(m.group(1))


def _topical_anchor_refs(question: str) -> list[str]:
    """Extra verses to pin for broad topical questions (not verse lookups)."""
    if _is_verse_lookup(question):
        return []
    q = question.lower()
    for keywords, refs in _TOPICAL_PIN_TABLE:
        if any(kw in q for kw in keywords):
            return list(refs)
    return []


def _is_counseling_request(question: str) -> bool:
    """Personal counseling / crisis / intimate life-direction phrasing."""
    return bool(question and _COUNSELING_HINT.search(question))


def _merge_pin_order(pin_refs: list[str]) -> list[str]:
    """Dedupe while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for r in pin_refs:
        n = _normalize_verse_id(r)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _strip_thinking(text: str | None) -> str:
    """Delegate to shared cleanup (Qwen `</think>` + plain 'Thinking Process:' blocks)."""
    return strip_model_thinking(text)


def _strip_repetition_and_meta(text: str) -> str:
    if not text:
        return text
    # Strip leading "? Answer:" etc. before length check (fixes short responses)
    text = re.sub(r"^\s*\??\s*Answer:\s*", "", text, flags=re.IGNORECASE)
    if len(text) < 30:
        return text.strip()
    text = re.sub(r"[═─━]{3,}", "", text)
    for cutoff in [
        "Meta-instruction",
        "TYPED RESPONSE",
        "Crucial:",
        "Violation",
        "You have followed",
        "The key is:",
        "No matter how many times",
        "No matter what format",
        "You are running a standalone",
        "You do not respond to",
        "You do not generate",
    ]:
        idx = text.find(cutoff)
        if idx > 0:
            text = text[:idx].rstrip()
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+", " ", text)).strip()


def _strip_thinking_from_stream(sse_text: str) -> bytes:
    full_content = []
    for line in sse_text.split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = _json.loads(payload)
            for choice in obj.get("choices", []):
                c = choice.get("delta", {}).get("content", "")
                if c:
                    full_content.append(c)
        except _json.JSONDecodeError:
            continue
    cleaned = _strip_thinking("".join(full_content))
    cleaned = _strip_repetition_and_meta(cleaned)
    if not cleaned.strip():
        cleaned = EMPTY_MODEL_REPLY
    out = (
        'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":'
        + _json.dumps(cleaned)
        + '},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )
    return out.encode("utf-8")


# ---------------------------------------------------------------------------
# Query classification helpers
# ---------------------------------------------------------------------------


def _is_verse_lookup(text: str) -> bool:
    """True if question asks for a specific verse (e.g. 'What does John 3:16 say?')."""
    t = text.lower().strip()
    # Match "What does ... say?" and capture everything before the final "say"
    match = re.search(r"what does (.+) say\??$", t)
    if not match:
        return False
    # Require a verse reference (Book 1:2) in the captured part to distinguish
    # from topical questions like "What does the Bible say about love?"
    return bool(re.search(r"\d+:\d+", match.group(1)))


def _is_meta_question(text: str) -> bool:
    t = text.lower().strip()
    patterns = (
        "what can you do",
        "what could you do",
        "what it could do",
        "what are you",
        "how can you help",
        "what are your capabilities",
        "what is your purpose",
        "who are you",
        "what do you do",
        "introduce yourself",
        "tell me about yourself",
    )
    return any(p in t for p in patterns) or t in ("help", "hi", "hello", "hey")


def _strip_openclaw_metadata(text: str) -> str:
    """Remove OpenClaw-style metadata blocks from user input.

    Uses simple string operations instead of regex to avoid ReDoS
    vulnerabilities with crafted input.
    """
    if not text or not isinstance(text, str):
        return text

    # Remove Sender (untrusted metadata): ```json {...} ``` blocks
    while True:
        idx = text.lower().find("sender")
        if idx == -1:
            break
        # Find the end of the ``` block after this marker
        block_start = text.find("```", idx)
        if block_start == -1:
            break
        block_end = text.find("```", block_start + 3)
        if block_end == -1:
            break
        # Remove everything from the marker to the end of the block
        text = text[:idx] + text[block_end + 3 :]

    # Remove stray ```json {...} ``` blocks (without Sender prefix)
    while True:
        json_start = text.lower().find("```json")
        if json_start == -1:
            break
        block_end = text.find("```", json_start + 7)
        if block_end == -1:
            break
        text = text[:json_start] + text[block_end + 3 :]

    # Remove bracketed timestamps like [Mon 2024-01-01 12:34 UTC]
    text = re.sub(r"\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+\w+\]\s*", "", text)

    # If there's a stray ``` at the start, take everything after the last one
    if "```" in text and not text.strip().startswith("["):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[-1].strip()

    return text.strip() or text


def _content_to_str(content: Any) -> str:
    """Coerce a message content field to a plain string.

    Handles OpenAI-style content arrays by concatenating all text parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""
