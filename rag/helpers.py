"""Pure helper functions for the Bible AI RAG server.

No ChromaDB, sentence-transformers, or external I/O — safe to import in tests.
"""

from __future__ import annotations

import json as _json
import re
from typing import Any
from urllib.parse import urlparse

from rag.response_cleanup import strip_model_thinking

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reciprocal Rank Fusion smoothing constant (standard default from the RRF paper)
RRF_K = 60

VERSES_COLLECTION = "bible_verses"
PASSAGES_COLLECTION = "bible_passages"
QUERY_PREFIX = "search_query: "

# Default retrieval tuning (overridden by settings/env at runtime)
HYBRID_CANDIDATES = 20

# Topical questions: pin a few high-signal verses so hybrid retrieval + passage expansion
# cannot drown the topic (e.g. marriage → unrelated "love" parables).
_TOPICAL_PIN_TABLE: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
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
        frozenset({"forgiveness", "forgive", "forgiving"}),
        ("Matthew 6:14", "Ephesians 4:32", "Colossians 3:13"),
    ),
    (
        frozenset({"money", "wealth", "rich", "greed", "steward"}),
        ("1 Timothy 6:10", "Matthew 6:24", "Proverbs 3:9"),
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

_VERSE_REF_IN_QUESTION = re.compile(
    r"\b((?:[123]\s)?[A-Za-z][A-Za-z]+(?:\s[A-Za-z]+){0,3}\s\d{1,3}:\d{1,3})\b",
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
_EVAL_SUFFIXES = (
    "? answer in quotes, then add explanation.",
    "? answer in quotes, then add explanation",
    ". answer in quotes, then add explanation.",
    ". answer in quotes, then add explanation",
    " answer in quotes, then add explanation.",
    " answer in quotes, then add explanation",
    "answer in quotes, then add explanation.",
    "answer in quotes, then add explanation",
)


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


def _validate_ollama_url(url: str) -> str:
    """Validate that OLLAMA_URL is a well-formed HTTP(S) URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"OLLAMA_URL must use http or https scheme, got: {url!r}")
    if not parsed.netloc:
        raise ValueError(f"OLLAMA_URL has no host: {url!r}")
    return url


# ---------------------------------------------------------------------------
# Text / document helpers
# ---------------------------------------------------------------------------


def _clean_doc_text(doc: str, ref: str) -> str:
    """Strip embedding prefix and reference prefix from stored document text."""
    text = doc
    if text.startswith("search_document: "):
        text = text[len("search_document: "):]
    if ref and text.startswith(ref + ": "):
        text = text[len(ref) + 2:]
    return text.strip()


def _normalize_verse_id(ref: str) -> str:
    """Map common aliases to Chroma ids (e.g. Psalm 1:1 → Psalms 1:1)."""
    ref = re.sub(r"\s+", " ", (ref or "").strip())
    if not ref:
        return ref
    m = re.match(r"^(.+?)\s+(\d{1,3}:\d{1,3})$", ref)
    if not m:
        return ref
    book, cv = m.group(1).strip(), m.group(2)
    if book.lower() == "psalm":
        book = "Psalms"
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
            t = t[len(prefix):].strip()
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
    if not re.search(r"what does .+ say\??", t):
        return False
    # Require a verse reference (Book 1:2) to distinguish from topical questions
    # e.g. "What does the Bible say about love?" is topical, not a verse lookup
    return bool(re.search(r"\d+:\d+", t))


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
    if not text or not isinstance(text, str):
        return text
    text = re.sub(
        r"Sender\s*\(untrusted\s*metadata\)\s*:\s*```json\s*\{[^}]{0,2000}\}\s*```\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"```json\s*\{[^}]{0,2000}\}\s*```\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+\w+\]\s*", "", text)
    if "```" in text and not text.strip().startswith("["):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[-1].strip()
    return text.strip() or text


def _content_to_str(content: Any) -> str:
    """Coerce a message content field to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
        return ""
    return str(content) if content is not None else ""
