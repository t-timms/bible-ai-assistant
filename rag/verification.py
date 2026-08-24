"""Real citation verification for Bible references in model output.

Existing hallucination detection (`training/evaluate.py::check_hallucination`,
pre-fix) only validated that a cited *book name* is real (against a hardcoded set
of the 66 canonical book names). It never checked that the cited chapter:verse
actually exists, or that the quoted text is close to the real verse. A response
citing "1 Corinthians 47:99" — a real book, a nonexistent verse — passed as
"not hallucinated".

This module verifies each extracted reference against an actual verse lookup
(book+chapter+verse -> text), so both nonexistent books *and* nonexistent
verses within real books are caught. It is a pure-function module (no I/O) —
callers supply the lookup function, which in production is backed by ChromaDB
(`rag.retrieval._fetch_verses_by_refs`) and in evaluation/tests by a plain
dict loaded from the raw Bible JSON.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import NamedTuple

# Verse reference pattern: optional leading number (1/2/3), one or more capitalized
# words, then chapter:verse. Mirrors the pattern used across rag/helpers.py and
# training/evaluate.py so extraction behaves identically everywhere it's used.
VERSE_REF_PATTERN = re.compile(
    r"(?:[123]?\s?[A-Za-z]+(?:\s[A-Za-z]+){0,3})\s\d+:\d+",
)

# Regex false positives from prose immediately before a real reference sweep a
# connective word into the match, e.g. "... and Psalms 27:1" or "As Hezekiah
# 3:5". Stripped from the front of a match (not dropped outright) so the real
# book name — "Psalms", "Hezekiah" — is recovered rather than lost.
_LEADING_CONNECTIVES = {"and", "or", "the", "of", "in", "to", "as", "see", "cf", "per"}

# Below this ratio, quoted text is considered too different from the real verse
# to be a paraphrase — flagged as a possible misquote rather than silently passed.
_TEXT_SIMILARITY_FLOOR = 0.5


class CitationIssue(NamedTuple):
    """One problem found with a cited reference."""

    ref: str
    reason: str  # "unknown_reference" | "possible_misquote"


VerseLookup = Callable[[str], "str | None"]


def _split_ref(ref: str) -> tuple[str, str] | None:
    """'John 3:16' -> ('John', '3:16'). None if it doesn't parse.

    Strips a leading run of connective words swept into the match by the
    greedy book-name group (e.g. "and Romans" -> "Romans", "As Hezekiah" ->
    "Hezekiah") so the real book name is recovered rather than the whole
    match being discarded. Always keeps at least one word as the book.
    """
    m = re.match(r"^(.+?)\s+(\d+:\d+)$", ref.strip())
    if not m:
        return None
    words = m.group(1).strip().split(" ")
    while len(words) > 1 and words[0].lower() in _LEADING_CONNECTIVES:
        words.pop(0)
    book = " ".join(words).strip()
    if not book:
        return None
    return book, m.group(2)


def extract_verse_refs(text: str) -> list[str]:
    """Extract candidate verse references from response text, recovering the
    real book name when a regex match swept in a leading connective word."""
    refs = []
    for raw in VERSE_REF_PATTERN.findall(text):
        parsed = _split_ref(raw)
        if not parsed:
            continue
        book, chapter_verse = parsed
        refs.append(f"{book} {chapter_verse}")
    return refs


def _normalize_for_compare(text: str) -> str:
    """Lowercase, strip punctuation/whitespace variance for fuzzy text comparison."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _quoted_spans(text: str) -> list[str]:
    """Extract double/curly-quoted spans from response text (candidate verse quotes)."""
    return re.findall(r'["“]([^"”]{10,400})["”]', text)


def verify_citations(
    response_text: str,
    verse_lookup: VerseLookup,
) -> list[CitationIssue]:
    """Check every cited reference in `response_text` against `verse_lookup`.

    verse_lookup(ref) should return the real verse text for a valid reference
    (any reasonable book-name spelling/aliasing is the caller's concern — see
    `rag.retrieval._fetch_verses_by_refs` for the production implementation)
    or None if the reference does not exist.

    Returns one CitationIssue per problem found:
      - "unknown_reference": the book:chapter:verse does not resolve at all —
        this is the real hallucination signal existing code was missing.
      - "possible_misquote": the reference resolves, but a quoted span in the
        response near that reference looks nothing like the real verse text
        (below _TEXT_SIMILARITY_FLOOR). Flagged separately from unknown
        references since paraphrase is expected and legitimate — this is a
        weaker signal, not proof of fabrication.
    """
    issues: list[CitationIssue] = []
    quotes = _quoted_spans(response_text)
    seen: set[str] = set()

    for ref in extract_verse_refs(response_text):
        if ref in seen:
            continue
        seen.add(ref)

        real_text = verse_lookup(ref)
        if real_text is None:
            issues.append(CitationIssue(ref=ref, reason="unknown_reference"))
            continue

        if not quotes:
            continue
        norm_real = _normalize_for_compare(real_text)
        best_ratio = max(
            SequenceMatcher(None, norm_real, _normalize_for_compare(q)).ratio() for q in quotes
        )
        if best_ratio < _TEXT_SIMILARITY_FLOOR:
            issues.append(CitationIssue(ref=ref, reason="possible_misquote"))

    return issues


def annotate_unverified_citations(text: str, issues: list[CitationIssue]) -> str:
    """Append a bracketed warning after each unverified reference's exact text.

    Deliberately non-destructive: only appends a marker next to the offending
    reference rather than deleting or rewriting surrounding text, so a false
    positive degrades the response's polish, not its content. Only touches
    "unknown_reference" issues — "possible_misquote" is a weaker signal and
    left to logging only (see rag_server.py) to avoid over-flagging valid
    paraphrase.
    """
    if not issues:
        return text
    out = text
    for issue in issues:
        if issue.reason != "unknown_reference":
            continue
        marker = f"{issue.ref} [⚠ reference not found in indexed text]"
        out = out.replace(issue.ref, marker)
    return out
