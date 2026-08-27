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
# Word reps are capped ({1,20}) to bound backtracking on adversarial inputs —
# no real book name exceeds 20 characters.
VERSE_REF_PATTERN = re.compile(
    r"(?:[123]?\s?[A-Za-z]{1,20}(?:\s[A-Za-z]{1,20}){0,3})\s\d+:\d+",
)

# Regex false positives from prose immediately before a real reference sweep a
# connective word into the match, e.g. "... and Psalms 27:1" or "As Hezekiah
# 3:5". Stripped from the front of a match (not dropped outright) so the real
# book name — "Psalms", "Hezekiah" — is recovered rather than lost.
_LEADING_CONNECTIVES = {
    "and",
    "or",
    "the",
    "of",
    "in",
    "to",
    "as",
    "see",
    "cf",
    "per",
    "about",
    "from",
    "with",
    "through",
    "for",
    "on",
    "at",
    "by",
    "into",
    "does",
    "do",
    "did",
    "say",
    "says",
    "said",
    "wrote",
    "teach",
    "teaches",
    "what",
    "where",
    "when",
    "who",
    "how",
    "which",
    "whom",
    "that",
    "this",
    "those",
    "these",
    "love",
    "reads",
    "means",
    "explains",
    "describes",
    "according",
    "gives",
    "tells",
    "shows",
    "proves",
    "reveals",
}

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

    When the greedy regex sweeps prose into the match (e.g. "love in John 3:16"
    from "God's love in John 3:16"), the real book name is recovered by scanning
    right-to-left for the last word that looks like a book name (capitalized or
    a numeric-prefix abbreviation).  Possessive fragments (bare ``s``) are
    discarded.  Always keeps at least one word as the book.
    """
    m = re.match(r"^(.+?)\s+(\d+:\d+)$", ref.strip())
    if not m:
        return None
    raw_words = m.group(1).strip()
    words: list[str] = []
    for w in raw_words.split(" "):
        if w.endswith("'s"):
            w = w[:-2]
        if w and w != "s":
            words.append(w)
    if not words:
        return None
    book_idx = len(words) - 1
    while book_idx > 0 and not words[book_idx][0].isupper():
        book_idx -= 1
    while book_idx > 0 and words[book_idx - 1].isdigit():
        book_idx -= 1
    book = " ".join(words[book_idx:]).strip()
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
    bad_refs = sorted(
        {issue.ref for issue in issues if issue.reason == "unknown_reference"},
        key=len,
        reverse=True,
    )
    if not bad_refs:
        return text
    # Single pass over an alternation of offending refs, longest first with word
    # boundaries: sequential str.replace would corrupt "1 John 3:16" when
    # annotating "John 3:16" and double-mark overlapping occurrences.
    pattern = re.compile(
        r"(?<!\w)(?<!\d )(" + "|".join(re.escape(r) for r in bad_refs) + r")(?!\w)"
    )
    return pattern.sub(lambda m: f"{m.group(1)} [⚠ reference not found in indexed text]", text)
