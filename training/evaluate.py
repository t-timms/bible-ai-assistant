#!/usr/bin/env python3
"""Evaluate the Bible assistant via RAG server.

Two modes:
  --judge     LLM-as-judge (default: qwen3.5:27b via Ollama; override with --judge-model)
  (default)   Fast keyword-overlap scoring (for quick checks)

Usage:
  python training/evaluate.py                          # fast keyword scoring
  python training/evaluate.py --judge                  # LLM-as-judge (thorough)
  python training/evaluate.py --judge --model-tag base # evaluate base model
  python training/evaluate.py --ollama-model bible-assistant-orpo  # A/B variant
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_eval_logger = logging.getLogger(__name__)

import httpx

# Repo root on path when invoked as `python training/evaluate.py` (script dir is `training/`)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from rag.helpers import _normalize_verse_id
from rag.response_cleanup import strip_model_thinking
from rag.verification import verify_citations
from scripts.benchmark_stats import wilson_interval

RAG_URL_DEFAULT = "http://localhost:8081/v1/chat/completions"
# Prefer 127.0.0.1: on Windows, "localhost" can hit ::1 while Ollama listens on IPv4 only.
JUDGE_URL_DEFAULT = "http://127.0.0.1:11434/v1/chat/completions"
# Default judge: general instruct model already common on project machines (~17GB). Override: --judge-model
DEFAULT_JUDGE_MODEL = "qwen3.5:27b"
DEFAULT_OLLAMA_MODEL = "bible-assistant"

# --- Protocol v3 pinned constants (benchmarks/manifest.v3.yaml) -----------------
# FUZZY_PASS_THRESHOLD must equal manifest.v3.yaml metric_constants.fuzzy_pass_threshold;
# tests/test_benchmark_manifest.py cross-checks the two so they cannot drift.
FUZZY_PASS_THRESHOLD = 0.85
# Decoding is pinned for reproducibility (manifest v3 decoding.temperature=0,
# seed_required=true). Recorded verbatim in every summary JSON artifact.
EVAL_TEMPERATURE = 0.0
EVAL_SEED = 42
JUDGE_TEMPERATURE = 0.1  # judge rubric sampling; unchanged from prior protocol
# Refusal-category questions test boundary behavior, not quote accuracy — scoring
# them with verse/citation metrics produced structural zeros (manifest v3). They
# are counted only and excluded from all rate denominators.
REFUSAL_CATEGORY = "refusal"

BIBLE_BOOKS = {
    "genesis",
    "exodus",
    "leviticus",
    "numbers",
    "deuteronomy",
    "joshua",
    "judges",
    "ruth",
    "1 samuel",
    "2 samuel",
    "1 kings",
    "2 kings",
    "1 chronicles",
    "2 chronicles",
    "ezra",
    "nehemiah",
    "esther",
    "job",
    "psalm",
    "psalms",
    "proverbs",
    "ecclesiastes",
    "song of solomon",
    "isaiah",
    "jeremiah",
    "lamentations",
    "ezekiel",
    "daniel",
    "hosea",
    "joel",
    "amos",
    "obadiah",
    "jonah",
    "micah",
    "nahum",
    "habakkuk",
    "zephaniah",
    "haggai",
    "zechariah",
    "malachi",
    "matthew",
    "mark",
    "luke",
    "john",
    "acts",
    "romans",
    "1 corinthians",
    "2 corinthians",
    "galatians",
    "ephesians",
    "philippians",
    "colossians",
    "1 thessalonians",
    "2 thessalonians",
    "1 timothy",
    "2 timothy",
    "titus",
    "philemon",
    "hebrews",
    "james",
    "1 peter",
    "2 peter",
    "1 john",
    "2 john",
    "3 john",
    "jude",
    "revelation",
}

VERSE_REF_PATTERN = re.compile(
    r"(?:[123]?\s?[A-Za-z]+(?:\s[A-Za-z]+){0,3})\s\d+:\d+",
)

JUDGE_SYSTEM = """You are an expert evaluator for a Bible AI assistant. Score responses on 5 dimensions (1-5 each).

Scoring guidelines:
- 5 = Excellent, no issues
- 4 = Good, minor issues
- 3 = Acceptable, some issues
- 2 = Poor, significant issues
- 1 = Failing, major issues

You MUST respond with valid JSON only, no other text."""

JUDGE_TEMPLATE = """Score this Bible assistant response.

**Question:** {question}
**Expected answer:** {expected}
**Actual response:** {response}

Score each dimension 1-5:
1. **faithfulness**: Does the response use provided context accurately? No unsupported claims?
2. **citation**: Are Bible references real, correctly formatted, and verse text accurate?
3. **hallucination**: Any fabricated verses, misquoted text, or invented references? (5=none, 1=severe)
4. **helpfulness**: Does the answer address the question? Is it useful and complete?
5. **conciseness**: Clean output? No repetition, no leaked instructions, no filler?

Return ONLY this JSON:
{{"faithfulness": N, "citation": N, "hallucination": N, "helpfulness": N, "conciseness": N, "reasoning": "brief explanation"}}"""


def _ollama_base_url(judge_openai_url: str) -> str:
    """http://host:11434/v1/chat/completions -> http://host:11434"""
    u = urlparse(judge_openai_url)
    return f"{u.scheme}://{u.netloc}"


def _extract_scores_json(content: str) -> dict | None:
    """Parse first JSON object (handles nested braces in strings via raw_decode)."""
    content = strip_model_thinking(content) or ""
    # Strip ```json ... ``` if present
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if fence:
        content = fence.group(1)
    start = content.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(content[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def query_rag(question: str, rag_url: str, ollama_model: str | None = None) -> str:
    if not question or not question.strip():
        return "[ERROR: empty question]"
    model = ollama_model if ollama_model else DEFAULT_OLLAMA_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    # Pinned decoding (manifest v3): temperature=0 + fixed seed. Some strict
    # OpenAI-compatible servers reject unknown fields like `seed` with 4xx —
    # retry once without the extras so evals still run on those.
    decoding = {"temperature": EVAL_TEMPERATURE, "seed": EVAL_SEED}
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(rag_url, json={**payload, **decoding})
            if r.status_code in (400, 415, 422):
                r = client.post(rag_url, json=payload)
            r.raise_for_status()
            data = r.json()
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return strip_model_thinking(raw) if raw else raw
    except Exception as e:
        return f"[ERROR: {e}]"


# ---------------------------------------------------------------------------
# Fast keyword scoring (original mode)
# ---------------------------------------------------------------------------


def has_citation(response: str) -> bool:
    return bool(VERSE_REF_PATTERN.search(response))


def check_verse_accuracy(response: str, expected: str) -> float:
    """Exact-substring key-phrase overlap (legacy metric).

    Penalizes valid paraphrase — e.g. "his one and only Son" vs. "his only
    born Son" scores 0 even though both are faithful renderings. Kept for
    continuity with historical benchmark runs; see check_verse_accuracy_fuzzy
    for a metric that doesn't have this failure mode.
    """
    if not expected:
        return 0.0
    key_phrases = [p.strip().lower() for p in expected.split(".") if len(p.strip()) > 10]
    if not key_phrases:
        key_phrases = [expected.lower()[:60]]
    hits = sum(1 for p in key_phrases if p in response.lower())
    return hits / len(key_phrases) if key_phrases else 0.0


def _normalize_for_fuzzy_compare(text: str) -> str:
    t = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def check_verse_accuracy_fuzzy(response: str, expected: str) -> float:
    """Best-match fuzzy overlap between `expected` and any sentence in `response`.

    Where check_verse_accuracy requires an exact substring, this scores how
    close the *closest* sentence in the response is to the expected text
    (difflib.SequenceMatcher ratio on normalized text), so a faithful
    paraphrase scores near 1.0 instead of 0. Use alongside, not instead of,
    the exact metric — report both (see docs/MODEL_COMPARISON.md).
    """
    if not expected or not response:
        return 0.0
    norm_expected = _normalize_for_fuzzy_compare(expected)
    if not norm_expected:
        return 0.0
    candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
    candidates.append(response)  # whole response as a fallback candidate
    best = 0.0
    for cand in candidates:
        norm_cand = _normalize_for_fuzzy_compare(cand)
        if not norm_cand:
            continue
        ratio = SequenceMatcher(None, norm_expected, norm_cand).ratio()
        best = max(best, ratio)
    return round(best, 3)


_semantic_scorer_cache: object | None = None
_SEMANTIC_SCORER_TRIED = False


def _get_semantic_scorer() -> object | None:
    """Lazy-load the cross-encoder reranker (bge-reranker-v2-m3, revision-pinned
    in rag/settings.py) as a general-purpose semantic-similarity scorer.

    Reused rather than adding a new dependency (e.g. bert-score): it's the same
    already-audited model this repo already runs for retrieval reranking, and a
    cross-encoder trained for relevance judgment is a documented alternative to
    raw BERTScore for exactly this — judging whether two texts say the same
    thing regardless of phrasing/sentence-structure. Returns None (metric
    unavailable, not zero) if sentence-transformers/the model can't load, e.g.
    outside the .venv-rag environment."""
    global _semantic_scorer_cache, _SEMANTIC_SCORER_TRIED
    if _SEMANTIC_SCORER_TRIED:
        return _semantic_scorer_cache
    _SEMANTIC_SCORER_TRIED = True
    try:
        from rag.retrieval import _get_reranker

        _semantic_scorer_cache = _get_reranker()
    except Exception:  # noqa: BLE001 - optional dependency; metric degrades to None
        _semantic_scorer_cache = None
    return _semantic_scorer_cache


def check_verse_accuracy_semantic(response: str, expected: str, scorer: Any = None) -> float | None:
    """Cross-encoder semantic-similarity score in [0, 1], or None if no scorer
    is available (caller should omit the metric, not treat None as 0).

    Protocol v5 (2026-09-04): added after auditing check_verse_accuracy_fuzzy on
    the v3.1/v3.2 re-evals. That metric is a best-single-sentence
    difflib.SequenceMatcher ratio against the whole `expected` string — pure
    character overlap. Two equally-correct, equally-cited answers with the same
    facts differently worded scored 0.11 vs 0.28 and 0.25 vs 0.59 purely from
    sentence-chunking luck (see docs/V3_STATUS.md "RE-EVAL DONE" / benchmarks/
    manifest.v5.yaml). A cross-encoder judges semantic equivalence, not
    character sequence overlap, so phrasing/sentence-structure luck no longer
    swings the score. Kept alongside, not instead of, the exact/fuzzy metrics —
    report all three; do not silently replace one with another (same rule v4
    used for all-in vs exposition-excluded).
    """
    if not response or not expected:
        return 0.0 if (response or expected) else None
    scorer = scorer if scorer is not None else _get_semantic_scorer()
    if scorer is None:
        return None
    # sentence-transformers' CrossEncoder applies bge-reranker-v2-m3's own
    # default_activation_function (Sigmoid, per its config_sentence_transformers.json)
    # inside .predict() -- the output is ALREADY a calibrated [0, 1] score, not a raw
    # logit. Verified empirically 2026-09-04: identical-text pair -> 0.9999,
    # unrelated-text pair -> 0.0000163. An earlier version of this function applied
    # a second sigmoid on top, which crushed that [0, 1] spread into [0.5, 0.73] and
    # made the metric non-discriminative (every unrelated/wrong-fact/refusal response
    # scored ~0.5 regardless of content) -- do not reapply sigmoid here.
    raw = float(scorer.predict([(expected, response)])[0])
    return round(max(0.0, min(1.0, raw)), 3)


# Prefixes that indicate a regex false positive (e.g. " and Psalms 27:1" matched as one ref)
_NON_BOOK_PREFIXES = ("and ", "or ", "the ", "of ", "in ", "to ")

_verse_lookup_cache: dict[str, str] | None = None


def _flatten_nested_bible(raw: object) -> list[dict]:
    """Flatten {book: {chapter: {verse: text}}} nested-dict corpora to verse dicts.

    Mirrors training/build_preference_data.load_verses' nested handling so
    evaluate.py accepts the same raw corpus shapes the trainers do.
    """
    if not isinstance(raw, dict):
        return []
    flat: list[dict] = []
    for book, chapters in raw.items():
        if not isinstance(chapters, dict):
            continue
        for ch, vdict in chapters.items():
            if not isinstance(vdict, dict):
                continue
            for v, text in vdict.items():
                if not text or not len(str(text).strip()):
                    continue
                try:
                    flat.append(
                        {
                            "book": str(book),
                            "chapter": int(ch),
                            "verse": int(v),
                            "text": str(text).strip(),
                        }
                    )
                except (TypeError, ValueError):
                    continue
    return flat


def _load_verses_via_preference_builder() -> list[dict] | None:
    """Reuse build_preference_data.load_verses read-only when importable.

    Returns None on any import/runtime failure (mid-flight edits upstream,
    missing corpus) — callers fall back to the local loader below.
    """
    try:
        from training.build_preference_data import load_verses as _builder_load_verses
    except Exception:
        return None
    try:
        return _builder_load_verses()
    except Exception:
        return None


def load_verse_lookup(project_root: Path | None = None) -> dict[str, str]:
    """Build a {normalized ref: text} lookup from data/raw/bible_web.json (or
    bible.json), for real per-verse citation verification (see
    rag.verification). Returns {} if no raw corpus is present locally — callers
    then fall back to the weaker book-name-only check.

    Tolerates both flat list-of-verse JSON and nested {book:{chapter:{verse:text}}}
    corpora; prefers the shared trainer loader when importable.
    """
    global _verse_lookup_cache
    if _verse_lookup_cache is not None:
        return _verse_lookup_cache

    verses: list[dict] | None = None
    if project_root is None:
        verses = _load_verses_via_preference_builder()
    if verses is None:
        verses = []
        root = project_root or Path(__file__).resolve().parents[1]
        raw_dir = root / "data" / "raw"
        for name in ("bible_web.json", "bible.json"):
            p = raw_dir / name
            if not p.exists():
                continue
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(raw, list):
                verses = [v for v in raw if isinstance(v, dict)]
            else:
                verses = _flatten_nested_bible(raw)
            break

    lookup: dict[str, str] = {}
    for v in verses:
        book, chapter, verse, text = (
            v.get("book"),
            v.get("chapter"),
            v.get("verse"),
            v.get("text"),
        )
        if not (book and chapter and verse and text):
            continue
        ref = _normalize_verse_id(f"{book} {chapter}:{verse}")
        lookup[ref] = str(text)
    _verse_lookup_cache = lookup
    return lookup


def check_hallucination(response: str, verse_lookup: dict[str, str] | None = None) -> bool:
    """True if `response` cites a Bible reference that doesn't check out.

    When a verse corpus is available (see `load_verse_lookup`), this verifies
    each cited chapter:verse actually exists — not just that the book name is
    real, which the previous implementation checked. Falls back to the
    book-name-only check when no corpus is available locally.
    """
    lookup = load_verse_lookup() if verse_lookup is None else verse_lookup
    if lookup:

        def _lookup(ref: str) -> str | None:
            return lookup.get(_normalize_verse_id(ref))

        issues = verify_citations(response, _lookup)
        return any(i.reason == "unknown_reference" for i in issues)

    # Legacy fallback: book-name-only check (weaker — misses fabricated verse
    # numbers within a real book) for environments without a local Bible corpus.
    refs = VERSE_REF_PATTERN.findall(response)
    for ref in refs:
        book_part = re.sub(r"\s+\d+:\d+$", "", ref).strip()
        # Skip regex false positives (e.g. " and Psalms 27:1" yields "and psalms")
        if book_part.lower().startswith(_NON_BOOK_PREFIXES):
            continue
        book_part_lower = book_part.lower()
        if book_part_lower and book_part_lower not in BIBLE_BOOKS:
            normalized = re.sub(r"^[123]\s+", "", book_part_lower)
            if normalized not in BIBLE_BOOKS:
                return True
    return False


# ---------------------------------------------------------------------------
# LLM-as-judge scoring
# ---------------------------------------------------------------------------


def _apply_score_clamps(scores: dict) -> dict:
    out = dict(scores)
    dims = ["faithfulness", "citation", "hallucination", "helpfulness", "conciseness"]
    for d in dims:
        val = out.get(d, 0)
        if isinstance(val, (int, float)):
            iv = int(round(float(val)))
            out[d] = max(1, min(5, iv))
        else:
            out[d] = 0
    return out


def judge_response(
    question: str,
    expected: str,
    response: str,
    judge_url: str,
    judge_model: str,
) -> dict:
    """Send response to the judge model and parse 5-dimension scores."""
    prompt = JUDGE_TEMPLATE.format(question=question, expected=expected, response=response)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    base = _ollama_base_url(judge_url)
    content = ""
    attempts: list[str] = []

    def _empty_scores(msg: str) -> dict:
        return {
            "error": msg,
            "faithfulness": 0,
            "citation": 0,
            "hallucination": 0,
            "helpfulness": 0,
            "conciseness": 0,
            "reasoning": "",
        }

    try:
        # trust_env=False: corporate HTTP_PROXY often breaks localhost:11434 (404 / wrong host).
        with httpx.Client(timeout=180.0, trust_env=False) as client:
            # 1) OpenAI-compatible (Ollama /v1/chat/completions)
            try:
                r = client.post(
                    judge_url,
                    json={
                        "model": judge_model,
                        "messages": messages,
                        "stream": False,
                        "temperature": 0.1,
                    },
                )
                if r.status_code == 200:
                    content = (r.json().get("choices", [{}])[0].get("message", {}) or {}).get(
                        "content", ""
                    ) or ""
                else:
                    attempts.append(f"openai-compat: HTTP {r.status_code}")
            except Exception as e:
                attempts.append(f"openai-compat: {e}")

            # 2) Native Ollama /api/chat
            if not content.strip():
                try:
                    r2 = client.post(
                        f"{base}/api/chat",
                        json={
                            "model": judge_model,
                            "messages": messages,
                            "stream": False,
                            "options": {"temperature": 0.1},
                        },
                    )
                    if r2.status_code == 200:
                        content = (r2.json().get("message", {}) or {}).get("content", "") or ""
                    else:
                        attempts.append(f"/api/chat: HTTP {r2.status_code}")
                except Exception as e:
                    attempts.append(f"/api/chat: {e}")

            # 3) /api/generate (single prompt; works on older/minimal setups)
            if not content.strip():
                combined = f"{JUDGE_SYSTEM}\n\nUser:\n{prompt}"
                try:
                    r3 = client.post(
                        f"{base}/api/generate",
                        json={
                            "model": judge_model,
                            "prompt": combined,
                            "stream": False,
                            "options": {"temperature": 0.1},
                        },
                    )
                    if r3.status_code == 200:
                        content = (r3.json().get("response", "") or "") or ""
                    else:
                        attempts.append(f"/api/generate: HTTP {r3.status_code}")
                except Exception as e:
                    attempts.append(f"/api/generate: {e}")
    except Exception as e:
        return _empty_scores(f"{e}. Tried judge at {base}. Verify Ollama: curl {base}/api/tags")

    if not content.strip():
        msg = (
            "LLM judge unavailable — all endpoints failed. "
            + ("; ".join(attempts) if attempts else "unknown failure")
            + f". Verify Ollama at {base}: run `ollama list` and confirm {judge_model!r} is available."
        )
        _eval_logger.error(msg)
        raise RuntimeError(msg)

    scores = _extract_scores_json(content)
    if not scores:
        return _empty_scores(f"JSON parse failed: {content[:200]!r}")

    out = _apply_score_clamps(scores)
    out["reasoning"] = str(out.get("reasoning", ""))[:500]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def summarize_keyword_results(results: list[dict]) -> dict[str, dict]:
    """Aggregate per-item keyword results into category buckets (pure — no I/O).

    Refusal-category items are counted only (``count_only`` marker); every other
    category accumulates sums for means plus binary counts for Wilson rates.
    """
    category_scores: dict[str, dict] = {}
    for r in results:
        category = r.get("category", "unknown")
        if category == REFUSAL_CATEGORY:
            cs = category_scores.setdefault(category, {"total": 0, "count_only": True})
            cs["total"] += 1
            continue
        cs = category_scores.setdefault(
            category,
            {
                "total": 0,
                "verse_accuracy_sum": 0.0,
                "verse_accuracy_fuzzy_sum": 0.0,
                "fuzzy_passes": 0,
                "citations": 0,
                "hallucinations": 0,
            },
        )
        cs["total"] += 1
        cs["verse_accuracy_sum"] += float(r.get("verse_accuracy", 0.0))
        cs["verse_accuracy_fuzzy_sum"] += float(r.get("verse_accuracy_fuzzy", 0.0))
        cs["fuzzy_passes"] += int(bool(r.get("fuzzy_pass", False)))
        cs["citations"] += int(bool(r.get("citation_present", False)))
        cs["hallucinations"] += int(bool(r.get("hallucination_detected", False)))
    return category_scores


def _rate_cell(successes: int, n: int) -> str:
    """Table cell: rate with Wilson 95% CI and n."""
    if not n:
        return "--"
    lo, hi = wilson_interval(successes, n)
    return f"{successes / n:.0%} [{lo:.0%},{hi:.0%}] n={n}"


def _run_keyword_eval(
    questions: list[dict],
    rag_url: str,
    output_path: Path,
    ollama_model: str,
    benchmark_protocol_id: str,
) -> None:
    """Keyword-overlap evaluation with pinned decoding + verified hallucination checks."""
    verse_lookup = load_verse_lookup()
    verification_mode = "corpus" if verse_lookup else "book_name_fallback"
    if verification_mode == "book_name_fallback":
        msg = (
            "Hallucination check running in book_name_fallback mode: no local Bible "
            "corpus found under data/raw/, so fabricated verse numbers within real "
            "books CANNOT be detected. Results are not comparable to corpus-verified "
            "runs (manifest v3 records this per artifact)."
        )
        _eval_logger.warning(msg)
        print("\n!!! WARNING: " + msg + "\n")

    results = []
    for i, q in enumerate(questions):
        question = q["question"]
        expected = q.get("expected_answer", "")
        category = q.get("category", "unknown")

        print(f"[{i + 1}/{len(questions)}] ({category}) {question}")
        response = query_rag(question, rag_url, ollama_model)
        preview = response[:150].encode("ascii", errors="replace").decode("ascii")
        print(f"  -> {preview}{'...' if len(response) > 150 else ''}")

        verse_score = check_verse_accuracy(response, expected)
        verse_score_fuzzy = check_verse_accuracy_fuzzy(response, expected)
        citation = has_citation(response)
        hallucinated = check_hallucination(response, verse_lookup)

        results.append(
            {
                "question": question,
                "expected_answer": expected,
                "response": response,
                "category": category,
                "verse_accuracy": round(verse_score, 2),
                "verse_accuracy_fuzzy": round(verse_score_fuzzy, 3),
                "fuzzy_pass": verse_score_fuzzy >= FUZZY_PASS_THRESHOLD,
                "citation_present": citation,
                "hallucination_detected": hallucinated,
                "hallucination_verification_mode": verification_mode,
            }
        )

    category_scores = summarize_keyword_results(results)
    _print_keyword_summary(category_scores)
    _save_keyword_results(
        category_scores,
        results,
        output_path,
        ollama_model,
        benchmark_protocol_id,
        verification_mode=verification_mode,
    )


def summarize_judge_results(results: list[dict], dims: list[str]) -> dict[str, dict]:
    """Aggregate judge results per category, excluding parse failures from sums.

    Parse failures (items whose ``judge_scores`` carry an ``error`` key) keep
    their per-item record intact but are excluded from dimension sums, so a
    judge outage cannot drag means toward zero. Each bucket reports ``total``,
    ``scored``, ``parse_failures`` and ``{dim}_sum`` over scored items only.
    """
    buckets: dict[str, dict] = {}
    for r in results:
        category = r.get("category", "unknown")
        scores = r.get("judge_scores") or {}
        failed = bool(r.get("judge_parse_failed")) or bool(scores.get("error"))
        cs = buckets.setdefault(
            category,
            {"total": 0, "scored": 0, "parse_failures": 0},
        )
        cs["total"] += 1
        if failed:
            cs["parse_failures"] += 1
            continue
        cs["scored"] += 1
        for d in dims:
            cs[f"{d}_sum"] = cs.get(f"{d}_sum", 0.0) + float(scores.get(d, 0))
    return buckets


def _run_judge_eval(
    questions: list[dict],
    rag_url: str,
    judge_url: str,
    judge_model: str,
    model_tag: str,
    output_path: Path,
    ollama_model: str,
    benchmark_protocol_id: str,
) -> None:
    """LLM-as-judge evaluation with 5-dimension scoring."""
    dims = ["faithfulness", "citation", "hallucination", "helpfulness", "conciseness"]
    results = []

    for i, q in enumerate(questions):
        question = q["question"]
        expected = q.get("expected_answer", "")
        category = q.get("category", "unknown")

        print(f"[{i + 1}/{len(questions)}] ({category}) {question}")
        response = query_rag(question, rag_url, ollama_model)
        print(f"  -> {response[:120]}{'...' if len(response) > 120 else ''}")

        # Judge has no streaming progress; first call can take minutes (model load + 27B infer).
        print(f"  Judge: scoring with {judge_model} (wait — no output until done)...", flush=True)
        scores = judge_response(question, expected, response, judge_url, judge_model)
        parse_failed = bool(scores.get("error"))
        if parse_failed:
            print(f"  Judge: PARSE FAILURE — excluded from means ({scores.get('error', '')[:80]})")
        else:
            print(
                f"  Judge: F={scores.get('faithfulness', '?')} C={scores.get('citation', '?')} "
                f"H={scores.get('hallucination', '?')} He={scores.get('helpfulness', '?')} "
                f"Co={scores.get('conciseness', '?')}"
            )

        results.append(
            {
                "question": question,
                "expected_answer": expected,
                "response": response,
                "category": category,
                "model_tag": model_tag,
                "judge_scores": scores,
                "judge_parse_failed": parse_failed,
            }
        )

    category_scores = summarize_judge_results(results, dims)

    # Summary table
    print("\n" + "=" * 110)
    header = f"{'Category':<18} {'N':>4} {'Scored':>6} {'ParseFail':>16}"
    for d in dims:
        header += f" {d[:8]:>9}"
    header += f" {'avg':>7}"
    print(header)
    print("-" * 110)

    totals = dict.fromkeys(dims, 0.0)
    total_n = 0
    total_scored = 0
    total_parse_failures = 0
    for cat, cs in sorted(category_scores.items()):
        n = cs["total"]
        scored = cs["scored"]
        total_n += n
        total_scored += scored
        total_parse_failures += cs["parse_failures"]
        pf_cell = f"{cs['parse_failures']}/{n} ({cs['parse_failures'] / n:.0%})" if n else "--"
        row = f"{cat:<18} {n:>4} {scored:>6} {pf_cell:>16}"
        cat_avg = 0.0
        for d in dims:
            avg = cs.get(f"{d}_sum", 0.0) / scored if scored else 0
            totals[d] += cs.get(f"{d}_sum", 0.0)
            cat_avg += avg
            row += f" {avg:>9.2f}"
        row += f" {cat_avg / len(dims):>7.2f}" if scored else f" {'--':>7}"
        print(row)

    print("-" * 110)
    pf_overall = (
        f"{total_parse_failures}/{total_n} ({total_parse_failures / total_n:.0%})"
        if total_n
        else "--"
    )
    row = f"{'OVERALL':<18} {total_n:>4} {total_scored:>6} {pf_overall:>16}"
    overall_avg = 0.0
    for d in dims:
        avg = totals[d] / total_scored if total_scored else 0
        overall_avg += avg
        row += f" {avg:>9.2f}"
    row += f" {overall_avg / len(dims):>7.2f}" if total_scored else f" {'--':>7}"
    print(row)
    print("=" * 110)

    # Save results
    summary = {
        "eval_mode": "llm-as-judge",
        "judge_model": judge_model,
        "ollama_model": ollama_model,
        "model_tag": model_tag,
        "decoding": {
            "rag_temperature": EVAL_TEMPERATURE,
            "rag_seed": EVAL_SEED,
            "judge_temperature": JUDGE_TEMPERATURE,
        },
        "total_questions": total_n,
        "scored_questions": total_scored,
        "parse_failure_rate": {
            "value": round(total_parse_failures / total_n, 4) if total_n else 0.0,
            "n": total_n,
        },
        "overall_scores": {
            d: round(totals[d] / total_scored, 3) if total_scored else 0 for d in dims
        },
        "category_summary": {
            cat: {
                "count": cs["total"],
                "scored": cs["scored"],
                "parse_failures": cs["parse_failures"],
                "parse_failure_rate": round(cs["parse_failures"] / cs["total"], 4)
                if cs["total"]
                else 0.0,
                **{
                    d: round(cs.get(f"{d}_sum", 0.0) / cs["scored"], 3) if cs["scored"] else 0
                    for d in dims
                },
            }
            for cat, cs in sorted(category_scores.items())
        },
        "results": results,
    }
    if benchmark_protocol_id:
        summary["benchmark_protocol_id"] = benchmark_protocol_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def _print_keyword_summary(category_scores: dict) -> None:
    print("\n" + "=" * 118)
    print(
        f"{'Category':<20} {'N':>4} {'VerseAcc':>9} {'FuzzyMean':>9} "
        f"{'Fuzzy>=' + f'{FUZZY_PASS_THRESHOLD:.2f}':>24} {'Citations':>24} {'Halluc':>22}"
    )
    print("-" * 118)
    total_all = 0
    scored_all = 0
    acc_all = 0.0
    fuzzy_all = 0.0
    fuzzy_pass_all = 0
    cite_all = 0
    hall_all = 0
    refusal_total = 0
    for cat, cs in sorted(category_scores.items()):
        n = cs["total"]
        total_all += n
        if cs.get("count_only"):
            refusal_total += n
            print(f"{cat:<20} {n:>4}   (count-only — excluded from verse/citation rates)")
            continue
        scored_all += n
        avg_acc = cs.get("verse_accuracy_sum", 0.0) / n if n else 0
        avg_fuzzy = cs.get("verse_accuracy_fuzzy_sum", 0.0) / n if n else 0
        print(
            f"{cat:<20} {n:>4} {avg_acc:>8.0%} {avg_fuzzy:>9.3f} "
            f"{_rate_cell(cs.get('fuzzy_passes', 0), n):>24} "
            f"{_rate_cell(cs.get('citations', 0), n):>24} "
            f"{_rate_cell(cs.get('hallucinations', 0), n):>22}"
        )
        acc_all += cs.get("verse_accuracy_sum", 0.0)
        fuzzy_all += cs.get("verse_accuracy_fuzzy_sum", 0.0)
        fuzzy_pass_all += cs.get("fuzzy_passes", 0)
        cite_all += cs.get("citations", 0)
        hall_all += cs.get("hallucinations", 0)
    print("-" * 118)
    if scored_all:
        overall_acc = acc_all / scored_all
        overall_fuzzy = fuzzy_all / scored_all
        print(
            f"{'OVERALL':<20} {scored_all:>4} {overall_acc:>8.0%} {overall_fuzzy:>9.3f} "
            f"{_rate_cell(fuzzy_pass_all, scored_all):>24} "
            f"{_rate_cell(cite_all, scored_all):>24} "
            f"{_rate_cell(hall_all, scored_all):>22}"
        )
    else:
        print(f"{'OVERALL':<20} {scored_all:>4}   (no rate-scored questions)")
    if refusal_total:
        print(
            f"{'refusal questions':<20} {refusal_total:>4}   "
            "(counted only; excluded from all rates per protocol v3)"
        )
    print("=" * 118)


def _rate_summary(successes: int, n: int) -> dict:
    """JSON artifact form of a binary rate: value + Wilson 95% CI + n."""
    lo, hi = wilson_interval(successes, n)
    return {
        "value": round(successes / n, 4) if n else 0.0,
        "wilson95": {"lo": round(lo, 4), "hi": round(hi, 4)},
        "n": n,
    }


def _save_keyword_results(
    category_scores: dict,
    results: list,
    output_path: Path,
    ollama_model: str,
    benchmark_protocol_id: str,
    verification_mode: str = "corpus",
) -> None:
    total_all = sum(cs["total"] for cs in category_scores.values())
    scored_cats = [cs for cs in category_scores.values() if not cs.get("count_only")]
    scored_all = sum(cs["total"] for cs in scored_cats)
    acc_all = sum(cs.get("verse_accuracy_sum", 0.0) for cs in scored_cats)
    fuzzy_all = sum(cs.get("verse_accuracy_fuzzy_sum", 0.0) for cs in scored_cats)
    fuzzy_pass_all = sum(cs.get("fuzzy_passes", 0) for cs in scored_cats)
    cite_all = sum(cs.get("citations", 0) for cs in scored_cats)
    hall_all = sum(cs.get("hallucinations", 0) for cs in scored_cats)
    refusal_total = total_all - scored_all

    def _cat_entry(cs: dict) -> dict:
        n = cs["total"]
        if cs.get("count_only"):
            return {"count": n, "count_only": True}
        return {
            "count": n,
            "avg_verse_accuracy": round(cs.get("verse_accuracy_sum", 0.0) / n, 3) if n else 0,
            "avg_verse_accuracy_fuzzy": round(cs.get("verse_accuracy_fuzzy_sum", 0.0) / n, 3)
            if n
            else 0,
            "fuzzy_pass_rate": _rate_summary(cs.get("fuzzy_passes", 0), n),
            "citations": cs.get("citations", 0),
            "citation_rate": _rate_summary(cs.get("citations", 0), n),
            "hallucinations": cs.get("hallucinations", 0),
            "hallucination_rate": _rate_summary(cs.get("hallucinations", 0), n),
        }

    summary = {
        "eval_mode": "keyword",
        "ollama_model": ollama_model,
        "decoding": {"temperature": EVAL_TEMPERATURE, "seed": EVAL_SEED},
        "hallucination_verification_mode": verification_mode,
        "fuzzy_pass_threshold": FUZZY_PASS_THRESHOLD,
        "total_questions": total_all,
        "rates_denominator_note": (
            "refusal-category questions are count-only and excluded from rate denominators"
            if refusal_total
            else ""
        ),
        "refusal_count": refusal_total,
        "overall_verse_accuracy": round(acc_all / scored_all, 3) if scored_all else 0,
        "overall_verse_accuracy_fuzzy": round(fuzzy_all / scored_all, 3) if scored_all else 0,
        "overall_fuzzy_pass_rate": _rate_summary(fuzzy_pass_all, scored_all),
        "total_citations": cite_all,
        "overall_citation_rate": _rate_summary(cite_all, scored_all),
        "total_hallucinations": hall_all,
        "overall_hallucination_rate": _rate_summary(hall_all, scored_all),
        "category_summary": {cat: _cat_entry(cs) for cat, cs in sorted(category_scores.items())},
        "results": results,
    }
    if benchmark_protocol_id:
        summary["benchmark_protocol_id"] = benchmark_protocol_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Bible AI assistant.")
    parser.add_argument("--rag-url", type=str, default=RAG_URL_DEFAULT)
    parser.add_argument(
        "--judge", action="store_true", help="Use LLM-as-judge instead of keyword scoring"
    )
    parser.add_argument(
        "--judge-url", type=str, default=JUDGE_URL_DEFAULT, help="Ollama URL for judge model"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Ollama model name for judge (must exist: ollama list). Default: qwen3.5:27b",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="sft+orpo",
        help="Tag for this model variant (e.g. base, sft, sft+orpo)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name passed to RAG server (must match `ollama list`)",
    )
    parser.add_argument(
        "--protocol-id",
        type=str,
        default="",
        help="Benchmark protocol id saved in JSON (e.g. bible_assistant_baseline_v1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: docs/evaluation_results.json)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    questions_path = project_root / "prompts" / "evaluation_questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(f"Evaluation questions not found: {questions_path}")

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} evaluation questions.")
    print(f"RAG server: {args.rag_url}")
    print(f"Ollama model (via API): {args.ollama_model}")
    print(f"Mode: {'LLM-as-judge' if args.judge else 'keyword scoring'}")
    if args.judge:
        print(f"Judge: {args.judge_model} at {args.judge_url}")
        print(f"Model tag: {args.model_tag}")
    if args.protocol_id:
        print(f"Benchmark protocol: {args.protocol_id}")
    print()

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"_{args.model_tag}" if args.judge else ""
        output_path = project_root / "docs" / f"evaluation_results{suffix}.json"

    proto = (args.protocol_id or "").strip()
    if args.judge:
        _run_judge_eval(
            questions,
            args.rag_url,
            args.judge_url,
            args.judge_model,
            args.model_tag,
            output_path,
            args.ollama_model,
            proto,
        )
    else:
        _run_keyword_eval(
            questions,
            args.rag_url,
            output_path,
            args.ollama_model,
            proto,
        )


if __name__ == "__main__":
    main()
