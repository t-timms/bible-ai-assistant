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
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

# Repo root on path when invoked as `python training/evaluate.py` (script dir is `training/`)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from rag.response_cleanup import strip_model_thinking

RAG_URL_DEFAULT = "http://localhost:8081/v1/chat/completions"
# Prefer 127.0.0.1: on Windows, "localhost" can hit ::1 while Ollama listens on IPv4 only.
JUDGE_URL_DEFAULT = "http://127.0.0.1:11434/v1/chat/completions"
# Default judge: general instruct model already common on project machines (~17GB). Override: --judge-model
DEFAULT_JUDGE_MODEL = "qwen3.5:27b"
DEFAULT_OLLAMA_MODEL = "bible-assistant"

BIBLE_BOOKS = {
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy", "joshua", "judges",
    "ruth", "1 samuel", "2 samuel", "1 kings", "2 kings", "1 chronicles", "2 chronicles",
    "ezra", "nehemiah", "esther", "job", "psalm", "psalms", "proverbs", "ecclesiastes",
    "song of solomon", "isaiah", "jeremiah", "lamentations", "ezekiel", "daniel", "hosea",
    "joel", "amos", "obadiah", "jonah", "micah", "nahum", "habakkuk", "zephaniah",
    "haggai", "zechariah", "malachi", "matthew", "mark", "luke", "john", "acts",
    "romans", "1 corinthians", "2 corinthians", "galatians", "ephesians", "philippians",
    "colossians", "1 thessalonians", "2 thessalonians", "1 timothy", "2 timothy", "titus",
    "philemon", "hebrews", "james", "1 peter", "2 peter", "1 john", "2 john", "3 john",
    "jude", "revelation",
}

VERSE_REF_PATTERN = re.compile(
    r"(?:[123]?\s*[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+\d+:\d+",
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
    model = ollama_model if ollama_model else DEFAULT_OLLAMA_MODEL
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                rag_url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": question}],
                    "stream": False,
                },
            )
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
    if not expected:
        return 0.0
    key_phrases = [p.strip().lower() for p in expected.split(".") if len(p.strip()) > 10]
    if not key_phrases:
        key_phrases = [expected.lower()[:60]]
    hits = sum(1 for p in key_phrases if p in response.lower())
    return hits / len(key_phrases) if key_phrases else 0.0


# Prefixes that indicate a regex false positive (e.g. " and Psalms 27:1" matched as one ref)
_NON_BOOK_PREFIXES = ("and ", "or ", "the ", "of ", "in ", "to ")


def check_hallucination(response: str) -> bool:
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
    prompt = JUDGE_TEMPLATE.format(
        question=question, expected=expected, response=response[:2000]
    )
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
        return _empty_scores(
            f"{e}. Tried judge at {base}. Verify Ollama: curl {base}/api/tags"
        )

    if not content.strip():
        return _empty_scores(
            "No judge response. "
            + ("; ".join(attempts) if attempts else "unknown failure")
            + f". Verify Ollama at {base} (curl {base}/api/tags)."
        )

    scores = _extract_scores_json(content)
    if not scores:
        return _empty_scores(f"JSON parse failed: {content[:200]!r}")

    out = _apply_score_clamps(scores)
    out["reasoning"] = str(out.get("reasoning", ""))[:500]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_keyword_eval(
    questions: list[dict],
    rag_url: str,
    output_path: Path,
    ollama_model: str,
    benchmark_protocol_id: str,
) -> None:
    """Original fast keyword-overlap evaluation."""
    results = []
    category_scores: dict[str, dict] = {}

    for i, q in enumerate(questions):
        question = q["question"]
        expected = q.get("expected_answer", "")
        category = q.get("category", "unknown")

        print(f"[{i + 1}/{len(questions)}] ({category}) {question}")
        response = query_rag(question, rag_url, ollama_model)
        print(f"  -> {response[:150]}{'...' if len(response) > 150 else ''}")

        verse_score = check_verse_accuracy(response, expected)
        citation = has_citation(response)
        hallucinated = check_hallucination(response)

        result = {
            "question": question, "expected_answer": expected,
            "response": response[:1000], "category": category,
            "verse_accuracy": round(verse_score, 2),
            "citation_present": citation, "hallucination_detected": hallucinated,
        }
        results.append(result)

        if category not in category_scores:
            category_scores[category] = {
                "total": 0, "verse_accuracy_sum": 0.0,
                "citations": 0, "hallucinations": 0,
            }
        cs = category_scores[category]
        cs["total"] += 1
        cs["verse_accuracy_sum"] += verse_score
        cs["citations"] += int(citation)
        cs["hallucinations"] += int(hallucinated)

    _print_keyword_summary(category_scores)
    _save_keyword_results(
        category_scores, results, output_path, ollama_model, benchmark_protocol_id
    )


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
    category_scores: dict[str, dict] = {}

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
        print(f"  Judge: F={scores.get('faithfulness', '?')} C={scores.get('citation', '?')} "
              f"H={scores.get('hallucination', '?')} He={scores.get('helpfulness', '?')} "
              f"Co={scores.get('conciseness', '?')}")

        result = {
            "question": question, "expected_answer": expected,
            "response": response[:1000], "category": category,
            "model_tag": model_tag, "judge_scores": scores,
        }
        results.append(result)

        if category not in category_scores:
            category_scores[category] = {"total": 0}
            for d in dims:
                category_scores[category][f"{d}_sum"] = 0.0
        cs = category_scores[category]
        cs["total"] += 1
        for d in dims:
            cs[f"{d}_sum"] += scores.get(d, 0)

        time.sleep(0.5)

    # Summary table
    print("\n" + "=" * 100)
    header = f"{'Category':<18} {'N':>3}"
    for d in dims:
        header += f" {d[:8]:>9}"
    header += f" {'avg':>7}"
    print(header)
    print("-" * 100)

    totals = dict.fromkeys(dims, 0.0)
    total_n = 0
    for cat, cs in sorted(category_scores.items()):
        n = cs["total"]
        total_n += n
        row = f"{cat:<18} {n:>3}"
        cat_avg = 0.0
        for d in dims:
            avg = cs[f"{d}_sum"] / n if n else 0
            totals[d] += cs[f"{d}_sum"]
            cat_avg += avg
            row += f" {avg:>9.2f}"
        row += f" {cat_avg / len(dims):>7.2f}"
        print(row)

    print("-" * 100)
    row = f"{'OVERALL':<18} {total_n:>3}"
    overall_avg = 0.0
    for d in dims:
        avg = totals[d] / total_n if total_n else 0
        overall_avg += avg
        row += f" {avg:>9.2f}"
    row += f" {overall_avg / len(dims):>7.2f}"
    print(row)
    print("=" * 100)

    # Save results
    summary = {
        "eval_mode": "llm-as-judge",
        "judge_model": judge_model,
        "ollama_model": ollama_model,
        "model_tag": model_tag,
        "total_questions": total_n,
        "overall_scores": {d: round(totals[d] / total_n, 3) if total_n else 0 for d in dims},
        "category_summary": {
            cat: {
                "count": cs["total"],
                **{d: round(cs[f"{d}_sum"] / cs["total"], 3) if cs["total"] else 0 for d in dims},
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
    print("\n" + "=" * 80)
    print(f"{'Category':<20} {'Count':>5} {'Verse Acc':>10} {'Citations':>10} {'Halluc':>8}")
    print("-" * 80)
    total_all, acc_all, cite_all, hall_all = 0, 0.0, 0, 0
    for cat, cs in sorted(category_scores.items()):
        n = cs["total"]
        avg_acc = cs["verse_accuracy_sum"] / n if n else 0
        print(f"{cat:<20} {n:>5} {avg_acc:>9.0%} {cs['citations']:>7}/{n:<2} {cs['hallucinations']:>5}/{n}")
        total_all += n
        acc_all += cs["verse_accuracy_sum"]
        cite_all += cs["citations"]
        hall_all += cs["hallucinations"]
    print("-" * 80)
    overall_acc = acc_all / total_all if total_all else 0
    print(f"{'OVERALL':<20} {total_all:>5} {overall_acc:>9.0%} {cite_all:>7}/{total_all:<2} {hall_all:>5}/{total_all}")
    print("=" * 80)


def _save_keyword_results(
    category_scores: dict,
    results: list,
    output_path: Path,
    ollama_model: str,
    benchmark_protocol_id: str,
) -> None:
    total_all = sum(cs["total"] for cs in category_scores.values())
    acc_all = sum(cs["verse_accuracy_sum"] for cs in category_scores.values())
    cite_all = sum(cs["citations"] for cs in category_scores.values())
    hall_all = sum(cs["hallucinations"] for cs in category_scores.values())
    summary = {
        "eval_mode": "keyword",
        "ollama_model": ollama_model,
        "total_questions": total_all,
        "overall_verse_accuracy": round(acc_all / total_all, 3) if total_all else 0,
        "total_citations": cite_all,
        "total_hallucinations": hall_all,
        "category_summary": {
            cat: {
                "count": cs["total"],
                "avg_verse_accuracy": round(cs["verse_accuracy_sum"] / cs["total"], 3) if cs["total"] else 0,
                "citations": cs["citations"],
                "hallucinations": cs["hallucinations"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Bible AI assistant.")
    parser.add_argument("--rag-url", type=str, default=RAG_URL_DEFAULT)
    parser.add_argument("--judge", action="store_true",
                        help="Use LLM-as-judge instead of keyword scoring")
    parser.add_argument("--judge-url", type=str, default=JUDGE_URL_DEFAULT,
                        help="Ollama URL for judge model")
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Ollama model name for judge (must exist: ollama list). Default: qwen3.5:27b",
    )
    parser.add_argument("--model-tag", type=str, default="sft+orpo",
                        help="Tag for this model variant (e.g. base, sft, sft+orpo)")
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
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: docs/evaluation_results.json)")
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
