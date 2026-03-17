#!/usr/bin/env python3
"""
Evaluate the Bible assistant via RAG server (http://localhost:8081/v1/chat/completions).

Loads questions from prompts/evaluation_questions.json, sends each to the RAG server,
scores responses (verse accuracy, citation, hallucination), and saves results to
docs/evaluation_results.json with a per-category summary table.

Usage:
  python training/evaluate.py
  python training/evaluate.py --rag-url http://localhost:8081/v1/chat/completions
"""
import json
import re
import argparse
from pathlib import Path

import httpx

RAG_URL_DEFAULT = "http://localhost:8081/v1/chat/completions"
MODEL_NAME = "bible-assistant"

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


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def query_rag(question: str, rag_url: str) -> str:
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                rag_url,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": question}],
                    "stream": False,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def has_citation(response: str) -> bool:
    """Check if the response contains at least one Bible reference (Book Chapter:Verse)."""
    return bool(VERSE_REF_PATTERN.search(response))


def check_verse_accuracy(response: str, expected: str) -> float:
    """Score 0.0–1.0: fraction of key phrases from expected_answer found in response."""
    if not expected:
        return 0.0
    key_phrases = [p.strip().lower() for p in expected.split(".") if len(p.strip()) > 10]
    if not key_phrases:
        key_phrases = [expected.lower()[:60]]
    hits = sum(1 for p in key_phrases if p in response.lower())
    return hits / len(key_phrases) if key_phrases else 0.0


def check_hallucination(response: str) -> bool:
    """True if the response likely contains a fabricated verse reference."""
    refs = VERSE_REF_PATTERN.findall(response)
    for ref in refs:
        book_part = re.sub(r"\s+\d+:\d+$", "", ref).strip().lower()
        if book_part and book_part not in BIBLE_BOOKS:
            normalized = re.sub(r"^[123]\s+", "", book_part)
            if normalized not in BIBLE_BOOKS:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-url", type=str, default=RAG_URL_DEFAULT)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    questions_path = project_root / "prompts" / "evaluation_questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(f"Evaluation questions not found: {questions_path}")

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} evaluation questions.")
    print(f"RAG server: {args.rag_url}\n")

    results = []
    category_scores: dict[str, dict] = {}

    for i, q in enumerate(questions):
        question = q["question"]
        expected = q.get("expected_answer", "")
        category = q.get("category", "unknown")

        print(f"[{i+1}/{len(questions)}] ({category}) {question}")
        response = query_rag(question, args.rag_url)
        print(f"  -> {response[:150]}{'...' if len(response) > 150 else ''}")

        verse_score = check_verse_accuracy(response, expected)
        citation = has_citation(response)
        hallucinated = check_hallucination(response)

        result = {
            "question": question,
            "expected_answer": expected,
            "response": response[:1000],
            "category": category,
            "verse_accuracy": round(verse_score, 2),
            "citation_present": citation,
            "hallucination_detected": hallucinated,
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

    print("\n" + "=" * 80)
    print(f"{'Category':<20} {'Count':>5} {'Verse Acc':>10} {'Citations':>10} {'Halluc':>8}")
    print("-" * 80)
    total_all = 0
    acc_all = 0.0
    cite_all = 0
    hall_all = 0
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

    output_path = project_root / "docs" / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_questions": total_all,
        "overall_verse_accuracy": round(overall_acc, 3),
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
