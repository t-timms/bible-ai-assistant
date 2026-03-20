"""Validate evaluation_questions.json schema and content."""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "prompts" / "evaluation_questions.json"

REQUIRED_CATEGORIES = {"verse_lookup", "topical", "character", "cross_reference", "context"}
REQUIRED_KEYS = {"question", "expected_answer", "category"}


@pytest.fixture
def questions() -> list[dict]:
    """Load evaluation questions."""
    if not QUESTIONS_PATH.exists():
        pytest.skip(f"Evaluation questions not found: {QUESTIONS_PATH}")
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_questions_is_list(questions: list[dict]) -> None:
    """Questions must be a JSON array."""
    assert isinstance(questions, list), "evaluation_questions.json must be an array"


def test_questions_not_empty(questions: list[dict]) -> None:
    """At least one question required."""
    assert len(questions) > 0, "At least one evaluation question required"


def test_each_question_has_required_keys(questions: list[dict]) -> None:
    """Each question must have question, expected_answer, category."""
    for i, q in enumerate(questions):
        for key in REQUIRED_KEYS:
            assert key in q, f"Question {i} missing required key: {key}"


def test_question_text_non_empty(questions: list[dict]) -> None:
    """Question text must not be empty."""
    for i, q in enumerate(questions):
        assert q.get("question"), f"Question {i} has empty question text"


def test_categories_are_valid(questions: list[dict]) -> None:
    """Categories should be known types (verse_lookup, topical, etc.)."""
    valid = REQUIRED_CATEGORIES | {"meta", "refusal", "off_topic"}
    for i, q in enumerate(questions):
        cat = q.get("category", "")
        assert cat in valid, f"Question {i} has unknown category: {cat}"


def test_minimum_category_coverage(questions: list[dict]) -> None:
    """Eval set should cover main categories for balanced scoring."""
    cats = {q.get("category") for q in questions}
    assert "verse_lookup" in cats, "Need verse_lookup questions for faithfulness/citation"
    assert "topical" in cats, "Need topical questions for thematic reasoning"
