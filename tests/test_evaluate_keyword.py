"""Unit tests for evaluate.py keyword scoring logic (no network)."""

import json
import sys
from pathlib import Path

import pytest

# Add project root so we can import training.evaluate
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate import (
    BIBLE_BOOKS,
    _print_keyword_summary,
    _save_keyword_results,
    check_hallucination,
    check_verse_accuracy,
    has_citation,
    load_questions,
    query_rag,
)


class TestHasCitation:
    """Tests for has_citation()."""

    def test_returns_true_when_verse_ref_present(self) -> None:
        assert has_citation("John 3:16 says that God so loved the world.")
        assert has_citation("As Romans 8:28 indicates, all things work together.")
        assert has_citation("1 Corinthians 13:4 describes love as patient.")

    def test_returns_false_when_no_ref(self) -> None:
        assert not has_citation("The Bible teaches about love.")
        assert not has_citation("")
        assert not has_citation("Genesis chapter 1")


class TestCheckVerseAccuracy:
    """Tests for check_verse_accuracy()."""

    def test_full_match_returns_one(self) -> None:
        expected = "For God so loved the world, that he gave his only Son."
        response = "For God so loved the world, that he gave his only Son. John 3:16."
        assert check_verse_accuracy(response, expected) == 1.0

    def test_partial_match_returns_fraction(self) -> None:
        expected = "First phrase. Second phrase here. Third phrase."
        response = "First phrase and Second phrase here."
        # Key phrases (len>10): "Second phrase here", "Third phrase"; "First phrase" too short
        # Actually expected.split(".") -> ["First phrase", " Second phrase here", " Third phrase", ""]
        # len > 10: " Second phrase here".strip() = "Second phrase here", " Third phrase".strip() = "Third phrase"
        # So key_phrases = [" second phrase here", " third phrase"] after strip/lower - wait, split gives " Second phrase here"
        # strip() makes "Second phrase here" (11 chars), "Third phrase" (12 chars). "First phrase" is 12 chars.
        # len(p.strip()) > 10: "First phrase" = 12, " Second phrase here" = 18, " Third phrase" = 12
        # So key_phrases = ["first phrase", "second phrase here", "third phrase"]
        # response has "first phrase" and "second phrase here" -> 2/3 = 0.666...
        result = check_verse_accuracy(response, expected)
        assert 0 < result < 1

    def test_empty_expected_returns_zero(self) -> None:
        assert check_verse_accuracy("John 3:16 says...", "") == 0.0

    def test_no_overlap_returns_zero(self) -> None:
        expected = "Completely different phrase about something else."
        response = "John 3:16 is about love."
        assert check_verse_accuracy(response, expected) == 0.0

    def test_short_expected_uses_fallback_slice(self) -> None:
        """When no phrase is >10 chars after split, falls back to expected[:60]."""
        # All dot-separated segments are <=10 chars → key_phrases will be empty → fallback
        expected = "Short"  # no period, 5 chars → filtered out → fallback to expected.lower()[:60]
        response = "Short answer here."
        # Fallback key_phrases = ["short"] (lowercase slice of expected)
        result = check_verse_accuracy(response, expected)
        assert result == 1.0  # "short" is in "short answer here."


class TestCheckHallucination:
    """Tests for check_hallucination()."""

    def test_real_book_not_hallucination(self) -> None:
        assert not check_hallucination("John 3:16 says God loved the world.")
        assert not check_hallucination("Romans 8:28 and 1 Corinthians 13:4.")
        assert not check_hallucination("Psalm 23:1 and Psalms 27:1.")
        assert not check_hallucination("Psalm 23:1. The Lord is my shepherd.")

    def test_fake_book_is_hallucination(self) -> None:
        assert check_hallucination("As Fakebook 1:2 says, this is made up.")
        assert check_hallucination("According to Invalid 5:10...")

    def test_no_ref_not_hallucination(self) -> None:
        assert not check_hallucination("The Bible teaches love.")
        assert not check_hallucination("")


class TestBibleBooks:
    """Sanity check BIBLE_BOOKS constant."""

    def test_contains_major_books(self) -> None:
        assert "genesis" in BIBLE_BOOKS
        assert "john" in BIBLE_BOOKS
        assert "revelation" in BIBLE_BOOKS
        assert "1 corinthians" in BIBLE_BOOKS
        assert "psalm" in BIBLE_BOOKS or "psalms" in BIBLE_BOOKS


class TestLoadQuestions:
    """Tests for load_questions()."""

    def test_loads_valid_json_list(self, tmp_path: Path) -> None:
        questions = [
            {"question": "Who was Moses?", "expected_answer": "A prophet."},
            {"question": "What is John 3:16?", "expected_answer": "God so loved..."},
        ]
        f = tmp_path / "questions.json"
        f.write_text(json.dumps(questions), encoding="utf-8")
        result = load_questions(f)
        assert len(result) == 2
        assert result[0]["question"] == "Who was Moses?"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "nonexistent.json")


class TestQueryRagErrorPaths:
    """Tests for query_rag() — only error paths, no network required."""

    def test_empty_question_returns_error_string(self) -> None:
        result = query_rag("", "http://127.0.0.1:8081/v1/chat/completions")
        assert result == "[ERROR: empty question]"

    def test_whitespace_question_returns_error_string(self) -> None:
        result = query_rag("   ", "http://127.0.0.1:8081/v1/chat/completions")
        assert result == "[ERROR: empty question]"

    def test_unreachable_server_returns_error_string(self) -> None:
        # Port 1 is reserved and always refused — triggers the except branch
        result = query_rag("Who was Moses?", "http://127.0.0.1:1/v1/chat/completions")
        assert result.startswith("[ERROR:")


class TestPrintKeywordSummary:
    """Tests for _print_keyword_summary() — pure print, just verify it runs."""

    def _sample_scores(self) -> dict:
        return {
            "history": {
                "total": 5,
                "verse_accuracy_sum": 3.5,
                "citations": 4,
                "hallucinations": 1,
            },
            "doctrine": {
                "total": 3,
                "verse_accuracy_sum": 2.1,
                "citations": 2,
                "hallucinations": 0,
            },
        }

    def test_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        _print_keyword_summary(self._sample_scores())
        out = capsys.readouterr().out
        assert "OVERALL" in out
        assert "history" in out

    def test_empty_scores_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        _print_keyword_summary({})
        out = capsys.readouterr().out
        assert "OVERALL" in out


class TestSaveKeywordResults:
    """Tests for _save_keyword_results() — file output, use tmp_path."""

    def _sample_scores(self) -> dict:
        return {
            "doctrine": {
                "total": 2,
                "verse_accuracy_sum": 1.5,
                "citations": 2,
                "hallucinations": 0,
            }
        }

    def _sample_results(self) -> list:
        return [
            {
                "question": "What is faith?",
                "response": "Faith is ...",
                "verse_accuracy": 0.75,
                "citation_present": True,
                "hallucination_detected": False,
            }
        ]

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        output = tmp_path / "results.json"
        _save_keyword_results(
            self._sample_scores(),
            self._sample_results(),
            output,
            "bible-assistant",
            "test_proto_v1",
        )
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["eval_mode"] == "keyword"
        assert data["ollama_model"] == "bible-assistant"
        assert data["benchmark_protocol_id"] == "test_proto_v1"
        assert data["total_questions"] == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "deep" / "results.json"
        _save_keyword_results(
            self._sample_scores(),
            [],
            output,
            "bible-assistant",
            "",
        )
        assert output.exists()

    def test_no_protocol_id_omits_key(self, tmp_path: Path) -> None:
        output = tmp_path / "results.json"
        _save_keyword_results(self._sample_scores(), [], output, "model", "")
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "benchmark_protocol_id" not in data


class TestQueryRagSuccessPath:
    """Tests for query_rag() success path — mocked httpx, no network."""

    def test_returns_model_response_on_success(self) -> None:
        """Covers the success branch of query_rag (lines 187-190)."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "For God so loved the world."}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value = mock_client
            result = query_rag(
                "What does John 3:16 say?",
                "http://localhost:8081/v1/chat/completions",
            )

        assert "For God so loved the world." in result

    def test_empty_content_returns_raw_empty(self) -> None:
        """When model returns empty content, raw empty string is returned directly."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value = mock_client
            result = query_rag("Who was Moses?", "http://localhost:8081/v1/chat/completions")

        assert result == ""
