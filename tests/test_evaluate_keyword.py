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
    check_verse_accuracy_fuzzy,
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


class TestCheckVerseAccuracyFuzzy:
    """Tests for check_verse_accuracy_fuzzy() — doesn't penalize valid paraphrase."""

    def test_exact_match_scores_high(self) -> None:
        expected = "For God so loved the world, that he gave his only Son."
        response = 'He said: "For God so loved the world, that he gave his only Son."'
        assert check_verse_accuracy_fuzzy(response, expected) > 0.9

    def test_valid_paraphrase_scores_higher_than_exact_metric(self) -> None:
        # Exact-substring metric fails this (different wording); fuzzy should not.
        expected = "his one and only Son"
        response = "The verse speaks of his only born Son, given out of love."
        fuzzy = check_verse_accuracy_fuzzy(response, expected)
        exact = check_verse_accuracy(response, expected)
        assert fuzzy > exact

    def test_empty_expected_returns_zero(self) -> None:
        assert check_verse_accuracy_fuzzy("Some response.", "") == 0.0

    def test_empty_response_returns_zero(self) -> None:
        assert check_verse_accuracy_fuzzy("", "Expected text.") == 0.0

    def test_unrelated_text_scores_low(self) -> None:
        expected = "For God so loved the world, that he gave his only Son."
        response = "The capital of France is Paris."
        assert check_verse_accuracy_fuzzy(response, expected) < 0.4


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
                "verse_accuracy_fuzzy_sum": 4.0,
                "citations": 4,
                "hallucinations": 1,
            },
            "doctrine": {
                "total": 3,
                "verse_accuracy_sum": 2.1,
                "verse_accuracy_fuzzy_sum": 2.5,
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
                "verse_accuracy_fuzzy_sum": 1.8,
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


# ---------------------------------------------------------------------------
# Canned-response regression tests — pin current metric semantics.
#
# Every expected value below was hand-verified against the CURRENT scoring
# implementation. If one of these fails after a "harmless" refactor, the
# metric's meaning changed — bump the benchmark protocol (new manifest.vN.yaml)
# before merging, per docs/BENCHMARK_PROTOCOL.md.
# ---------------------------------------------------------------------------

_JOHN_3_16_EXPECTED = (
    "For God so loved the world, that he gave his only born Son, that whoever "
    "believes in him should not perish, but have eternal life."
)
_ROMANS_8_28_EXPECTED = (
    "We know that all things work together for good for those who love God, for "
    "those who are called according to his purpose."
)


class TestCannedVerseAccuracyExact:
    """Pins check_verse_accuracy: fraction of dot-separated key phrases (len>10)
    present as exact lowercase substrings."""

    RESPONSE_FULL = (
        "John 3:16 says: For God so loved the world, that he gave his only born "
        "Son, that whoever believes in him should not perish, but have eternal life."
    )

    def test_full_quote_scores_exactly_one(self) -> None:
        assert check_verse_accuracy(self.RESPONSE_FULL, _JOHN_3_16_EXPECTED) == 1.0

    def test_paraphrase_scores_exactly_zero(self) -> None:
        """The known all-or-nothing-per-phrase weakness: a faithful paraphrase
        with zero exact phrase hits scores 0.0. Pinned so any softening of the
        legacy metric is a visible protocol change, not silent drift."""
        response = (
            "The verse teaches that God loved the world so much that he gave his one and only Son."
        )
        assert check_verse_accuracy(response, _JOHN_3_16_EXPECTED) == 0.0

    def test_truncated_phrase_misses_exact_match(self) -> None:
        """A response quoting the first half of the single key phrase still
        scores 0.0 — partial phrase overlap is not credited."""
        response = (
            "Romans 8:28 tells us: We know that all things work together for good "
            "for those who love God."
        )
        assert check_verse_accuracy(response, _ROMANS_8_28_EXPECTED) == 0.0


class TestCannedVerseAccuracyFuzzy:
    """Pins check_verse_accuracy_fuzzy: best SequenceMatcher ratio over response
    sentences on normalized text (rounded to 3 decimals)."""

    RESPONSE_FULL = (
        'He said: "For God so loved the world, that he gave his only born Son, '
        'that whoever believes in him should not perish, but have eternal life."'
    )

    def test_verbatim_quote_pinned(self) -> None:
        assert check_verse_accuracy_fuzzy(self.RESPONSE_FULL, _JOHN_3_16_EXPECTED) == 0.969

    def test_paraphrase_pinned(self) -> None:
        response = (
            "The verse teaches that God loved the world so much that he gave his one and only Son."
        )
        assert check_verse_accuracy_fuzzy(response, _JOHN_3_16_EXPECTED) == 0.448

    def test_unrelated_text_pinned(self) -> None:
        response = "The capital of France is Paris."
        assert check_verse_accuracy_fuzzy(response, _JOHN_3_16_EXPECTED) == 0.205

    def test_threshold_routing_at_0_85(self) -> None:
        """Only the verbatim quote clears FUZZY_PASS_THRESHOLD=0.85; the faithful
        paraphrase does not — this is exactly why the threshold is reported
        alongside the mean."""
        from training.evaluate import FUZZY_PASS_THRESHOLD

        assert (
            check_verse_accuracy_fuzzy(self.RESPONSE_FULL, _JOHN_3_16_EXPECTED)
            >= FUZZY_PASS_THRESHOLD
        )
        paraphrase = check_verse_accuracy_fuzzy(
            "God loved the world and gave his Son.", _JOHN_3_16_EXPECTED
        )
        assert paraphrase < FUZZY_PASS_THRESHOLD


class TestCannedHasCitation:
    """Pins has_citation: VERSE_REF_PATTERN substring match."""

    def test_positive_and_negative_fixtures(self) -> None:
        assert has_citation("See John 3:16.") is True
        assert has_citation("As 1 Corinthians 13:4 states, love is patient.") is True
        assert has_citation("No reference in this sentence.") is False
        assert has_citation("") is False


class TestCannedHallucinationCorpusMode:
    """Pins check_hallucination WITH an explicit corpus lookup ('corpus' mode):
    fabricated verse numbers inside real books ARE caught (protocol v2+)."""

    LOOKUP = {
        "John 3:16": "For God so loved the world",
        "Psalms 23:1": "Yahweh is my shepherd",
    }

    def test_real_verse_not_hallucination(self) -> None:
        assert check_hallucination("John 3:16 says God loved the world.", self.LOOKUP) is False

    def test_fabricated_verse_number_in_real_book_is_hallucination(self) -> None:
        assert check_hallucination("John 119:9 is fabricated.", self.LOOKUP) is True

    def test_fake_book_is_hallucination(self) -> None:
        assert check_hallucination("Fakebook 1:2 says so.", self.LOOKUP) is True

    def test_no_citations_not_hallucination(self) -> None:
        assert check_hallucination("The Bible teaches love.", self.LOOKUP) is False


class TestCannedHallucinationFallbackMode:
    """Pins check_hallucination with an EMPTY lookup ('book_name_fallback' mode):
    only fake book names are caught; fabricated verse numbers within real books
    pass silently. This documented weakness is why evaluate.py records the
    verification mode in every artifact."""

    FALLBACK_LOOKUP: dict[str, str] = {}

    def test_fake_book_still_caught(self) -> None:
        assert check_hallucination("Fakebook 1:2 says so.", self.FALLBACK_LOOKUP) is True

    def test_real_book_with_fabricated_verse_number_passes(self) -> None:
        """Known blind spot of fallback mode — pinned deliberately."""
        assert check_hallucination("John 119:9 passes in fallback.", self.FALLBACK_LOOKUP) is False


class TestProtocolV3Constants:
    """New v3 knobs must keep their pinned values."""

    def test_fuzzy_pass_threshold_is_manifest_value(self) -> None:
        from training.evaluate import FUZZY_PASS_THRESHOLD

        assert FUZZY_PASS_THRESHOLD == 0.85

    def test_decoding_constants_pinned(self) -> None:
        from training.evaluate import EVAL_SEED, EVAL_TEMPERATURE

        assert EVAL_TEMPERATURE == 0.0
        assert EVAL_SEED == 42


class TestQueryRagDecodingParams:
    """query_rag must send pinned decoding params (manifest v3)."""

    def _capture_post(self) -> tuple:
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        return mock_client, patch

    def test_payload_carries_temperature_and_seed(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value = mock_client
            query_rag("Who was Moses?", "http://localhost:8081/v1/chat/completions")

        sent = mock_client.post.call_args.kwargs["json"]
        assert sent["temperature"] == 0.0
        assert sent["seed"] == 42

    def test_retries_without_decoding_extras_on_422(self) -> None:
        """Strict servers rejecting `seed` get a clean retry without extras."""
        from unittest.mock import MagicMock, patch

        rejected = MagicMock()
        rejected.status_code = 422
        ok = MagicMock()
        ok.status_code = 200
        ok.raise_for_status = MagicMock()
        ok.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_client = MagicMock()
        mock_client.post.side_effect = [rejected, ok]

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value = mock_client
            result = query_rag("Who was Moses?", "http://localhost:8081/v1/chat/completions")

        assert result == "ok"
        assert mock_client.post.call_count == 2
        second_payload = mock_client.post.call_args_list[1].kwargs["json"]
        assert "seed" not in second_payload
        assert "temperature" not in second_payload


class TestSummarizeKeywordResultsRefusalRouting:
    """Refusal category is count-only: no sums enter its bucket, so no verse/
    citation percentages can be computed from it."""

    def _results(self) -> list[dict]:
        return [
            {
                "category": "refusal",
                "verse_accuracy": 0.0,
                "verse_accuracy_fuzzy": 0.1,
                "fuzzy_pass": False,
                "citation_present": True,
                "hallucination_detected": False,
            },
            {
                "category": "verse_lookup",
                "verse_accuracy": 1.0,
                "verse_accuracy_fuzzy": 0.9,
                "fuzzy_pass": True,
                "citation_present": True,
                "hallucination_detected": False,
            },
        ]

    def test_refusal_bucket_is_count_only(self) -> None:
        from training.evaluate import summarize_keyword_results

        cats = summarize_keyword_results(self._results())
        assert cats["refusal"]["total"] == 1
        assert cats["refusal"].get("count_only") is True
        assert "citations" not in cats["refusal"]
        assert "verse_accuracy_sum" not in cats["refusal"]

    def test_non_refusal_bucket_accumulates(self) -> None:
        from training.evaluate import summarize_keyword_results

        cats = summarize_keyword_results(self._results())
        cs = cats["verse_lookup"]
        assert cs["total"] == 1
        assert cs["verse_accuracy_sum"] == 1.0
        assert cs["fuzzy_passes"] == 1
        assert cs["citations"] == 1
        assert cs["hallucinations"] == 0
        assert "count_only" not in cs

    def test_saved_summary_excludes_refusal_from_rates(self, tmp_path: Path) -> None:
        from training.evaluate import summarize_keyword_results

        output = tmp_path / "out.json"
        cats = summarize_keyword_results(self._results())
        _save_keyword_results(cats, self._results(), output, "model", "", "corpus")
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["refusal_count"] == 1
        # Rates computed over the 1 non-refusal item only.
        assert data["overall_citation_rate"]["n"] == 1
        assert data["overall_citation_rate"]["value"] == 1.0
        assert data["overall_hallucination_rate"]["value"] == 0.0
        assert data["overall_fuzzy_pass_rate"]["value"] == 1.0
        assert "wilson95" in data["overall_citation_rate"]
        assert data["category_summary"]["refusal"]["count_only"] is True

    def test_saved_summary_records_verification_mode_and_decoding(self, tmp_path: Path) -> None:
        from training.evaluate import summarize_keyword_results

        output = tmp_path / "out.json"
        cats = summarize_keyword_results(self._results())
        _save_keyword_results(cats, self._results(), output, "model", "", "book_name_fallback")
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["hallucination_verification_mode"] == "book_name_fallback"
        assert data["decoding"] == {"temperature": 0.0, "seed": 42}
        assert data["fuzzy_pass_threshold"] == 0.85


class TestSummarizeJudgeResultsParseFailures:
    """Judge parse failures are excluded from means but preserved per item."""

    def test_parse_failure_excluded_from_sums(self) -> None:
        from training.evaluate import summarize_judge_results

        results = [
            {
                "category": "topical",
                "judge_scores": {"faithfulness": 5, "citation": 4},
                "judge_parse_failed": False,
            },
            {
                "category": "topical",
                "judge_scores": {"error": "JSON parse failed", "faithfulness": 0},
                "judge_parse_failed": True,
            },
        ]
        buckets = summarize_judge_results(results, ["faithfulness", "citation"])
        cs = buckets["topical"]
        assert cs["total"] == 2
        assert cs["scored"] == 1
        assert cs["parse_failures"] == 1
        assert cs["faithfulness_sum"] == 5.0

    def test_error_key_implies_parse_failure_without_flag(self) -> None:
        from training.evaluate import summarize_judge_results

        results = [
            {"category": "c", "judge_scores": {"error": "boom"}, "judge_parse_failed": False}
        ]
        buckets = summarize_judge_results(results, ["faithfulness"])
        assert buckets["c"]["parse_failures"] == 1
        assert buckets["c"]["scored"] == 0
