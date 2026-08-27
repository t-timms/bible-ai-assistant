"""Pure-data tests for training.dataset_builder (T1/T2/T3).

No torch, no unsloth, no data/raw required - synthetic verse fixtures only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


from training.dataset_builder import (  # noqa: E402 -- sys.path bootstrap above
    THEME_KEYWORDS,
    build_rag_grounded,
    build_rag_multiturn,
    dedupe_by_normalized_question,
    detect_theme,
    filter_contaminated,
    iter_user_contents,
    load_contamination_questions,
    normalize_question,
    primary_question_key,
)


def _make_verses(n: int = 12) -> list[dict]:
    books = ["John", "Psalms"]
    return [
        {
            "book": books[i % len(books)],
            "chapter": 1,
            "verse": (i // 2) + 1,
            "text": f"{books[i % 2]} verse {(i // 2) + 1} about faith and love, long enough text.",
        }
        for i in range(n)
    ]


class TestNormalizeQuestion:
    def test_lowercases(self) -> None:
        assert normalize_question("What Does John 3:16 Say?") == "what does john 3:16 say"

    def test_collapses_whitespace(self) -> None:
        assert normalize_question("what   does\n\tjohn say") == "what does john say"

    def test_strips_single_trailing_punctuation(self) -> None:
        assert normalize_question("who is God?") == "who is god"

    def test_strips_stacked_punctuation(self) -> None:
        assert normalize_question("who is God?!..") == "who is god"

    def test_strips_trailing_quotes(self) -> None:
        assert normalize_question('who is God?"') == "who is god"

    def test_idempotent(self) -> None:
        once = normalize_question("What is grace?")
        assert normalize_question(once) == once


class TestFilterContaminated:
    def test_messages_format_excluded(self) -> None:
        contaminated = {"what is sin"}
        examples = [
            {
                "messages": [
                    {"role": "system", "content": "SYS"},
                    {"role": "user", "content": "What is Sin?"},
                    {"role": "assistant", "content": "..."},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "SYS"},
                    {"role": "user", "content": "Explain John 3:16."},
                    {"role": "assistant", "content": "..."},
                ]
            },
        ]
        kept, n = filter_contaminated(examples, contaminated)
        assert n == 1
        assert len(kept) == 1
        assert "John 3:16" in kept[0]["messages"][1]["content"]

    def test_prompt_string_format_excluded(self) -> None:
        contaminated = {"what is prayer"}
        examples = [{"prompt": "What is prayer?"}, {"prompt": "Tell me about Psalms 23."}]
        kept, n = filter_contaminated(examples, contaminated)
        assert n == 1
        assert kept[0]["prompt"] == "Tell me about Psalms 23."

    def test_checks_all_user_turns(self) -> None:
        contaminated = {"who was pilate"}
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Hello there."},
                    {"role": "assistant", "content": "Hi!"},
                    {"role": "user", "content": "Who Was Pilate?"},
                ]
            }
        ]
        _, n = filter_contaminated(examples, contaminated)
        assert n == 1

    def test_clean_corpus_untouched(self) -> None:
        examples = [{"prompt": f"Question number {i} about topic {i}."} for i in range(5)]
        kept, n = filter_contaminated(examples, set())
        assert n == 0 and len(kept) == 5


class TestDedupeByNormalizedQuestion:
    def test_keeps_first_and_counts(self) -> None:
        examples = [
            {"prompt": "What is faith?"},
            {"prompt": "What is Faith?"},
            {"prompt": "What is hope?"},
        ]
        kept, n = dedupe_by_normalized_question(examples)
        assert n == 1
        assert [e["prompt"] for e in kept] == ["What is faith?", "What is hope?"]

    def test_empty(self) -> None:
        assert dedupe_by_normalized_question([]) == ([], 0)


class TestUserContentExtraction:
    def test_iter_user_contents_yields_only_user_turns(self) -> None:
        example = {
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second question"},
            ]
        }
        assert list(iter_user_contents(example)) == ["first question", "second question"]

    def test_primary_question_key_prompt_string(self) -> None:
        assert primary_question_key({"prompt": "hello there"}) == "hello there"

    def test_primary_question_key_none_when_no_user(self) -> None:
        assert primary_question_key({"messages": [{"role": "assistant", "content": "x"}]}) is None


class TestDetectTheme:
    def test_returns_theme_for_keyword_text(self) -> None:
        theme, keywords = next(iter(THEME_KEYWORDS.items()))
        assert detect_theme(f"Something about {keywords[0]} today") == theme

    def test_returns_none_for_unrelated_text(self) -> None:
        assert detect_theme("zzz qqq xyzzy") is None


class TestBuildRagGroundedFormat:
    def test_uses_shared_context_format(self) -> None:
        ex = build_rag_grounded(_make_verses(), "SYS", n=4)
        assert len(ex) == 4
        roles = [m["role"] for m in ex[0]["messages"]]
        assert roles == ["system", "user", "assistant"]
        user_content = ex[0]["messages"][1]["content"]
        assert user_content.startswith("Context:\n- **")
        assert "\n\nQ: " in user_content

    def test_assistant_cites_web_reference(self) -> None:
        ex = build_rag_grounded(_make_verses(6), "SYS", n=3)
        for e in ex:
            assistant = e["messages"][2]["content"]
            assert "(WEB)" in assistant


class TestBuildRagMultiturn:
    def test_conversational_shape(self) -> None:
        ex = build_rag_multiturn(_make_verses(), "SYS", n=4)
        assert len(ex) == 4
        roles = [m["role"] for m in ex[0]["messages"]]
        assert roles[0] == "system"
        assert roles[-1] == "assistant"
        assert roles.count("user") >= 2


class TestLoadContaminationQuestions:
    def test_suite_files_win_over_eval_fallback(self, tmp_path: Path) -> None:
        suites = tmp_path / "benchmarks" / "suites"
        suites.mkdir(parents=True)
        (suites / "a.json").write_text(json.dumps([{"question": "suite question one"}]))
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "evaluation_questions.json").write_text(
            json.dumps([{"question": "fallback question"}])
        )
        qs = load_contamination_questions(tmp_path)
        assert any(q == "suite question one" for q in qs)
        assert all(q != "fallback question" for q in qs)

    def test_falls_back_to_eval_questions_without_suites(self, tmp_path: Path) -> None:
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "evaluation_questions.json").write_text(
            json.dumps([{"question": "fallback question"}])
        )
        qs = load_contamination_questions(tmp_path)
        assert any(q == "fallback question" for q in qs)

    def test_walker_collects_known_keys_only(self, tmp_path: Path) -> None:
        suites = tmp_path / "benchmarks" / "suites"
        suites.mkdir(parents=True)
        payload = [
            {"question": "alpha beta", "answer": "gamma delta"},
            {"input": "epsilon zeta"},
            {"prompt": "eta theta"},
        ]
        (suites / "s.json").write_text(json.dumps(payload))
        qs = load_contamination_questions(tmp_path)
        collected = set(qs)
        assert "alpha beta" in collected
        assert "epsilon zeta" in collected
        assert "eta theta" in collected
        assert "gamma delta" not in collected

    def test_empty_project_returns_empty_set(self, tmp_path: Path) -> None:
        assert load_contamination_questions(tmp_path) == set()
