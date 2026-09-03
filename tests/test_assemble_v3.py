"""Offline unit tests for training/assemble_v3.py — merges teacher-distilled
answers with freshly-built keep-as-is categories into train_v3.json."""

from __future__ import annotations

import json
from pathlib import Path

from training.assemble_v3 import BLEND_N, KEEP_BUDGETS, TRIAGE_N, load_distilled, reuse_blend


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_distilled_keeps_only_ok_rows(tmp_path: Path) -> None:
    src = tmp_path / "distill_out.jsonl"
    _write_jsonl(
        src,
        [
            {
                "status": "ok",
                "category": "topical_collections",
                "context": "- **John 3:16**: For God so loved the world",
                "question": "What is the theme here?",
                "answer": "John 3:16 shows God's love expressed in giving his Son.",
            },
            {
                "status": "dropped",
                "category": "topical_collections",
                "context": "ctx",
                "question": "q?",
                "answer": "bad",
                "issues": ["unknown_reference:Hezekiah 3:5"],
            },
        ],
    )
    out = load_distilled(src)
    assert len(out) == 1
    ex = out[0]
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert "Context:" in ex["messages"][1]["content"]
    assert "Q: What is the theme here?" in ex["messages"][1]["content"]
    assert ex["messages"][2]["content"].startswith("John 3:16")
    assert ex["category"] == "topical_collections"


def test_load_distilled_tolerates_blank_lines(tmp_path: Path) -> None:
    src = tmp_path / "d.jsonl"
    src.write_text(
        '\n{"status": "ok", "category": "c", "context": "x", "question": "y?", "answer": "z"}\n\n',
        encoding="utf-8",
    )
    assert len(load_distilled(src)) == 1


def test_reuse_blend_filters_and_caps(tmp_path: Path) -> None:
    prior = [
        {"messages": [{"role": "user", "content": "a"}], "category": "general_blend"},
        {"messages": [{"role": "user", "content": "b"}], "category": "verse_recall"},
        {"messages": [{"role": "user", "content": "c"}], "category": "general_blend"},
        {"category": "general_blend"},  # no messages -> skipped
        {"messages": [{"role": "user", "content": "d"}], "category": "general_blend"},
    ]
    p = tmp_path / "train_v3.json"
    p.write_text(json.dumps(prior), encoding="utf-8")

    out = reuse_blend(p, cap=BLEND_N)
    assert [m["messages"][0]["content"] for m in out] == ["a", "c", "d"]
    assert all(set(r) == {"messages"} for r in out)  # category stripped

    assert len(reuse_blend(p, cap=2)) == 2


def test_keep_budgets_are_sane() -> None:
    assert set(KEEP_BUDGETS) == {
        "verse_recall",
        "translation_specific",
        "reverse_lookup",
        "passage_recall",
        "near_miss_guard",
    }
    verse_drill = sum(
        KEEP_BUDGETS[k]
        for k in ("verse_recall", "translation_specific", "reverse_lookup", "passage_recall")
    )
    # v2 shipped ~18k verse-drill; v3 cuts it ~60%.
    assert 6000 <= verse_drill <= 9000
    assert TRIAGE_N > 0
    # general_blend must stay above the ~20-25% catastrophic-forgetting floor.
    assert BLEND_N >= 10000
