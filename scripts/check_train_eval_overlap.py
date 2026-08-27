#!/usr/bin/env python3
"""Verify zero train/eval question overlap (audit F-1 follow-up).

Regenerates the training pools via the dataset builders' public generation
functions, applies their own decontamination filter, and lists any normalized
question that still appears in ALL benchmarks/suites/*.json snapshots.

Normalization is the shared contract from scripts/benchmark_stats.normalize_question
(identical to training/dataset_builder.normalize_question; equivalence is
pinned by tests/test_stats.py).

Exit codes: 0 = clean (or environment cannot build pools — reasons printed),
1 = overlap found. The pytest wrapper in tests/test_evaluation_questions.py
skips with a clear reason when the builders are not yet importable or no raw
corpus is present.

Usage:
  python scripts/check_train_eval_overlap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_stats import normalize_question  # noqa: E402

SUITES_DIR = PROJECT_ROOT / "benchmarks" / "suites"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
QUESTION_KEYS = {"question", "prompt", "query", "input", "user_message"}

# Public pool-builder probes tried on build_preference_data before falling back
# to its per-category generators (the training-side agent may add a public
# wrapper at any time; prefer it when present).
_PREFERENCE_POOL_PROBES = (
    "build_preference_pairs",
    "build_all_pairs",
    "build_pairs",
    "generate_pairs",
)


def read_json_tolerant(path: Path):
    """Parse JSON regardless of UTF-8 / UTF-8-BOM / UTF-16 encoding.

    The frozen suite snapshots must never be re-encoded (their sha256 pins the
    exact bytes — v1 is UTF-16), so readers have to cope, not fix.
    """
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return json.loads(data.decode("utf-16"))
    if data.startswith(b"\xef\xbb\xbf"):
        return json.loads(data.decode("utf-8-sig"))
    return json.loads(data.decode("utf-8"))


def _walk_questions(node: object, found: set[str]) -> None:
    """Collect question-ish strings; mirrors dataset_builder's tolerant walk."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in QUESTION_KEYS and isinstance(value, str):
                found.add(value)
            else:
                _walk_questions(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_questions(item, found)


def load_suite_questions(project_root: Path = PROJECT_ROOT) -> dict[str, list[str]]:
    """normalized question -> original texts, across ALL suite snapshots."""
    suites_dir = project_root / "benchmarks" / "suites"
    originals: list[str] = []
    for path in sorted(suites_dir.glob("*.json")):
        originals.extend(_collect_from_file(path))
    by_normalized: dict[str, list[str]] = {}
    for q in originals:
        if q.strip():
            by_normalized.setdefault(normalize_question(q), []).append(q)
    return by_normalized


def _collect_from_file(path: Path) -> set[str]:
    found: set[str] = set()
    _walk_questions(read_json_tolerant(path), found)
    return found


def _find_raw_bible_json(project_root: Path) -> Path | None:
    raw_dir = project_root / "data" / "raw"
    for name in ("bible.json", "bible_web.json", "bible_kjv.json", "en_bbe.json"):
        p = raw_dir / name
        if p.exists():
            return p
    for f in raw_dir.glob("*.json"):
        return f
    return None


def _dataset_builder_pool(project_root: Path) -> tuple[set[str], str | None]:
    """Normalized questions surviving the builder's own decontamination filter."""
    try:
        import training.dataset_builder as db
    except Exception as e:
        return set(), f"dataset_builder unimportable: {e}"

    input_path = _find_raw_bible_json(project_root)
    if input_path is None:
        return set(), "no raw Bible corpus under data/raw/ — cannot generate SFT pool"

    try:
        system_prompt = db.load_system_prompt(project_root, for_training=True)
        verses = db.load_verses(input_path)
        categories: dict[str, list[dict]] = {
            "verse_lookups": db.build_verse_lookups(verses, system_prompt),
            "rag_grounded": db.build_rag_grounded(verses, system_prompt),
            "rag_multiturn": db.build_rag_multiturn(verses, system_prompt),
            "thematic": db.build_thematic(system_prompt),
            "general_assistant": db.build_general_assistant(system_prompt),
            "meta_questions": db.build_meta_questions(system_prompt),
            "multi_turn": db.build_multiturn(system_prompt),
            "refusals": db.build_refusals(system_prompt),
            "cross_reference": db.build_cross_reference(system_prompt),
        }
        contaminated = db.load_contamination_questions(project_root)
        keys: set[str] = set()
        for _name, examples in categories.items():
            kept, _n_excluded = db.filter_contaminated(examples, contaminated)
            for example in kept:
                key = db.primary_question_key(example)
                if key:
                    keys.add(key)
        return keys, None
    except Exception as e:
        return set(), f"dataset_builder pool generation failed: {type(e).__name__}: {e}"


def _preference_pool(project_root: Path) -> tuple[set[str], str | None]:
    try:
        import training.build_preference_data as bpd
    except Exception as e:
        return set(), f"build_preference_data unimportable: {e}"

    try:
        pairs: list[dict] = []
        public_builder = next(
            (name for name in _PREFERENCE_POOL_PROBES if callable(getattr(bpd, name, None))),
            None,
        )
        if public_builder is not None:
            built = getattr(bpd, public_builder)()
            pairs = list(built)
        else:
            verses = bpd.load_verses()
            counts = bpd.resolve_pair_counts()
            generators = {
                "hard_negative": (bpd._build_hard_negative_pairs, verses),
                "hallucination_fake_book": (bpd._build_hallucination_pairs, verses),
                "instruction_leak": (bpd._build_instruction_leak_pairs, verses),
                "repetition": (bpd._build_repetition_pairs, verses),
                "answer_prefix": (bpd._build_answer_prefix_pairs, verses),
                "verbose": (bpd._build_verbose_pairs, verses),
                "think_tag_leak": (bpd._build_think_tag_pairs, verses),
                "bible_for_everything": (bpd._build_bible_for_everything_pairs, None),
            }
            for name, (build_fn, verses_arg) in generators.items():
                args = (counts[name],) if verses_arg is None else (verses_arg, counts[name])
                pairs.extend(build_fn(*args))

        # filter_contaminated / primary_question_key / load_contamination_questions
        # live in dataset_builder only — build_preference_data has no copies.
        from training.dataset_builder import (
            filter_contaminated,
            load_contamination_questions,
            primary_question_key,
        )

        contaminated = load_contamination_questions(project_root)
        kept, _n_excluded = filter_contaminated(pairs, contaminated)
        keys = {k for k in (primary_question_key(p) for p in kept) if k}
        return keys, None
    except Exception as e:
        return set(), f"preference pool generation failed: {type(e).__name__}: {e}"


def find_overlaps(
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Returns (overlaps, skip_reasons).

    overlaps: (pool_name, normalized_question, original_eval_question)
    skip_reasons: why a pool could not be checked (environmental, not a failure).
    """
    eval_questions = load_suite_questions(project_root)
    if not eval_questions:
        return [], ["no benchmark suite snapshots found under benchmarks/suites/"]

    pools: dict[str, set[str]] = {}
    skips: list[str] = []
    for pool_name, builder in (
        ("sft(dataset_builder)", _dataset_builder_pool),
        ("orpo(build_preference_data)", _preference_pool),
    ):
        keys, reason = builder(project_root)
        if reason:
            skips.append(f"{pool_name}: {reason}")
        else:
            pools[pool_name] = keys

    overlaps: list[tuple[str, str, str]] = []
    for pool_name, keys in pools.items():
        for normalized in sorted(keys & set(eval_questions)):
            for original in eval_questions[normalized]:
                overlaps.append((pool_name, normalized, original))
    return overlaps, skips


def main() -> int:
    print("Loading eval questions from benchmarks/suites/*.json ...")
    eval_questions = load_suite_questions()
    print(
        f"  {sum(len(v) for v in eval_questions.values())} questions "
        f"({len(eval_questions)} unique normalized)"
    )

    overlaps, skips = find_overlaps()
    for reason in skips:
        print(f"SKIPPED: {reason}")

    if not eval_questions and not skips:
        print("No suites to check against.")

    if overlaps:
        print(f"\nFAIL: {len(overlaps)} train/eval overlap(s) survived decontamination:\n")
        for pool_name, normalized, original in overlaps[:50]:
            print(f"  [{pool_name}] {original[:90]!r}  ->  {normalized[:90]!r}")
        if len(overlaps) > 50:
            print(f"  ... and {len(overlaps) - 50} more")
        print(
            "\nTraining builders MUST exclude every snapshot question "
            "(see training/dataset_builder.filter_contaminated)."
        )
        return 1

    print(
        "\nOK: zero normalized-question overlap between filtered training pools "
        "and all benchmark suites."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
