#!/usr/bin/env python3
"""Stage 3 — GRPO with fully verifiable rewards (RLVR).

Starts from the SFT (or SFT+ORPO) adapter and optimises it against a *programmatic*
reward — no LLM judge, no human labels — so the signal is exactly the thing this
project is graded on:

    R(response) = w_cite * citation_exists_in_index
                + w_text * quoted_text_matches_a_real_verse
                + w_fmt  * format_compliance (prompt_format contract)

Weights come from the ``grpo:`` block of the training config
(``training/config.v2-9b.yaml`` / ``training/config.v2.yaml``).

Environment: conda ``bible-orpo`` (transformers >= 5.1, native Qwen3.5) + Unsloth,
which ships GRPO/DAPO support on top of TRL's GRPOTrainer. Blackwell: bf16 (or fp8
where available) — never fp16.

Usage:
    python training/train_grpo.py \
        --policy-path models/qwen3.5-9b-bible-v2-orpo \
        --config training/config.v2-9b.yaml \
        --run-name qwen3.5-9b-bible-v2-grpo
    python training/train_grpo.py --policy-path ... --dry-run   # reward wiring only, no GPU

STATUS: scaffold. The reward functions and config plumbing are complete and unit-
testable; the training loop needs a real smoke run (``--max-steps 2``) on the target
GPU before any overnight launch. Not exercised by CI (needs torch/unsloth/CUDA).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.prompt_format import extract_question  # noqa: E402
from rag.verification import (  # noqa: E402
    _normalize_for_compare,
    _quoted_spans,
    extract_verse_refs,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DEFAULT_WEIGHTS = {"citation_exists": 0.5, "text_match_exact": 0.35, "format_compliance": 0.15}
DEFAULT_GRPO = {
    "num_generations": 8,
    "beta_kl": 0.02,
    "max_completion_length": 512,
    "learning_rate": 1.0e-6,
    "temperature": 1.0,
}


def load_grpo_config(cfg_path: Path) -> tuple[dict, dict, str]:
    """Return (reward_weights, grpo_params, train_file) from a training YAML."""
    import yaml

    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    grpo = cfg.get("grpo") or {}
    weights = {**DEFAULT_WEIGHTS, **(grpo.get("reward_weights") or {})}
    params = {**DEFAULT_GRPO, **{k: v for k, v in grpo.items() if k != "reward_weights"}}
    train_file = ((cfg.get("data") or {}).get("train_file")) or "data/processed/train_v2.json"
    return weights, params, train_file


# --------------------------------------------------------------------------- #
# Verse lookup (dict-backed; no ChromaDB / RAG server dependency for training)
# --------------------------------------------------------------------------- #

_REF_KEY = re.compile(r"^\s*([1-3]?\s?[A-Za-z][A-Za-z ]*?)\s+(\d+):(\d+)\s*$")


def build_verse_lookup(corpus_path: Path):
    """`ref -> verse text` from a flat Bible JSON (data/raw/bible_web.json shape).

    Accepts either a list of {book,chapter,verse,text} rows or a nested
    {book: {chapter: {verse: text}}} mapping. Returns a callable compatible with
    rag.verification.verify_citations' `verse_lookup`.
    """
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    table: dict[str, str] = {}

    def _put(book: str, ch, vs, text: str) -> None:
        table[f"{book.strip().lower()} {int(ch)}:{int(vs)}"] = text

    if isinstance(raw, list):
        for row in raw:
            _put(row["book"], row["chapter"], row["verse"], row["text"])
    elif isinstance(raw, dict):
        for book, chapters in raw.items():
            for ch, verses in chapters.items():
                for vs, text in verses.items():
                    _put(book, ch, vs, text)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unrecognised corpus shape in {corpus_path}")

    # Book-name spellings that differ between the training data ("Psalm",
    # "Song of Songs") and data/raw/bible_web.json ("Psalms", "Song of Solomon").
    # Without this the reward silently under-credits correct citations to those books.
    _ALIASES = {
        "psalm": "psalms",
        "psalms": "psalm",
        "song of songs": "song of solomon",
        "song of solomon": "song of songs",
    }

    def lookup(ref: str) -> str | None:
        m = _REF_KEY.match(ref)
        if not m:
            return table.get(ref.strip().lower())
        book, ch, vs = m.group(1).strip().lower(), int(m.group(2)), int(m.group(3))
        hit = table.get(f"{book} {ch}:{vs}")
        if hit is None and book in _ALIASES:
            hit = table.get(f"{_ALIASES[book]} {ch}:{vs}")
        return hit

    return lookup


# --------------------------------------------------------------------------- #
# Reward components — each returns a float in [0, 1]
# --------------------------------------------------------------------------- #


def r_citation_exists(text: str, verse_lookup) -> float:
    refs = extract_verse_refs(text)
    if not refs:
        return 0.0
    good = sum(1 for r in refs if verse_lookup(r) is not None)
    return good / len(refs)


def r_text_match(text: str, verse_lookup) -> float:
    """Best fuzzy overlap between any quoted span and the verse it's attributed to."""
    quotes = _quoted_spans(text)
    refs = [r for r in extract_verse_refs(text) if verse_lookup(r) is not None]
    if not quotes or not refs:
        return 0.0
    from difflib import SequenceMatcher

    best = 0.0
    for ref in refs:
        real = _normalize_for_compare(verse_lookup(ref) or "")
        for q in quotes:
            best = max(best, SequenceMatcher(None, real, _normalize_for_compare(q)).ratio())
    return best


_ANSWER_PREFIX = re.compile(r"^\s*(answer|response)\s*:", re.IGNORECASE)


def r_format_compliance(text: str) -> float:
    """Cheap proxy for the prompt_format contract: no 'Answer:' prefix, no leaked
    Context block, at least one verse reference, not pathologically long."""
    score = 1.0
    if _ANSWER_PREFIX.match(text):
        score -= 0.5
    if "Context:" in text or "<|im_start|>" in text:
        score -= 0.5
    if not extract_verse_refs(text):
        score -= 0.25
    if len(text) > 2000:
        score -= 0.25
    return max(0.0, score)


def make_reward_fn(weights: dict, verse_lookup):
    """Build the TRL GRPOTrainer-compatible reward function."""
    w_c = weights["citation_exists"]
    w_t = weights["text_match_exact"]
    w_f = weights["format_compliance"]

    def reward_fn(completions=None, **_kwargs) -> list[float]:
        out: list[float] = []
        for comp in completions or []:
            text = comp if isinstance(comp, str) else comp[-1]["content"]
            out.append(
                w_c * r_citation_exists(text, verse_lookup)
                + w_t * r_text_match(text, verse_lookup)
                + w_f * r_format_compliance(text)
            )
        return out

    reward_fn.__name__ = "verifiable_bible_reward"
    return reward_fn


# --------------------------------------------------------------------------- #
# Prompt dataset — GRPO consumes prompts only
# --------------------------------------------------------------------------- #


def load_prompt_dataset(train_file: Path, limit: int | None):
    from datasets import load_dataset

    if not train_file.is_absolute():
        train_file = PROJECT_ROOT / train_file
    ds = load_dataset("json", data_files=str(train_file), split="train")

    def to_prompt(ex):
        msgs = ex.get("messages") or []
        user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
        return {"prompt": [{"role": "user", "content": extract_question(user) or user}]}

    ds = ds.map(to_prompt, remove_columns=[c for c in ds.column_names if c != "prompt"])
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--policy-path", required=True, help="SFT or SFT+ORPO adapter dir to start from"
    )
    ap.add_argument("--config", default="training/config.v2-9b.yaml")
    ap.add_argument(
        "--data", default=None, help="Override prompt dataset (default: config data.train_file)"
    )
    ap.add_argument(
        "--corpus", default="data/raw/bible_web.json", help="Verse-text corpus for the reward"
    )
    ap.add_argument("--run-name", default="qwen3.5-9b-bible-v2-grpo")
    ap.add_argument(
        "--max-steps", type=int, default=-1, help="-1 = full run; use 2 for a smoke test"
    )
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="Build reward + dataset, score a stub, exit (no GPU)"
    )
    args = ap.parse_args()

    weights, gp, cfg_train_file = load_grpo_config(Path(args.config))
    train_file = Path(args.data or cfg_train_file)
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = PROJECT_ROOT / corpus

    print(f"[grpo] reward weights: {weights}")
    print(f"[grpo] grpo params:    {gp}")
    print(f"[grpo] prompts:        {train_file}")
    print(f"[grpo] reward corpus:  {corpus}")

    if not corpus.exists():
        if not args.dry_run:
            raise FileNotFoundError(
                f"Verse corpus not found: {corpus}. Fetch data/raw/ (see docs/WALKTHROUGH.md) "
                "or pass --corpus."
            )
        print(f"[dry-run] {corpus} missing — using a 1-verse stub lookup")
        _stub = {"john 3:16": "For God so loved the world, that he gave his one and only Son..."}
        verse_lookup = lambda ref: _stub.get(ref.strip().lower())  # noqa: E731
    else:
        verse_lookup = build_verse_lookup(corpus)
    reward_fn = make_reward_fn(weights, verse_lookup)

    if args.dry_run:
        good = 'John 3:16 (WEB): "For God so loved the world, that he gave his one and only Son..."'
        bad = "Answer: According to Hesitations 9:99 the sky is green."
        print(f"[dry-run] reward(good) = {reward_fn(completions=[good])[0]:.3f}")
        print(f"[dry-run] reward(bad)  = {reward_fn(completions=[bad])[0]:.3f}")
        print("[dry-run] OK — reward wiring is sane; run without --dry-run on the GPU box.")
        return

    # ---- real training path (bible-orpo env) ----
    import torch  # noqa: F401
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.policy_path,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,  # bf16 auto on Blackwell
    )
    FastLanguageModel.for_training(model)

    dataset = load_prompt_dataset(train_file, args.limit_prompts)

    grpo_cfg = GRPOConfig(
        output_dir=f"checkpoints_v2_grpo/{args.run_name}",
        run_name=args.run_name,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=int(gp["num_generations"]),
        max_completion_length=int(gp["max_completion_length"]),
        learning_rate=float(gp["learning_rate"]),
        beta=float(gp["beta_kl"]),
        temperature=float(gp.get("temperature", 1.0)),
        bf16=True,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=50,
        report_to=("none" if args.no_wandb else "wandb"),
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=dataset,
    )
    trainer.train()
    out = Path(grpo_cfg.output_dir) / "final"
    trainer.save_model(str(out))
    print(f"[grpo] saved -> {out}")


if __name__ == "__main__":
    main()
