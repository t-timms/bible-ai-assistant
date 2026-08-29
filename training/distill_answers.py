#!/usr/bin/env python3
"""V3 answer distillation — regenerate templated answers with a stronger teacher.

Reads distillation *inputs* (the ``Context:`` block + ``Q:`` the dataset already
builds) and writes teacher-written answers that synthesize rather than dump a list.
Every answer is validated against the real verse corpus before it is kept: a cited
``Book C:V`` that does not resolve is a hard fail (retried once, then dropped).

Teacher-agnostic. Pick a backend:

    # dry run — no network, deterministic stub, exercises the whole pipeline
    python training/distill_answers.py --backend echo \
        --in data/raw_v3/distill_inputs.jsonl --out data/raw_v3/distill_out.jsonl

    # real runs
    ANTHROPIC_API_KEY=...  --backend anthropic --model claude-...
    OPENAI_API_KEY=...      --backend openai    --model gpt-...
    HF_TOKEN=...            --backend hf        --model Qwen/Qwen3.8-27B
    --backend vllm --vllm-url http://127.0.0.1:8001/v1 --model local

Resumable: re-running skips ``id``s already present in ``--out``.

Input JSONL line:  {"id": str, "category": str, "context": str, "question": str}
Output JSONL line: {"id", "category", "question", "context", "answer",
                    "teacher", "status": "ok"|"dropped", "issues": [...]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.verification import verify_citations  # noqa: E402

try:  # reuse the exact corpus lookup the GRPO reward uses
    from training.train_grpo import build_verse_lookup  # noqa: E402
except Exception:  # pragma: no cover - fallback if import path differs
    build_verse_lookup = None  # type: ignore[assignment]

TeacherFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> answer

SYSTEM_CONTRACT = (
    "You are a careful Bible study assistant. Answer the question using ONLY the "
    "verses in CONTEXT for any scripture quotation or citation. Synthesize: explain "
    "the idea in your own words and weave the cited verses in as support. Cite "
    "inline as `Book Chapter:Verse` (e.g. Romans 8:28). Never quote or cite a verse "
    "that is not in CONTEXT. If CONTEXT is insufficient to answer, say so briefly "
    "rather than guess. Keep it to 2-5 sentences unless the question genuinely needs "
    "more. Do not answer with a bare bullet list of verses. Tone: pastoral, plain, "
    "non-sectarian."
)

STRICTER_SUFFIX = (
    "\n\nYour previous attempt cited or quoted something not present in CONTEXT. "
    "Redo the answer. Every reference and every quoted phrase must come from a verse "
    "printed in CONTEXT below. If CONTEXT does not support an answer, say that."
)

# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _echo_teacher(_system: str, user: str) -> str:
    """Deterministic offline stub: quote the first CONTEXT verse back, plainly.

    Not a real answer — just enough structure (one real citation, prose, no list)
    to exercise validation and assembly end to end.
    """
    ref, text = _first_context_verse(user)
    if ref is None:
        return "The provided context does not contain enough to answer that."
    return (
        f"{ref} speaks to this directly: “{text}” The passage frames the "
        f"question in terms of God's dealing with his people rather than abstract "
        f"principle, which is how the surrounding verses develop it."
    )


def _anthropic_teacher(model: str) -> TeacherFn:
    import anthropic  # lazy

    client = anthropic.Anthropic()

    def call(system: str, user: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    return call


def _openai_teacher(model: str) -> TeacherFn:
    from openai import OpenAI  # lazy

    client = OpenAI()

    def call(system: str, user: str) -> str:
        r = client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (r.choices[0].message.content or "").strip()

    return call


def _hf_teacher(model: str) -> TeacherFn:
    from huggingface_hub import InferenceClient  # lazy

    client = InferenceClient(model=model, token=os.environ.get("HF_TOKEN"))

    def call(system: str, user: str) -> str:
        r = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=600,
        )
        return (r.choices[0].message.content or "").strip()

    return call


def _vllm_teacher(model: str, base_url: str) -> TeacherFn:
    import requests  # lazy; already a project dep

    url = base_url.rstrip("/") + "/chat/completions"

    def call(system: str, user: str) -> str:
        resp = requests.post(
            url,
            json={
                "model": model,
                "max_tokens": 600,
                "temperature": 0.7,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        return (body["choices"][0]["message"]["content"] or "").strip()

    return call


def make_teacher(args: argparse.Namespace) -> TeacherFn:
    if args.backend == "echo":
        return _echo_teacher
    if args.backend == "anthropic":
        return _anthropic_teacher(args.model)
    if args.backend == "openai":
        return _openai_teacher(args.model)
    if args.backend == "hf":
        return _hf_teacher(args.model)
    if args.backend == "vllm":
        return _vllm_teacher(args.model, args.vllm_url)
    raise ValueError(f"unknown backend {args.backend!r}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _first_context_verse(user_prompt: str) -> tuple[str | None, str | None]:
    """Pull the first ``- **Book C:V**: text`` line out of a Context block."""
    for line in user_prompt.splitlines():
        line = line.strip()
        if line.startswith("- **") and "**:" in line:
            ref, _, text = line[4:].partition("**:")
            return ref.strip(), text.strip()
    return None, None


def _looks_like_list_dump(text: str) -> bool:
    bullets = sum(1 for ln in text.splitlines() if ln.lstrip().startswith(("•", "- ", "* ")))
    return bullets >= 4


def validate(answer: str, verse_lookup: Callable[[str], str | None]) -> list[str]:
    """Return a list of problem strings; empty == good."""
    problems: list[str] = []
    if len(answer.strip()) < 20:
        problems.append("too_short")
    if _looks_like_list_dump(answer):
        problems.append("list_dump")
    for issue in verify_citations(answer, verse_lookup):
        if issue.reason == "unknown_reference":
            problems.append(f"unknown_reference:{issue.ref}")
        # possible_misquote is a weak signal (paraphrase is legitimate) — not fatal
    return problems


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for ln in out_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            done.add(json.loads(ln)["id"])
        except Exception:  # noqa: BLE001 - a half-written trailing line
            continue
    return done


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument(
        "--backend", choices=["echo", "anthropic", "openai", "hf", "vllm"], default="echo"
    )
    ap.add_argument("--model", default="", help="teacher model id for the chosen backend")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--corpus", default="data/raw/bible_web.json", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    args = ap.parse_args()

    corpus = args.corpus if args.corpus.is_absolute() else PROJECT_ROOT / args.corpus
    if build_verse_lookup is None:
        raise SystemExit("could not import training.train_grpo.build_verse_lookup")
    verse_lookup = build_verse_lookup(corpus)

    teacher = make_teacher(args)
    done = load_done_ids(args.out_path)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(ln) for ln in args.in_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    kept = dropped = skipped = 0
    with args.out_path.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(rows, 1):
            rid = row["id"]
            if rid in done:
                skipped += 1
                continue
            user = f"CONTEXT:\n{row['context']}\n\nQUESTION: {row['question']}"
            system = SYSTEM_CONTRACT
            answer, problems = "", ["not_attempted"]
            for attempt in range(args.max_retries + 1):
                try:
                    answer = teacher(system, user)
                except Exception as e:  # noqa: BLE001 - network/rate-limit; back off and retry
                    print(f"[{i}/{len(rows)}] {rid} call error: {e!r}; backing off", flush=True)
                    time.sleep(5 * (attempt + 1))
                    continue
                problems = validate(answer, verse_lookup)
                if not problems:
                    break
                system = SYSTEM_CONTRACT + STRICTER_SUFFIX
            status = "ok" if not problems else "dropped"
            rec = {
                "id": rid,
                "category": row.get("category", "?"),
                "question": row["question"],
                "context": row["context"],
                "answer": answer,
                "teacher": f"{args.backend}:{args.model}" if args.model else args.backend,
                "status": status,
                "issues": problems,
            }
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            if status == "ok":
                kept += 1
            else:
                dropped += 1
                print(f"[{i}/{len(rows)}] {rid} DROPPED: {problems}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)

    total_new = kept + dropped
    rate = (kept / total_new * 100) if total_new else 0.0
    print(
        f"done: kept={kept} dropped={dropped} skipped={skipped} "
        f"keep_rate={rate:.1f}% -> {args.out_path}",
        flush=True,
    )
    if total_new and rate < 90.0:
        print("WARNING: keep-rate below 90% — check the teacher prompt / model", flush=True)


if __name__ == "__main__":
    main()
