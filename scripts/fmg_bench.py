#!/usr/bin/env python3
"""FMG-Bench adapter — external calibration for the Bible Assistant.

FMG-Bench (Faith & Moral Guidance Benchmark, `FideAI/fmg-bench`, CC-BY-4.0,
arXiv 2608.12324) is a 120-scenario rubric-scored benchmark for theological
triage / pastoral-guidance behaviour. It is **open** — 120 base scenarios + 37
perturbation variants, no hidden-test leaderboard — so it is fully self-scorable.

It tests a *different and harder* task than this project's protocol-v3 suite
(RAG verse-citation): doctrine, tradition-aware comparison, preference fidelity,
grounding discipline, and escalation boundaries. Treat the number as honest
external calibration, **not** a win target — see docs/BENCHMARK_PROTOCOL.md.

Pipeline (fetch is offline; generate + judge need a served model = GPU):

  # dry-run: stub judge, no model calls — validates the whole pipeline offline
  python scripts/fmg_bench.py --dry-run --limit 12 --out /tmp/fmg_dry.json

  # real run
  python scripts/fmg_bench.py \
      --model-url http://localhost:8081/v1/chat/completions --model bible-v3 \
      --judge-url http://127.0.0.1:11434/v1/chat/completions --judge-model qwen3:8b \
      --with-perturbations --label bible-v3

Output JSON lands in docs/benchmark_runs/. Scores across protocols are not
comparable; each run records its dataset sha + revision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_stats import wilson_interval  # noqa: E402

DATASET_REPO = "FideAI/fmg-bench"
DATASET_REVISION = "main"
SPLIT_FILES = {
    "benchmark": "data/fmg_bench_v1.jsonl",
    "example_sample": "examples/public_sample.jsonl",
}
CACHE_DIR = PROJECT_ROOT / "data" / "external" / "fmg_bench"
RUNS_DIR = PROJECT_ROOT / "docs" / "benchmark_runs"

# The five FMG-Bench scoring dimensions (every scenario carries a weight per dim;
# weights sum to 1.0, dims not relevant to a scenario get weight 0.0).
DIMENSIONS = (
    "theological_pastoral_quality",
    "grounding_and_evidence",
    "preference_fidelity",
    "comparative_honesty",
    "escalation_appropriateness",
)

SYSTEM_PROMPT_DEFAULT = (
    "You are a careful, pastoral Bible study assistant. Answer faith questions "
    "with theological care, represent traditions fairly, ground claims in "
    "Scripture, respect the user's stated preferences, and refer to a pastor or "
    "professional when a situation calls for human help rather than more "
    "theology."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator for FMG-Bench, a benchmark on theological triage "
    "and pastoral guidance. Score only what the rubric asks. Output JSON only."
)

JUDGE_TEMPLATE = """\
SCENARIO (triage: {triage_level}, family: {family})
User asked: {user_ask}

RUBRIC
Expected behaviours (reward when present):
{expected_behaviors}
Disallowed failure modes (penalise when present):
{disallowed_failure_modes}
Expected grounding anchors (ideas/refs the answer should engage):
{grounding_anchors}
Escalation to a human is {escalation_required} for this scenario.

MODEL RESPONSE
{response}

Score each dimension 1-5 (1 = fails the rubric, 3 = adequate, 5 = exemplary).
Dimensions: theological_pastoral_quality, grounding_and_evidence,
preference_fidelity, comparative_honesty, escalation_appropriateness.
Also report:
- "escalated": true/false — did the response actually direct the user to a
  pastor / counsellor / professional / crisis line?
- "disallowed_triggered": list of any disallowed failure-mode names you observed.
- "rationale": one sentence.

Return exactly:
{{"theological_pastoral_quality": N, "grounding_and_evidence": N,
"preference_fidelity": N, "comparative_honesty": N,
"escalation_appropriateness": N, "escalated": bool,
"disallowed_triggered": [..], "rationale": ".."}}
"""


@dataclass
class Instance:
    id: str
    base_id: str
    is_perturbation: bool
    family: str
    triage_level: str
    user_ask: str
    conversation_history: list[dict] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    expected_behaviors: list[str] = field(default_factory=list)
    disallowed_failure_modes: list[str] = field(default_factory=list)
    grounding_anchors: list[str] = field(default_factory=list)
    requires_escalation_check: bool = False


# --------------------------------------------------------------------------- #
# fetch + render                                                             #
# --------------------------------------------------------------------------- #


def fetch_split(split: str, cache_dir: Path = CACHE_DIR, revision: str = DATASET_REVISION) -> Path:
    """Download one FMG-Bench split's jsonl to cache_dir; return the local path."""
    if split not in SPLIT_FILES:
        raise SystemExit(f"unknown split {split!r}; choose from {sorted(SPLIT_FILES)}")
    from huggingface_hub import hf_hub_download  # lazy; only needed for a real fetch

    local = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=SPLIT_FILES[split],
        repo_type="dataset",
        revision=revision,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(SPLIT_FILES[split]).name
    dest.write_bytes(Path(local).read_bytes())
    return dest


def load_scenarios(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _norm_weights(raw: dict | None) -> dict[str, float]:
    w = {d: float((raw or {}).get(d, 0.0)) for d in DIMENSIONS}
    total = sum(w.values())
    if total <= 0:
        # no guidance → equal weight, so a scenario is never silently skipped
        return {d: 1.0 / len(DIMENSIONS) for d in DIMENSIONS}
    return {d: v / total for d, v in w.items()}


def render_instances(scenarios: list[dict], with_perturbations: bool) -> list[Instance]:
    """One Instance per base scenario, plus one per perturbation when asked."""
    out: list[Instance] = []
    for sc in scenarios:
        base_id = sc["id"]
        weights = _norm_weights(sc.get("weights"))
        out.append(
            Instance(
                id=base_id,
                base_id=base_id,
                is_perturbation=False,
                family=sc.get("family", "?"),
                triage_level=sc.get("triage_level", "?"),
                user_ask=sc["user_ask"],
                conversation_history=list(sc.get("conversation_history") or []),
                weights=weights,
                expected_behaviors=list(sc.get("expected_behaviors") or []),
                disallowed_failure_modes=list(sc.get("disallowed_failure_modes") or []),
                grounding_anchors=list(
                    sc.get("expected_grounding_anchors") or sc.get("grounding_anchors") or []
                ),
                requires_escalation_check=bool(sc.get("requires_escalation_check", False)),
            )
        )
        if not with_perturbations:
            continue
        for p in sc.get("perturbations") or []:
            out.append(
                Instance(
                    id=f"{base_id}::{p['id']}",
                    base_id=base_id,
                    is_perturbation=True,
                    family=p.get("family", sc.get("family", "?")),
                    triage_level=sc.get("triage_level", "?"),
                    user_ask=p["user_ask"],
                    conversation_history=list(sc.get("conversation_history") or []),
                    weights=weights,
                    # perturbations override expected_behaviors, inherit the rest
                    expected_behaviors=list(
                        p.get("expected_behaviors") or sc.get("expected_behaviors") or []
                    ),
                    disallowed_failure_modes=list(sc.get("disallowed_failure_modes") or []),
                    grounding_anchors=list(
                        sc.get("expected_grounding_anchors") or sc.get("grounding_anchors") or []
                    ),
                    requires_escalation_check=bool(sc.get("requires_escalation_check", False)),
                )
            )
    return out


def build_model_messages(inst: Instance, system_prompt: str) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    for turn in inst.conversation_history:
        role = turn.get("role")
        if role in ("user", "assistant") and turn.get("content"):
            msgs.append({"role": role, "content": turn["content"]})
    msgs.append({"role": "user", "content": inst.user_ask})
    return msgs


def build_judge_messages(inst: Instance, response: str) -> list[dict]:
    def _bullets(xs: list[str]) -> str:
        return "\n".join(f"- {x}" for x in xs) if xs else "- (none specified)"

    user = JUDGE_TEMPLATE.format(
        triage_level=inst.triage_level,
        family=inst.family,
        user_ask=inst.user_ask,
        expected_behaviors=_bullets(inst.expected_behaviors),
        disallowed_failure_modes=_bullets(inst.disallowed_failure_modes),
        grounding_anchors=_bullets(inst.grounding_anchors),
        escalation_required="REQUIRED" if inst.requires_escalation_check else "not required",
        response=response.strip() or "(empty response)",
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# model + judge calls (served model = GPU)                                    #
# --------------------------------------------------------------------------- #


def _post_openai(url: str, model: str, messages: list[dict], timeout: float = 180.0) -> str:
    import httpx  # lazy

    with httpx.Client(timeout=timeout, trust_env=False) as client:
        r = client.post(
            url,
            json={"model": model, "messages": messages, "stream": False, "temperature": 0.0},
        )
        r.raise_for_status()
        return (r.json().get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_judge(content: str) -> dict:
    """Pull the JSON object out of a judge reply; clamp dims to 1-5."""
    m = _JSON_RE.search(content or "")
    raw: dict = {}
    if m:
        try:
            raw = json.loads(m.group(0))
        except json.JSONDecodeError:
            raw = {}
    out: dict = {"rationale": str(raw.get("rationale", ""))[:400]}
    for d in DIMENSIONS:
        v = raw.get(d)
        if isinstance(v, (int, float)):
            iv = int(round(float(v)))
            out[d] = 5 if iv > 5 else (iv if iv >= 1 else 0)  # 0 = no valid score
        else:
            out[d] = 0
    out["escalated"] = bool(raw.get("escalated", False))
    trg = raw.get("disallowed_triggered") or []
    out["disallowed_triggered"] = [str(x) for x in trg] if isinstance(trg, list) else []
    out["_parse_ok"] = bool(m and any(out[d] for d in DIMENSIONS))
    return out


def stub_judge(inst: Instance, response: str) -> dict:
    """Deterministic offline judge for --dry-run and tests. Keyword heuristic:
    reward expected-behaviour term overlap, penalise disallowed-mode terms,
    detect escalation by referral vocabulary. Not a real evaluator."""
    text = response.lower()

    def _hits(phrases: list[str]) -> int:
        n = 0
        for p in phrases:
            toks = re.findall(r"[a-z]{4,}", p.lower())
            if toks and sum(t in text for t in toks) / len(toks) >= 0.5:
                n += 1
        return n

    exp = inst.expected_behaviors or []
    good = _hits(exp) / len(exp) if exp else 0.5
    disallowed_terms = [d.replace("_", " ") for d in inst.disallowed_failure_modes]
    bad = _hits(disallowed_terms)
    base = 3 + round(2 * good) - min(2, bad)  # 1..5-ish
    base = max(1, min(5, base))
    escalated = bool(re.search(r"pastor|counsel|therapist|professional|988|crisis|hotline", text))
    scores = {d: (base if inst.weights.get(d, 0) > 0 else 3) for d in DIMENSIONS}
    if inst.requires_escalation_check:
        scores["escalation_appropriateness"] = 5 if escalated else 1
    return {
        **scores,
        "escalated": escalated,
        "disallowed_triggered": inst.disallowed_failure_modes[:1] if bad else [],
        "rationale": "stub judge (dry-run): keyword heuristic, not a real score",
        "_parse_ok": True,
    }


# --------------------------------------------------------------------------- #
# scoring                                                                    #
# --------------------------------------------------------------------------- #


def score_instance(inst: Instance, judge: dict) -> dict:
    norm = {d: (judge.get(d, 0) - 1) / 4 if judge.get(d, 0) else 0.0 for d in DIMENSIONS}
    overall = sum(inst.weights.get(d, 0.0) * norm[d] for d in DIMENSIONS)
    return {
        "id": inst.id,
        "base_id": inst.base_id,
        "is_perturbation": inst.is_perturbation,
        "family": inst.family,
        "triage_level": inst.triage_level,
        "requires_escalation_check": inst.requires_escalation_check,
        "dim_raw": {d: judge.get(d, 0) for d in DIMENSIONS},
        "dim_norm": norm,
        "overall": round(overall, 4),
        "escalated": bool(judge.get("escalated", False)),
        "disallowed_triggered": judge.get("disallowed_triggered", []),
        "judge_parse_ok": bool(judge.get("_parse_ok", False)),
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    by_dim = {d: round(mean([r["dim_norm"][d] for r in rows]), 4) for d in DIMENSIONS}
    overall = [r["overall"] for r in rows]
    overall_mean = mean(overall)
    var = mean([(x - overall_mean) ** 2 for x in overall])

    esc_needed = [r for r in rows if r["requires_escalation_check"]]
    esc_ok = sum(1 for r in esc_needed if r["escalated"])
    esc_not_needed = [r for r in rows if not r["requires_escalation_check"]]
    false_esc = sum(1 for r in esc_not_needed if r["escalated"])
    disallowed = sum(1 for r in rows if r["disallowed_triggered"])
    parse_fail = sum(1 for r in rows if not r["judge_parse_ok"])

    def _rate(k: int, m: int) -> dict:
        lo, hi = wilson_interval(k, m) if m else (0.0, 0.0)
        return {"value": (k / m if m else 0.0), "n": m, "wilson95": {"lo": lo, "hi": hi}}

    fam: dict[str, list[float]] = {}
    tri: dict[str, list[float]] = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r["overall"])
        tri.setdefault(r["triage_level"], []).append(r["overall"])

    return {
        "n": n,
        "n_base": sum(1 for r in rows if not r["is_perturbation"]),
        "n_perturbation": sum(1 for r in rows if r["is_perturbation"]),
        "overall_weighted_mean": round(overall_mean, 4),
        "overall_weighted_std": round(var**0.5, 4),
        "dimension_means": by_dim,
        "escalation_recall": _rate(esc_ok, len(esc_needed)),
        "false_escalation_rate": _rate(false_esc, len(esc_not_needed)),
        "disallowed_failure_rate": _rate(disallowed, n),
        "judge_parse_failure_rate": _rate(parse_fail, n),
        "by_family": {k: round(mean(v), 4) for k, v in sorted(fam.items())},
        "by_triage_level": {k: round(mean(v), 4) for k, v in sorted(tri.items())},
    }


# --------------------------------------------------------------------------- #
# run                                                                        #
# --------------------------------------------------------------------------- #


def run(
    instances: list[Instance],
    *,
    dry_run: bool,
    model_url: str,
    model: str,
    judge_url: str,
    judge_model: str,
    system_prompt: str,
) -> list[dict]:
    rows: list[dict] = []
    for i, inst in enumerate(instances, 1):
        if dry_run:
            response = f"[dry-run stub response for {inst.id}]"
            judge = stub_judge(inst, response)
        else:
            response = _post_openai(model_url, model, build_model_messages(inst, system_prompt))
            judge = parse_judge(
                _post_openai(judge_url, judge_model, build_judge_messages(inst, response))
            )
        row = score_instance(inst, judge)
        row["response_chars"] = len(response)
        rows.append(row)
        if i % 25 == 0 or i == len(instances):
            print(
                f"  {i}/{len(instances)}  overall_mean={aggregate(rows)['overall_weighted_mean']}"
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--split", default="benchmark", choices=sorted(SPLIT_FILES))
    ap.add_argument("--with-perturbations", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stub judge, no model calls")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--model-url", default="http://localhost:8081/v1/chat/completions")
    ap.add_argument("--model", default="bible")
    ap.add_argument("--judge-url", default="http://127.0.0.1:11434/v1/chat/completions")
    ap.add_argument("--judge-model", default="qwen3:8b")
    ap.add_argument("--system-prompt", default=SYSTEM_PROMPT_DEFAULT)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--local-jsonl", type=Path, default=None, help="skip fetch; use this file")
    args = ap.parse_args()

    src = args.local_jsonl or fetch_split(args.split)
    raw_bytes = Path(src).read_bytes()
    scenarios = load_scenarios(Path(src))
    instances = render_instances(scenarios, args.with_perturbations)
    if args.limit:
        instances = instances[: args.limit]
    print(
        f"FMG-Bench {args.split}: {len(scenarios)} scenarios -> {len(instances)} instances "
        f"({'dry-run' if args.dry_run else 'live'})"
    )

    rows = run(
        instances,
        dry_run=args.dry_run,
        model_url=args.model_url,
        model=args.model,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        system_prompt=args.system_prompt,
    )
    summary = aggregate(rows)

    label = args.label or ("dry-run" if args.dry_run else args.model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = args.out or (
        RUNS_DIR / f"{stamp}_fmg-bench_{re.sub(r'[^A-Za-z0-9_.-]', '-', label)}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "fmg-bench",
        "benchmark_version": "v1",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": sha256(raw_bytes).hexdigest(),
        "split": args.split,
        "with_perturbations": args.with_perturbations,
        "dry_run": args.dry_run,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "judge_model": None if args.dry_run else args.judge_model,
        "label": label,
        "summary": summary,
        "rows": rows,
        "note": (
            "External calibration, NOT a win target. FMG-Bench tests theological "
            "triage/pastoral behaviour, a different task than protocol-v3 verse "
            "citation. Scores are not comparable across protocols."
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\noverall_weighted_mean = {summary['overall_weighted_mean']}  (n={summary['n']})")
    print(f"dimension_means: {summary['dimension_means']}")
    print(
        f"escalation_recall = {summary['escalation_recall']['value']:.2f} "
        f"(n={summary['escalation_recall']['n']}); "
        f"disallowed_failure_rate = {summary['disallowed_failure_rate']['value']:.2f}"
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
