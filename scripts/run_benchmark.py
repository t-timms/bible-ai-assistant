#!/usr/bin/env python3
"""
Run the versioned Bible Assistant benchmark and save JSON under docs/benchmark_runs/.

  python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo
  python scripts/run_benchmark.py --label orpo-f16 --ollama-model bible-assistant-orpo-f16 --judge

See docs/BENCHMARK_PROTOCOL.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = PROJECT_ROOT / "training" / "evaluate.py"
DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "manifest.v1.yaml"
RUNS_DIR = PROJECT_ROOT / "docs" / "benchmark_runs"


def _load_protocol_id(manifest_path: Path) -> str:
    try:
        import yaml  # type: ignore
    except ImportError:
        return "bible_assistant_baseline_v1"
    if not manifest_path.is_file():
        return "bible_assistant_baseline_v1"
    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return str(data.get("protocol_id") or "bible_assistant_baseline_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run versioned benchmark → docs/benchmark_runs/")
    parser.add_argument("--label", type=str, required=True, help="Short run label, e.g. orpo-q4, orpo-f16")
    parser.add_argument(
        "--ollama-model",
        type=str,
        required=True,
        help="Ollama model name (must exist: ollama list)",
    )
    parser.add_argument("--judge", action="store_true", help="Use LLM-as-judge (slower)")
    parser.add_argument(
        "--judge-url",
        type=str,
        default="",
        help="Forwarded to evaluate.py --judge-url (default: evaluate.py default, 127.0.0.1:11434)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="",
        help="Forwarded to evaluate.py --judge-model (default: evaluate.py, qwen3.5:27b)",
    )
    parser.add_argument(
        "--rag-url",
        type=str,
        default="http://localhost:8081/v1/chat/completions",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="YAML manifest with protocol_id",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="",
        help="Tag inside JSON (default: same as --label)",
    )
    args = parser.parse_args()

    protocol_id = _load_protocol_id(args.manifest)
    tag = args.model_tag.strip() or args.label
    mode = "judge" if args.judge else "keyword"
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{date}_{args.label}_{mode}.json"

    cmd = [
        sys.executable,
        str(EVALUATE),
        "--rag-url",
        args.rag_url,
        "--ollama-model",
        args.ollama_model,
        "--protocol-id",
        protocol_id,
        "--model-tag",
        tag,
        "--output",
        str(out),
    ]
    if args.judge:
        cmd.append("--judge")
        if args.judge_url.strip():
            cmd.extend(["--judge-url", args.judge_url.strip()])
        if args.judge_model.strip():
            cmd.extend(["--judge-model", args.judge_model.strip()])

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
