#!/usr/bin/env python3
"""
Run the versioned Bible Assistant benchmark and save JSON under docs/benchmark_runs/.

  python scripts/run_benchmark.py --label orpo-q4 --ollama-model bible-assistant-orpo
  python scripts/run_benchmark.py --label orpo-f16 --ollama-model bible-assistant-orpo-f16 --judge

The --manifest default resolves the LATEST benchmarks/manifest.v*.yaml by version
number (v3 today). The manifest's suite_sha256 is verified against the actual
snapshot file bytes before anything runs — a mismatch aborts. Running an older
manifest than the latest requires an explicit --allow-older-manifest.

See docs/BENCHMARK_PROTOCOL.md
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = PROJECT_ROOT / "training" / "evaluate.py"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
MANIFEST_GLOB = "manifest.v*.yaml"
RUNS_DIR = PROJECT_ROOT / "docs" / "benchmark_runs"

_MANIFEST_VERSION_RE = re.compile(r"manifest\.v(\d+)\.ya?ml$")


def manifest_version(path: Path) -> int | None:
    """manifest.v3.yaml → 3; None when the name doesn't carry a version."""
    m = _MANIFEST_VERSION_RE.search(path.name)
    return int(m.group(1)) if m else None


def find_manifests(benchmarks_dir: Path = BENCHMARKS_DIR) -> list[Path]:
    return sorted(benchmarks_dir.glob(MANIFEST_GLOB))


def resolve_latest_manifest(benchmarks_dir: Path = BENCHMARKS_DIR) -> Path:
    """Highest-versioned manifest.vN.yaml; raises SystemExit when none exist."""
    versioned = [(manifest_version(p), p) for p in find_manifests(benchmarks_dir)]
    versioned = [(v, p) for v, p in versioned if v is not None]
    if not versioned:
        raise SystemExit(
            f"No {MANIFEST_GLOB} manifests found under {benchmarks_dir}. "
            "A versioned protocol manifest is required to run the benchmark."
        )
    return max(versioned, key=lambda vp: vp[0])[1]


def load_manifest(manifest_path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit("PyYAML is required to read benchmark manifests") from e
    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Manifest is not a YAML mapping: {manifest_path}")
    return data


def verify_suite_sha256(manifest_path: Path, manifest_data: dict) -> str:
    """Fail fast when the pinned suite hash doesn't match the snapshot bytes.

    Suite paths are resolved relative to the manifest's project root (the
    manifest lives in ``<root>/benchmarks/``). Returns the resolved suite path
    (relative). Manifests without a suite_sha256 pin (v1/v2) only warn — they
    cannot be byte-verified by design.
    """
    suite_rel = str(manifest_data.get("suite_path") or "").strip()
    pinned = str(manifest_data.get("suite_sha256") or "").strip().lower()
    if not suite_rel:
        raise SystemExit(f"Manifest has no suite_path: {manifest_path}")
    suite_path = manifest_path.resolve().parent.parent / suite_rel
    if not suite_path.is_file():
        raise SystemExit(f"Pinned suite snapshot missing: {suite_path}")
    if not pinned:
        print(
            f"WARNING: {manifest_path.name} pins no suite_sha256 — results from it "
            "are not byte-reproducible."
        )
        return suite_rel
    actual = hashlib.sha256(suite_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != pinned:
        raise SystemExit(
            f"ABORT: suite snapshot hash mismatch for {suite_path}\n"
            f"  manifest {manifest_path.name} pins: {pinned}\n"
            f"  actual file bytes hash:   {actual}\n"
            "The frozen snapshot was modified or corrupted. Never edit snapshots — "
            "publish a new suite + manifest instead (docs/BENCHMARK_PROTOCOL.md)."
        )
    print(f"Suite sha256 verified OK: {suite_rel}")
    return suite_rel


def ensure_not_older(chosen_version: int | None, latest_version: int, allow_older: bool) -> None:
    """Refuse older-than-latest manifests without an explicit opt-in."""
    if chosen_version is None or chosen_version >= latest_version or allow_older:
        return
    raise SystemExit(
        f"ABORT: manifest.v{chosen_version} is older than the latest "
        f"(manifest.v{latest_version}). Scores across protocol versions are NOT "
        "comparable. Re-run against the latest manifest (omit --manifest), or pass "
        "--allow-older-manifest explicitly if you really mean to reproduce history."
    )


def _load_protocol_id(manifest_data: dict, manifest_path: Path) -> str:
    protocol_id = str(manifest_data.get("protocol_id") or "").strip()
    if not protocol_id:
        raise SystemExit(f"Manifest has no protocol_id: {manifest_path}")
    return protocol_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run versioned benchmark → docs/benchmark_runs/")
    parser.add_argument(
        "--label", type=str, required=True, help="Short run label, e.g. orpo-q4, orpo-f16"
    )
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
        default=None,
        help="YAML manifest with protocol_id (default: latest benchmarks/manifest.v*.yaml)",
    )
    parser.add_argument(
        "--allow-older-manifest",
        action="store_true",
        help="Explicitly allow running a manifest older than the latest (non-comparable scores)",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="",
        help="Tag inside JSON (default: same as --label)",
    )
    args = parser.parse_args()

    manifest_path = (
        args.manifest if args.manifest is not None else resolve_latest_manifest(BENCHMARKS_DIR)
    ).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    chosen_version = manifest_version(manifest_path)
    latest_version = manifest_version(resolve_latest_manifest(BENCHMARKS_DIR))
    ensure_not_older(chosen_version, latest_version or 0, args.allow_older_manifest)

    manifest_data = load_manifest(manifest_path)
    verify_suite_sha256(manifest_path, manifest_data)
    protocol_id = _load_protocol_id(manifest_data, manifest_path)

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

    print(f"Manifest: {manifest_path.name} (protocol {protocol_id})")
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
