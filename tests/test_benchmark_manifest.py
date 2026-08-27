"""Benchmark manifest schema validation using Pydantic models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
# Every versioned protocol file is checked, not just v1 — new manifest.vN.yaml files
# (frozen once published, per docs/BENCHMARK_PROTOCOL.md) are picked up automatically.
MANIFEST_PATHS = sorted(BENCHMARKS_DIR.glob("manifest.v*.yaml"))


def _load(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        pytest.skip("PyYAML not installed")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate(manifest_data: dict) -> None:
    """Validate raw manifest dict against the BenchmarkManifest schema."""
    from scripts.benchmark_schema import BenchmarkManifest

    BenchmarkManifest.model_validate(manifest_data)


@pytest.mark.parametrize(
    "path", MANIFEST_PATHS, ids=[p.name for p in MANIFEST_PATHS] or ["no-manifest"]
)
def test_manifest_validates_against_schema(path: Path) -> None:
    """Every versioned manifest YAML must pass full Pydantic schema validation."""
    if not MANIFEST_PATHS:
        pytest.skip("No manifest.v*.yaml files found")
    _validate(_load(path))


@pytest.mark.parametrize(
    "path", MANIFEST_PATHS, ids=[p.name for p in MANIFEST_PATHS] or ["no-manifest"]
)
def test_suite_path_exists(path: Path) -> None:
    """suite_path must reference an existing file relative to the project root."""
    if not MANIFEST_PATHS:
        pytest.skip("No manifest.v*.yaml files found")
    manifest_data = _load(path)
    rel = manifest_data.get("suite_path")
    assert rel, "manifest must set suite_path"
    assert (PROJECT_ROOT / rel).is_file(), f"suite_path must exist: {rel}"


class TestManifestSchemaValidation:
    """Unit tests for the BenchmarkManifest Pydantic schema itself."""

    def test_valid_minimal_manifest(self) -> None:
        _validate(
            {
                "protocol_id": "test_v1",
                "protocol_version": 1,
                "suite_path": "prompts/evaluation_questions.json",
                "metrics": {
                    "keyword": {"summary": "fast", "cli": "python eval.py"},
                },
            }
        )

    def test_missing_protocol_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="protocol_id"):
            _validate(
                {
                    "protocol_version": 1,
                    "suite_path": "prompts/evaluation_questions.json",
                    "metrics": {"keyword": {"summary": "fast", "cli": "python eval.py"}},
                }
            )

    def test_zero_protocol_version_raises(self) -> None:
        with pytest.raises(ValidationError, match="protocol_version"):
            _validate(
                {
                    "protocol_id": "test_v1",
                    "protocol_version": 0,
                    "suite_path": "prompts/evaluation_questions.json",
                    "metrics": {"keyword": {"summary": "fast", "cli": "python eval.py"}},
                }
            )

    def test_negative_protocol_version_raises(self) -> None:
        with pytest.raises(ValidationError, match="protocol_version"):
            _validate(
                {
                    "protocol_id": "test_v1",
                    "protocol_version": -1,
                    "suite_path": "prompts/evaluation_questions.json",
                    "metrics": {"keyword": {"summary": "fast", "cli": "python eval.py"}},
                }
            )

    def test_no_metrics_raises(self) -> None:
        with pytest.raises(ValidationError, match="metrics"):
            _validate(
                {
                    "protocol_id": "test_v1",
                    "protocol_version": 1,
                    "suite_path": "prompts/evaluation_questions.json",
                    "metrics": {},
                }
            )

    def test_missing_keyword_raises(self) -> None:
        with pytest.raises(ValidationError, match="keyword"):
            _validate(
                {
                    "protocol_id": "test_v1",
                    "protocol_version": 1,
                    "suite_path": "prompts/evaluation_questions.json",
                    "metrics": {"judge": {"summary": "slow", "cli": "python eval.py --judge"}},
                }
            )


class TestRunBenchmarkManifestResolution:
    """run_benchmark.py must default to the LATEST manifest and fail fast on
    suite-hash mismatches / silent older-protocol runs."""

    @pytest.fixture
    def bench_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "benchmarks"
        d.mkdir()
        for v in (1, 2, 3):
            (d / f"manifest.v{v}.yaml").write_text(
                f"protocol_id: proto_v{v}\nprotocol_version: {v}\nsuite_path: suites/q.v{v}.json\n",
                encoding="utf-8",
            )
        return d

    def test_resolves_highest_version(self, bench_dir: Path) -> None:
        from scripts.run_benchmark import resolve_latest_manifest

        assert resolve_latest_manifest(bench_dir).name == "manifest.v3.yaml"

    def test_no_manifests_exits(self, tmp_path: Path) -> None:
        from scripts.run_benchmark import resolve_latest_manifest

        empty = tmp_path / "none"
        empty.mkdir()
        with pytest.raises(SystemExit):
            resolve_latest_manifest(empty)

    def test_manifest_version_parsing(self, tmp_path: Path) -> None:
        from scripts.run_benchmark import manifest_version

        assert manifest_version(Path("manifest.v12.yaml")) == 12
        assert manifest_version(Path("unversioned.yaml")) is None

    def _write_suite(self, bench_dir: Path, content: bytes) -> Path:
        suites = bench_dir.parent / "suites"
        suites.mkdir(exist_ok=True)
        suite = suites / "q.json"
        suite.write_bytes(content)
        return suite

    def test_sha256_match_passes(self, bench_dir: Path) -> None:
        import hashlib

        from scripts.run_benchmark import verify_suite_sha256

        content = b'[{"question": "q?"}]'
        self._write_suite(bench_dir, content)
        manifest = {
            "suite_path": "suites/q.json",
            "suite_sha256": hashlib.sha256(content).hexdigest(),
        }
        assert verify_suite_sha256(bench_dir / "manifest.v3.yaml", manifest) == "suites/q.json"

    def test_sha256_mismatch_aborts(self, bench_dir: Path) -> None:
        from scripts.run_benchmark import verify_suite_sha256

        self._write_suite(bench_dir, b"tampered bytes")
        manifest = {
            "suite_path": "suites/q.json",
            "suite_sha256": "0" * 64,
        }
        with pytest.raises(SystemExit, match="hash mismatch"):
            verify_suite_sha256(bench_dir / "manifest.v3.yaml", manifest)

    def test_missing_snapshot_aborts(self, bench_dir: Path) -> None:
        from scripts.run_benchmark import verify_suite_sha256

        manifest = {"suite_path": "suites/absent.json", "suite_sha256": "0" * 64}
        with pytest.raises(SystemExit, match="missing"):
            verify_suite_sha256(bench_dir / "manifest.v3.yaml", manifest)

    def test_unpinned_suite_warns_but_proceeds(self, bench_dir: Path, capsys) -> None:
        from scripts.run_benchmark import verify_suite_sha256

        self._write_suite(bench_dir, b"[]")
        manifest = {"suite_path": "suites/q.json"}
        rel = verify_suite_sha256(bench_dir / "manifest.v1.yaml", manifest)
        assert rel == "suites/q.json"
        assert "no suite_sha256" in capsys.readouterr().out

    def test_older_manifest_requires_explicit_flag(self) -> None:
        from scripts.run_benchmark import ensure_not_older

        with pytest.raises(SystemExit, match="allow-older-manifest"):
            ensure_not_older(chosen_version=2, latest_version=3, allow_older=False)
        ensure_not_older(chosen_version=2, latest_version=3, allow_older=True)
        ensure_not_older(chosen_version=3, latest_version=3, allow_older=False)
        ensure_not_older(chosen_version=None, latest_version=3, allow_older=False)


class TestShippedSuiteIntegrity:
    """Live check: the latest SHIPPED manifest's pinned sha256 must match the
    actual snapshot bytes in this repo — catches snapshot tampering/decay."""

    def test_latest_shipped_manifest_hash_matches_snapshot(self) -> None:

        from scripts.run_benchmark import (
            BENCHMARKS_DIR,
            load_manifest,
            resolve_latest_manifest,
            verify_suite_sha256,
        )

        if not list(BENCHMARKS_DIR.glob("manifest.v*.yaml")):
            pytest.skip("No shipped manifests found")
        latest = resolve_latest_manifest(BENCHMARKS_DIR)
        data = load_manifest(latest)
        # Returns without raising only on a verified hash (or unpinned manifest).
        rel = verify_suite_sha256(latest, data)
        assert (PROJECT_ROOT / rel).is_file()

    def test_evaluate_threshold_matches_latest_manifest_constant(self) -> None:
        """FUZZY_PASS_THRESHOLD and manifest metric_constants must never drift apart."""
        try:
            import yaml  # type: ignore
        except ImportError:
            pytest.skip("PyYAML not installed")
        from training.evaluate import FUZZY_PASS_THRESHOLD

        manifests = sorted(BENCHMARKS_DIR.glob("manifest.v*.yaml"))
        if not manifests:
            pytest.skip("No manifests found")
        latest = manifests[-1]
        data = yaml.safe_load(latest.read_text(encoding="utf-8")) or {}
        pinned = (data.get("metric_constants") or {}).get("fuzzy_pass_threshold")
        if pinned is None:
            pytest.skip(f"{latest.name} pins no fuzzy_pass_threshold")
        assert float(pinned) == FUZZY_PASS_THRESHOLD
