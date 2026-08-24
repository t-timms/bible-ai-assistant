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
