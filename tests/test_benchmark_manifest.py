"""Benchmark manifest schema (protocol versioning)."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "benchmarks" / "manifest.v1.yaml"


@pytest.fixture
def manifest() -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        pytest.skip("PyYAML not installed")
    if not MANIFEST.exists():
        pytest.skip(f"Missing {MANIFEST}")
    with open(MANIFEST, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_manifest_has_protocol_id(manifest: dict) -> None:
    assert manifest.get("protocol_id"), "manifest must set protocol_id"


def test_manifest_has_suite_path(manifest: dict) -> None:
    rel = manifest.get("suite_path")
    assert rel, "manifest must set suite_path"
    assert (PROJECT_ROOT / rel).is_file(), f"suite_path must exist: {rel}"


def test_manifest_version_field(manifest: dict) -> None:
    assert "protocol_version" in manifest


def test_manifest_version_is_positive_int(manifest: dict) -> None:
    """Protocol version must be a positive integer to prevent accidental rollback."""
    version = manifest.get("protocol_version")
    assert isinstance(version, int) and version >= 1, (
        f"protocol_version must be a positive integer, got: {version}"
    )
