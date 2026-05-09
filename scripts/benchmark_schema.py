"""Pydantic models for validating benchmark manifest YAML files."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MetricEntry(BaseModel):
    summary: str
    cli: str


class JudgeConfig(BaseModel):
    default_model: str
    default_url_env: str = "OLLAMA_URL"
    dimensions: list[str] = Field(min_length=1)

    @field_validator("dimensions")
    @classmethod
    def _valid_dimensions(cls, v: list[str]) -> list[str]:
        valid = {"faithfulness", "citation", "hallucination", "helpfulness", "conciseness"}
        for d in v:
            if d not in valid:
                raise ValueError(f"Unknown judge dimension: {d!r}; valid: {valid}")
        return v


class BenchmarkManifest(BaseModel):
    protocol_id: str = Field(min_length=1)
    protocol_version: int = Field(ge=1)
    suite_path: str = Field(min_length=1)
    description: str = ""
    metrics: dict[str, MetricEntry] = Field(min_length=1)
    judge: JudgeConfig | None = None
    reproducibility_checklist: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_suite_path(self) -> BenchmarkManifest:
        """suite_path must reference an existing file relative to the manifest."""
        if not self.suite_path:
            raise ValueError("suite_path must not be empty")
        return self

    @field_validator("suite_path")
    @classmethod
    def _suite_path_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("suite_path must not be empty")
        return v.strip()

    @field_validator("metrics")
    @classmethod
    def _has_keyword_metric(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "keyword" not in v:
            raise ValueError("manifest must define at least a 'keyword' metric")
        return v
