"""Centralised configuration for the RAG server (reads from env / .env file)."""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All RAG-server configuration in one validated model.

    Reads from environment variables (case-insensitive) and an optional
    ``.env`` file in the project root.  Set ``LOG_JSON=true`` in production.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Ollama / LLM
    # ------------------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "bible-assistant"

    # ------------------------------------------------------------------
    # RAG server network
    # ------------------------------------------------------------------
    rag_host: str = "127.0.0.1"
    rag_port: int = 8081

    # ------------------------------------------------------------------
    # Retrieval tuning
    # ------------------------------------------------------------------
    rag_top_k: int = 5
    hybrid_candidates: int = 20

    # ------------------------------------------------------------------
    # Security  (empty string = auth disabled; fine for localhost dev)
    # ------------------------------------------------------------------
    api_key: str = ""

    # ------------------------------------------------------------------
    # Rate limiting (slowapi format, e.g. "60/minute", "5/second")
    # ------------------------------------------------------------------
    rate_limit: str = "60/minute"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False  # set True in production for structured JSON logs

    # ------------------------------------------------------------------
    # Model identifiers (override to swap embedding / reranker models)
    # ------------------------------------------------------------------
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("ollama_url")
    @classmethod
    def _check_ollama_url(cls, v: str) -> str:
        u = urlparse(v)
        if u.scheme not in ("http", "https"):
            raise ValueError(f"OLLAMA_URL must use http or https, got {u.scheme!r}")
        if not u.netloc:
            raise ValueError("OLLAMA_URL has no host")
        return v

    @field_validator("rag_top_k", "hybrid_candidates")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        up = v.upper()
        if up not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return up


# Module-level singleton — import this everywhere instead of calling os.getenv().
settings = Settings()
