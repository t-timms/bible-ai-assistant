"""Centralised configuration for the RAG server (reads from env / .env file)."""

from __future__ import annotations

import re
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
    # 2026-09-04: was 5. scripts/retrieval_metrics.py (after fixing two real bugs
    # in it — a _get_rag() tuple-unpack mismatch and a column-order mislabel)
    # measured fused_rerank recall@5=0.20 / recall@10=0.25 for `topical` and
    # recall@5=0.22 / recall@10=0.33 for `character` — real, comfortably-sized
    # recall gains between depth 5 and 10 for exactly the categories weakest in
    # generation. Also fixes a train/serve mismatch: training/build_v3_thematic.py
    # already retrieved at top_k 7-9. 8 stays well under context_max_chars (3500).
    rag_top_k: int = 8
    hybrid_candidates: int = 20
    # Wall-clock budget for the dense ChromaDB query (H-8). A soft timeout — see
    # comment in rag/retrieval.py on why a truly hung call can't be hard-killed
    # from a thread in CPython — but bounds how long a slow query dominates the
    # response, and unblocks the pipeline to proceed on BM25 results alone.
    chroma_query_timeout_seconds: float = 10.0

    # ------------------------------------------------------------------
    # Context budget / query limits
    # ------------------------------------------------------------------
    # Max chars of the rendered context block injected into a user turn.
    # Enforced after pinning + rerank selection; lowest-ranked extras are
    # truncated first so pinned verses always survive.
    context_max_chars: int = 3500
    # Hard cap on incoming query length, applied before classification,
    # embedding, and BM25 scoring.
    max_query_chars: int = 2000

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
    # Application environment (development / production)
    # ------------------------------------------------------------------
    app_env: str = "production"

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    title: str = "Bible AI RAG Server"

    # ------------------------------------------------------------------
    # Request limits
    # ------------------------------------------------------------------
    max_request_body_bytes: int = 1_048_576  # 1 MB
    # Ceiling applied to client-supplied max_tokens before forwarding to Ollama.
    max_tokens_ceiling: int = 4096

    # ------------------------------------------------------------------
    # CORS (empty list = disabled; use ["*"] to allow all origins in dev)
    # ------------------------------------------------------------------
    cors_origins: list[str] = []

    # ------------------------------------------------------------------
    # ChromaDB path (override if index lives outside project root)
    # ------------------------------------------------------------------
    chroma_db_path: str = ""

    # ------------------------------------------------------------------
    # Model identifiers (override to swap embedding / reranker models)
    # ------------------------------------------------------------------
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Pinned commit SHAs (H-5 — trust_remote_code=True on an unpinned model id is a
    # supply-chain risk: a compromised upstream repo could push malicious code to
    # "main" and it would execute here on next load). Verified against the HF Hub
    # API directly (not just an LLM summary of it) on 2026-08-24 — re-verify before
    # bumping. Empty string = unpinned (only if you deliberately want latest).
    embed_model_revision: str = "e9b6763023c676ca8431644204f50c2b100d9aab"
    reranker_model_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

    # ------------------------------------------------------------------
    # Citation verification (rag/verification.py) — checks that cited verse
    # references actually exist in the indexed Bible text, not just that the
    # book name is real. "log" only records issues (structured log line);
    # "annotate" additionally appends a visible warning marker next to any
    # reference that doesn't resolve. Default "log" until validated against
    # live model traffic — see docs/MODEL_COMPARISON.md.
    # ------------------------------------------------------------------
    citation_verification_enabled: bool = True
    citation_verification_mode: str = "log"

    @field_validator("citation_verification_mode")
    @classmethod
    def _valid_citation_mode(cls, v: str) -> str:
        valid = {"log", "annotate"}
        low = v.lower()
        if low not in valid:
            raise ValueError(f"CITATION_VERIFICATION_MODE must be one of {valid}")
        return low

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

    @field_validator(
        "rag_top_k",
        "hybrid_candidates",
        "max_request_body_bytes",
        "context_max_chars",
        "max_query_chars",
        "max_tokens_ceiling",
    )
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("chroma_query_timeout_seconds")
    @classmethod
    def _positive_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("app_env")
    @classmethod
    def _valid_app_env(cls, v: str) -> str:
        up = v.lower()
        if up not in ("development", "production"):
            raise ValueError("APP_ENV must be 'development' or 'production'")
        return up

    @field_validator("rate_limit")
    @classmethod
    def _valid_rate_limit(cls, v: str) -> str:
        if not re.match(r"^\d+/(second|minute|hour|day)$", v.strip()):
            raise ValueError(
                f"RATE_LIMIT must be a valid slowapi format like '60/minute', got: {v!r}"
            )
        return v.strip()

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
