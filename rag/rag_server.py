"""
FastAPI RAG middleware with hybrid retrieval (Dense + BM25 + RRF + Cross-Encoder Reranking).

Pipeline:
  1. Dense search (nomic-embed-text-v1.5 via ChromaDB) -> top 20
  2. BM25 sparse search (rank_bm25) -> top 20
  3. Reciprocal Rank Fusion to merge results
  4. Cross-encoder reranking (bge-reranker-v2-m3) -> top K
  5. Parent-child passage expansion for thematic questions

Run: uvicorn rag.rag_server:app --host 127.0.0.1 --port 8081

Security notes:
  - Set API_KEY env var to require X-API-Key header authentication.
  - RATE_LIMIT defaults to 60/minute per IP (slowapi).
  - All requests receive a unique X-Request-ID correlation header.
  - Set LOG_JSON=true for structured JSON logs in production.
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from rag.helpers import (
    _COUNSELING_SYSTEM_GUARD,
    _EVAL_SUFFIXES,
    EMPTY_MODEL_REPLY,
    _content_to_str,
    _extract_verse_ref_from_lookup,
    _is_counseling_request,
    _is_meta_question,
    _is_verse_lookup,
    _strip_openclaw_metadata,
    _strip_repetition_and_meta,
    _strip_thinking,
    _strip_thinking_from_stream,
    _topical_anchor_refs,
)
from rag.retrieval import _retrieve, release_resources
from rag.settings import settings

# ---------------------------------------------------------------------------
# Request-ID context — injected by middleware, read by logging filter
# ---------------------------------------------------------------------------

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class _RequestIDFilter(logging.Filter):
    """Injects the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get("-")
        return True


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Set up structured JSON logging (production) or plain text (development)."""
    log_level = getattr(logging, settings.log_level, logging.INFO)
    req_filter = _RequestIDFilter()

    if settings.log_json:
        try:
            from pythonjsonlogger import jsonlogger

            handler = logging.StreamHandler()
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(request_id)s %(message)s"
            )
            handler.setFormatter(formatter)
            handler.addFilter(req_filter)
            logging.basicConfig(level=log_level, handlers=[handler], force=True)
        except ImportError:
            logging.basicConfig(level=log_level)
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s [%(request_id)s] %(message)s"
        )
        handler.setFormatter(formatter)
        handler.addFilter(req_filter)
        logging.basicConfig(level=log_level, handlers=[handler], force=True)


_configure_logging()
logger = logging.getLogger(__name__)

# Maximum request body size (bytes) to prevent DoS via oversized payloads
MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MB

# ---------------------------------------------------------------------------
# Startup security warnings
# ---------------------------------------------------------------------------

if not settings.api_key and settings.rag_host not in ("127.0.0.1", "localhost", "::1"):
    logger.warning(
        "SECURITY: API_KEY is not set but RAG server is binding to %s — "
        "any client on this network can call the API without authentication. "
        "Set the API_KEY environment variable to require authentication.",
        settings.rag_host,
    )

# ---------------------------------------------------------------------------
# Rate limiting (slowapi)
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# API key auth (optional — disabled when settings.api_key == "")
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Dependency: enforce API key when settings.api_key is set."""
    if not settings.api_key:
        return  # auth disabled in dev/localhost mode
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request/response and inject it into
    the logging context so all log lines for a request share the same ID."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = _request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
        finally:
            _request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class _ChatMessage(BaseModel):
    role: str = "user"
    content: str | list[Any] = ""

    @field_validator("role")
    @classmethod
    def _normalize_role(cls, v: str) -> str:
        return v if v in ("system", "user", "assistant") else "user"


class _ChatCompletionRequest(BaseModel):
    model: str = "bible-assistant"
    messages: list[_ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    think: bool | None = None


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """Release heavy model objects on shutdown."""
    yield
    release_resources()


app = FastAPI(
    title="Bible AI RAG Server",
    version="1.0.0",
    description="Hybrid RAG middleware (Dense + BM25 + RRF + Reranking) for Bible AI Assistant.",
    lifespan=_lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(_RequestIDMiddleware)

# ---------------------------------------------------------------------------
# Prometheus metrics (optional — graceful no-op if package not installed)
# ---------------------------------------------------------------------------

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus metrics available at /metrics")
except ImportError:
    logger.debug("prometheus-fastapi-instrumentator not installed — /metrics disabled")


@app.exception_handler(Exception)
async def _handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    summary="Health check",
    tags=["ops"],
    response_model=dict[str, str],
)
def health() -> dict[str, str]:
    """Returns service status. Used by load balancers and readiness probes."""
    return {"status": "ok", "service": "rag", "version": "1.0.0"}


@app.post(
    "/v1/chat/completions",
    summary="Chat with Bible AI (RAG-augmented)",
    tags=["chat"],
    dependencies=[Depends(_require_api_key)],
)
@limiter.limit(settings.rate_limit)
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with hybrid RAG retrieval.

    Augments the last user message with retrieved Bible passages before
    forwarding to Ollama. Applies constitutional guardrails for counseling
    requests and strips model chain-of-thought from responses.

    Errors:
      - 413: Request body exceeds 1 MB
      - 415: Content-Type is not application/json
      - 422: Malformed JSON or schema validation failure
      - 401: Invalid or missing API key (when API_KEY is set)
      - 429: Rate limit exceeded
      - 502: Ollama unreachable or returned an error
      - 504: Ollama request timed out
    """
    # Enforce Content-Type before reading the body
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/json",
        )

    # Guard against oversized payloads — read body first; size is the authoritative check
    raw = await request.body()
    if len(raw) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")

    try:
        parsed = _ChatCompletionRequest.model_validate_json(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    messages = [m.model_dump() for m in parsed.messages]
    model = parsed.model
    stream = parsed.stream
    last_q_for_policy: str | None = None

    last_user_content = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_content = m.get("content")
            if isinstance(last_user_content, list):
                for part in last_user_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        last_user_content = part.get("text", "")
                        break
                else:
                    last_user_content = ""
            break

    if last_user_content and last_user_content.strip():
        q = _strip_openclaw_metadata(last_user_content.strip())
        for suffix in _EVAL_SUFFIXES:
            if q.lower().endswith(suffix.lower()):
                q = q[: -len(suffix)].strip()
                break
        last_q_for_policy = q
        if _is_meta_question(q):
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = {**messages[i], "content": q}
                    break
        else:
            pin_refs: list[str] = []
            vr = _extract_verse_ref_from_lookup(q)
            if vr:
                pin_refs.append(vr)
            topical_pins = _topical_anchor_refs(q)
            pin_refs.extend(topical_pins)
            context = _retrieve(q, top_k=settings.rag_top_k, pin_refs=pin_refs or None)
            if context and context.strip():
                notes: list[str] = []
                if vr:
                    notes.append(
                        "The context includes the verse you were asked about; quote it exactly."
                    )
                if topical_pins and not _is_verse_lookup(q):
                    notes.append(
                        "Use only passages that fit the topic; do not substitute unrelated stories."
                    )
                note_block = ("\n\nNote: " + " ".join(notes)) if notes else ""
                augmented = "Context:\n" + context + "\n\nQ: " + q + note_block
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i] = {**messages[i], "content": augmented}
                        break

    # Normalise all message roles and content types
    normalized = []
    for m in messages:
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = _content_to_str(m.get("content"))
        normalized.append({"role": role, "content": content})
    messages = normalized

    if last_q_for_policy and _is_counseling_request(last_q_for_policy):
        messages.insert(0, {"role": "system", "content": _COUNSELING_SYSTEM_GUARD})

    ollama_payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": parsed.max_tokens or 2048,
        # Qwen3+ in Ollama: suppress chain-of-thought by default
        "think": False,
    }
    # Allow clients to re-enable thinking (e.g. debugging) via "think": true
    if parsed.think is not None:
        ollama_payload["think"] = parsed.think

    url = settings.ollama_url.rstrip("/") + "/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                think_enabled = bool(ollama_payload.get("think", False))
                req = client.build_request("POST", url, json=ollama_payload)
                response = await client.send(req, stream=True)
                if response.status_code != 200:
                    err_body = (await response.aread()).decode("utf-8", errors="replace")
                    await response.aclose()
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama {response.status_code}: {err_body}",
                    )

                if think_enabled:
                    # think=True (debug only): buffer full response to strip thinking tokens
                    raw_chunks = []
                    async for chunk in response.aiter_bytes():
                        raw_chunks.append(chunk)
                    await response.aclose()
                    full_sse = b"".join(raw_chunks).decode("utf-8", errors="replace")
                    cleaned_bytes = _strip_thinking_from_stream(full_sse)

                    async def _stream_buffered():
                        yield cleaned_bytes

                    return StreamingResponse(
                        _stream_buffered(),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache"},
                    )
                else:
                    # think=False (default): proxy chunks directly — no thinking tokens expected
                    async def _stream_direct():
                        try:
                            async for chunk in response.aiter_bytes():
                                yield chunk
                        finally:
                            await response.aclose()

                    return StreamingResponse(
                        _stream_direct(),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache"},
                    )

            r = await client.post(url, json=ollama_payload)
            if r.status_code != 200:
                raise HTTPException(
                    status_code=502, detail=f"Ollama error {r.status_code}: {r.text}"
                )
            data = r.json()
            try:
                raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                content = _strip_thinking(raw_text) if raw_text else ""
                content = _strip_repetition_and_meta(content) if content else ""
                out = content.rstrip()
                if out:
                    if out[-1] in ",;:":
                        out = out[:-1] + "."
                    elif out[-1] not in ".?!\"'":
                        for end in (". ", "? ", "! "):
                            idx = out.rfind(end)
                            if idx != -1:
                                out = out[: idx + 1].rstrip()
                                break
                    data["choices"][0]["message"]["content"] = out
                else:
                    data["choices"][0]["message"]["content"] = EMPTY_MODEL_REPLY
            except (IndexError, KeyError, TypeError) as e:
                logger.debug("Post-processing skipped: %s", e)
            return data
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach Ollama at {settings.ollama_url}. Is it running?",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="Ollama request timed out.") from e


# ---------------------------------------------------------------------------
# CLI entry point  (registered as `rag-server` via pyproject.toml)
# ---------------------------------------------------------------------------


def serve() -> None:
    """Start the RAG server with uvicorn.  Respects settings / env vars."""
    import uvicorn

    uvicorn.run(
        "rag.rag_server:app",
        host=settings.rag_host,
        port=settings.rag_port,
        log_level=settings.log_level.lower(),
    )
