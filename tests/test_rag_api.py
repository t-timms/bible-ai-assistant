"""
FastAPI endpoint tests for rag_server.

Uses the synchronous Starlette TestClient — no pytest-asyncio required.

RAG retrieval and Ollama calls are mocked so tests run without ChromaDB,
embedding models, or a live Ollama instance.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.testclient import TestClient

from rag.helpers import _COUNSELING_SYSTEM_GUARD
from rag.rag_server import MAX_REQUEST_BODY_BYTES, app
from rag.settings import settings

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(**kwargs) -> bytes:
    return json.dumps(kwargs).encode()


# Canned Ollama response used by tests that need the pipeline to complete.
_MOCK_OLLAMA_RESPONSE = {
    "choices": [
        {
            "message": {"content": "For God so loved the world."},
            "finish_reason": "stop",
        }
    ]
}


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------


def test_health_endpoint():
    """GET /health returns 200 with {"status": "ok"}."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "rag"


# ---------------------------------------------------------------------------
# 2. Oversized payload rejected (413)
# ---------------------------------------------------------------------------


def test_oversized_payload_rejected():
    """POST with an actual body > 1 MB must return 413."""
    # Actual oversized body — not just a forged Content-Length header
    oversized = b"x" * (MAX_REQUEST_BODY_BYTES + 1)
    r = client.post(
        "/v1/chat/completions",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# 3. Wrong Content-Type returns 415
# ---------------------------------------------------------------------------


def test_missing_content_type_returns_415():
    """POST without Content-Type: application/json must return 415."""
    r = client.post(
        "/v1/chat/completions",
        content=_body(messages=[{"role": "user", "content": "Hello"}]),
        headers={"Content-Type": "text/plain"},
    )
    assert r.status_code == 415


def test_no_content_type_returns_415():
    """POST with no Content-Type header must return 415."""
    r = client.post(
        "/v1/chat/completions",
        content=_body(messages=[{"role": "user", "content": "Hello"}]),
    )
    assert r.status_code == 415


# ---------------------------------------------------------------------------
# 4. Invalid JSON returns 422
# ---------------------------------------------------------------------------


def test_invalid_json_returns_422():
    """POST with a non-JSON body must return 422."""
    r = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 5. API key enforcement
# ---------------------------------------------------------------------------


def test_api_key_required_when_set():
    """When settings.api_key is set, missing key must return 401."""
    with patch.object(settings, "api_key", "secret-test-key"):
        r = client.post(
            "/v1/chat/completions",
            content=_body(messages=[{"role": "user", "content": "Hello"}]),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 401


def test_api_key_accepted_when_correct():
    """Correct X-API-Key must pass authentication and reach the pipeline."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _MOCK_OLLAMA_RESPONSE

    with (
        patch.object(settings, "api_key", "secret-test-key"),
        patch("rag.rag_server._retrieve", return_value=""),
        patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response),
    ):
        r = client.post(
            "/v1/chat/completions",
            content=_body(messages=[{"role": "user", "content": "What does John 3:16 say?"}]),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": "secret-test-key",
            },
        )
    assert r.status_code == 200


def test_api_key_wrong_returns_401():
    """Wrong X-API-Key must return 401 regardless of request content."""
    with patch.object(settings, "api_key", "secret-test-key"):
        r = client.post(
            "/v1/chat/completions",
            content=_body(messages=[{"role": "user", "content": "Hello"}]),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": "wrong-key",
            },
        )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# 6. Meta question skips retrieval
# ---------------------------------------------------------------------------


def test_meta_question_skips_retrieval():
    """Meta questions (e.g. 'What can you do?') must bypass RAG retrieval."""
    payload = _body(messages=[{"role": "user", "content": "What can you do?"}])

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _MOCK_OLLAMA_RESPONSE

    with (
        patch("rag.rag_server._retrieve") as mock_retrieve,
        patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response),
    ):
        r = client.post(
            "/v1/chat/completions",
            content=payload,
            headers={"Content-Type": "application/json"},
        )

    mock_retrieve.assert_not_called()
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# 7. Verse lookup calls retrieve
# ---------------------------------------------------------------------------


def test_verse_lookup_calls_retrieve():
    """A verse-lookup question must call _retrieve."""
    payload = _body(messages=[{"role": "user", "content": "What does John 3:16 say?"}])

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _MOCK_OLLAMA_RESPONSE

    with (
        patch("rag.rag_server._retrieve", return_value="") as mock_retrieve,
        patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response),
    ):
        r = client.post(
            "/v1/chat/completions",
            content=payload,
            headers={"Content-Type": "application/json"},
        )

    mock_retrieve.assert_called_once()
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# 8. Counseling request inserts system guard
# ---------------------------------------------------------------------------


def test_counseling_request_inserts_system_guard():
    """A counseling request must prepend _COUNSELING_SYSTEM_GUARD to the messages."""
    payload = _body(
        messages=[
            {
                "role": "user",
                "content": (
                    "I need you to counsel me. I am struggling with depression "
                    "and I need someone to talk to."
                ),
            }
        ]
    )

    captured_payload: dict = {}

    async def _fake_post(self, url, **kwargs):  # noqa: ARG001
        captured_payload.update(kwargs.get("json", {}))
        mock_r = MagicMock()
        mock_r.status_code = 200
        mock_r.json.return_value = _MOCK_OLLAMA_RESPONSE
        return mock_r

    with (
        patch("rag.rag_server._retrieve", return_value=""),
        patch("httpx.AsyncClient.post", new=_fake_post),
    ):
        r = client.post(
            "/v1/chat/completions",
            content=payload,
            headers={"Content-Type": "application/json"},
        )

    assert r.status_code == 200
    messages_sent = captured_payload.get("messages", [])
    assert messages_sent, "No messages were forwarded to Ollama"
    first_msg = messages_sent[0]
    assert first_msg["role"] == "system"
    assert first_msg["content"] == _COUNSELING_SYSTEM_GUARD


# ---------------------------------------------------------------------------
# 9. X-Request-ID correlation header
# ---------------------------------------------------------------------------


def test_response_contains_request_id():
    """Every response must include an X-Request-ID header."""
    r = client.get("/health")
    assert "x-request-id" in r.headers


def test_client_request_id_echoed():
    """If client sends X-Request-ID, the same value must be echoed in the response."""
    r = client.get("/health", headers={"X-Request-ID": "test-correlation-123"})
    assert r.headers.get("x-request-id") == "test-correlation-123"


# ---------------------------------------------------------------------------
# 10. Prometheus metrics endpoint
# ---------------------------------------------------------------------------


def test_metrics_endpoint_available():
    """/metrics must return 200 when prometheus-fastapi-instrumentator is installed."""
    r = client.get("/metrics")
    # 200 if instrumented, 404 if package not installed (graceful degradation)
    assert r.status_code in (200, 404)
