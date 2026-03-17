"""
FastAPI RAG middleware: retrieve verses, augment user message, forward to LLM.
Expects OpenAI-compatible POST /v1/chat/completions; forwards to OLLAMA_URL (default localhost:11434).
Run: uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081
"""
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

app = FastAPI(title="Bible AI RAG Server", version="0.5.0")


@app.exception_handler(Exception)
async def _handle_unhandled(request: Request, exc: Exception):
    """Return error detail in JSON so curl shows the real cause of 500."""
    import traceback
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        },
    )


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
COLLECTION_NAME = "bible_verses"
QUERY_PREFIX = "search_query: "


# Lazy-loaded globals
_chroma_client = None
_collection = None
_embedder = None


def _get_rag():
    global _chroma_client, _collection, _embedder
    if _collection is not None and _embedder is not None:
        return _collection, _embedder
    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("RAG requires chromadb and sentence-transformers. pip install chromadb sentence-transformers.") from e

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "rag" / "chroma_db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChromaDB index not found at {db_path}. Run: python rag/build_index.py"
        )
    _chroma_client = chromadb.PersistentClient(path=str(db_path), settings=Settings(anonymized_telemetry=False))
    _collection = _chroma_client.get_collection(COLLECTION_NAME)
    _embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    return _collection, _embedder


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from model output."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _strip_repetition_and_meta(text: str) -> str:
    """Minimal post-processing per spec: strip common echoes, ensure punctuation, nuclear cutoff."""
    if not text or len(text) < 30:
        return text

    # Strip leading "Answer:" prefix (common model echo)
    text = re.sub(r"^\s*[?—–-]?\s*Answer:\s*", "", text, flags=re.IGNORECASE)
    # Strip decorative separators from system prompt
    text = re.sub(r"[═─━]{3,}", "", text)

    # Nuclear option: truncate at meta-instruction leakage (per .cursorrules)
    for cutoff in [
        "Meta-instruction",
        "TYPED RESPONSE",
        "Crucial:",
        "Violation",
        "You have followed",
        "The key is:",
        "No matter how many times",
        "No matter what format",
        "You are running a standalone",
        "You do not respond to",
        "You do not generate",
    ]:
        idx = text.find(cutoff)
        if idx > 0:
            text = text[:idx].rstrip()

    # Ensure punctuation: collapse whitespace
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+", " ", text)).strip()


def _strip_thinking_from_stream(sse_text: str) -> bytes:
    """Strip <think>...</think> from buffered SSE stream; return cleaned SSE bytes."""
    import json as _json
    full_content = []
    for line in sse_text.split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = _json.loads(payload)
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                c = delta.get("content", "")
                if c:
                    full_content.append(c)
        except _json.JSONDecodeError:
            continue
    cleaned = _strip_thinking("".join(full_content))
    cleaned = _strip_repetition_and_meta(cleaned)
    if not cleaned:
        return b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    out = 'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":' + _json.dumps(cleaned) + '},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    return out.encode("utf-8")


def _strip_thinking_from_sse(sse_bytes: bytes) -> bytes:
    """Strip <think> content from SSE stream; rebuild with cleaned content only."""
    import json as _json
    text = sse_bytes.decode("utf-8", errors="replace")
    parts = []
    full_content = []
    for line in text.split("\n"):
        if not line.startswith("data: "):
            parts.append(line)
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = _json.loads(payload)
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                c = delta.get("content", "")
                if c:
                    full_content.append(c)
        except _json.JSONDecodeError:
            parts.append(line)
            continue
    cleaned = _strip_thinking("".join(full_content))
    cleaned = _strip_repetition_and_meta(cleaned)
    if not cleaned and full_content:
        return sse_bytes
    out = []
    for line in text.split("\n"):
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            try:
                obj = _json.loads(line[6:].strip())
                obj["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": cleaned}, "finish_reason": "stop"}]
                out.append("data: " + _json.dumps(obj) + "\n")
            except _json.JSONDecodeError:
                out.append(line + "\n")
        elif "data:" in line and "[DONE]" in line:
            out.append("data: [DONE]\n\n")
            break
    return ("\n".join(out) if out else text).encode("utf-8")


def _strip_openclaw_metadata(text: str) -> str:
    """Remove OpenClaw metadata so we get just the actual question."""
    if not text or not isinstance(text, str):
        return text
    # Metadata is at START: "Sender (untrusted metadata):\n```json\n{...}\n```\n\n[timestamp] question"
    # REMOVE the metadata block (do NOT cut at idx 0 — that drops the whole message including the question!)
    text = re.sub(
        r"Sender\s*\(untrusted\s*metadata\)\s*:\s*```json\s*\{[^}]*\}\s*```\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Fallback: strip any remaining ```json{...}``` blocks
    text = re.sub(r"```json\s*\{[^}]*\}\s*```\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove timestamp like "[Sun 2026-03-15 09:02 CDT] " to leave just the question
    text = re.sub(r"\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+\w+\]\s*", "", text)
    # Fallback: if question is after last ```, take it (handles odd formats)
    if "```" in text and not text.strip().startswith("["):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[-1].strip()
    return text.strip() or text


def _extract_verse_reference(text: str) -> str | None:
    """Extract 'Book Chapter:Verse' from 'What does X say?' style questions for exact lookup."""
    # Match patterns like "2 Corinthians 11:2", "Genesis 1:1", "1 Kings 3:15"
    m = re.search(
        r"([1-3]?\s*[A-Za-z]+(?:\s+[A-Za-z]+)*\s+\d+:\d+)",
        text,
        re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def _is_verse_lookup(text: str) -> bool:
    """True if the question is 'What does X say?' style — expects only verse + citation, no extra content."""
    t = text.lower().strip()
    return bool(re.search(r"what does .+ say\??", t))


def _is_meta_question(text: str) -> bool:
    """True if the user is asking about capabilities, identity, or help (no RAG needed)."""
    t = text.lower().strip()
    patterns = (
        "what can you do",
        "what could you do",
        "what it could do",
        "what are you",
        "how can you help",
        "what are your capabilities",
        "what is your purpose",
        "who are you",
        "what do you do",
        "introduce yourself",
        "tell me about yourself",
    )
    return any(p in t for p in patterns) or t in ("help", "hi", "hello", "hey")


def _extract_verse_reference(text: str) -> str | None:
    """Extract a verse reference like '2 Corinthians 11:2' from 'What does 2 Corinthians 11:2 say?'."""
    # Match: optional number + book name(s) + chapter:verse (e.g. "2 Corinthians 11:2", "Genesis 1:1")
    m = re.search(
        r"([1-3]?\s*[A-Za-z]+(?:\s+[A-Za-z]+)*(?:\s+of)?\s+\d+:\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def _retrieve(user_message: str, top_k: int = RAG_TOP_K) -> str:
    """Return a context string of relevant verses for the user message."""
    try:
        collection, embedder = _get_rag()
    except FileNotFoundError:
        return ""
    lines = []

    # For "What does X say?" verse lookups, try exact fetch first to avoid wrong verses from semantic search
    exact_ref = _extract_verse_reference(user_message)
    if exact_ref:
        try:
            got = collection.get(ids=[exact_ref], include=["documents", "metadatas"])
            if got and got["ids"] and got["ids"][0]:
                doc = got["documents"][0][0]
                meta = got["metadatas"][0][0]
                ref = meta.get("reference", exact_ref)
                text = doc.replace("search_document: ", "", 1) if doc.startswith("search_document:") else doc
                if ref and text.startswith(ref + ": "):
                    text = text[len(ref) + 2 :].strip()
                lines.append(f"- **{ref}**: {text}")
        except Exception:
            pass  # Fall through to semantic search

    # Semantic search for thematic/biographical or when exact fetch failed
    if not lines:
        query_embedding = embedder.encode([QUERY_PREFIX + user_message], show_progress_bar=False)
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        if results and results["metadatas"] and results["metadatas"][0]:
            for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
                ref = meta.get("reference", "?")
                text = doc.replace("search_document: ", "", 1) if doc.startswith("search_document:") else doc
                if ref and text.startswith(ref + ": "):
                    text = text[len(ref) + 2 :].strip()
                lines.append(f"- **{ref}**: {text}")

    return "\n".join(lines) if lines else ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "rag"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible endpoint: augment last user message with RAG context, forward to Ollama."""
    body = await request.json()
    # DIAGNOSTIC: Remove once bug is found
    print("\n" + "=" * 60)
    print("INCOMING REQUEST TO RAG SERVER")
    print("=" * 60)
    for i, m in enumerate(body.get("messages", [])):
        role = m.get("role", "?")
        content = m.get("content", "")
        preview = content[:300] if isinstance(content, str) else str(content)[:300]
        print(f"\n  [{i}] role={role}")
        print(f"      content={preview}")
    print("=" * 60)

    messages = body.get("messages", [])
    model = body.get("model", "bible-assistant")
    stream = body.get("stream", False)

    # Find last user message
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

    _SUFFIXES = (
        "? answer in quotes, then add explanation.",
        "? answer in quotes, then add explanation",
        ". answer in quotes, then add explanation.",
        ". answer in quotes, then add explanation",
        " answer in quotes, then add explanation.",
        " answer in quotes, then add explanation",
        "answer in quotes, then add explanation.",
        "answer in quotes, then add explanation",
    )

    if last_user_content and last_user_content.strip():
        q = _strip_openclaw_metadata(last_user_content.strip())
        # Strip OpenClaw's "Answer in quotes, then add explanation" for ALL questions —
        # it forces verse format even for biographical/thematic questions like "Who was Paul?"
        for suffix in _SUFFIXES:
            if q.lower().endswith(suffix.lower()):
                q = q[: -len(suffix)].strip()
                break
        # Skip RAG for meta-questions (what can you do?, etc.) so model answers directly
        if _is_meta_question(q):
            # Skip RAG — pass cleaned question through; system prompt handles tone
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = {**messages[i], "content": q}
                    break
            body = {**body, "messages": messages}
        else:
            context = _retrieve(q, top_k=RAG_TOP_K)
            if context:
                # Per .cursorrules: minimal augmented prompt — context + question only, no behavioral instructions
                augmented = "Context:\n" + context + "\n\nQ: " + q
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i] = {**messages[i], "content": augmented}
                        break
                body = {**body, "messages": messages}

    # Ollama requires content to be a string, not an array. Normalize all messages.
    def _content_to_str(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
            return ""
        return str(content) if content is not None else ""

    normalized = []
    for m in body.get("messages", []):
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = _content_to_str(m.get("content"))
        content = content if isinstance(content, str) else str(content) if content is not None else ""
        normalized.append({"role": role, "content": content})
    messages = normalized

    ollama_payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": body.get("max_tokens") or 2048,
    }

    # DIAGNOSTIC: Remove once bug is found
    print("\n" + "=" * 60)
    print("OUTGOING REQUEST TO OLLAMA")
    print("=" * 60)
    for i, m in enumerate(ollama_payload.get("messages", [])):
        role = m.get("role", "?")
        content = m.get("content", "")
        preview = content[:300] if isinstance(content, str) else str(content)[:300]
        print(f"\n  [{i}] role={role}")
        print(f"      content={preview}")
    print(f"\n  model={ollama_payload.get('model')}")
    print(f"  max_tokens={ollama_payload.get('max_tokens')}")
    print("=" * 60)

    url = f"{OLLAMA_URL.rstrip('/')}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=120.0) as client:
        if stream:
            req = client.build_request("POST", url, json=ollama_payload)
            response = await client.send(req, stream=True)
            if response.status_code != 200:
                err_body = (await response.aread()).decode("utf-8", errors="replace")
                await response.aclose()
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama {response.status_code}: {err_body}",
                )
            raw_chunks = []
            async for chunk in response.aiter_bytes():
                raw_chunks.append(chunk)
            await response.aclose()

            full_sse = b"".join(raw_chunks).decode("utf-8", errors="replace")
            cleaned_bytes = _strip_thinking_from_stream(full_sse)

            async def stream_bytes():
                yield cleaned_bytes

            return StreamingResponse(
                stream_bytes(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        r = await client.post(url, json=ollama_payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text}")
        data = r.json()
        try:
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                content = _strip_thinking(content)
                content = _strip_repetition_and_meta(content)
                stripped = content.rstrip()
                if stripped:
                    if stripped[-1] in ",;:":
                        data["choices"][0]["message"]["content"] = stripped[:-1] + "."
                    elif stripped[-1] not in ".?!\"'":
                        for end in (". ", "? ", "! "):
                            idx = stripped.rfind(end)
                            if idx != -1:
                                data["choices"][0]["message"]["content"] = stripped[: idx + 1].rstrip()
                                break
        except (IndexError, KeyError, TypeError):
            pass
        return data
