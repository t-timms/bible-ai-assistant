"""
Gradio Web UI for Bible AI Assistant: landing page, text chat, and voice.
Expects RAG server at localhost:8081; Ollama behind RAG; optional Kokoro TTS at 8880.
"""
from __future__ import annotations

import logging
import os
import socket
import tempfile
from pathlib import Path

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

RAG_BASE = os.getenv("RAG_SERVER_URL", "http://127.0.0.1:8081").rstrip("/")
RAG_URL = RAG_BASE + "/v1/chat/completions"
TTS_URL = os.getenv("TTS_URL", "http://localhost:8880") + "/v1/audio/speech"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "bible-assistant-orpo")
WHISPER_MODEL = "large-v3-turbo"

# trust_env=False: corporate HTTP_PROXY often breaks localhost on Windows
_http_client = httpx.Client(timeout=120.0, trust_env=False)

_whisper_model = None


def _find_free_port(host: str, start: int, attempts: int = 30) -> int:
    """Use `start`, then start+1, … if the port is already taken (e.g. another Gradio)."""
    bind_host = host or "127.0.0.1"
    for i in range(attempts):
        port = start + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((bind_host, port))
        except OSError:
            continue
        return port
    return start


# ---------------------------------------------------------------------------
# Landing / stack status
# ---------------------------------------------------------------------------


def check_stack_status() -> str:
    """Lightweight health ping for demo landing (RAG + optional TTS)."""
    lines = []
    try:
        r = httpx.get(f"{RAG_BASE}/health", timeout=3.0, trust_env=False)
        if r.status_code == 200:
            lines.append("**RAG** (port 8081): connected — hybrid retrieval ready.")
        else:
            lines.append(f"**RAG** (port 8081): returned HTTP {r.status_code}.")
    except Exception as e:
        lines.append(
            f"**RAG** (port 8081): not reachable (`{type(e).__name__}`). "
            "Start: `uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081`"
        )
    try:
        tts_base = os.getenv("TTS_URL", "http://localhost:8880").rstrip("/")
        t = httpx.get(f"{tts_base}/health", timeout=2.0, trust_env=False)
        if t.status_code == 200:
            lines.append("**TTS** (port 8880): Kokoro endpoint looks up (voice tab).")
        else:
            lines.append("**TTS** (8880): service responded but check logs (optional for text-only demo).")
    except Exception:
        lines.append(
            "**TTS** (8880): optional — use local `kokoro` pip package or Docker; voice tab still tries in-process TTS."
        )
    lines.append(f"**Ollama model** (via RAG): `{MODEL_NAME}` — ensure `ollama list` includes it.")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# STT / chat / TTS (unchanged behavior)
# ---------------------------------------------------------------------------


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        try:
            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device="cuda",
                compute_type="float16",
            )
            logger.info("Faster-Whisper using GPU (cuda)")
        except (RuntimeError, OSError) as e:
            logger.warning("Faster-Whisper GPU failed (%s), falling back to CPU", e)
            _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio_path: str | None) -> str:
    if not audio_path:
        return ""
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(audio_path, language="en")
        return " ".join(seg.text for seg in segments).strip()
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("STT transcription failed: %s", e, exc_info=True)
        return "[Transcription failed. Please try again.]"


def chat_with_rag(message: str, model_override: str) -> str:
    if not message.strip():
        return ""
    model = (model_override or "").strip() or MODEL_NAME
    try:
        r = _http_client.post(
            RAG_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "stream": False,
            },
        )
        r.raise_for_status()
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (content or "").strip()
    except httpx.HTTPStatusError as e:
        logger.error("RAG server returned %s", e.response.status_code, exc_info=True)
        return "[Could not reach the assistant. Is RAG on 8081 and Ollama running?]"
    except (httpx.RequestError, KeyError, IndexError) as e:
        logger.error("RAG request failed: %s", e, exc_info=True)
        return "[Could not reach the assistant. Please try again.]"


_kokoro_pipeline = None


def _synthesize_local(text: str) -> str | None:
    global _kokoro_pipeline
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError:
        return None
    try:
        if _kokoro_pipeline is None:
            _kokoro_pipeline = KPipeline(lang_code="a")
            print("[Voice] Kokoro TTS using local pipeline (GPU if available)")
        chunks = []
        for _gs, _ps, audio in _kokoro_pipeline(text, voice="af_bella"):
            chunks.append(audio)
        if not chunks:
            return None
        audio_arr = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="tts_")
        sf.write(tmp.name, audio_arr, 24000)
        return tmp.name
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning("Local TTS failed: %s", e)
        return None


def synthesize(text: str) -> str | None:
    if not text.strip():
        return None
    out = _synthesize_local(text)
    if out:
        return out
    try:
        r = _http_client.post(
            TTS_URL,
            json={"model": "kokoro", "input": text, "voice": "af_bella", "response_format": "wav"},
        )
        if r.status_code != 200:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="tts_")
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    except (httpx.RequestError, OSError) as e:
        logger.warning("HTTP TTS failed: %s", e)
        return None


def text_chat(message: str, history: list, model_override: str) -> tuple[list, str]:
    history = history or []
    if not message.strip():
        return history, ""
    reply = chat_with_rag(message, model_override)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def _audio_to_path(audio) -> str | None:
    if audio is None:
        return None
    if isinstance(audio, str) and Path(audio).exists():
        return audio
    if isinstance(audio, dict):
        path = audio.get("path") or audio.get("name")
        if path and Path(path).exists():
            return str(path)
    if isinstance(audio, tuple) and len(audio) >= 2:
        sr, arr = audio[0], audio[1]
        if arr is None or (hasattr(arr, "size") and arr.size == 0):
            return None
        import numpy as np
        import soundfile as sf

        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        data = arr.astype(np.float32) / 32768.0 if arr.dtype == np.int16 else np.asarray(arr, dtype=np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="voice_")
        sf.write(tmp.name, data, int(sr))
        return tmp.name
    return None


def voice_chat(audio, history: list, model_override: str) -> tuple[list, str | None]:
    history = history or []
    try:
        wav_path = _audio_to_path(audio)
    except (ValueError, OSError, TypeError) as e:
        logger.error("Audio processing failed: %s", e, exc_info=True)
        history = history + [
            {"role": "user", "content": "[Voice] (error)"},
            {"role": "assistant", "content": "Audio processing failed. Please try recording again."},
        ]
        return history, None
    if not wav_path:
        history = history + [
            {"role": "user", "content": "[Voice] (no audio)"},
            {
                "role": "assistant",
                "content": "No audio captured. Use the microphone, speak, then stop—or upload a file.",
            },
        ]
        return history, None
    transcript = transcribe(wav_path)
    if not transcript:
        history = history + [
            {"role": "user", "content": "[Voice] (no speech detected)"},
            {"role": "assistant", "content": "No speech detected. Try again and speak clearly."},
        ]
        return history, None
    reply = chat_with_rag(transcript, model_override)
    audio_path = synthesize(reply)
    history = history + [
        {"role": "user", "content": f"[Voice] {transcript}"},
        {"role": "assistant", "content": reply},
    ]
    return history, audio_path


# ---------------------------------------------------------------------------
# Theme + custom CSS (landing page)
# ---------------------------------------------------------------------------

DEMO_CSS = """
:root {
  --ink: #1c1917;
  --muted: #57534e;
  --cream: #fffbeb;
  --gold: #d97706;
  --gold-dim: rgba(217, 119, 6, 0.12);
  --card: #ffffff;
  --border: #e7e5e4;
}
.hero-wrap {
  background: linear-gradient(145deg, #1c1917 0%, #292524 40%, #422006 100%);
  border-radius: 16px;
  padding: 2rem 2.25rem;
  margin-bottom: 1.25rem;
  color: #fafaf9;
  box-shadow: 0 12px 40px rgba(28, 25, 23, 0.35);
}
.hero-wrap h1 {
  font-size: 1.85rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
  margin: 0 0 0.5rem 0 !important;
  color: #fff !important;
  border: none !important;
}
.hero-sub {
  font-size: 1.05rem;
  opacity: 0.92;
  line-height: 1.55;
  max-width: 52rem;
  margin-bottom: 1rem;
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.75rem;
}
.badge {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  background: rgba(255, 251, 235, 0.12);
  border: 1px solid rgba(251, 191, 36, 0.35);
  color: #fde68a;
}
.status-panel {
  background: var(--cream);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.15rem;
  margin-bottom: 1rem;
}
.status-panel p, .status-panel li { color: var(--ink) !important; }
.chat-card {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem;
  background: var(--card);
  box-shadow: 0 2px 12px rgba(28, 25, 23, 0.06);
}
.footer-note {
  font-size: 0.85rem;
  color: var(--muted);
  margin-top: 1.25rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=gr.themes.GoogleFont("Source Sans 3"),
).set(
    body_background_fill="#fafaf9",
    block_title_text_weight="600",
    button_primary_background_fill="linear-gradient(90deg, #b45309, #d97706)",
    button_primary_background_fill_hover="linear-gradient(90deg, #92400e, #b45309)",
)

# Gradio 6+: theme/css are app-level and belong on launch(), not Blocks().
with gr.Blocks(title="Bible AI Assistant") as demo:
    gr.HTML(
        """
        <div class="hero-wrap">
          <h1>Scripture, grounded.</h1>
          <p class="hero-sub">
            Ask verse lookups or topical questions. Answers go through a <strong>local RAG</strong> layer
            (hybrid retrieval + reranking) and your <strong>Ollama</strong> model—nothing leaves your machine
            except optional cloud model pulls you already chose.
          </p>
          <div class="badge-row">
            <span class="badge">Local-first</span>
            <span class="badge">RAG + Ollama</span>
            <span class="badge">Study aid</span>
          </div>
        </div>
        """
    )

    status_md = gr.Markdown(elem_classes=["status-panel"])
    demo.load(check_stack_status, outputs=status_md)
    refresh_btn = gr.Button("Refresh connection status", size="sm", variant="secondary")
    refresh_btn.click(check_stack_status, outputs=status_md)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Chat")
            # Gradio 6: tuple history removed; use OpenAI-style {"role","content"} dicts only.
            # No type=/format= kwarg in recent 6.x. Copy: use buttons= instead of show_copy_button.
            chatbot = gr.Chatbot(
                height=420,
                label="Conversation",
                buttons=["copy"],
            )
            model_box = gr.Textbox(
                label="Ollama model name (via RAG)",
                value=MODEL_NAME,
                placeholder="bible-assistant-orpo",
                info="Must match `ollama list`. Passed as `model` to the RAG API.",
            )
            msg = gr.Textbox(
                placeholder="e.g. What does John 3:16 say?",
                label="Your question",
                lines=2,
            )
            clear = gr.Button("Clear chat", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### How to run the stack")
            gr.Markdown(
                """
1. **Ollama** — `ollama serve` (or app) with your Bible model created.
2. **RAG** — from repo root:  
   `uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081`
3. **This UI** — `python ui/app.py` → open the URL below.

*Voice tab:* Faster-Whisper runs in-process; TTS uses pip `kokoro` or Docker on **8880** (see `deployment/pc/voice_setup.md`).
                """
            )
            gr.Markdown("### Try asking")
            gr.Examples(
                examples=[
                    ["What does John 3:16 say?"],
                    ["What does the Bible say about forgiveness?"],
                    ["Who was Ruth?"],
                ],
                inputs=msg,
                label="Examples (click to fill)",
            )

    with gr.Tabs():
        with gr.Tab("Text"):
            msg.submit(text_chat, [msg, chatbot, model_box], [chatbot, msg])
            clear.click(lambda: ([], ""), outputs=[chatbot, msg])

        with gr.Tab("Voice"):
            gr.Markdown(
                "Record a question or upload audio. Replies play as audio when TTS is available."
            )
            voice_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your voice question",
            )
            voice_resp = gr.Audio(label="Spoken reply", autoplay=True)
            voice_btn = gr.Button("Send voice message", variant="primary")
            voice_btn.click(voice_chat, [voice_in, chatbot, model_box], [chatbot, voice_resp])
            voice_in.stop_recording(voice_chat, [voice_in, chatbot, model_box], [chatbot, voice_resp])

    gr.Markdown(
        """
<div class="footer-note">

**Disclaimer:** This tool is a **Bible study assistant**, not pastoral counseling, therapy, or a substitute for the local church.
For spiritual care or crisis support, speak with a pastor or licensed professional.

</div>
        """,
        elem_classes=["footer-note"],
    )

if __name__ == "__main__":
    _host = os.getenv("GRADIO_HOST", "127.0.0.1")
    # Gradio also documents GRADIO_SERVER_PORT; prefer explicit GRADIO_PORT then that, then 7860
    _port_raw = os.getenv("GRADIO_PORT") or os.getenv("GRADIO_SERVER_PORT") or "7860"
    _preferred = int(_port_raw)
    _port = _find_free_port(_host, _preferred)
    if _port != _preferred:
        print(
            f"[Gradio] Port {_preferred} was busy; using {_port} instead. "
            f"Open http://{_host}:{_port}/",
            flush=True,
        )
    else:
        print(f"[Gradio] http://{_host}:{_port}/", flush=True)
    demo.launch(
        server_name=_host,
        server_port=_port,
        show_error=True,
        theme=theme,
        css=DEMO_CSS,
    )
