"""
Gradio Web UI for Bible AI Assistant: text and voice chat.
Expects RAG server at localhost:8081, Kokoro TTS at localhost:8880.
"""
import tempfile
from pathlib import Path

import gradio as gr
import httpx

RAG_URL = "http://localhost:8081/v1/chat/completions"
TTS_URL = "http://localhost:8880/v1/audio/speech"
MODEL_NAME = "bible-assistant"
WHISPER_MODEL = "large-v3-turbo"

# Lazy-load Whisper (heavy)
_whisper_model = None


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
            print("[Voice] Faster-Whisper using GPU (cuda)")
        except Exception as e:
            print(f"[Voice] Faster-Whisper GPU failed ({e}), falling back to CPU")
            _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio_path: str | None) -> str:
    """Transcribe audio with Faster-Whisper."""
    if not audio_path:
        return ""
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(audio_path, language="en")
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        return f"[STT error: {e}]"


def chat_with_rag(message: str) -> str:
    """Send message to RAG server, return assistant reply."""
    if not message.strip():
        return ""
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                RAG_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": message}],
                    "stream": False,
                },
            )
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip()
    except Exception as e:
        return f"[RAG error: {e}]"


# Optional: local Kokoro (pip install kokoro) uses conda PyTorch = GPU if supported
_kokoro_pipeline = None


def _synthesize_local(text: str) -> str | None:
    """Use hexgrad/kokoro package locally (GPU if PyTorch supports it). Falls back to HTTP."""
    global _kokoro_pipeline
    try:
        from kokoro import KPipeline
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    try:
        if _kokoro_pipeline is None:
            _kokoro_pipeline = KPipeline(lang_code="a")  # American English
            print("[Voice] Kokoro TTS using local pipeline (GPU if available)")
        chunks = []
        for _gs, _ps, audio in _kokoro_pipeline(text, voice="af_bella"):
            chunks.append(audio)
        if not chunks:
            return None
        audio_arr = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        out_path = Path(tempfile.gettempdir()) / f"tts_{hash(text) & 0x7FFFFFFF}.wav"
        sf.write(str(out_path), audio_arr, 24000)
        return str(out_path)
    except Exception:
        return None


def synthesize(text: str) -> str | None:
    """Convert text to speech via Kokoro TTS. Returns path to WAV file."""
    if not text.strip():
        return None
    # Try local Kokoro first (GPU when pip install kokoro in conda env)
    out = _synthesize_local(text)
    if out:
        return out
    # Fall back to Docker/HTTP
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                TTS_URL,
                json={"model": "kokoro", "input": text, "voice": "af_bella", "response_format": "wav"},
            )
            if r.status_code != 200:
                return None
            out_path = Path(tempfile.gettempdir()) / f"tts_{hash(text) & 0x7FFFFFFF}.wav"
            out_path.write_bytes(r.content)
            return str(out_path)
    except Exception:
        return None


def text_chat(message: str, history: list) -> tuple[list, str]:
    """Handle text tab: send to RAG, append to history."""
    history = history or []
    if not message.strip():
        return history, ""
    reply = chat_with_rag(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def _audio_to_path(audio) -> str | None:
    """Extract a usable file path from Gradio's Audio output (tuple, dict, or path string)."""
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
        wav_path = Path(tempfile.gettempdir()) / f"voice_{hash(str(arr.shape)) & 0x7FFFFFFF}.wav"
        sf.write(str(wav_path), data, int(sr))
        return str(wav_path)
    return None


def voice_chat(audio, history: list) -> tuple[list, str | None]:
    """Handle voice tab: transcribe -> chat -> synthesize -> return history + audio."""
    history = history or []
    try:
        wav_path = _audio_to_path(audio)
    except Exception as e:
        history = history + [
            {"role": "user", "content": "[Voice] (error)"},
            {"role": "assistant", "content": f"[Voice error] {type(e).__name__}: {e}"},
        ]
        return history, None
    if not wav_path:
        history = history + [
            {"role": "user", "content": "[Voice] (no audio)"},
            {"role": "assistant", "content": "No audio captured. Click the microphone, record your question, then click Stop. Make sure your browser has mic permission."},
        ]
        return history, None
    transcript = transcribe(wav_path)
    if not transcript:
        history = history + [
            {"role": "user", "content": "[Voice] (no speech detected)"},
            {"role": "assistant", "content": "No speech was detected. Please try again and speak clearly."},
        ]
        return history, None
    reply = chat_with_rag(transcript)
    audio_path = synthesize(reply)
    history = history + [
        {"role": "user", "content": f"[Voice] {transcript}"},
        {"role": "assistant", "content": reply},
    ]
    return history, audio_path


with gr.Blocks(title="Bible AI Assistant") as demo:
    gr.Markdown("# Bible AI Assistant")
    gr.Markdown(
        "Ask about Scripture via text or voice. Requires RAG server (port 8081) and Kokoro TTS (port 8880). "
        "**Disclaimer:** This is a study aid, not pastoral care or counseling. For spiritual direction, consult a pastor or trusted believer. "
        "Built to serve God by pointing people to his Word—free, local, and accessible. "
        "Accessibility: Tab to navigate, Enter to send."
    )
    chatbot = gr.Chatbot(height=400, label="Chat")
    msg = gr.Textbox(placeholder="Ask a question...", label="Message")

    with gr.Tabs():
        with gr.Tab("Text"):
            msg.submit(text_chat, [msg, chatbot], [chatbot, msg])
            gr.Examples(
                ["What does John 3:16 say?", "Who was the Apostle Paul?", "What does the Bible say about forgiveness?"],
                inputs=msg,
            )
        with gr.Tab("Voice"):
            voice_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record your question (click mic, speak, then stop) or upload an audio file",
            )
            voice_resp = gr.Audio(label="Response (audio)", autoplay=True)
            voice_in.stop_recording(voice_chat, [voice_in, chatbot], [chatbot, voice_resp])
            voice_in.change(voice_chat, [voice_in, chatbot], [chatbot, voice_resp])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
