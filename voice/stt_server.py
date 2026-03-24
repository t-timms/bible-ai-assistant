"""
STT (Speech-to-Text) module — Faster-Whisper.

The Whisper model is loaded in-process by ui/app.py (see _get_whisper() there),
which handles GPU/CPU fallback automatically.  A standalone HTTP STT server is
not required for the default Gradio UI workflow.

To run a standalone HTTP STT endpoint (e.g. for integrating with other clients):

    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    def transcribe(audio_path: str) -> str:
        segments, _ = model.transcribe(audio_path, language="en")
        return " ".join(seg.text for seg in segments).strip()

Expose `transcribe` via FastAPI or any other HTTP framework as needed.
"""
