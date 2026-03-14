# Voice (STT + TTS)

Speech-to-text and text-to-speech for the PC development stack. **Runs on PC only** (GPU required); Jetson deployment is text-only (e.g. Telegram).

## Components

| Component | Model | VRAM | Notes |
|-----------|--------|------|--------|
| STT | Faster-Whisper (large-v3-turbo) | ~2 GB | `voice/stt_server.py` or in-process in Gradio |
| TTS | Kokoro 82M (Docker) | ~0.4 GB | OpenAI-compatible API on port 8880 |

## STT: Faster-Whisper

```bash
pip install faster-whisper
# Load once: WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
# Transcribe with beam_size=5, language="en", vad_filter=True
```

## TTS: Kokoro

Run via Docker (Kokoro-FastAPI):

```bash
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

Request: `POST http://localhost:8880/v1/audio/speech` with `model`, `input`, `voice`, `response_format`, `speed`. See `tts_config.md` for details.

## VRAM Budget (PC)

LLM (~3 GB) + STT (~2 GB) + TTS (~0.4 GB) + embeddings (~0.5 GB) ≈ 6 GB; 10 GB headroom on 16 GB GPU.

Checkpoint: **v0.6.5** when voice mode is integrated (e.g. in Gradio).
