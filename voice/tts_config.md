# Kokoro TTS Configuration

- **Image:** `ghcr.io/remsky/kokoro-fastapi-gpu:latest`
- **Port:** 8880
- **Endpoint:** `POST /v1/audio/speech` (OpenAI-compatible)

## Request body

```json
{
  "model": "kokoro",
  "input": "Text to speak (max ~1000 chars for UI).",
  "voice": "af_bella",
  "response_format": "wav",
  "speed": 1.0
}
```

54 voices available (see Kokoro-FastAPI docs). Use `af_bella` or choose per locale.

## Docker

```bash
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

Ensure Docker Desktop is running and `--gpus all` is set on Windows.
