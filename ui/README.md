# Gradio Web UI

Web interface with **text** and **voice** tabs. Voice: microphone → Faster-Whisper → RAG+LLM → Kokoro TTS → playback.

## Run

```bash
conda activate bible-ai
python ui/app.py
# Open http://localhost:7860
```

Requires: RAG server on 8081, Kokoro TTS on 8880, Whisper (in-process). See guide Section 18.

Checkpoint: **v0.9.0** when Gradio UI with voice is ready for demo.
