# Gradio Web UI

Landing page + **text** and **voice** chat. All LLM traffic goes through the **RAG server** (hybrid retrieval, response cleanup), then **Ollama**.

## Dependencies

If you used **`requirements-rag.txt`** only, Gradio is not installed yet:

```powershell
pip install -r requirements-ui.txt
```

The UI is tested with **Gradio 6.x** (`requirements-ui.txt`). Gradio 6 moved `theme` / `css` to `launch()` and adjusted `Chatbot` props.

## Launch

**Full stack:** see **[docs/DEMO_LAUNCH.md](../docs/DEMO_LAUNCH.md)** or run `.\scripts\start_demo.ps1` from the repo root for copy-paste steps.

Minimal (after RAG + Ollama are already up):

```powershell
conda activate <your-env>
cd bible-ai-assistant
python ui/app.py
# http://127.0.0.1:7860
```

## Environment

| Variable | Default |
|----------|---------|
| `RAG_SERVER_URL` | `http://127.0.0.1:8081` |
| `OLLAMA_MODEL` | `bible-assistant-orpo` |
| `GRADIO_HOST` | `127.0.0.1` |
| `GRADIO_PORT` or `GRADIO_SERVER_PORT` | `7860` — if that port is busy, the app tries **7861, 7862, …** automatically and prints the URL |
| `TTS_URL` | `http://localhost:8880` |

The UI pings RAG `/health` on load and offers **Refresh connection status**. You can change the **Ollama model name** in the sidebar without restarting the app.

## Voice

- **STT:** Faster-Whisper in-process (GPU if available).
- **TTS:** pip `kokoro` (local) or HTTP Kokoro on port **8880** (Docker). See `deployment/pc/voice_setup.md`.

Checkpoint: **v0.9.0** when Gradio demo is showcase-ready.
