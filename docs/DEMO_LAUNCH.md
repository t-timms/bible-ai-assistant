# Demo launch (Ollama + RAG + Gradio)

Use this when you want the **full local demo**: fine-tuned model in Ollama, hybrid RAG on **8081**, and the **Gradio** landing UI on **7860**.

## Prerequisites

- Chroma index built: `python rag/build_index.py` (once)
- Ollama model created (e.g. `bible-assistant-orpo`) per `deployment/pc/README.md`
- Conda env with **RAG** deps: `pip install -r requirements-rag.txt` (or full `requirements.txt`)
- **Gradio UI** deps: `pip install -r requirements-ui.txt` (needed if you did not install the full `requirements.txt`)

## Three terminals (Windows PowerShell)

| # | What | Command |
|---|------|---------|
| 1 | **Ollama** | Run the Ollama app or `ollama serve`. Pull/run is automatic when the model is used. |
| 2 | **RAG** | `cd` to repo root, activate env, then: `uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081` |
| 3 | **Gradio** | Same env, repo root: `python ui/app.py` → open **http://127.0.0.1:7860** |

Optional **4th** — Kokoro TTS on **8880** for HTTP TTS (see `deployment/pc/voice_setup.md`). Text chat works without it; the UI also tries in-process `kokoro` if installed.

## Environment variables (optional)

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAG_SERVER_URL` | `http://127.0.0.1:8081` | RAG base URL (UI health check + chat) |
| `OLLAMA_MODEL` | `bible-assistant-orpo` | Model name sent to RAG (must exist in Ollama) |
| `GRADIO_HOST` | `127.0.0.1` | Bind address |
| `GRADIO_PORT` | `7860` | Gradio port |
| `TTS_URL` | `http://localhost:8880` | Kokoro HTTP service |

## Quick helper script

From repo root:

```powershell
.\scripts\start_demo.ps1
```

Prints the same steps and optional URLs (does not start processes for you).

## UI features

- **Landing hero** + **live status** for RAG (and optional TTS)
- **Model name** field to A/B Ollama tags without restarting the UI
- **Text** and **Voice** tabs; disclaimer footer

See `ui/README.md` for Gradio-only notes.
