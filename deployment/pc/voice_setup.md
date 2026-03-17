# Voice Mode Setup (Windows PC)

Voice tab in Gradio: microphone → Faster-Whisper (STT) → RAG → Kokoro TTS → playback.

---

## Prerequisites

- **Docker Desktop** — Start it from the Start menu before running any `docker` commands
- **Conda env** with `faster-whisper`, `soundfile`, `gradio` (already in requirements.txt)
- **RAG server** running on port 8081
- **Ollama** with `bible-assistant` model

---

## Step 1: Start Kokoro TTS (Docker)

**With NVIDIA GPU:**
```powershell
docker run -d --gpus all -p 8880:8880 --name kokoro-tts ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

**Without GPU (CPU only):**
```powershell
docker run -d -p 8880:8880 --name kokoro-tts ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

Check it's running:
```powershell
Invoke-RestMethod -Uri http://localhost:8880/health -ErrorAction SilentlyContinue
# Or: curl http://localhost:8880/health
```

---

## Step 2: Start the Stack (4 terminals)

| Terminal | Command |
|----------|---------|
| **1. Ollama** | Already running (or `ollama serve`) |
| **2. RAG server** | `conda activate bible-ai-assistant` → `cd bible-ai-assistant` → `uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081` |
| **3. Kokoro TTS** | `docker run -d --gpus all -p 8880:8880 --name kokoro-tts ghcr.io/remsky/kokoro-fastapi-gpu` (or CPU command above) |
| **4. Gradio** | `conda activate bible-ai-assistant` → `cd bible-ai-assistant` → `python ui/app.py` |

---

## Step 3: Test Voice

1. Open http://localhost:7860
2. Go to **Voice** tab
3. Record a question (e.g. "What does John 3:16 say?")
4. First use will download Faster-Whisper (~2 GB); wait 30–60 seconds
5. Response should transcribe → RAG → TTS → play

---

---

## Optional: Run on GPU (Faster)

**Faster-Whisper:** Already tries GPU first. If your conda env has PyTorch with CUDA 12.8+ (for RTX 5070 Ti / Blackwell), it will use GPU automatically. Check the Gradio terminal on first voice use—it will print either "Faster-Whisper using GPU" or "falling back to CPU".

**Kokoro TTS:** The Docker GPU image does not support RTX 5070 Ti (Blackwell sm_120) yet. For local GPU TTS, install the Python package:

```powershell
conda activate bible-ai-assistant
pip install kokoro
```

Then stop the Kokoro Docker container (`docker stop kokoro-tts`). The app will use local Kokoro with your conda PyTorch, which may work on GPU if your env has CUDA 12.8+.

---

## GPU Acceleration (RTX 5070 Ti / Blackwell)

The Docker Kokoro image uses an older PyTorch that does not support sm_120 (Blackwell). To use your GPU:

**1. Faster-Whisper** — Already tries CUDA first. If your conda env has PyTorch nightly with CUDA 12.8+, it will use GPU. Check the Gradio terminal on first voice use: you should see `[Voice] Faster-Whisper using GPU (cuda)` or the fallback message.

**2. Kokoro TTS (local, no Docker)** — Install the hexgrad package so it uses your conda PyTorch:
```powershell
conda activate bible-ai-assistant
pip install kokoro>=0.9.4
```
Then stop the Docker Kokoro container (`docker stop kokoro-tts`). The app will use local Kokoro, which runs on GPU if PyTorch supports Blackwell. You should see `[Voice] Kokoro TTS using local pipeline (GPU if available)` on first use.

---

## Optional: Run on GPU (Faster)

If your conda env has PyTorch with CUDA 12.8+ (e.g. for RTX 5070 Ti), you can run both STT and TTS on GPU:

1. **Faster-Whisper** — Already tries GPU first. Check the Gradio terminal on first voice use: you'll see either `[Voice] Faster-Whisper using GPU (cuda)` or `falling back to CPU`. If you see the latter, your PyTorch may not support your GPU (e.g. Blackwell sm_120); ensure you have PyTorch nightly with CUDA 12.8+.

2. **Kokoro TTS (local)** — Install the Python package instead of using Docker:
   ```powershell
   conda activate bible-ai-assistant
   pip install kokoro
   ```
   The app will use local Kokoro with your conda PyTorch (GPU if supported). You can stop the Kokoro Docker container. First run downloads the model (~350 MB).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No audio" or voice tab doesn't respond | Kokoro not running. Start Docker container. |
| `[STT error: ...]` | Faster-Whisper failed. Check conda env has `faster-whisper`, `soundfile`. First run downloads model. |
| TTS returns nothing | Kokoro may be slow on CPU. Try GPU image. Check `http://localhost:8880/health`. |
| `--gpus all` fails | Use CPU image: `ghcr.io/remsky/kokoro-fastapi-cpu:latest` |
| Stop Kokoro container | `docker stop kokoro-tts` |

---

## GPU Mode (Faster, RTX 5070 Ti)

**Faster-Whisper:** Already tries GPU first. Check the Gradio terminal on first voice use — you'll see either `[Voice] Faster-Whisper using GPU (cuda)` or a fallback message. Your conda env needs PyTorch with CUDA 12.8+ for Blackwell (sm_120).

**Kokoro TTS (local GPU):** The Docker GPU image fails on RTX 5070 Ti (Blackwell) because its PyTorch doesn't support sm_120. Use the **local Kokoro** package instead — it uses your conda PyTorch:

```powershell
conda activate bible-ai-assistant
pip install kokoro
```

Restart Gradio. The app will try local Kokoro first (GPU if PyTorch supports it), then fall back to Docker. You can stop the Kokoro Docker container when using local Kokoro.

---

## GPU Acceleration (RTX 5070 Ti / Blackwell)

The Docker Kokoro image does **not** support Blackwell (sm_120). For GPU:

### Faster-Whisper (STT)
Uses conda PyTorch. If your env has PyTorch nightly + CUDA 12.8+, it may use GPU. On first voice use, check the Gradio terminal:
- `[Voice] Faster-Whisper using GPU (cuda)` = GPU
- `[Voice] Faster-Whisper GPU failed (...), falling back to CPU` = CPU

### Kokoro TTS (local, no Docker)
```powershell
conda activate bible-ai-assistant
pip install kokoro
```

Then **stop** the Kokoro Docker container. The app will use the local Kokoro package, which uses conda PyTorch (GPU if supported). On first voice TTS, you’ll see:
- `[Voice] Kokoro TTS using local pipeline (GPU if available)`

If local Kokoro fails, the app falls back to Docker (port 8880).
