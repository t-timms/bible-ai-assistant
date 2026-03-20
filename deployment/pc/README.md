# PC deployment (Ollama)

## Modelfile generator

`generate_modelfile.py` writes `Modelfile` from `prompts/system_prompt.txt` and a **`models/*.gguf`** path.

```powershell
# Default: Q4 ORPO (recommended for daily use — faster import, less VRAM)
python deployment/pc/generate_modelfile.py

# F16 ORPO (often slightly higher quality; slow first-time `ollama create`)
python deployment/pc/generate_modelfile.py --gguf qwen3.5-4b-bible-John-v8-orpo-f16.gguf
```

Use an **absolute path** in `FROM` (the script emits one) so Ollama does not try to pull from the registry.

## A/B test: Q4 vs F16 (same ORPO merge)

Build **two** Ollama models with **different names** so both stay installed:

```powershell
cd "path\to\bible-ai-assistant"

# 1) Q4 — import is quicker
python deployment/pc/generate_modelfile.py --gguf qwen3.5-4b-bible-John-v8-orpo-q4_k_m.gguf
ollama create bible-assistant-orpo -f deployment/pc/Modelfile

# 2) F16 — expect several minutes on “gathering model components”
python deployment/pc/generate_modelfile.py --gguf qwen3.5-4b-bible-John-v8-orpo-f16.gguf
ollama create bible-assistant-orpo-f16 -f deployment/pc/Modelfile

# 3) (Optional) Restore Modelfile to Q4 for docs/repo default
python deployment/pc/generate_modelfile.py
```

Compare:

```powershell
ollama run bible-assistant-orpo "What does John 3:16 say?"
ollama run bible-assistant-orpo-f16 "What does John 3:16 say?"
```

Point the RAG server / `evaluate.py` at the Ollama model name you want to test.

## Qwen “thinking” / chain-of-thought

Qwen3-class models in Ollama can emit ` ` blocks, “Thinking Process:”, or `**Retrieve Verse:**` planning text.

- **RAG server** (`rag/rag_server.py`) sends `"think": false` on Ollama chat requests by default (override with a top-level `"think": true` in your client JSON if you need it). Requires a recent Ollama build; if the field is ignored, upgrade Ollama or rely on post-processing.
- **System prompt** (`prompts/system_prompt.txt`) instructs the model not to show chain-of-thought; re-run `generate_modelfile.py` and `ollama create …` when you change it.
- **Post-process** (`rag/response_cleanup.py`) strips leaked planning if the model still emits it. The RAG server always writes the final cleaned assistant string into JSON for non-streaming chat completions (including when the answer already ends with normal sentence punctuation).

## SFT-only vs SFT+ORPO

- SFT merged GGUFs use names like `qwen3.5-4b-bible-John-v8-*-f16.gguf` / `*-q4_k_m.gguf`.
- ORPO merged GGUFs use `qwen3.5-4b-bible-John-v8-orpo-*` — pass the matching file to `--gguf`.

See **[docs/training_results/POST_TRAINING_CHECKLIST.md](../../docs/training_results/POST_TRAINING_CHECKLIST.md)** for the full pipeline.
