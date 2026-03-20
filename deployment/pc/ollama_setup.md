# Ollama Setup (Windows PC)

1. **Install:** https://ollama.com/download/windows  
2. **Serve:** In a terminal, run `ollama serve` (or rely on Ollama app to start it).  
3. **Load model:** After creating the GGUF and Modelfile (see guide Section 11):
   ```bash
   ollama create bible-assistant -f Modelfile
   ollama run bible-assistant "What does John 3:16 say?"
   ```
4. **Clients:** Point any OpenAI-compatible client (Gradio, curl, API) to the RAG server at `http://localhost:8081/v1` for verse-grounded answers. The RAG server forwards to Ollama at `http://localhost:11434`.

---

### Using the model in other apps (LM Studio, llama.cpp, etc.)

For **consistent behavior** (verse lookups, no refusals, no rambling), the model must use the **same system prompt and parameters** as in Ollama. Otherwise it may refuse follow-up verse requests or "go out of control."

- **System prompt:** Use the full contents of `prompts/system_prompt.txt` as the system/instruction prompt in your app.
- **Parameters:** Prefer temperature ~0.2, max new tokens ~256 (or 512 if you want slightly longer answers).
- **Ollama (recommended)** uses these automatically via the Modelfile; other apps require you to set the system prompt and params yourself.
