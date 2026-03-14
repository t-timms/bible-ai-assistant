# Ollama Setup (Windows PC)

1. **Install:** https://ollama.com/download/windows  
2. **Serve:** In a terminal, run `ollama serve` (or rely on Ollama app to start it).  
3. **Load model:** After creating the GGUF and Modelfile (see guide Section 11):
   ```bash
   ollama create bible-assistant -f Modelfile
   ollama run bible-assistant "What does John 3:16 say?"
   ```
4. **OpenClaw:** Point OpenClaw to the RAG server at `http://localhost:8081/v1`, not directly to Ollama. The RAG server forwards to Ollama at `http://localhost:11434`.
