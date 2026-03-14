# Jetson Orin Nano Super Deployment

1. **Transfer model and RAG DB**
   ```bash
   scp models\bible-qwen3-q4km.gguf USER@JETSON_IP:~/models/
   scp -r rag\chroma_db USER@JETSON_IP:~/rag/
   ```

2. **Install llama.cpp:** Run `install.sh`, then copy `llama_server.service` to `/etc/systemd/system/`. Replace `USER` with your Jetson username. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable llama_server
   sudo systemctl start llama_server
   ```

3. **Tailscale:** On Jetson: `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`. Use the Jetson Tailscale IP (100.x.x.x) from the VPS for the LLM endpoint.

4. **RAG (optional on Jetson):** If running RAG on Jetson, install Python env, build index or copy chroma_db, then enable `rag_server.service`. Otherwise the VPS can call the Jetson for LLM only and run RAG elsewhere.

5. **OOM:** If OOM, reduce `--ctx-size` to 8192 or `--n-gpu-layers` to 24.

Checkpoint: **v0.7.0** when Jetson deployment is live.
