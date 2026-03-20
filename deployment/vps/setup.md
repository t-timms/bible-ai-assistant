# VPS Setup (Future Use)

This folder is reserved for production deployment. The Bible AI project focuses on the model and RAG stack; deployment targets include:

- **Jetson Orin Nano** — Edge inference with llama.cpp
- **Cloud VM** — RAG server + Ollama on a VPS

Agent frameworks (e.g. OpenClaw) and Telegram integration can be built as a **separate project** that calls this RAG server at `http://your-server:8081/v1`.
