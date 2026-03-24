# Bible AI Assistant

RAG-based question answering system over the Bible corpus using local LLMs.

# Environments
- `conda activate bible-ai-assistant` — main development
- `conda activate bible-orpo` — ORPO fine-tuning experiments

# Commands
- `conda run -n bible-ai-assistant jupyter lab` — launch notebooks
- `python -m pytest tests/` — run tests
- `ollama run llama3.2` — local inference testing
- `ollama run llama3.1:8b` — higher quality inference

# Architecture
- /bible-ai-assistant — main RAG pipeline code
- /world-english-bible — source corpus (READ-ONLY)
- /llama.cpp — local inference engine

# GPU Context
- RTX 5070 Ti, 16GB VRAM, Blackwell sm_120
- PyTorch 2.10+cu128 (conda env: mlenv or bible-ai-assistant)
- For large models use 4-bit quantization via BitsAndBytes

# IMPORTANT Rules
- Bible corpus in /world-english-bible is READ-ONLY — never modify
- NEVER commit API keys or .env files
- All LLM inference tested locally before any deployment
- Use parameterized queries for any database operations
- Log retrieval quality metrics when testing RAG changes
