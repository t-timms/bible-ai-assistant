# Prompts

- **`system_prompt.txt`** — Full system prompt for inference (Ollama Modelfile, RAG server). Used at runtime.
- **`system_prompt_training.txt`** — Short system prompt for dataset building. The dataset builder embeds this in every training example. A long prompt causes **instruction leaking** in 4B models (they echo rules instead of following them). Keep it to 1–2 sentences.
