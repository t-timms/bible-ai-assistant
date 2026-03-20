# Environment Requirements by Phase

This document specifies when conda environments are required versus when they are not. Use it to avoid activation errors and ensure correct tooling for each pipeline stage.

---

## Summary

| Phase | Conda Env Required? | Notes |
|-------|---------------------|-------|
| **SFT training** | Yes — `bible-orpo` | transformers ≥5.1, native Qwen3.5; see [QWEN35_MERGE_AND_DEPLOYMENT.md](QWEN35_MERGE_AND_DEPLOYMENT.md) |
| **ORPO training** | Yes — `bible-orpo` | transformers ≥5.1, native Qwen3.5 |
| **Merge adapters** | Yes — `bible-orpo` | Qwen3.5-4B needs transformers 5.x |
| **GGUF conversion** | No | Any Python with torch, or base Python |
| **Quantization (llama-quantize)** | No | C++ executable; standalone |
| **Ollama create/run** | No | Ollama CLI; no Python |
| **Evaluation** | No (any Python) | `evaluate.py` uses httpx only |

---

## When Conda Environments Are Required

### SFT (Supervised Fine-Tuning)

```bash
conda activate bible-orpo
python training/train_unsloth.py --run-name qwen3.5-4b-bible-John-v6
```

- **Environment:** `bible-orpo`
- **Why:** transformers ≥5.1 for native Qwen3.5; SFT in `bible-ai-assistant` produces incompatible adapter. See [QWEN35_MERGE_AND_DEPLOYMENT.md](QWEN35_MERGE_AND_DEPLOYMENT.md)

### ORPO (Preference Alignment)

```bash
conda activate bible-orpo
python training/train_orpo.py --sft-path models/qwen3.5-4b-bible-John-v6
```

- **Environment:** `bible-orpo`
- **Why:** transformers ≥5.1 for native Qwen3.5 support; see [ORPO_TWO_ENV_SETUP.md](ORPO_TWO_ENV_SETUP.md)

### Merge Adapters

```bash
conda activate bible-orpo
python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v6
```

- **Environment:** `bible-orpo` (Qwen3.5-4B) or `bible-ai-assistant` (legacy Qwen3-4B; deprecated for Qwen3.5)
- **Why:** Merge uses Unsloth; Qwen3.5-4B requires transformers 5.x for correct loading

---

## When Conda Environments Are Not Required

### GGUF Conversion

```bash
python ../llama.cpp/convert_hf_to_gguf.py models/qwen3.5-4b-bible-John-v5-merged \
  --outfile models/qwen3.5-4b-bible-John-v5-f16.gguf --outtype f16
```

- **Environment:** None
- **Why:** Script needs `torch`, `transformers`, `gguf`; base Python or any env with these deps suffices. No training-specific env needed.

### Quantization (llama-quantize)

```powershell
.\llama-quantize.exe input.gguf output-q4_k_m.gguf Q4_K_M
```

- **Environment:** None
- **Why:** Built C++ executable; runs without Python or conda

### Ollama Create and Run

```bash
ollama create bible-assistant -f deployment/pc/Modelfile
ollama run bible-assistant
```

- **Environment:** None
- **Why:** Ollama is a standalone application; no Python or conda involvement

### Evaluation

```bash
python training/evaluate.py
```

- **Environment:** None (any Python with `httpx`)
- **Why:** Script calls RAG server and optionally Ollama via HTTP; no ML training stack required

---

## Quick Reference

**Use `bible-orpo` for:** SFT, ORPO, merge (Qwen3.5-4B)

**No env needed for:** GGUF conversion, llama-quantize, Ollama CLI, evaluation
