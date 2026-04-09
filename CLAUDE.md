# Bible AI Assistant — Claude Code Project Guide

Flagship portfolio project. Locally-hosted Bible Q&A assistant with fine-tuned Qwen3.5-4B,
hybrid RAG, ORPO preference alignment, and constitutional AI guardrails.
Read this before making any changes.

---

## What This Is

A domain-specific LLM application built for deep Biblical study. Combines a fine-tuned model
(ORPO alignment on Qwen3.5-4B) with hybrid RAG (ChromaDB dense + BM25 sparse + cross-encoder
reranking) and a voice pipeline (Faster-Whisper + Kokoro TTS). Gradio UI for local use.

Portfolio value: proves end-to-end fine-tuning, hybrid RAG, MLOps discipline (34 W&B runs),
constitutional AI guardrails, and production-quality Python. The faith angle is distinctive.

---

## Environments

| Env | Purpose | Activate |
|-----|---------|---------|
| `bible-ai-assistant` | Main development — RAG, API, UI | `conda activate bible-ai-assistant` |
| `bible-orpo` | ORPO / GRPO fine-tuning experiments | `conda activate bible-orpo` |

---

## Commands

```bash
# Launch Jupyter (main dev env)
conda run -n bible-ai-assistant jupyter lab

# Run tests
python -m pytest tests/ -v

# With coverage (183 tests, ~55% coverage)
python -m pytest tests/ --cov=. --cov-report=term-missing -v

# Local inference (quick prototyping)
ollama run llama3.2
ollama run llama3.1:8b  # higher quality

# Full stack (Docker)
docker-compose up

# Makefile shortcuts
make test
make lint
make run
```

---

## Architecture

```
User query (text or voice)
    → Faster-Whisper (STT, if voice)
    → Input validation + constitutional guardrails (CONSTITUTION.md)
    → Hybrid retrieval:
        - ChromaDB (dense, sentence-transformers embeddings)
        - BM25 (sparse, keyword matching)
        - RRF fusion (Reciprocal Rank Fusion)
        - Cross-encoder reranking (flashrank or Cohere)
    → Fine-tuned Qwen3.5-4B (ORPO-aligned, 4-bit QLoRA)
    → Constitutional filter (guardrails on output)
    → Gradio UI / API response
    → Kokoro TTS (if voice output requested)
```

---

## Repo Layout

```
bible-ai-assistant/
├── bible-ai-assistant/     # Main RAG pipeline code
├── world-english-bible/    # Source corpus (READ-ONLY — never modify)
├── llama.cpp/              # Local inference engine (vendored — READ-ONLY)
├── rag/                    # RAG pipeline: retrieval, reranking, fusion
├── checkpoints/            # Training checkpoints (1000–5925 steps, gitignored)
├── checkpoints-orpo/       # ORPO fine-tuning checkpoints (gitignored)
├── data/                   # Training data, preference pairs (gitignored)
├── models/                 # Model weights (gitignored)
├── benchmarks/             # lm-eval results, hallucination rate benchmarks
├── prompts/                # System prompts, constitutional AI rules
├── deployment/             # Docker, systemd, deployment configs
├── tests/                  # 183 tests across all modules
├── docker-compose.yml      # Full stack: API + UI + ChromaDB
├── CONSTITUTION.md         # Constitutional AI rules for output filtering
├── AGENTS.md               # Agent architecture notes
├── pyproject.toml          # v0.1.0, setuptools, Python 3.10+
└── CLAUDE.md               # This file
```

---

## Fine-Tuning Pipeline

### Base Model
**Qwen3.5-4B** — small enough for 16GB VRAM, strong instruction following, multilingual

### ORPO Training (conda: bible-orpo)
```bash
conda activate bible-orpo
# Training script
python train_orpo.py --config configs/orpo_config.yaml

# W&B tracking (mandatory)
export WANDB_PROJECT=bible-ai-assistant
wandb login
```

**Training stats**: 34 W&B runs tracked | 5925 steps to best checkpoint | QLoRA 4-bit quantization

### fp8 Training (Blackwell native)
```python
from trl import ORPOConfig
config = ORPOConfig(
    fp8=True,           # Blackwell sm_120 native fp8 kernels
    bf16=False,
    attn_implementation="flash_attention_2",
)
```

### Eval
```bash
conda run -n bible-orpo lm_eval \
  --model hf \
  --model_args pretrained=./checkpoints/best \
  --tasks arc_easy,hellaswag,mmlu,truthfulqa_mc1 \
  --device cuda
```

---

## GPU Context

- **RTX 5070 Ti**, 16GB VRAM, Blackwell sm_120
- **PyTorch**: 2.10+cu128 (conda env: `mlenv` or `bible-ai-assistant`)
- For large models: 4-bit quantization via BitsAndBytes
- Always set `attn_implementation="flash_attention_2"` in `from_pretrained()`

---

## W&B Integration

All training runs logged to W&B. 34 runs tracked.

```python
import wandb, os
wandb.init(
    project=os.environ["WANDB_PROJECT"],  # "bible-ai-assistant"
    config=config,
    tags=["orpo", "qwen3", "v2"]
)
# Log: loss, eval metrics, VRAM peak, hallucination rate
```

Dashboard: wandb.ai → Projects → bible-ai-assistant

---

## Testing

**183 tests, ~55% coverage**. Run before every commit.

```bash
python -m pytest tests/ -v --tb=short

# CI check (same as CI/CD)
python -m pytest tests/ -q --cov=. --cov-fail-under=50
```

Test conventions:
- Mock all external LLM calls — never hit Ollama in unit tests
- RAG tests: use a small in-memory ChromaDB collection with 5–10 test verses
- Log retrieval quality metrics when testing RAG changes (precision@k)
- Always test hallucination guardrails: assert the model refuses off-topic queries

---

## Known Issues

- `llama.cpp` binary requires CUDA 12.8 rebuild if CUDA toolkit version changes
- ChromaDB embedding dimension must match the model used at index time — if you change the embedding model, delete and rebuild the index
- ORPO training requires `bible-orpo` env specifically (different TRL version than `mlenv`)
- Voice pipeline (Kokoro TTS) requires espeak-ng on the system: `apt install espeak-ng`

---

## IMPORTANT Rules

- **`world-english-bible/` is READ-ONLY** — never modify the source corpus
- **`llama.cpp/` is READ-ONLY** — vendored copy, do not modify source files
- **Never commit API keys or `.env` files** — use environment variables
- **All LLM inference tested locally before any deployment** — use Ollama for dev
- **Use parameterized queries for any database operations** — no string-format SQL
- **Log retrieval quality metrics when testing RAG changes** — track precision@k
- **Constitutional guardrails must pass** — test off-topic rejection before every release
