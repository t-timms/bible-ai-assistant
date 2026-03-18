# ORPO Training: Two-Environment Setup

This document describes why ORPO (Odds Ratio Preference Optimization) for Qwen3.5-4B requires a separate conda environment, and how the SFT → ORPO pipeline produces a fully trained model.

---

## Overview

| Stage | Environment | Key Dependency | Output |
|-------|-------------|-----------------|--------|
| **SFT** (Supervised Fine-Tuning) | `bible-ai-assistant` | transformers ≤ 4.57.2 | `models/qwen3.5-4b-bible-John-v4/` |
| **ORPO** (Preference Alignment) | `bible-orpo` | transformers ≥ 5.1 | `models/qwen3.5-4b-bible-John-v5/` |
| **Merge** | `bible-orpo` | transformers ≥ 5.1 (Qwen3.5) | `models/...-merged/` |
| **Deploy** (GGUF, Ollama, eval) | none | — | Merged GGUF, Ollama, etc. |

Both environments operate on the same model artifact: the LoRA adapter saved to disk. The pipeline is file-based; no special conda integration is required.

---

## Why Two Environments?

### Technical Constraint

- **Qwen3.5-4B** uses a newer architecture (Gated DeltaNet, multimodal config). Native support for `model_type: qwen3_5` was added in **transformers ≥ 5**.

- **Unsloth** (used for SFT and ORPO) pins **transformers ≤ 4.57.2** in the SFT environment to preserve compatibility with its internal patches (e.g., RoPE handling).

- With transformers 4.57.2, Qwen3.5 is loaded as a fallback Qwen3 model. The weight layout is incompatible, so loading fails (e.g., `'weight' is not an nn.Module`).

### Resolution

| Environment | transformers | Purpose |
|-------------|--------------|---------|
| `bible-ai-assistant` | ≤ 4.57.2 | SFT training; stable for Unsloth |
| `bible-orpo` | ≥ 5.1 | ORPO training; native Qwen3.5 support |

SFT stays on the older transformers. ORPO runs in a separate env with newer transformers. Each env uses the versions required for its stage.

---

## Pipeline Flow

```
┌─────────────────────────────────┐
│  Env: bible-ai-assistant        │
│  transformers ≤ 4.57.2         │
│                                 │
│  1. Load Qwen3.5-4B base       │
│  2. Train LoRA (SFT)            │
│  3. Save adapter to disk       │
└─────────────────────────────────┘
                 │
                 │  models/qwen3.5-4b-bible-John-v4/
                 │  ├── adapter_model.safetensors
                 │  ├── adapter_config.json
                 │  └── tokenizer files
                 ▼
┌─────────────────────────────────┐
│  Env: bible-orpo                │
│  transformers ≥ 5.1             │
│                                 │
│  1. Load Qwen3.5-4B base        │
│  2. Apply same LoRA config      │
│  3. Load SFT adapter from disk  │
│  4. Train ORPO on preferences   │
│  5. Save updated adapter        │
└─────────────────────────────────┘
                 │
                 │  models/qwen3.5-4b-bible-John-v5/
                 │  ├── adapter_model.safetensors  (SFT + ORPO)
                 │  └── ...
                 ▼
         Merge → GGUF → Deploy
```

---

## Artifact Handoff

The handoff between environments is a standard LoRA adapter plus config:

1. **SFT saves** to `models/qwen3.5-4b-bible-John-v4/`:
   - `adapter_model.safetensors` — LoRA weights
   - `adapter_config.json` — rank, target modules, etc.
   - Tokenizer files

2. **ORPO script**:
   - Loads the base model (Qwen3.5-4B)
   - Applies LoRA with the same config as SFT
   - Calls `set_peft_model_state_dict(model, sft_state)` to load SFT weights
   - Continues training with ORPO
   - Saves to `models/qwen3.5-4b-bible-John-v5/`

ORPO never modifies the base model directly; it reads the SFT adapter from disk and writes an updated adapter.

---

## Setup Instructions

### Environment 1: SFT (Existing)

Use the existing `bible-ai-assistant` environment with `transformers<=4.57.2`:

```bash
conda activate bible-ai-assistant
python training/train_unsloth.py --run-name qwen3.5-4b-bible-John-v4
```

### Environment 2: ORPO (New)

Create and use a separate environment with newer transformers:

```bash
# Create environment
conda create -n bible-orpo python=3.11 -y
conda activate bible-orpo

# Install PyTorch (CUDA 12.8 for RTX 5070 Ti)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install training deps with newer transformers
pip install "transformers>=5.1" unsloth trl datasets wandb bitsandbytes accelerate

# Run ORPO (from project root)
cd /path/to/bible-ai-assistant
python training/train_orpo.py --sft-path models/qwen3.5-4b-bible-John-v4
```

Output will be written to `models/qwen3.5-4b-bible-John-v5/` (or the run name specified).

### Merge and Deploy

**Merge** (requires `bible-orpo` for Qwen3.5-4B):

```bash
conda activate bible-orpo
python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v5
```

**Deploy** (no conda env required): GGUF conversion, quantize, `ollama create`, `ollama run`, evaluation. See [ENVIRONMENT_REQUIREMENTS.md](ENVIRONMENT_REQUIREMENTS.md).

---

## Script Behavior

`train_orpo.py` checks for native Qwen3.5 support:

- **transformers ≥ 5** (qwen3_5 in `CONFIG_MAPPING`): loads directly from Hugging Face; no config patching.
- **transformers < 5**: exits with instructions to create and use the `bible-orpo` environment.

---

## References

- [ENVIRONMENT_REQUIREMENTS.md](ENVIRONMENT_REQUIREMENTS.md) — When conda envs are vs are not required
- [ORPO paper](https://arxiv.org/abs/2403.07691)
- [Transformers Qwen3.5 docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- `training/train_orpo.py` — ORPO training script
- `data/processed/preferences.json` — preference pairs (run `build_preference_data.py` to generate)
