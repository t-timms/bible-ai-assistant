# ORPO Training: Two-Environment Setup

This document describes why ORPO (Odds Ratio Preference Optimization) for Qwen3.5-4B requires a separate conda environment, and how the SFT → ORPO pipeline produces a fully trained model.

---

## Overview

| Stage | Environment | Key Dependency | Output |
|-------|-------------|-----------------|--------|
| **SFT** (Supervised Fine-Tuning) | `bible-orpo` | transformers ≥ 5.1 | `models/qwen3.5-4b-bible-John-v6/` |
| **ORPO** (Preference Alignment) | `bible-orpo` | transformers ≥ 5.1 | `models/qwen3.5-4b-bible-John-v7/` (optional) |
| **Merge** | `bible-orpo` | transformers ≥ 5.1 (Qwen3.5) | `models/...-merged/` |
| **Deploy** (GGUF, Ollama, eval) | none | — | Merged GGUF, Ollama, etc. |

**Critical:** SFT must run in `bible-orpo` (transformers 5.x) so the adapter uses native Qwen3.5 layout. SFT in `bible-ai-assistant` (transformers 4.57.2) produces a Qwen3-style adapter that corrupts the model on merge. See [QWEN35_MERGE_AND_DEPLOYMENT.md](QWEN35_MERGE_AND_DEPLOYMENT.md).

---

## Why Two Environments?

### Technical Constraint

- **Qwen3.5-4B** uses a newer architecture (Gated DeltaNet, multimodal config). Native support for `model_type: qwen3_5` was added in **transformers ≥ 5**.

- **Unsloth** (used for SFT and ORPO) pins **transformers ≤ 4.57.2** in the SFT environment to preserve compatibility with its internal patches (e.g., RoPE handling).

- With transformers 4.57.2, Qwen3.5 is loaded as a fallback Qwen3 model. The weight layout is incompatible, so loading fails (e.g., `'weight' is not an nn.Module`).

### Resolution

| Environment | transformers | Purpose |
|-------------|--------------|---------|
| `bible-orpo` | ≥ 5.1 | SFT and ORPO; native Qwen3.5 for both |
| `bible-ai-assistant` | ≤ 4.57.2 | Deprecated for Qwen3.5; produces incompatible adapter |

SFT and ORPO both run in `bible-orpo` for consistent native Qwen3.5 architecture. Merge then works correctly.

---

## Pipeline Flow

```
┌─────────────────────────────────┐
│  Env: bible-orpo                │
│  transformers ≥ 5.1             │
│                                 │
│  1. Load Qwen3.5-4B base        │
│  2. Train LoRA (SFT)            │
│  3. Save adapter to disk        │
└─────────────────────────────────┘
                 │
                 │  models/qwen3.5-4b-bible-John-v6/
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
                 │  models/qwen3.5-4b-bible-John-v7/
                 │  ├── adapter_model.safetensors  (SFT + ORPO)
                 │  └── ...
                 ▼
         Merge → GGUF → Deploy
```

---

## Artifact Handoff

The handoff between environments is a standard LoRA adapter plus config:

1. **SFT saves** to `models/qwen3.5-4b-bible-John-v6/`:
   - `adapter_model.safetensors` — LoRA weights
   - `adapter_config.json` — rank, target modules, etc.
   - Tokenizer files

2. **ORPO script**:
   - Loads the base model (Qwen3.5-4B)
   - Applies LoRA with the same config as SFT
   - Calls `set_peft_model_state_dict(model, sft_state)` to load SFT weights
   - Continues training with ORPO
   - Saves to `models/qwen3.5-4b-bible-John-v7/`

ORPO never modifies the base model directly; it reads the SFT adapter from disk and writes an updated adapter.

---

## Setup Instructions

### Environment: bible-orpo (SFT + ORPO)

Use `bible-orpo` for both SFT and ORPO. Native Qwen3.5 (transformers 5.x) is required for a working merge.

**SFT:**

```bash
conda activate bible-orpo
python training/train_unsloth.py --run-name qwen3.5-4b-bible-John-v6
```

**ORPO** (after SFT):

```bash
python training/train_orpo.py --sft-path models/qwen3.5-4b-bible-John-v6
```

Output: `models/qwen3.5-4b-bible-John-v7/` (SFT + ORPO).

### One-time setup for bible-orpo

If `bible-orpo` does not exist:

```bash
conda create -n bible-orpo python=3.11 -y
conda activate bible-orpo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5.1" unsloth trl datasets wandb bitsandbytes accelerate
```

### Merge and Deploy

**Merge** (requires `bible-orpo` for Qwen3.5-4B):

```bash
conda activate bible-orpo
python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v6
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
