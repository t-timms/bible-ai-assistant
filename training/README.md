# Training

bf16 LoRA fine-tuning of Qwen3.5-4B on the Bible dataset.

## Scripts

| Script | Purpose |
|--------|---------|
| `train_unsloth.py` | Main fine-tuning script (Unsloth + bf16 LoRA). |
| `train_orpo.py` | ORPO preference alignment (runs after SFT; requires separate env). See [ORPO Two-Env Setup](../docs/ORPO_TWO_ENV_SETUP.md). |
| `merge_adapters.py` | Merge LoRA adapters into full model for export. |
| `dataset_builder.py` | Build Q&A dataset from raw sources → `data/processed/train.json`. |
| `evaluate.py` | Run evaluation test set; `--ollama-model`, `--protocol-id` for A/B. See `docs/BENCHMARK_PROTOCOL.md`. |
| `config.yaml` | Hyperparameters (batch size, LR, epochs, etc.). |

## Environment

- Conda env `bible-ai-assistant` with PyTorch nightly (CUDA 12.8+) and Unsloth. See main guide Section 6.
- **Critical:** Use `bf16=True`, not `fp16`, on RTX 5070 Ti (Blackwell).

## Quick Run

```bash
conda activate bible-ai-assistant
# After dataset is ready (data/processed/train.json). Default run name: qwen3.5-4b-bible-John-v4
python training/train_unsloth.py
# Optional: use local base model or a different run name
python training/train_unsloth.py --model-path models/base_model --run-name qwen3.5-4b-bible-John-v4
python training/merge_adapters.py
python training/evaluate.py
```

Hyperparameters are read from `config.yaml` when available (PyYAML required). LoRA adapters are saved to `models/<run_name>` (e.g. `models/qwen3.5-4b-bible-John-v4`). For a full step-by-step walkthrough, see **docs/WALKTHROUGH.md** (Step 9).

## ORPO (Preference Alignment)

After SFT, run ORPO in a **separate environment** with `transformers>=5.1` (Qwen3.5 requires native support). Full details: **[docs/ORPO_TWO_ENV_SETUP.md](../docs/ORPO_TWO_ENV_SETUP.md)**.

```bash
# Create ORPO env (one-time)
conda create -n bible-orpo python=3.11 -y
conda activate bible-orpo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5.1" unsloth trl datasets wandb bitsandbytes accelerate

# Run ORPO
python training/train_orpo.py --sft-path models/qwen3.5-4b-bible-John-v4
```

Output: `models/qwen3.5-4b-bible-John-v5/` (SFT + ORPO adapter). Merge with `merge_adapters.py` as usual. **Env requirements:** See [ENVIRONMENT_REQUIREMENTS.md](../docs/ENVIRONMENT_REQUIREMENTS.md) — merge needs `bible-orpo`; GGUF/Ollama/eval require no conda env.

## Checkpoints

- **v0.3.0:** Fine-tuning complete — Qwen3.5-4B trained on Bible Q&A. Push merged model to Hugging Face for backup.
