# Training

QLoRA fine-tuning of Qwen3 4B on the Bible dataset.

## Scripts

| Script | Purpose |
|--------|---------|
| `train_unsloth.py` | Main fine-tuning script (Unsloth + QLoRA). |
| `merge_adapters.py` | Merge LoRA adapters into full model for export. |
| `dataset_builder.py` | Build Q&A dataset from raw sources → `data/processed/train.json`. |
| `evaluate.py` | Run evaluation test set; check for verse fabrication and constitution compliance. |
| `config.yaml` | Hyperparameters (batch size, LR, epochs, etc.). |

## Environment

- Conda env `bible-ai-assistant` with PyTorch nightly (CUDA 12.8+) and Unsloth. See main guide Section 6.
- **Critical:** Use `bf16=True`, not `fp16`, on RTX 5070 Ti (Blackwell).

## Quick Run

```bash
conda activate bible-ai-assistant
# After dataset is ready (data/processed/train.json). Default run name: qwen3-4b-bible-John
python training/train_unsloth.py
# Optional: use local base model or a different run name
python training/train_unsloth.py --model-path models/base_model --run-name qwen3-4b-bible-John
python training/merge_adapters.py
python training/evaluate.py
```

Hyperparameters are read from `config.yaml` when available (PyYAML required). LoRA adapters are saved to `models/<run_name>` (e.g. `models/qwen3-4b-bible-John`). For a full step-by-step walkthrough, see **docs/WALKTHROUGH.md** (Step 9).

## Checkpoints

- **v0.3.0:** Fine-tuning complete — Qwen3 4B trained on Bible Q&A. Push merged model to Hugging Face for backup.
