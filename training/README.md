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

- Conda env `bible-ai` with PyTorch nightly (CUDA 12.8+) and Unsloth. See main guide Section 6.
- **Critical:** Use `bf16=True`, not `fp16`, on RTX 5070 Ti (Blackwell).

## Quick Run

```bash
conda activate bible-ai
# After dataset is ready:
python training/train_unsloth.py
python training/merge_adapters.py
python training/evaluate.py
```

## Checkpoints

- **v0.3.0:** Fine-tuning complete — Qwen3 4B trained on Bible Q&A. Push merged model to Hugging Face for backup.
