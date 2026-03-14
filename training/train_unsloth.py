#!/usr/bin/env python3
"""
QLoRA fine-tuning of Qwen3 4B with Unsloth.
Requires: conda env bible-ai, PyTorch nightly (CUDA 12.8+), data/processed/train.json.
Use bf16=True (never fp16) on RTX 5070 Ti (Blackwell).
"""
from pathlib import Path

# Uncomment and adapt after installing unsloth, trl, datasets
# from unsloth import FastModel
# from trl import SFTTrainer, SFTConfig
# from datasets import load_dataset
# import wandb

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_file = project_root / "data" / "processed" / "train.json"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}. Run dataset_builder.py first."
        )
    # TODO: Load config from config.yaml, init W&B, load model with FastModel.from_pretrained,
    #       apply get_peft_model (LoRA), SFTTrainer, train(), save_pretrained to models/bible-qwen3-4b-lora
    raise NotImplementedError(
        "Training script skeleton. Implement with Unsloth + SFTTrainer per guide Section 9. "
        "Use bf16=True, never fp16."
    )


if __name__ == "__main__":
    main()
