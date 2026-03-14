#!/usr/bin/env python3
"""
Merge LoRA adapters into full model (merged_16bit) for export and quantization.
Run after train_unsloth.py. Output: models/bible-qwen3-4b-merged.
"""
from pathlib import Path

# from unsloth import FastModel

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    lora_path = project_root / "models" / "bible-qwen3-4b-lora"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}. Run train_unsloth.py first.")
    out_path = project_root / "models" / "bible-qwen3-4b-merged"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: FastModel.from_pretrained(lora_path), save_pretrained_merged(..., save_method='merged_16bit')
    raise NotImplementedError(
        "Merge script skeleton. Implement per guide Section 9 (Merging LoRA Adapters)."
    )


if __name__ == "__main__":
    main()
