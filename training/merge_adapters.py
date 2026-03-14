#!/usr/bin/env python3
"""
Merge LoRA adapters into full model (merged_16bit) for export and quantization.
Run after train_unsloth.py. Output: models/<run_name>-merged (e.g. qwen3-4b-bible-John-merged).

Usage:
  python training/merge_adapters.py
  python training/merge_adapters.py --lora-path models/qwen3-4b-bible-John --base-model models/base_model
"""
from pathlib import Path
import argparse
import os

# Defaults aligned with train_unsloth.py (run name = adapter folder)
DEFAULT_LORA_NAME = "qwen3-4b-bible-John"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, default=None,
                        help=f"Path to LoRA adapter (default: models/{DEFAULT_LORA_NAME})")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Path to base model (default: models/base_model if exists, else HF)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output folder for merged model (default: <lora_path>-merged)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    lora_path = Path(args.lora_path) if args.lora_path else project_root / "models" / DEFAULT_LORA_NAME
    lora_path = lora_path.resolve()
    if not lora_path.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found: {lora_path}. Run train_unsloth.py first."
        )

    base_path = args.base_model
    if base_path is None:
        base_candidate = project_root / "models" / "base_model"
        base_path = str(base_candidate) if base_candidate.exists() else MODEL_NAME
    else:
        base_path = str(Path(base_path).resolve())

    out_path = args.output
    if out_path is None:
        out_path = project_root / "models" / f"{lora_path.name}-merged"
    else:
        out_path = Path(out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from unsloth import FastLanguageModel
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("Install unsloth and peft: pip install unsloth peft") from e

    # Blackwell: force SDPA so merge doesn't use xformers
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 12:
            import unsloth.models.llama as _llama
            import unsloth.models.qwen3 as _qwen3
            _llama.HAS_XFORMERS = False
            _qwen3.HAS_XFORMERS = False

    # When base is local, Unsloth loads tokenizer from that path and transformers can fail (dict vs object).
    # Temporarily hide local tokenizer files so tokenizer loads from HF.
    use_local_base = (base_path != MODEL_NAME)
    tokenizer_renames = []
    if use_local_base:
        base_dir = Path(base_path)
        for f in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
            p = base_dir / f
            if p.exists():
                bak = p.with_suffix(p.suffix + ".bak")
                os.rename(p, bak)
                tokenizer_renames.append((bak, p))
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_path,
            max_seq_length=2048,
            load_in_4bit=False,
            dtype="bfloat16",
            tokenizer_name=MODEL_NAME if use_local_base else None,
        )
    finally:
        for bak, orig in tokenizer_renames:
            if bak.exists():
                os.rename(bak, orig)
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, str(lora_path))

    # Merge LoRA into base model in memory (Unsloth's save_pretrained_merged is bound to base, so use PEFT merge)
    model = model.merge_and_unload()

    # Save merged model (bf16) and tokenizer for inference / GGUF conversion
    model.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer.save_pretrained(str(out_path))
    print(f"Merged model saved to {out_path}")


if __name__ == "__main__":
    main()
