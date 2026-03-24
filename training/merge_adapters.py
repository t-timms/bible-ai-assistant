#!/usr/bin/env python3
"""
Merge LoRA adapters into full model (merged bf16) for export and quantization.
Run after train_unsloth.py. Output: models/<run_name>-merged (e.g. qwen3.5-4b-bible-John-v8-merged).

Qwen3.5 + Unsloth: saved adapters use keys with ``language_model.layers`` and ``lora_A.weight``.
Native ``transformers`` + PEFT expect ``model.model.layers`` and ``lora_A.default.weight``.
``FastLanguageModel`` + ``PeftModel.from_pretrained`` therefore skips all LoRA weights (silent merge
failure → garbage output). This script remaps keys and uses ``get_peft_model`` + ``load_state_dict``.

Usage:
  python training/merge_adapters.py
  python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v8
  python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v8 --base-model Qwen/Qwen3.5-4B
  # If you trained with --model-path models/base_model, merge with the same path:
  python training/merge_adapters.py --lora-path ... --base-model models/base_model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Defaults aligned with train_unsloth.py (run name = adapter folder)
DEFAULT_LORA_NAME = "qwen3.5-4b-bible-John-v8"
MODEL_NAME = "Qwen/Qwen3.5-4B"

# Unsloth-saved Qwen3.5 LoRA → native HF + PEFT key layout
_LORA_KEY_REMAP_OLD = "base_model.model.model.language_model.layers"
_LORA_KEY_REMAP_NEW = "base_model.model.model.layers"


def _remap_lora_state_dict(state_dict: dict) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, tensor in state_dict.items():
        nk = key.replace(_LORA_KEY_REMAP_OLD, _LORA_KEY_REMAP_NEW)
        nk = nk.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        out[nk] = tensor
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help=f"Path to LoRA adapter (default: models/{DEFAULT_LORA_NAME})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=(
            "Path or HF id for base weights (default: Qwen/Qwen3.5-4B, same as train_unsloth "
            "without --model-path). Do NOT use models/base_model unless that exact folder "
            "was used for training — a mismatched base causes shape errors at merge."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder for merged model (default: <lora_path>-merged)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.lora_path:
        lora_path = Path(args.lora_path).resolve()
    else:
        lora_path = (project_root / "models" / DEFAULT_LORA_NAME).resolve()
        logger.warning(
            "No --lora-path provided. Using hardcoded default: %s — "
            "pass --lora-path <path> to merge a different adapter.",
            lora_path,
        )
    if not lora_path.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found: {lora_path}. Run train_unsloth.py first."
        )

    adapter_file = lora_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Missing {adapter_file}")

    # Default matches train_unsloth.py when --model-path is omitted (HF hub).
    # Auto-using models/base_model caused wrong-arch merges when that copy differed
    # from the checkpoint the LoRA was trained on (e.g. different revision).
    if args.base_model:
        raw = args.base_model
        p = Path(raw).expanduser()
        base_path = str(p.resolve()) if p.is_dir() else raw
    else:
        base_path = MODEL_NAME

    out_path: Path
    if args.output is None:
        out_path = project_root / "models" / f"{lora_path.name}-merged"
    else:
        out_path = Path(args.output).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from peft import PeftConfig, get_peft_model
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Install peft, safetensors, transformers: pip install peft safetensors transformers"
        ) from e

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    logger.info("Loading base model from %r (device_map=%r)...", base_path, device_map)
    # trust_remote_code required by Qwen3.5 for custom architecture modules
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info("Loading PEFT config from %s...", lora_path)
    peft_config = PeftConfig.from_pretrained(str(lora_path))

    logger.info("Building PEFT model and loading remapped LoRA weights...")
    model = get_peft_model(base_model, peft_config)
    raw_sd = load_file(str(adapter_file))
    remapped = _remap_lora_state_dict(raw_sd)
    load_result = model.load_state_dict(remapped, strict=False)
    n_lora_missing = sum(1 for k in load_result.missing_keys if "lora" in k)
    if n_lora_missing:
        raise RuntimeError(
            f"LoRA weights missing after remap ({n_lora_missing} keys). "
            "Check Unsloth/PEFT versions or adapter layout."
        )
    if load_result.unexpected_keys:
        logger.warning("Unexpected keys ignored: %s...", load_result.unexpected_keys[:5])

    logger.info("Merging LoRA into base weights...")
    model = model.merge_and_unload()

    # trust_remote_code required by Qwen3.5 for custom tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    logger.info("Saving merged model to %s...", out_path)
    model.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer.save_pretrained(str(out_path))
    logger.info("Merged model saved to %s", out_path)


if __name__ == "__main__":
    main()
