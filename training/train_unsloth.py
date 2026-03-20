#!/usr/bin/env python3
"""
bf16 LoRA fine-tuning of Qwen3.5-4B with Unsloth.
Requires: conda env bible-orpo (transformers 5.x), PyTorch (CUDA 12.8+), data/processed/train.json.
Use bf16=True (never fp16) on RTX 5070 Ti (Blackwell).

Usage:
  python training/train_unsloth.py
  python training/train_unsloth.py --run-name qwen3.5-4b-bible-John-v4
  python training/train_unsloth.py --no-wandb   # Skip W&B (fallback if W&B has issues)
"""
# Fix Windows console encoding and W&B service timeout (must run before other imports)
import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("WANDB__SERVICE_WAIT", "90")  # Give W&B service more time to start
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import argparse
from pathlib import Path

# Training config (aligned with config.yaml). Edit here or in config.yaml for reference.
MODEL_NAME = "Qwen/Qwen3.5-4B"
# Qwen3.5: Unsloth does NOT recommend QLoRA 4-bit (quantization differences cause garbage output). Use bf16 LoRA.
LOAD_IN_4BIT = False
MAX_SEQ_LENGTH = 4096
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.15
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OUTPUT_DIR = "checkpoints"
NUM_EPOCHS = 2
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1.0e-4
WARMUP_STEPS = 50
EVAL_SPLIT = 0.1
SAVE_STEPS = 500
LOGGING_STEPS = 50
BF16 = True  # REQUIRED for Blackwell. Do not use fp16.


def _load_config_yaml(project_root: Path) -> None:
    """Override module constants from training/config.yaml if present and PyYAML available."""
    try:
        import yaml
    except ImportError:
        return
    cfg_path = project_root / "training" / "config.yaml"
    if not cfg_path.exists():
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        return
    global MODEL_NAME, LOAD_IN_4BIT, MAX_SEQ_LENGTH, LORA_R, LORA_ALPHA, LORA_DROPOUT
    global LORA_TARGET_MODULES, OUTPUT_DIR, NUM_EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION
    global LEARNING_RATE, WARMUP_STEPS, SAVE_STEPS, LOGGING_STEPS, BF16, EVAL_SPLIT
    if "model" in cfg:
        m = cfg["model"]
        MODEL_NAME = m.get("name", MODEL_NAME)
        LOAD_IN_4BIT = m.get("load_in_4bit", LOAD_IN_4BIT)
        MAX_SEQ_LENGTH = m.get("max_seq_length", MAX_SEQ_LENGTH)
    if "lora" in cfg:
        lora_cfg = cfg["lora"]
        LORA_R = lora_cfg.get("r", LORA_R)
        LORA_ALPHA = lora_cfg.get("alpha", LORA_ALPHA)
        LORA_DROPOUT = lora_cfg.get("dropout", LORA_DROPOUT)
        LORA_TARGET_MODULES = lora_cfg.get("target_modules", LORA_TARGET_MODULES)
    if "training" in cfg:
        t = cfg["training"]
        OUTPUT_DIR = t.get("output_dir", OUTPUT_DIR)
        NUM_EPOCHS = t.get("num_train_epochs", NUM_EPOCHS)
        BATCH_SIZE = t.get("per_device_train_batch_size", BATCH_SIZE)
        GRADIENT_ACCUMULATION = t.get("gradient_accumulation_steps", GRADIENT_ACCUMULATION)
        LEARNING_RATE = float(t.get("learning_rate", LEARNING_RATE))
        WARMUP_STEPS = t.get("warmup_steps", WARMUP_STEPS)
        SAVE_STEPS = t.get("save_steps", SAVE_STEPS)
        LOGGING_STEPS = t.get("logging_steps", LOGGING_STEPS)
        BF16 = t.get("bf16", BF16)
        EVAL_SPLIT = float(t.get("eval_split", EVAL_SPLIT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="qwen3.5-4b-bible-John-v4", help="W&B run name and folder for saved adapter (e.g. models/qwen3.5-4b-bible-John-v4)")
    parser.add_argument("--model-path", type=str, default=None, help="Local path to base model (default: use HF MODEL_NAME)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging (use if W&B service fails on Windows)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    _load_config_yaml(project_root)
    train_file = project_root / "data" / "processed" / "train.json"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}. Run dataset_builder.py first.")

    try:
        import os

        import torch
        import wandb
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "Install training deps: pip install unsloth trl datasets wandb. "
            "PyTorch: use nightly with CUDA 12.8+ for RTX 5070 Ti."
        ) from e

    # Blackwell (RTX 50xx, capability 12.x): xformers has no operator; force PyTorch SDPA
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 12:
            import unsloth.models.llama as _llama
            _llama.HAS_XFORMERS = False
            try:
                import unsloth.models.qwen3 as _qwen3
                _qwen3.HAS_XFORMERS = False
            except ImportError:
                pass
            try:
                import unsloth.models.qwen3_5 as _qwen3_5
                _qwen3_5.HAS_XFORMERS = False
            except ImportError:
                pass

    model_path = args.model_path or MODEL_NAME
    use_local_model = bool(args.model_path)
    if args.model_path:
        model_path = str(Path(args.model_path).resolve())

    if args.no_wandb:
        wandb.init(project="bible-ai", name=args.run_name, mode="disabled")
    else:
        # On Windows, give W&B service extra time to start; UTF-8 fix is at top of file
        wandb.init(
            project="bible-ai",
            name=args.run_name,
            settings=wandb.Settings(_service_wait=90),
        )

    # When loading from a local path, Unsloth loads the tokenizer from that path too.
    # Some transformers versions fail on local tokenizer config (dict vs object). Workaround:
    # temporarily hide local tokenizer files so Unsloth loads the tokenizer from HF instead.
    tokenizer_renames = []
    try:
        if use_local_model:
            for f in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
                p = Path(model_path) / f
                if p.exists():
                    bak = p.with_suffix(p.suffix + ".bak")
                    os.rename(p, bak)
                    tokenizer_renames.append((bak, p))
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype="bfloat16",  # Required for Blackwell (RTX 5070 Ti)
            load_in_4bit=LOAD_IN_4BIT,
            load_in_16bit=not LOAD_IN_4BIT,  # bf16 LoRA when not using 4-bit (required for Qwen3.5)
            tokenizer_name=MODEL_NAME if use_local_model else None,
        )
    finally:
        for bak, orig in tokenizer_renames:
            if bak.exists():
                os.rename(bak, orig)

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Qwen3.5 tokenizer from Unsloth is a VL processor that treats text as images. Use text-only tokenizer for dataset.
    from transformers import AutoTokenizer
    # trust_remote_code required by Qwen3.5 tokenizer for custom chat template
    text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load dataset (messages format)
    full_dataset = load_dataset("json", data_files=str(train_file), split="train")

    # Add "text" column: apply chat template so the trainer has one string per example
    def format_messages(examples):
        texts = []
        for messages in examples["messages"]:
            text = text_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    full_dataset = full_dataset.map(format_messages, batched=True, remove_columns=full_dataset.column_names)

    # Pre-tokenize with text-only tokenizer (avoids VL processor treating prompt text as base64 images)
    # Pad to max_length so collator gets same-length sequences; mask padding in labels with -100
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    def tokenize_fn(examples):
        out = text_tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors=None,
        )
        pad_id = text_tokenizer.pad_token_id
        out["labels"] = [
            [idx if idx != pad_id else -100 for idx in ids]
            for ids in out["input_ids"]
        ]
        return out

    full_dataset = full_dataset.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing")

    # Train/eval split to monitor overfitting via W&B
    split = full_dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")

    # Training args — bf16 required for Blackwell
    # skip_prepare_dataset: we already tokenized above; avoids Unsloth tokenization map (Windows spawn issue)
    training_args = SFTConfig(
        output_dir=str(project_root / OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none" if args.no_wandb else "wandb",
        bf16=BF16,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    # Save LoRA adapter: folder name matches run name (e.g. qwen3.5-4b-bible-John)
    out_path = project_root / "models" / args.run_name
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"Saved LoRA adapter to {out_path}")


if __name__ == "__main__":
    main()
