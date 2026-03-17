#!/usr/bin/env python3
"""
QLoRA fine-tuning of Qwen3 4B with Unsloth.
Requires: conda env bible-ai-assistant, PyTorch nightly (CUDA 12.8+), data/processed/train.json.
Use bf16=True (never fp16) on RTX 5070 Ti (Blackwell).

Usage:
  python training/train_unsloth.py
  python training/train_unsloth.py --run-name qwen3-4b-run-1
"""
from pathlib import Path
import argparse

# Training config (aligned with config.yaml). Edit here or in config.yaml for reference.
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LOAD_IN_4BIT = True
MAX_SEQ_LENGTH = 2048
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
        l = cfg["lora"]
        LORA_R = l.get("r", LORA_R)
        LORA_ALPHA = l.get("alpha", LORA_ALPHA)
        LORA_DROPOUT = l.get("dropout", LORA_DROPOUT)
        LORA_TARGET_MODULES = l.get("target_modules", LORA_TARGET_MODULES)
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
    parser.add_argument("--run-name", type=str, default="qwen3-4b-bible-John", help="W&B run name and folder for saved adapter (e.g. models/qwen3-4b-bible-John)")
    parser.add_argument("--model-path", type=str, default=None, help="Local path to base model (default: use HF MODEL_NAME)")
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
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
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
            import unsloth.models.qwen3 as _qwen3
            _llama.HAS_XFORMERS = False
            _qwen3.HAS_XFORMERS = False

    model_path = args.model_path or MODEL_NAME
    use_local_model = bool(args.model_path)
    if args.model_path:
        model_path = str(Path(args.model_path).resolve())

    wandb.init(project="bible-ai", name=args.run_name)

    # When loading from a local path, Unsloth loads the tokenizer from that path too.
    # Some transformers versions fail on local tokenizer config (dict vs object). Workaround:
    # temporarily hide local tokenizer files so Unsloth loads the tokenizer from HF instead.
    tokenizer_renames = []
    if use_local_model:
        for f in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
            p = Path(model_path) / f
            if p.exists():
                bak = p.with_suffix(p.suffix + ".bak")
                os.rename(p, bak)
                tokenizer_renames.append((bak, p))
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype="bfloat16",  # Required for Blackwell (RTX 5070 Ti)
            load_in_4bit=LOAD_IN_4BIT,
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

    # Load dataset (messages format)
    full_dataset = load_dataset("json", data_files=str(train_file), split="train")

    # Add "text" column: apply chat template so the trainer has one string per example
    def format_messages(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    full_dataset = full_dataset.map(format_messages, batched=True, remove_columns=full_dataset.column_names)

    # Pre-tokenize in the main process to avoid Windows multiprocessing (UnslothSFTTrainer not found in workers)
    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
            return_tensors=None,
        )
        out["labels"] = [ids[:] for ids in out["input_ids"]]
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
        report_to="wandb",
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

    # Save LoRA adapter: folder name matches run name (e.g. qwen3-4b-bible-John)
    out_path = project_root / "models" / args.run_name
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"Saved LoRA adapter to {out_path}")


if __name__ == "__main__":
    main()
