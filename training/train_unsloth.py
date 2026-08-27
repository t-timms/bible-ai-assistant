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
import hashlib
import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Training config defaults — must match training/config.yaml.
# YAML values override these at runtime via _load_config_yaml().
MODEL_NAME = "Qwen/Qwen3.5-4B"
# Pinned commit SHA for reproducible loads (H-5 supply-chain hardening — see
# rag/settings.py for the same rationale). Passed to FastLanguageModel.from_pretrained
# via load_model_pinned(): if Unsloth's signature rejects `revision`, we fall back
# UNPINNED with a loud warning instead of breaking a multi-hour run (audit T7).
# Verified against the HF Hub API directly on 2026-08-24 — re-verify before bumping MODEL_NAME.
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# Qwen3.5: Unsloth does NOT recommend QLoRA 4-bit (quantization differences cause garbage output). Use bf16 LoRA.
LOAD_IN_4BIT = False
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OUTPUT_DIR = "checkpoints"
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2.0e-4
LR_SCHEDULER_TYPE = "cosine"  # audit T5: replaces linear drift
WARMUP_RATIO = 0.03  # ratio-based; absolute warmup_steps drifted on short runs
MAX_EVAL_STEPS = 50  # cap for derived eval/save cadence
EVAL_SPLIT = 0.1
LOGGING_STEPS = 50
BF16 = True  # REQUIRED for Blackwell. Do not use fp16.
RANDOM_STATE = 3407

# ChatML assistant-turn start marker: loss is masked (-100) up to and including
# this span so training signal applies to completions only (audit T6).
ASSISTANT_START_SPAN = "<|im_start|>assistant\n"
IGNORE_INDEX = -100
WEIGHTS_FILE_SUFFIXES = frozenset({".safetensors", ".bin", ".gguf"})


def estimate_total_steps(
    n_train_examples: int, epochs: float, batch_size: int, grad_accum_steps: int
) -> int:
    """Optimizer-step estimate: ceil(n * epochs / effective_batch)."""
    effective_batch = max(1, batch_size * grad_accum_steps)
    return max(1, math.ceil(n_train_examples * epochs / effective_batch))


def suggest_eval_steps(total_steps: int, cap: int = MAX_EVAL_STEPS) -> int:
    """Eval/save cadence sized to the run: min(cap, total_steps // 6), at least 1.

    //6 guarantees >=6 eval/save points so load_best_model_at_end +
    metric_for_best_model have something meaningful to select from.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    return max(1, min(cap, total_steps // 6))


def find_last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Start index of the LAST occurrence of ``needle`` in ``haystack``, or -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    first, n = needle[0], len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i] == first and list(haystack[i : i + n]) == list(needle):
            return i
    return -1


def build_completion_labels(
    input_ids: Sequence[int],
    assistant_span_ids: Sequence[int],
    pad_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Labels masking everything up to and including the LAST assistant-start span.

    Completion tokens (and <|im_end|>) keep their ids; padding is masked. Raises
    when the span is absent so chat-template drift fails loudly instead of
    silently training on prompt tokens.
    """
    span_start = find_last_subsequence(input_ids, assistant_span_ids)
    if span_start == -1:
        raise ValueError(
            "Assistant start marker not found in tokenized sample — chat template "
            f"drift? Expected span tokens {list(assistant_span_ids)} present in input."
        )
    labels = [ignore_index] * len(input_ids)
    for i in range(span_start + len(assistant_span_ids), len(input_ids)):
        if input_ids[i] != pad_id:
            labels[i] = input_ids[i]
    return labels


def load_model_pinned(
    from_pretrained: Callable[..., Any], model_name: str, revision: str | None, **kwargs: Any
) -> Any:
    """``from_pretrained`` with revision pinning; loud UNPINNED fallback on TypeError.

    Unsloth's FastLanguageModel.from_pretrained has not always accepted `revision`;
    rather than risk breaking a run on an unverified kwarg, we attempt pinned,
    catch TypeError, warn, and retry unpinned (audit T7).
    """
    if not revision:
        return from_pretrained(model_name=model_name, **kwargs)
    try:
        return from_pretrained(model_name=model_name, revision=revision, **kwargs)
    except TypeError as exc:
        logger.warning(
            "REPRO WARNING: %s rejected revision=%r (%s). Continuing UNPINNED — "
            "weights may differ from the verified snapshot. Investigate Unsloth "
            "support for `revision` before trusting this run.",
            getattr(from_pretrained, "__name__", "from_pretrained"),
            revision,
            exc,
        )
        return from_pretrained(model_name=model_name, **kwargs)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights_dir(dir_path: Path) -> dict[str, str]:
    """Hash weight files under ``dir_path``; verify *.sha256 sidecars when present.

    Returns {relative_path: sha256}. Raises FileNotFoundError when no weight
    files exist and ValueError on any sidecar mismatch.
    """
    results: dict[str, str] = {}
    for path in sorted(dir_path.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in WEIGHTS_FILE_SUFFIXES):
            continue
        digest = sha256_file(path)
        results[path.relative_to(dir_path).as_posix()] = digest
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.exists():
            expected = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
            if expected != digest:
                raise ValueError(
                    f"Checksum mismatch for {path}: sidecar expects {expected}, got {digest}"
                )
    if not results:
        raise FileNotFoundError(f"No weight files (*.safetensors/*.bin/*.gguf) in {dir_path}")
    return results


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
    global LEARNING_RATE, LR_SCHEDULER_TYPE, WARMUP_RATIO, MAX_EVAL_STEPS, LOGGING_STEPS
    global BF16, EVAL_SPLIT, RANDOM_STATE
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
        LR_SCHEDULER_TYPE = t.get("lr_scheduler_type", LR_SCHEDULER_TYPE)
        WARMUP_RATIO = float(t.get("warmup_ratio", WARMUP_RATIO))
        MAX_EVAL_STEPS = int(t.get("max_eval_steps", MAX_EVAL_STEPS))
        LOGGING_STEPS = t.get("logging_steps", LOGGING_STEPS)
        BF16 = t.get("bf16", BF16)
        EVAL_SPLIT = float(t.get("eval_split", EVAL_SPLIT))
        RANDOM_STATE = int(t.get("random_state", RANDOM_STATE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        type=str,
        default="qwen3.5-4b-bible-John-v4",
        help="W&B run name and folder for saved adapter (e.g. models/qwen3.5-4b-bible-John-v4)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local path to base model (default: use HF MODEL_NAME)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging (use if W&B service fails on Windows)",
    )
    parser.add_argument(
        "--verify-weights",
        type=str,
        default=None,
        metavar="DIR",
        help="Hash weight files under DIR (verifying *.sha256 sidecars), print results, exit.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    _load_config_yaml(project_root)

    if args.verify_weights:
        hashes = verify_weights_dir(Path(args.verify_weights))
        for rel, digest in hashes.items():
            print(f"{digest}  {rel}")
        return
    train_file = project_root / "data" / "processed" / "train.json"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}. Run dataset_builder.py first."
        )

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

    wandb_project = os.getenv("WANDB_PROJECT", "bible-ai")
    if args.no_wandb:
        wandb.init(project=wandb_project, name=args.run_name, mode="disabled")
    else:
        # On Windows, give W&B service extra time to start; UTF-8 fix is at top of file
        wandb.init(
            project=wandb_project,
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
        # Pinned load (audit T7): pass MODEL_REVISION when Unsloth accepts it;
        # fall back unpinned with a loud warning otherwise.
        model, tokenizer = load_model_pinned(
            FastLanguageModel.from_pretrained,
            model_name=model_path,
            revision=None if use_local_model else MODEL_REVISION,
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
        random_state=RANDOM_STATE,
    )

    # Qwen3.5 tokenizer from Unsloth is a VL processor that treats text as images. Use text-only tokenizer for dataset.
    from transformers import AutoTokenizer

    # trust_remote_code required by Qwen3.5 tokenizer for custom chat template
    text_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )

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

    full_dataset = full_dataset.map(
        format_messages, batched=True, remove_columns=full_dataset.column_names
    )

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
        # Loss masking (audit T6): -100 up to and including the assistant-start
        # span so only completion tokens contribute to loss.
        assistant_span_ids = text_tokenizer(ASSISTANT_START_SPAN, add_special_tokens=False)[
            "input_ids"
        ]
        out["labels"] = [
            build_completion_labels(ids, assistant_span_ids, pad_id) for ids in out["input_ids"]
        ]
        return out

    full_dataset = full_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing"
    )

    # Train/eval split to monitor overfitting via W&B
    split = full_dataset.train_test_split(test_size=EVAL_SPLIT, seed=RANDOM_STATE)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")

    # Derived cadence (audit T5): absolute save/eval steps drifted off-peak on
    # short runs and left load_best_model_at_end nothing to select from.
    total_steps_estimate = estimate_total_steps(
        len(train_dataset), NUM_EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION
    )
    eval_steps = suggest_eval_steps(total_steps_estimate, MAX_EVAL_STEPS)
    if eval_steps > total_steps_estimate:
        raise RuntimeError(
            f"Derived eval_steps ({eval_steps}) exceeds estimated total optimizer "
            f"steps ({total_steps_estimate}) — adjust batch size or epochs."
        )
    print(
        f"Schedule: ~{total_steps_estimate} optimizer steps, "
        f"eval/save every {eval_steps}, scheduler={LR_SCHEDULER_TYPE}, "
        f"warmup_ratio={WARMUP_RATIO}"
    )

    # Training args — bf16 required for Blackwell
    # skip_prepare_dataset: we already tokenized above; avoids Unsloth tokenization map (Windows spawn issue)
    training_args = SFTConfig(
        output_dir=str(project_root / OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        save_steps=eval_steps,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=eval_steps,
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
