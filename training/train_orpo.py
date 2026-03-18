#!/usr/bin/env python3
"""
ORPO preference alignment stage (runs after SFT).

Loads the SFT-trained LoRA adapter and fine-tunes it further with ORPO
(Odds Ratio Preference Optimization) using preference pairs from
data/processed/preferences.json.

Usage:
  python training/train_orpo.py --sft-path models/qwen3.5-4b-bible-John-v4
  python training/train_orpo.py --sft-path models/qwen3.5-4b-bible-John-v4 --run-name v4-orpo
  python training/train_orpo.py --sft-path models/qwen3.5-4b-bible-John-v4 --no-wandb
"""
# Fix Windows console encoding and W&B service timeout (must run before other imports)
import sys
import os
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("WANDB__SERVICE_WAIT", "90")
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
import argparse

MODEL_NAME = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH = 4096
BF16 = True


def main() -> None:
    parser = argparse.ArgumentParser(description="ORPO preference training on SFT model.")
    parser.add_argument("--sft-path", type=str, required=True,
                        help="Path to SFT LoRA adapter (e.g. models/qwen3.5-4b-bible-John-v4)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name (default: <sft-folder>-orpo)")
    parser.add_argument("--preferences", type=str, default=None,
                        help="Path to preferences JSON (default: data/processed/preferences.json)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="ORPO odds ratio weight")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (use if W&B service fails on Windows)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sft_path = Path(args.sft_path)
    if not sft_path.is_absolute():
        sft_path = project_root / sft_path
    sft_path = sft_path.resolve()
    if not sft_path.exists():
        raise FileNotFoundError(f"SFT adapter not found: {sft_path}. Run train_unsloth.py first.")

    # v5 = SFT+ORPO; v4 = SFT-only (distinguish in output name)
    if args.run_name:
        run_name = args.run_name
    elif sft_path.name.endswith("-v4"):
        run_name = sft_path.name.replace("-v4", "-v5")
    else:
        run_name = f"{sft_path.name}-orpo"
    pref_path = Path(args.preferences) if args.preferences else (
        project_root / "data" / "processed" / "preferences.json"
    )
    if not pref_path.exists():
        raise FileNotFoundError(
            f"Preference data not found: {pref_path}. Run build_preference_data.py first."
        )

    try:
        import os
        import torch
        import wandb
        from unsloth import FastLanguageModel
        from trl import ORPOTrainer, ORPOConfig
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install deps: pip install unsloth trl datasets wandb. "
            "Requires trl >= 0.8.0 for ORPOTrainer."
        ) from e

    # Qwen3.5-4B requires transformers>=5 for native qwen3_5 support. transformers<=4.57.2 loads it as
    # Qwen3 (wrong arch) and weights fail to load. Check for native support first.
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        supports_qwen3_5 = "qwen3_5" in CONFIG_MAPPING
    except Exception:
        supports_qwen3_5 = False

    if supports_qwen3_5:
        # Native path: load directly from hub, no config patching
        model_name_to_load = MODEL_NAME
        print(f"Using native Qwen3.5 support (transformers supports qwen3_5). Loading {MODEL_NAME}")
    else:
        # transformers<5: config patching does NOT work - Qwen3.5 weights are incompatible with Qwen3 arch.
        # Provide clear instructions instead of failing mid-load.
        import transformers
        tf_ver = getattr(transformers, "__version__", "?")
        raise RuntimeError(
            f"ORPO with Qwen3.5-4B requires transformers>=5 for native qwen3_5 support.\n"
            f"Current: transformers=={tf_ver}. The config-patch workaround fails because Qwen3.5 "
            f"weights (Gated DeltaNet, etc.) are incompatible with the Qwen3 architecture.\n\n"
            f"To run ORPO, create a separate env with newer transformers:\n"
            f'  conda create -n bible-orpo python=3.11 -y\n'
            f'  conda activate bible-orpo\n'
            f'  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n'
            f'  pip install "transformers>=5.1" unsloth trl datasets wandb\n'
            f"  python training/train_orpo.py --sft-path {sft_path}\n\n"
            f"Note: SFT (train_unsloth.py) stays on transformers<=4.57.2; ORPO uses the separate env."
        )

    # When native qwen3_5 support: model_name_to_load = MODEL_NAME already set above.
    # Legacy patch block removed: config patching fails because Qwen3.5 weights != Qwen3 arch.

    # Blackwell xformers workaround
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

    if args.no_wandb:
        wandb.init(project="bible-ai", name=run_name, mode="disabled")
    else:
        wandb.init(
            project="bible-ai",
            name=run_name,
            settings=wandb.Settings(_service_wait=90),
        )

    # Load base model + apply SFT LoRA (from patched local path so model_type=qwen3)
    # use_exact_model_name=True prevents Unsloth from rewriting our local path to a hub model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_to_load,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype="bfloat16",
        load_in_4bit=True,
        use_exact_model_name=True,
    )

    # Re-apply LoRA for continued training (ORPO stage)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load SFT weights into the LoRA adapter
    from peft import set_peft_model_state_dict
    import safetensors.torch
    adapter_file = sft_path / "adapter_model.safetensors"
    if adapter_file.exists():
        sft_state = safetensors.torch.load_file(str(adapter_file))
        set_peft_model_state_dict(model, sft_state)
        print(f"Loaded SFT adapter weights from {adapter_file}")
    else:
        adapter_bin = sft_path / "adapter_model.bin"
        if adapter_bin.exists():
            sft_state = torch.load(str(adapter_bin), map_location="cpu")
            set_peft_model_state_dict(model, sft_state)
            print(f"Loaded SFT adapter weights from {adapter_bin}")
        else:
            raise FileNotFoundError(
                f"No adapter weights found in {sft_path}. "
                "Expected adapter_model.safetensors or adapter_model.bin."
            )

    # Use text-only tokenizer for ORPO: model tokenizer may be a processor that treats text as images
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=str(pref_path), split="train")
    print(f"Loaded {len(dataset)} preference pairs")

    # ORPO expects prompt/chosen/rejected as strings. Use prompt + assistant content for chosen/rejected.
    # Qwen3.5 chat template is strict; bypass it for chosen/rejected by concatenating manually.
    eos = tokenizer.eos_token or "<|im_end|>"

    def format_for_orpo(examples):
        prompts, chosens, rejecteds = [], [], []
        for prompt_msgs, chosen_msgs, rejected_msgs in zip(
            examples["prompt"], examples["chosen"], examples["rejected"]
        ):
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            chosen_content = chosen_msgs[0]["content"] if chosen_msgs else ""
            rejected_content = rejected_msgs[0]["content"] if rejected_msgs else ""
            chosen_text = prompt_text + chosen_content + eos + "\n"
            rejected_text = prompt_text + rejected_content + eos + "\n"
            prompts.append(prompt_text)
            chosens.append(chosen_text)
            rejecteds.append(rejected_text)
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    dataset = dataset.map(format_for_orpo, batched=True,
                          remove_columns=dataset.column_names, desc="Formatting")

    config = ORPOConfig(
        output_dir=str(project_root / "checkpoints-orpo"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=BF16,
        beta=args.beta,
        max_length=2048,
        max_prompt_length=512,
        warmup_steps=20,
        logging_steps=10,
        save_steps=200,
        report_to="none" if args.no_wandb else "wandb",
        optim="adamw_8bit",
    )

    trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=config,
    )

    trainer.train()

    # Save ORPO-tuned adapter
    out_path = project_root / "models" / run_name
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"Saved ORPO adapter to {out_path}")


if __name__ == "__main__":
    main()
