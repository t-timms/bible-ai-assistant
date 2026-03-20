#!/usr/bin/env python3
"""Quick test of merged model with transformers (no GGUF/Ollama)."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "models/qwen3.5-4b-bible-John-v8-merged"
PROMPT = "<|im_start|>system\nYou are a Bible AI assistant.<|im_end|>\n<|im_start|>user\nWhat does John 3:16 say?<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Merged HF folder (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    model_path = args.model_path
    print(f"Loading model and tokenizer from {model_path}...")
    # trust_remote_code required by Qwen3.5 for custom architecture modules
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Generating...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    response = full.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    print("--- Response ---")
    print(response[:500])
    print("---")

if __name__ == "__main__":
    main()
