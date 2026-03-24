#!/usr/bin/env python3
"""Test base Qwen3.5-4B (no fine-tuning) with transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-4B"
PROMPT = "<|im_start|>system\nYou are a Bible AI assistant.<|im_end|>\n<|im_start|>user\nWhat does John 3:16 say?<|im_end|>\n<|im_start|>assistant\n"


def main():
    print("Loading base model and tokenizer...")
    # trust_remote_code required by Qwen3.5 for custom architecture modules
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
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
    print(response[:600])
    print("---")


if __name__ == "__main__":
    main()
