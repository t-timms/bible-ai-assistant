# Bible AI Assistant — Project Journey & Key Learnings

This document captures the critical technical decisions, problems encountered, and solutions from building this project. Use it for interview prep, portfolio content, or understanding the "why" behind the architecture.

---

## Executive Summary

A fine-tuned Bible Q&A assistant built on Qwen3 4B, with RAG (ChromaDB), OpenClaw agent integration, and Telegram. The model is trained on ~1,600 diverse examples, quantized to Q4_K_M GGUF, and deployed via Ollama. Key lesson: **for small models (4B), less is more**—short system prompts, diverse training data, and server-side post-processing beat long instruction lists and massive repetitive datasets.

---

## Architecture at a Glance

```
User (Telegram / Web UI)
    → OpenClaw Gateway (openclaw gateway)
    → RAG Server (port 8081): retrieves verses, augments prompt, post-processes output
    → Ollama (bible-assistant, Q4_K_M GGUF)
    → Response back through RAG → OpenClaw → User
```

**Stack:** Qwen3 4B (base) → Unsloth QLoRA fine-tune → merge → llama.cpp convert → Q4_K_M quantize → Ollama. RAG: ChromaDB + nomic-embed-text-v1.5. Agent: OpenClaw.

---

## Phase 1: Initial Setup (What the WALKTHROUGH Covers)

- **Environment:** Conda, PyTorch nightly (CUDA 12.8 for RTX 5070 Ti Blackwell)
- **Base model:** Qwen3-4B-Instruct-2507 from Hugging Face
- **Data:** WEB Bible → `bible_web.json` (31k verses) → ChromaDB index
- **Training:** Unsloth QLoRA, LoRA r=8, 2 epochs, LR 1e-4, dropout 0.15
- **Export:** Merge LoRA → convert_hf_to_gguf.py (f16) → llama-quantize (Q4_K_M)
- **Deploy:** Ollama Modelfile with system prompt + parameters

---

## Phase 2: The Overfitting Crisis (Critical for Interviews)

### The Problem

The initial fine-tuned model had severe issues:

- **Verse fabrication** — Model would invent or misquote verses instead of using RAG context
- **Instruction leaking** — Model output meta-instructions: "Avoid repetition. Trim redundancy. Just answer. Then exit."
- **Repetition/looping** — Same phrase repeated 5–10 times in one response
- **Bible answers for non-Bible questions** — "What is 2+2?" → model would cite Scripture
- **"Answer:" prefix** — Model would start responses with "Answer:" before the actual content

### Root Cause

1. **Dataset:** 31,598 examples in a single format—"What does X say?" → "X says: [verse]. This verse is part of Scripture..." repeated 31k times. The model memorized the pattern instead of learning behavior.
2. **System prompt:** 157 lines (~2,500 tokens) of detailed instructions, forbidden phrases, and examples. A 4B model cannot reliably separate "instructions to follow silently" from "text to output." It learned to generate instruction-like language.
3. **Training signal:** ~80% of each training example was system prompt. The model spent capacity memorizing that text rather than learning diverse response patterns.

### The Fix (Quality Over Quantity)

| Before | After | Why |
|--------|-------|-----|
| 31,598 examples, 1 format | ~1,600 examples, 7 categories | Diverse behavior > memorization |
| 157-line system prompt | 15-line system prompt (~200 tokens) | Small models need concise prompts |
| LoRA r=16, 3 epochs, LR 2e-4 | LoRA r=8, 2 epochs, LR 1e-4, dropout 0.15 | Less capacity = less overfitting |
| No eval split | 10% eval split | Monitor train vs. eval loss for overfitting |
| No post-processing | _strip_thinking + _strip_repetition_and_meta | Safety net for residual issues |

### Dataset Categories (v2/v3)

- **Verse lookups** (~800) — "What does X say?" with 5–6 response formats
- **RAG-grounded** (~700) — Simulated RAG context: "Relevant Bible verses: ... User question: ..." so model learns to quote provided text exactly
- **Thematic** (~30) — "What does the Bible say about forgiveness?"
- **General assistant** (~35) — Non-Bible Q&A so model doesn't default to Scripture for everything
- **Meta-questions** (~18) — "What can you do?" with no verses
- **Multi-turn** (~8) — Follow-up questions, context awareness
- **Refusals** (~12) — "Make up a verse" → polite decline; "Role-play as God" → decline

### System Prompt Simplification

**Principle:** Identity + 4 rules + style in one short block. No forbidden-phrase lists (those teach the model those phrases). No exhaustive format examples. Let training data teach behavior.

---

## Phase 3: Inference-Time Fixes (RAG Server Post-Processing)

Even with a better model, Qwen3 outputs `<think>...</think>` reasoning blocks that consume token budget. We add server-side cleanup:

1. **`_strip_thinking`** — Remove `<think>...</think>` from model output (regex) so only the final answer is returned.
2. **`_strip_repetition_and_meta`** — Strip leaked instructions ("Avoid repetition", "Just answer. Exit.", etc.) and truncate when the same 60+ char block appears twice (loop detection).

Applied to both streaming and non-streaming responses before sending to OpenClaw.

### Modelfile Parameters

- `repeat_penalty 1.65` — Stronger penalty for repeated tokens
- `repeat_last_n 128` — Look at more recent tokens when penalizing
- `num_predict 512` — Cap output length to reduce runaway loops

---

## Key Technical Decisions (Interview Talking Points)

1. **Why Qwen3 4B?** — Good instruction-following at 4B scale, efficient for local inference, fine-tunes well with LoRA. Trade-off: small models need careful prompt and data design.
2. **Why RAG instead of training all verses?** — RAG retrieves at inference; fine-tuning teaches behavior. Training 31k verses caused memorization. RAG keeps the model grounded in retrieved context.
3. **Why LoRA and not full fine-tune?** — Full 4B fine-tune requires more VRAM and risks catastrophic forgetting. LoRA trains ~0.1% of params, preserves base capabilities, merges cleanly.
4. **Why Q4_K_M?** — Good quality/size trade-off for 4B. Fits in 16GB, runs fast. Q8 would be more accurate but larger.
5. **Why OpenClaw?** — Agent framework with Telegram, web UI, session memory. RAG server is a drop-in OpenAI-compatible LLM provider.

---

## Pipeline Commands (Quick Reference)

```powershell
# 1. Build dataset (after editing dataset_builder.py or system_prompt.txt)
conda activate bible-ai-assistant
cd C:\Users\ttimm\Desktop\John\bible-ai-assistant
python training/dataset_builder.py

# 2. Train
python training/train_unsloth.py --run-name qwen3-4b-bible-John-v3

# 3. Merge LoRA
python training/merge_adapters.py --lora-path models/qwen3-4b-bible-John-v3

# 4. Convert to GGUF (from project root, llama.cpp cloned alongside)
python ..\llama.cpp\convert_hf_to_gguf.py models\qwen3-4b-bible-John-v3-merged --outfile models\qwen3-4b-bible-John-v3-f16.gguf --outtype f16
..\llama.cpp\build\bin\Release\llama-quantize.exe models\qwen3-4b-bible-John-v3-f16.gguf models\qwen3-4b-bible-John-v3-q4_k_m.gguf Q4_K_M

# 5. Update Modelfile (edit generate_modelfile.py GGUF_PATH) and deploy
python deployment/pc/generate_modelfile.py
ollama rm bible-assistant
ollama create bible-assistant -f deployment/pc/Modelfile

# 6. Run the stack
# Terminal 1: RAG server
uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081
# Terminal 2: OpenClaw
openclaw gateway
```

---

## WALKTHROUGH.md Accuracy Gaps

| Section | Current State | Actual State |
|---------|---------------|--------------|
| Step 8c | `--max-examples 50000` | No such flag; dataset_builder.py produces ~1,595 diverse examples |
| Step 9 | 3 epochs | 2 epochs (train_unsloth.py) |
| Step 9 | LoRA r=16 (implied) | LoRA r=8 |
| Step 10 | merge_adapters.py (default) | Use `--lora-path models/qwen3-4b-bible-John-v3` for v3 |
| Step 11 | num_predict 300, repeat_penalty 1.45 | num_predict 512, repeat_penalty 1.65, repeat_last_n 128 |
| Step 12 | Basic RAG description | RAG server also has _strip_thinking, _strip_repetition_and_meta |
| Step 13 | openclaw start | openclaw gateway |
| System prompt | 157-line constitution | 15-line simplified prompt (prompts/system_prompt.txt) |

---

## Glossary (For Interviews)

- **LoRA** — Low-Rank Adaptation; trains small adapter weights instead of full model
- **QLoRA** — Quantized LoRA; base model in 4-bit, adapters in bf16
- **RAG** — Retrieval-Augmented Generation; inject retrieved docs (verses) into prompt
- **ChromaDB** — Vector database for semantic search
- **nomic-embed-text** — Embedding model for verse indexing
- **GGUF** — Model format for llama.cpp / Ollama (quantized)
- **Q4_K_M** — 4-bit quantization; good balance of quality and size
- **Overfitting** — Model memorizes training data instead of generalizing
- **Instruction leaking** — Model outputs its own instructions instead of following them silently

---

*"In Jesus' name. Amen! All glory to God!"*
