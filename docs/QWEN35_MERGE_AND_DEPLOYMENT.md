# Qwen3.5 Merge and Deployment Notes

This document describes the architecture mismatch between SFT/ORPO environments and its impact on merge and deployment. It records the diagnostic process and the corrected pipeline.

---

## Summary

| Issue | Root Cause | Resolution |
|-------|------------|------------|
| Merged model produces garbage | Adapter trained with one architecture, merged with another | Run SFT in `bible-orpo` (transformers 5.x) for native Qwen3.5 |
| v5 merge shows "missing adapter keys" | v5 adapter has `language_model.layers`; merge loads `layers` | SFT+ORPO both in `bible-orpo` |
| v6 still garbage (transformers + Ollama) | QLoRA 4-bit not recommended for Qwen3.5; Unsloth may load Qwen3 instead | Use `load_in_4bit: false`, `load_in_16bit: true` (bf16 LoRA); ~10GB VRAM for 4B |
| v7 garbage: echoes system prompt | **Instruction leaking** — full rules/style embedded in every training example; 4B overfits | Use `prompts/system_prompt_training.txt` (1–2 sentences); full prompt only in Modelfile at inference |

---

## Problem: Garbage Output After Merge

After merging the LoRA adapter and converting to GGUF, the model produced repetitive garbage (e.g., "answer.elf them", "implement the room theensive for", "2-3 passages and tie them together") instead of coherent Bible answers.

### Diagnostic Steps

1. **Ollama test** — Both v4 and v5 merged GGUF produced garbage.
2. **Modelfile tweaks** — `repeat_penalty`, `num_ctx`, `repeat_last_n` did not fix the issue.
3. **Transformers test** — `scripts/test_merged_model.py` loaded the merged model with `AutoModelForCausalLM` and generated; output was garbage.
4. **Base model test** — `scripts/test_base_model.py` loaded base Qwen3.5-4B from Hugging Face; output was coherent and correctly quoted John 3:16.

**Conclusion:** The merged model is corrupted. The issue is not GGUF conversion or Ollama; it is the merge step.

---

## Root Cause: Architecture Mismatch

| Stage | Environment | transformers | Model Layout |
|-------|--------------|--------------|--------------|
| **SFT (v4)** | `bible-ai-assistant` | ≤ 4.57.2 | Qwen3 fallback; adapter keys: `model.layers.X` |
| **ORPO (v5)** | `bible-orpo` | ≥ 5.1 | Native Qwen3.5; adapter keys: `model.language_model.layers.X` |
| **Merge** | `bible-orpo` | ≥ 5.1 | Loads native Qwen3.5; expects matching adapter keys |

- **v4 adapter:** Trained with transformers 4.57.2. Qwen3.5 is loaded as a Qwen3-compatible model. LoRA keys use `model.layers`.
- **Merge:** Runs in `bible-orpo` with transformers 5.x. The base model has native Qwen3.5 layout (`model.language_model.layers` or a different structure).
- **v4 merge:** Keys match structurally (no "missing" warning), but the weight layouts are incompatible—LoRA trained for Qwen3 layout applied to Qwen3.5 produces corrupted outputs.
- **v5 merge:** Adapter has `language_model.layers`; merge loads model with `layers` (Unsloth "Fast Qwen3 patching"). Keys do not match; PEFT reports "missing adapter keys"; merge effectively applies nothing.

---

## Corrected Pipeline

All training and merge must use **native Qwen3.5** (transformers ≥ 5.1) for consistent architecture:

| Stage | Environment | Output |
|-------|-------------|--------|
| **SFT** | `bible-orpo` | `models/qwen3.5-4b-bible-John-v7/` |
| **ORPO** | `bible-orpo` | `models/qwen3.5-4b-bible-John-v8/` (optional) |
| **Merge** | `bible-orpo` | `models/...-merged/` |
| **GGUF / Ollama** | none | Deploy |

**Important:** Use bf16 LoRA for Qwen3.5 (not 4-bit QLoRA). Set `load_in_4bit: false` in `training/config.yaml` and `train_unsloth.py`.

### Commands

```bash
# 1. SFT in bible-orpo (native Qwen3.5, bf16 LoRA)
conda activate bible-orpo
python training/train_unsloth.py --run-name qwen3.5-4b-bible-John-v7

# 2. Merge (adapter keys will match)
python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v7

# 3. GGUF, Modelfile, Ollama (no conda)
python ../llama.cpp/convert_hf_to_gguf.py models/qwen3.5-4b-bible-John-v7-merged \
  --outfile models/qwen3.5-4b-bible-John-v7-f16.gguf --outtype f16
python deployment/pc/generate_modelfile.py --gguf qwen3.5-4b-bible-John-v7-f16.gguf
ollama create bible-assistant -f deployment/pc/Modelfile
```

**SFT + ORPO (v8 example):** merge `models/qwen3.5-4b-bible-John-v8-orpo`, convert `...-orpo-merged` → `...-orpo-f16.gguf`, quantize to `...-orpo-q4_k_m.gguf`. For Ollama, `generate_modelfile.py` defaults to **Q4 ORPO** (faster `ollama create`). To **A/B Q4 vs F16**, create two names, e.g. `bible-assistant-orpo` and `bible-assistant-orpo-f16`, using `--gguf` for each GGUF — see [deployment/pc/README.md](../deployment/pc/README.md).

---

## Test Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_base_model.py` | Load base Qwen3.5-4B from HF; verify coherent generation |
| `scripts/test_merged_model.py` | Load merged model from disk; verify merge did not corrupt |

Run from `bible-orpo`:

```bash
conda activate bible-orpo
python scripts/test_base_model.py
python scripts/test_merged_model.py
```

---

## References

- [ORPO_TWO_ENV_SETUP.md](ORPO_TWO_ENV_SETUP.md) — Updated to recommend SFT in `bible-orpo` for Qwen3.5
- [ENVIRONMENT_REQUIREMENTS.md](ENVIRONMENT_REQUIREMENTS.md) — Conda env by phase
- [llama.cpp issue #10312](https://github.com/ggerganov/llama.cpp/issues/10312) — Qwen repetition loops in GGUF
