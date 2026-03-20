# Post-Training Checklist

Use this when SFT (and optionally ORPO) training finishes. Run all steps from project root.

Path examples below use **v8** and **v8 + ORPO**; swap folder names for other runs.

---

## 1. Merge LoRA → Full Model

**After SFT only:**

```powershell
conda activate bible-orpo
python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v8
```

**After SFT + ORPO** (merge the ORPO adapter folder, not the raw SFT folder):

```powershell
python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v8-orpo
```

**Output:** `models/<lora-folder-name>-merged/` (e.g. `...-v8-merged/` or `...-v8-orpo-merged/`)

**Base weights:** Merge defaults to **`Qwen/Qwen3.5-4B`** on Hugging Face (same as `train_unsloth.py` without `--model-path`). If you trained with `--model-path models/base_model`, add `--base-model models/base_model`. Using a different local copy than training can cause `size mismatch` errors or a bad merge.

---

## 2. Convert to GGUF (float16)

```powershell
# SFT+ORPO merged example:
python ..\llama.cpp\convert_hf_to_gguf.py models/qwen3.5-4b-bible-John-v8-orpo-merged `
  --outfile models/qwen3.5-4b-bible-John-v8-orpo-f16.gguf --outtype f16
```

---

## 3. Quantize to Q4_K_M (recommended for Ollama)

```powershell
..\llama.cpp\build\bin\Release\llama-quantize.exe models/qwen3.5-4b-bible-John-v8-orpo-f16.gguf `
  models/qwen3.5-4b-bible-John-v8-orpo-q4_k_m.gguf q4_k_m
```

**Windows note:** `llama-quantize` may be at `build\Release\` instead of `build\bin\Release\`.

**Why Q4 for Modelfile default:** `ollama create` with an **~8 GB F16** file can sit on “gathering model components” for a long time. Q4 (~2.6 GB) imports faster and uses less VRAM.

---

## 4. Modelfile & Ollama

`deployment/pc/generate_modelfile.py` accepts **`--gguf <filename under models/>`** (or an absolute path). Default is **Q4 ORPO** (`qwen3.5-4b-bible-John-v8-orpo-q4_k_m.gguf`).

**Single model (Q4 recommended):**

```powershell
python deployment/pc/generate_modelfile.py
ollama create bible-assistant-orpo -f deployment/pc/Modelfile
```

**A/B test Q4 vs F16** (two Ollama names; run both `ollama create` blocks):

```powershell
# Q4
python deployment/pc/generate_modelfile.py --gguf qwen3.5-4b-bible-John-v8-orpo-q4_k_m.gguf
ollama create bible-assistant-orpo -f deployment/pc/Modelfile

# F16 — be patient on first import
python deployment/pc/generate_modelfile.py --gguf qwen3.5-4b-bible-John-v8-orpo-f16.gguf
ollama create bible-assistant-orpo-f16 -f deployment/pc/Modelfile

# Restore repo default Modelfile (Q4)
python deployment/pc/generate_modelfile.py
```

More detail: **[deployment/pc/README.md](../../deployment/pc/README.md)**

---

## 5. Evaluate

```powershell
# Start RAG server and Ollama first; set the app to the model you want (e.g. bible-assistant-orpo).
python training/evaluate.py --ollama-model bible-assistant-orpo
```

**Versioned A/B benchmark** (saves under `docs/benchmark_runs/`): see **[../BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md)** — `scripts/run_benchmark.py` + `scripts/compare_benchmark_runs.py`.

**Pass:** Zero fabricated verses, constitution compliance, verse accuracy ≥ 85%.

---

## 6. Smoke Test

```powershell
ollama run bible-assistant-orpo "What does John 3:16 say?"
```

**Via RAG (recommended):** answers are post-processed by `rag/response_cleanup.py` (strips Qwen `</think>` blocks and plain “Thinking Process:” preambles). The non-streaming OpenAI-compatible handler **always** assigns the cleaned string to `choices[0].message.content` so clients never see a mix of raw and stripped text. Direct `ollama run` bypasses that layer.

**Without Ollama (transformers):** from project root, in `bible-orpo`:

```powershell
python scripts/test_merged_model.py --model-path models/qwen3.5-4b-bible-John-v8-orpo-merged
```

---

## W&B

If a shared link 404s, open [wandb.ai](https://wandb.ai), sign in, go to project **bible-ai**, and find your run by date or run id.

---

## Quick Reference

| Step | Command / Action |
|------|------------------|
| Merge SFT | `python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v8` |
| Merge SFT+ORPO | `python training/merge_adapters.py --lora-path models/qwen3.5-4b-bible-John-v8-orpo` |
| GGUF | `convert_hf_to_gguf.py ...-merged --outtype f16` |
| Quantize | `llama-quantize ... q4_k_m` |
| Modelfile | `python deployment/pc/generate_modelfile.py [--gguf ...]` |
| Ollama | `ollama create bible-assistant-orpo -f deployment/pc/Modelfile` |
| A/B F16 | `generate_modelfile.py --gguf ...-orpo-f16.gguf` → `ollama create bible-assistant-orpo-f16 ...` |
| Eval | `python training/evaluate.py` |
