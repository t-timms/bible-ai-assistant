# Bible AI Assistant — Cursor Completion Checklist

Read `.cursorrules` before starting ANY task. Every file must comply with the architecture defined there.

---

## PHASE 1: Dataset (Priority: HIGH)
**Goal:** Build ~1,800 diverse Bible Q&A examples for fine-tuning (quality over quantity to avoid overfitting).

- [ ] Create `data/build_dataset.py`
  - Loads raw Bible text from `data/raw/` (JSON or CSV, all 66 books, KJV + NIV + ESV)
  - Generates Q&A pairs in these categories:
    - Verse lookup: "What does [Book Chapter:Verse] say?" → exact verse text with citation
    - Topical: "What does the Bible say about [topic]?" → relevant verses + brief explanation
    - Character: "Who was [biblical figure]?" → summary with supporting verses
    - Cross-reference: "How does [verse A] relate to [verse B]?" → comparison with citations
    - Context: "What is the context of [verse]?" → surrounding passage explanation
  - Output format: JSONL with `{"instruction": "...", "input": "", "output": "..."}`
  - Save to `data/processed/bible_qa_dataset.jsonl`
  - Target: ~1,800 diverse examples (verse lookups, thematic, cross-reference, meta, refusals)
  - Include train/val split (90/10) saved as separate files

- [ ] Create `data/raw/download_bible.py`
  - Downloads KJV full text (public domain) in structured JSON
  - Format: `[{"book": "Genesis", "chapter": 1, "verse": 1, "text": "..."}]`
  - Save to `data/raw/bible_kjv.json`

- [ ] Create `data/sample.json`
  - 10 example Q&A pairs showing the expected format
  - This file IS tracked in git (see .gitignore)

---

## PHASE 2: RAG Index (Priority: HIGH)
**Goal:** Index all 31,000+ Bible verses into ChromaDB.

- [ ] Verify `rag/build_index.py` exists and is correct:
  - Uses `nomic-ai/nomic-embed-text-v1.5` (NOT v1)
  - Uses `"search_document: "` prefix when embedding verses
  - Stores in `rag/chroma_db/`
  - Metadata per verse: book, chapter, verse number, translation
  - Batch size: 500 (ChromaDB default limit)
  - Run it: `python rag/build_index.py` — should index all verses

- [ ] Verify `rag/rag_server.py` augmented prompt is MINIMAL:
  ```python
  augmented = "Context:\n" + context + "\n\nQ: " + q
  ```
  - NO behavioral instructions in the augmented prompt
  - Uses `"search_query: "` prefix when embedding user question
  - Post-processing: strip `<think>` blocks, strip decorative characters
  - Returns via OpenAI-compatible `/v1/chat/completions` endpoint

---

## PHASE 3: Training Scripts (Priority: HIGH)
**Goal:** Fine-tune Qwen3.5-4B on the Bible dataset.

- [ ] Verify `training/train_unsloth.py`:
  - Model: `Qwen/Qwen3.5-4B`
  - Uses `FastModel` from unsloth (or `FastLanguageModel` — both work)
  - `bf16=True` (NEVER `fp16=True`)
  - LoRA config: `r=16`, `alpha=32`, `dropout=0.1`
  - Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  - Dataset loaded from `data/processed/train.json`
  - Chat template applied (Qwen3 uses ChatML `<|im_start|>` format)
  - W&B logging enabled
  - Saves checkpoints to `checkpoints/`
  - Training args: 3 epochs, batch_size=4, gradient_accumulation=4, lr=2e-4
  - Max sequence length: 2048

- [ ] Verify `training/merge_adapters.py`:
  - Loads base model + LoRA adapter from `checkpoints/`
  - Merges to full 16-bit model
  - Saves to `models/qwen3.5-4b-bible-John-vN-merged/`

- [ ] Create `training/quantize_gguf.sh` (or .bat for Windows):
  - Converts merged model to GGUF Q4_K_M using llama.cpp
  - Saves to `models/qwen3.5-4b-bible-John-vN-q4_k_m.gguf`
  - Command: `python llama.cpp/convert_hf_to_gguf.py models/qwen3.5-4b-bible-John-vN-merged/ --outtype q4_k_m --outfile models/qwen3.5-4b-bible-John-vN-q4_k_m.gguf`

---

## PHASE 4: Evaluation (Priority: HIGH — this makes it a portfolio piece)
**Goal:** Quantitative evaluation of the fine-tuned model.

- [ ] Create `prompts/evaluation_questions.json`:
  - 50 test questions across 5 categories:
    - 10 verse lookups (exact match expected)
    - 10 topical questions
    - 10 character questions
    - 10 cross-reference questions
    - 10 context questions
  - Each has: `{"question": "...", "expected_answer": "...", "category": "..."}`

- [ ] Create `training/evaluate.py`:
  - Loads the fine-tuned model (via Ollama API at localhost:11434 or direct)
  - Also loads base Qwen3.5-4B (untuned) for comparison
  - Runs all 50 questions through both models
  - Scores each response:
    - Verse accuracy: does the cited verse text match the actual verse?
    - Citation present: did it include Book Chapter:Verse?
    - Hallucination check: did it invent verses that don't exist?
    - Relevance: is the answer on-topic? (simple keyword overlap score)
  - Saves results to `docs/evaluation_results.json`
  - Prints summary table: category, fine-tuned score, base score, improvement

---

## PHASE 5: Skill Routing Fix (Priority: HIGH)
**Goal:** Ensure Bible questions route through the fine-tuned model, not the 27B brain.

- [ ] Verify `~/.openclaw/workspace/skills/bible-lookup/SKILL.md` exists with correct content:
  ```markdown
  # Bible lookup

  Answers Bible questions using a specialized fine-tuned
  model with verse-accurate RAG retrieval.

  ## When to use
  - User asks about a Bible verse
  - User asks a theological question
  - User asks "what does the Bible say about..."
  - User asks about a biblical figure or event

  ## How to use
  Make an HTTP request to the local Bible API:

  curl -X POST http://localhost:8081/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"bible-assistant","messages":[{"role":"user","content":"USER_QUESTION"}]}'

  Return the assistant's response content directly to the user.
  ```

- [ ] Verify `~/.openclaw/SOUL.md` (or workspace equivalent) says:
  ```
  # Bible AI Assistant
  You are a personal AI assistant with deep Bible knowledge.
  For Bible questions, use the bible-lookup skill.
  For everything else, use your tools directly.
  Be warm, clear, and helpful.
  ```

- [ ] Test: send "What does Proverbs 3:5 say?" in Telegram
  - Check RAG server terminal — it MUST show incoming request
  - If no RAG activity, the skill is not being invoked

---

## PHASE 6: Voice Pipeline Verification (Priority: MEDIUM)
**Goal:** Confirm STT → RAG → Bible model → TTS works end-to-end.

- [ ] Verify `ui/app.py`:
  - Text tab: sends question to `http://localhost:8081/v1/chat/completions`
  - Voice tab: records audio → Faster-Whisper `large-v3-turbo` transcribes → sends to RAG server → response text → Kokoro TTS on port 8880
  - Both tabs display the response text
  - Voice tab plays audio response

- [ ] Test sequence:
  1. Start Kokoro Docker: `docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest`
  2. Start Gradio: `python ui/app.py`
  3. Open http://localhost:7860
  4. Type a question in text tab — confirm response
  5. Speak a question in voice tab — confirm transcription, response, and audio playback

---

## PHASE 7: GitHub Repository (Priority: HIGH)
**Goal:** Publishable portfolio piece.

- [ ] Create `README.md` with:
  - Project title and one-line description
  - Architecture diagram (Mermaid or image) showing:
    - Telegram → OpenClaw (27B brain) → Bible Skill → RAG Server → ChromaDB + Bible Model (4B)
    - Gradio UI → Whisper STT → RAG Server → Kokoro TTS
  - Tech stack list (one line each, no paragraphs)
  - Evaluation results summary table
  - Setup instructions (5 steps max)
  - Demo screenshot or GIF
  - Skills demonstrated section:
    - Fine-tuning (bf16 LoRA, Unsloth)
    - RAG (ChromaDB, nomic-embed-text-v1.5)
    - Constitutional AI alignment
    - Multi-model agent orchestration (OpenClaw)
    - Voice pipeline (Faster-Whisper + Kokoro TTS)
    - Edge deployment ready (Jetson Orin Nano Super)
    - Docker containerization
  - License: Apache 2.0

- [ ] Verify `.gitignore` excludes:
  - `*.gguf`, `*.safetensors`
  - `models/`, `checkpoints/`
  - `rag/chroma_db/`
  - `.env`, `__pycache__/`
  - `*.pyc`
  - Keeps: `!data/sample.json`

- [ ] Create `docs/ARCHITECTURE.md`:
  - Detailed explanation of the router pattern
  - Why constitutional AI approach was chosen
  - Model selection rationale
  - RAG design decisions

- [ ] Push to GitHub under `omnipotence-eth/bible-ai-assistant`

---

## PHASE 8: Demo Recording (Priority: HIGH)
**Goal:** 60-second video showing the full pipeline.

- [ ] Record screen capture showing:
  - Open Gradio UI
  - Speak a Bible question into the microphone
  - System transcribes (Whisper)
  - RAG retrieves relevant verses
  - Fine-tuned model generates response with citations
  - Kokoro TTS reads the response aloud
  - Total time: under 15 seconds end-to-end

- [ ] Upload to GitHub repo as `docs/demo.mp4` or link to YouTube

---

## PHASE 9: Jetson Deployment (Priority: LOW — do after job search stabilizes)

- [ ] Transfer `bible-qwen3-q4km.gguf` to Jetson Orin Nano Super
- [ ] Run via llama.cpp with `--ctx-size 16384 --cache-type-k q4_1 --cache-type-v q4_1`
- [ ] Point VPS OpenClaw at Jetson's Tailscale IP

---

## PHASE 10: VPS Always-On (Priority: LOW)

- [ ] Deploy OpenClaw to DigitalOcean $12/month droplet
- [ ] Configure Telegram bot to route through VPS
- [ ] VPS calls back to home PC (or Jetson) for Bible model inference

---

## CRITICAL RULES FOR ALL FILES:
1. Augmented RAG prompt is ONLY `"Context:\n" + context + "\n\nQ: " + q` — NO instructions
2. `bf16=True` everywhere — NEVER `fp16=True`
3. Embeddings use `nomic-embed-text-v1.5` with task prefixes
4. Bible model is accessed ONLY via RAG server on port 8081
5. OpenClaw 27B brain NEVER receives the system prompt directly
6. No decorative separators (═══) in any prompt — the model echoes them
