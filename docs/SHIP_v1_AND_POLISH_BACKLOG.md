# Ship v1 + polish backlog

**Purpose:** Close the loop after long training runs without losing track of follow-ups.  
**v1 = “I can run it locally, reproduce deploy, and describe what it is.”** Everything else is polish.

---

## What “finished” means for v1 (do these, then breathe)

| # | Done when… | Your notes |
|---|------------|------------|
| 1 | **Artifacts named:** merged HF dir, GGUF(s), Ollama model names (`bible-assistant-orpo`, `bible-assistant-orpo-f16`, etc.) | Paths under `models/` (not in git); names in `deployment/pc/README.md` and `POST_TRAINING_CHECKLIST.md` |
| 1b | **User-visible cleanup:** RAG + eval use `rag/response_cleanup.py` (`strip_model_thinking`) | Implemented; restart RAG after pulls |
| 2 | **One command path works:** RAG on 8081 + Ollama + smoke Q&A (UI or `curl` / `ollama run`) | `uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081` + API smoke (e.g. John 3:16) |
| 3 | **Training provenance:** W&B run link or `wandb/` run id, `config.yaml` snapshot or path, SFT + ORPO run dates | Example: run id `ey7znpz7` (2026-03-19) — see `docs/training_results/2026-03-19-run-notes.md`; open runs on [wandb.ai](https://wandb.ai) project **bible-ai** |
| 4 | **Repo state:** commit hash (or tag `v1-orpo` / similar) that matches deployed weights | Set after `git push`; optional: `git tag v1-orpo` |
| 5 | **Post-training doc touched:** `docs/training_results/POST_TRAINING_CHECKLIST.md` — check off what you actually ran | Use checklist merge → GGUF → Ollama → eval rows |

You do **not** need full LLM-judge benchmarks to call v1 done.

---

## Minimal validation (fast, before you shelve it)

Pick **one** — enough to trust nothing is catastrophically broken:

- [ ] `python scripts/run_benchmark.py --label sanity --ollama-model bible-assistant-orpo` (keyword, no judge), **or**
- [x] 5–10 manual questions you care about (John 3:16, one refusal, one topical), logged in a note below.

**Sanity log (paste results or “pass”):**

```
Date: 2026-03-17
Model: bible-assistant-orpo (via RAG 8081)
Notes: OpenAI-compatible /v1/chat/completions returns clean message.content (no </think> leakage) for John 3:16 smoke test.
```

---

## Polish backlog (do later — don’t block v1)

| Priority | Item | Why later |
|----------|------|-----------|
| P1 | Further reduce **thinking** leaks if new formats appear (extend `rag/response_cleanup.py`) | After user testing |
| P1 | **LLM-as-judge** full 54-Q run + `compare_benchmark_runs.py` | Hours of runtime; nice for A/B Q4 vs F16 |
| P2 | **Gradio / voice** pass (STT/TTS, latency) | Product layer |
| P2 | **More SFT/ORPO data** for observed failure modes | After you have real user traces |
| P3 | **Jetson / edge** notes | Separate deployment project |
| P3 | API **auth**, rate limits, logging | Production hardening |

Add rows as you discover issues.

---

## Intentionally skipped for v1

- [x] Full judge benchmark (optional follow-up)
- [ ] (add anything else you decided not to do now)

---

## One-liner “what this project is” (for README or friends)

**Bible AI Assistant** is a **Qwen3.5-4B** model fine-tuned with **bf16 LoRA (SFT)** and optional **ORPO** alignment, served through **Ollama** (GGUF). A **hybrid RAG** stack (ChromaDB dense + BM25 + reranking) on **~31k verses** grounds answers; a **FastAPI** server on **port 8081** exposes an OpenAI-compatible chat API and strips model chain-of-thought before returning text to clients.

---

## Related docs

- `docs/training_results/POST_TRAINING_CHECKLIST.md` — merge → GGUF → Ollama → smoke
- `docs/BENCHMARK_PROTOCOL.md` — when you return to judged eval
- `deployment/pc/README.md` — PC deploy details
- `CHANGELOG.md` — release-level technical notes
