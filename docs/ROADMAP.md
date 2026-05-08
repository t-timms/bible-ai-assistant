# Bible AI Assistant — Remediation Roadmap

**Purpose:** Reference for coding agents. Completed blocks are marked. Each block is a self-contained MR.

---

## Block 1 — Configurable title ✅

Add `TITLE` env var to RAG Settings, `GRADIO_TITLE` env var to Gradio UI.

- `rag/settings.py`: `title` field (default `"Bible AI RAG Server"`)
- `rag/rag_server.py`: uses `settings.title`
- `ui/app.py`: `GRADIO_TITLE` env var (default `"Bible AI Assistant"`)
- Tests: default, env override, FastAPI integration

---

## Block 2 — README Quick Start fix (C-1)

Replace `requirements.txt` reference in README with `pip install -e ".[rag,ui,train,dev]"`.

---

## Block 3 — ORPO precision fix (C-3)

Set `load_in_4bit=False` in `train_orpo.py` to match SFT precision. If VRAM constrained, document the tradeoff.

---

## Block 4 — Preference pair diversity (C-4)

Expand `_build_verbose_pairs` and `_build_bible_for_everything_pairs` in `build_preference_data.py` — currently 28% of pairs repeat only 12 unique prompts.

---

## Block 5 — ORPO validation split (H-1)

Add `test_size=0.1` split and `eval_dataset` to ORPOTrainer. Add `eval_steps=20`.

---

## Block 6 — ORPO warmup fix (H-2)

Change `warmup_steps=20` to `warmup_steps=5` (~8% of 63 total steps).

---

## Block 7 — WANDB_PROJECT env var (H-3)

Replace hardcoded `"bible-ai"` with `os.getenv("WANDB_PROJECT", "bible-ai")` in `train_orpo.py`.

---

## Block 8 — LLM judge truncation (H-4)

Remove 1000-char truncation in `training/evaluate.py` or raise to 4000+.

---

## Block 9 — Remove time.sleep in eval (H-5)

Remove `time.sleep(0.5)` from `training/evaluate.py`. Use `asyncio.Semaphore` if throttling needed.

---

## Block 10 — Judge failure error (H-6)

Raise `RuntimeError` or emit `logging.error` + `"judge_available": false` when all judge endpoints fail.

---

## Block 11 — Local docker-compose (H-7)

Add `docker-compose.yml` for local dev (RAG server + Ollama). Add `start.sh` script.

---

## Block 12 — Quality polish (P2)

| ID | Action | File |
|----|--------|------|
| M-2 | Move `random.seed(42)` into `main()` | `build_preference_data.py` |
| M-3 | Align MAX_SEQ_LENGTH (2048) between SFT and ORPO | `train_orpo.py` |
| M-4 | Print warning when using default adapter path | `merge_adapters.py` |
| M-5 | Explain counter-intuitive hallucination rate | `docs/MODEL_COMPARISON.md` |
| M-6 | Remove personal documents from public repo | `docs/`, root |
| O-2 | Gate traceback on `APP_ENV` | `rag/rag_server.py` |
| O-3 | Structured logging at RAG pipeline stages | `rag/rag_server.py` |
| O-4 | Gitignore checkpoint README stubs | `.gitignore` |

---

## Iteration loop (from OPTIMIZATION_PLAN.md)

```
1. Deploy current model (vN)
2. Run evaluate.py --judge --model-tag vN
3. Find worst category + worst questions
4. Add training examples for those failure modes
5. Rebuild data -> train vN+1
6. (Optional) Run ORPO on vN
7. Deploy vN+1 and vN-orpo, re-eval
8. Update leaderboard; compare
9. Repeat from step 2
```

---

## See also

- `docs/CODEBASE_AUDIT.md` — full audit with severity ratings
- `docs/OPTIMIZATION_PLAN.md` — strategies for maximizing domain scores
- `docs/SHIP_v1_AND_POLISH_BACKLOG.md` — v1 completion checklist
- `docs/DEVELOPMENT_WORKFLOW.md` — phase-gated workflow
