# Documentation

This folder contains the main guides and reference docs for the Bible AI Assistant project.

## Entry point

- **[WALKTHROUGH.md](WALKTHROUGH.md)** — Step-by-step walkthrough (Steps 1–12). Follow this in order for environment setup, dataset build, fine-tuning, GGUF/Ollama, and later phases (RAG, voice, deployment). **Start here** if you are building the project from scratch.

## Workflow and phases

- **[DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)** — Phase-gated workflow with version tags (v0.1.0 → v0.4.0+). Use with the walkthrough for milestones and checkpoints.

## Project status and shipping

- **[PROJECT_STATUS_AND_GOALS.md](PROJECT_STATUS_AND_GOALS.md)** — What we've done, what's left, and the end goal.
- **[SHIP_v1_AND_POLISH_BACKLOG.md](SHIP_v1_AND_POLISH_BACKLOG.md)** — Close v1 after training; backlog for later polish (judge eval, UI, etc.).
- **[../CHANGELOG.md](../CHANGELOG.md)** — Release-style technical changelog.

## Reference

- **[ENVIRONMENT_REQUIREMENTS.md](ENVIRONMENT_REQUIREMENTS.md)** — When conda envs are required (SFT, ORPO, merge) vs not (GGUF, Ollama, eval). Quick reference for each pipeline stage.
- **[ORPO_TWO_ENV_SETUP.md](ORPO_TWO_ENV_SETUP.md)** — SFT and ORPO in `bible-orpo` (transformers 5.x) for Qwen3.5-4B.
- **[QWEN35_MERGE_AND_DEPLOYMENT.md](QWEN35_MERGE_AND_DEPLOYMENT.md)** — Merge architecture, diagnostics, and corrected pipeline.
- **[deployment/pc/README.md](../deployment/pc/README.md)** — Ollama Modelfile, Q4 vs F16, A/B testing, thinking / cleanup notes.
- **[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)** — Versioned eval suite, `run_benchmark.py`, A/B comparison.
- **[architecture.md](architecture.md)** — Architecture and phase-by-phase deployment.
- **[evaluation_results.md](evaluation_results.md)** — Evaluation run results.
- **[OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)** — Strategy to maximize domain scores.
- **[CODEBASE_AUDIT.md](CODEBASE_AUDIT.md)** — Industry-standard audit and improvement roadmap.
- **[EVALUATION_CHECKLIST.md](EVALUATION_CHECKLIST.md)** — Score-maximizing checklist and leaderboard.
- **[PORTFOLIO.md](PORTFOLIO.md)** — Elevator pitch, resume bullets, and interview talking points.
- **[SERVING_GOD.md](SERVING_GOD.md)** — Design principles for serving God: boundaries, humility, community, multi-translation roadmap.
- **[training_results/README.md](training_results/README.md)** — Training run notes and post-training checklist links.

## File layout

```
docs/
├── README.md                      ← You are here
├── WALKTHROUGH.md                 ← Main ordered guide (Steps 1–12)
├── DEVELOPMENT_WORKFLOW.md        ← Phase checklist and version tags
├── PROJECT_STATUS_AND_GOALS.md    ← Status and roadmap
├── SHIP_v1_AND_POLISH_BACKLOG.md  ← v1 closure + polish table
├── BENCHMARK_PROTOCOL.md          ← Benchmarks and judge eval
├── EVALUATION_CHECKLIST.md
├── ENVIRONMENT_REQUIREMENTS.md
├── ORPO_TWO_ENV_SETUP.md
├── QWEN35_MERGE_AND_DEPLOYMENT.md
├── architecture.md
├── evaluation_results.md
├── OPTIMIZATION_PLAN.md
├── CODEBASE_AUDIT.md
├── PORTFOLIO.md
├── SERVING_GOD.md
├── benchmark_runs/                ← Versioned benchmark outputs (optional)
└── training_results/
    ├── README.md
    ├── POST_TRAINING_CHECKLIST.md
    └── 2026-03-19-run-notes.md  ← Example W&B / run log template
```
