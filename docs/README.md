# Documentation

This folder contains the main guides and reference docs for the Bible AI Assistant project.

## Entry point

- **[WALKTHROUGH.md](WALKTHROUGH.md)** — Step-by-step walkthrough (Steps 1–12). Follow this in order for environment setup, dataset build, fine-tuning, GGUF/Ollama, and later phases (RAG, voice, deployment). **Start here** if you are building the project from scratch.

## Workflow and phases

- **[DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)** — Phase-gated workflow with version tags (v0.1.0 → v0.4.0+). Use with the walkthrough for milestones and checkpoints.

## Project Status & Career

- **[PROJECT_STATUS_AND_GOALS.md](PROJECT_STATUS_AND_GOALS.md)** — What we've done, what's left, and the end goal.
- **[CAREER_AND_PORTFOLIO.md](CAREER_AND_PORTFOLIO.md)** — How this project helps your resume, portfolio, and career.
- **[INTERVIEW_PREP.md](INTERVIEW_PREP.md)** — How to speak to the project and study for interviews.

## Reference

- **[ENVIRONMENT_REQUIREMENTS.md](ENVIRONMENT_REQUIREMENTS.md)** — When conda envs are required (SFT, ORPO, merge) vs not (GGUF, Ollama, eval). Quick reference for each pipeline stage.
- **[ORPO_TWO_ENV_SETUP.md](ORPO_TWO_ENV_SETUP.md)** — Two-environment setup for ORPO (SFT in `bible-ai-assistant`, ORPO in `bible-orpo`). Required for Qwen3.5-4B preference alignment.
- **[architecture.md](architecture.md)** — Architecture and phase-by-phase deployment.
- **[evaluation_results.md](evaluation_results.md)** — Evaluation run results.
- **[training_results/README.md](training_results/README.md)** — Training run notes and screenshots.

## File layout

```
docs/
├── README.md                 ← You are here
├── WALKTHROUGH.md            ← Main ordered guide (Steps 1–12; all step details inline)
├── DEVELOPMENT_WORKFLOW.md   ← Phase checklist and version tags
├── ENVIRONMENT_REQUIREMENTS.md ← Conda env vs no-env by phase
├── ORPO_TWO_ENV_SETUP.md    ← Two-env setup for ORPO (Qwen3.5-4B)
├── architecture.md
├── evaluation_results.md
└── training_results/
    └── README.md
```
