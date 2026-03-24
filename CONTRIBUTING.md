# Contributing to Bible AI Assistant

Thank you for your interest in contributing to the Bible AI Assistant project. This document covers everything you need to get your development environment running, understand the project conventions, and submit quality contributions.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Environment Setup](#development-environment-setup)
3. [Running Tests](#running-tests)
4. [Running Linting](#running-linting)
5. [Pre-commit Hooks](#pre-commit-hooks)
6. [Branch Naming](#branch-naming)
7. [Commit Message Style](#commit-message-style)
8. [Pull Request Process](#pull-request-process)
9. [Project Structure](#project-structure)
10. [Bible Data](#bible-data)
11. [Reporting Bugs](#reporting-bugs)
12. [Do Not Contribute](#do-not-contribute)

---

## Prerequisites

Before setting up the project, make sure you have the following installed and available on your PATH:

- **Python 3.10 or higher** (3.11 recommended)
- **conda** (Miniconda or Anaconda) — used to manage the isolated Python environment
- **Ollama** — required to run the local LLM backend for inference and RAG responses
  - Install from [https://ollama.com](https://ollama.com)
  - Pull the required model: `ollama pull qwen2.5:7b` (or whichever model is configured)
- **Git** — for version control and submitting pull requests
- **Git LFS** (optional) — only needed if working with large binary assets

---

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/bible-ai-assistant.git
cd bible-ai-assistant
```

### 2. Create the Conda Environment

```bash
conda create -n bible-ai python=3.11 -y
conda activate bible-ai
```

### 3. Install the Package with All Dev Extras

```bash
pip install -e ".[rag,ui,dev]"
```

This installs the project in editable mode along with all optional dependency groups:
- `rag` — ChromaDB, sentence-transformers, BM25, FastAPI, slowapi, prometheus instrumentation
- `ui` — Gradio web interface + Faster-Whisper STT + Kokoro TTS
- `dev` — pytest, hypothesis, ruff, pre-commit, bandit, and other developer tools

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This registers the hooks defined in `.pre-commit-config.yaml` so they run automatically before each commit.

### 5. Verify the Setup

```bash
python -c "import rag; print('RAG module OK')"
ollama list
```

---

## Running Tests

### Run the Full Test Suite

```bash
pytest tests/
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=rag --cov=training --cov-report=term-missing --cov-report=html
```

The HTML coverage report is written to `htmlcov/index.html`.

### Run a Specific Test File or Test

```bash
pytest tests/test_rag_api.py
pytest tests/test_rag_api.py::test_verse_lookup_calls_retrieve
```

### Test Markers

Some tests require Ollama to be running locally and are marked accordingly. To skip them:

```bash
pytest tests/ -m "not requires_ollama"
```

---

## Running Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

### Check for Lint Errors

```bash
ruff check .
```

### Auto-fix Lint Errors

```bash
ruff check . --fix
```

### Check Formatting (without modifying files)

```bash
ruff format --check .
```

### Apply Formatting

```bash
ruff format .
```

Ruff is configured in `pyproject.toml` under `[tool.ruff]`. Do not introduce separate `.flake8` or `black` configuration — Ruff is the single source of truth for style.

---

## Pre-commit Hooks

Pre-commit hooks run automatically when you `git commit`. They enforce code quality before anything reaches the repository.

### Install Hooks

```bash
pre-commit install
```

### Run All Hooks Manually (against all files)

```bash
pre-commit run --all-files
```

### Hooks Configured

| Hook | Purpose |
|------|---------|
| `ruff` | Lint and auto-fix Python code |
| `ruff-format` | Format Python code |
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with a newline |
| `check-yaml` | Validate YAML syntax |
| `check-json` | Validate JSON syntax |
| `check-merge-conflict` | Detect unresolved merge conflict markers |
| `detect-private-key` | Prevent accidental commit of private keys |
| `bandit` | Static security analysis for Python code |

If a hook fails during a commit, fix the reported issue, re-stage your changes, and commit again.

---

## Branch Naming

Use the following naming conventions for branches:

| Prefix | When to Use | Example |
|--------|-------------|---------|
| `feature/` | New functionality | `feature/verse-citation-formatter` |
| `fix/` | Bug fixes | `fix/rag-empty-context-crash` |
| `docs/` | Documentation only | `docs/model-card-update` |
| `refactor/` | Code restructuring with no behavior change | `refactor/rag-pipeline-split` |
| `chore/` | Maintenance, dependency bumps, CI changes | `chore/bump-chromadb-0.5` |
| `test/` | Adding or fixing tests | `test/add-training-unit-tests` |

Branch names should be lowercase and use hyphens, not underscores or spaces.

---

## Commit Message Style

This project follows [Conventional Commits](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <short description>

[optional body]

[optional footer: e.g., Closes #42]
```

### Types

| Type | When to Use |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `chore` | Build process, dependency, or tooling changes |
| `perf` | Performance improvements |
| `style` | Formatting changes that do not affect logic |

### Examples

```
feat(rag): add BM25 hybrid search alongside vector retrieval

fix(ui): prevent crash when Ollama is not running on startup

docs(model-card): add quantization benchmark results table

test(training): add unit tests for preference data builder

chore(deps): bump sentence-transformers to 3.1.0
```

Keep the subject line under 72 characters. Use the body to explain the "why" when the change is not self-evident.

---

## Pull Request Process

1. **One concern per PR.** Do not bundle unrelated changes. A PR that fixes a bug and adds a new feature will be asked to split.

2. **Tests are required.** New behavior must have tests. Bug fixes should include a regression test that would have caught the original issue.

3. **CI must pass.** All GitHub Actions checks (lint, tests, security scan) must be green before a PR can be merged.

4. **Link related issues.** If your PR closes or relates to a GitHub Issue, include it in the PR description:
   ```
   Closes #42
   Related to #38
   ```

5. **Keep the diff focused.** Avoid reformatting files that are unrelated to your change. If you want to do bulk formatting, open a separate `style/` PR.

6. **Write a clear PR description.** Include:
   - What the change does
   - Why it is needed
   - How it was tested
   - Any known limitations or follow-up work

7. **Respond to review comments** within a reasonable time. If you need more time or disagree with a comment, say so — do not leave reviews unanswered.

---

## Project Structure

```
bible-ai-assistant/
├── rag/
│   ├── rag_server.py       # FastAPI app: routes, auth, rate limiting, middleware
│   ├── helpers.py          # Pure string/regex helpers (no I/O — fully unit-tested)
│   ├── retrieval.py        # Hybrid retrieval pipeline (dense + BM25 + RRF + rerank)
│   ├── settings.py         # Pydantic-validated config (reads from env / .env)
│   ├── response_cleanup.py # Thinking-block and repetition stripping
│   └── build_index.py      # ChromaDB index builder (run once)
├── training/               # SFT + ORPO training scripts and config
│   ├── train_unsloth.py    # Stage 1: supervised fine-tuning (requires GPU)
│   ├── train_orpo.py       # Stage 2: ORPO preference alignment (requires GPU)
│   ├── merge_adapters.py   # Merge LoRA adapters into full model for export
│   ├── build_preference_data.py
│   └── evaluate.py         # LLM-as-judge + keyword benchmark runner
├── ui/                     # Gradio 6 web interface (text + voice)
│   └── app.py
├── voice/                  # STT (Faster-Whisper) + TTS (Kokoro)
├── tests/                  # All tests (183 tests, 55% coverage)
├── deployment/             # PC, Jetson, and VPS Dockerfiles + configs
├── docs/                   # Guides, architecture, model card, training results
├── scripts/                # Benchmarking, leaderboard, utility scripts
├── benchmarks/             # Versioned evaluation protocol
├── prompts/                # System prompt + 54-question eval suite
└── pyproject.toml          # Project metadata, extras, tool config (ruff, bandit, pytest)
```

Add new modules in the appropriate top-level package. Avoid creating ad-hoc scripts at the project root — place them in `scripts/` instead.

---

## Bible Data

The Bible JSON data files are **not included in this repository** for size and licensing reasons.

To obtain the data:

1. **Use the download script:**
   ```bash
   python scripts/download_bible_data.py
   ```
   This fetches the World English Bible (WEB) in JSON format from a public domain source and places it in `data/`.

2. **Manually download a public domain translation:**
   - **KJV** (King James Version) — public domain in the US
   - **WEB** (World English Bible) — public domain worldwide, preferred
   - Place the JSON file at `data/web.json` (or as configured in `pyproject.toml`)

3. **Index the data into ChromaDB:**
   ```bash
   python scripts/index_bible.py
   ```

Do not commit Bible JSON files, ChromaDB collections, or any third-party translation that is not explicitly public domain.

---

## Reporting Bugs

Open a [GitHub Issue](https://github.com/your-org/bible-ai-assistant/issues) and include:

- **Steps to reproduce** — the exact commands or actions that trigger the bug
- **Expected behavior** — what should have happened
- **Actual behavior** — what actually happened, including full error output
- **Operating system** — e.g., Windows 11, Ubuntu 22.04, macOS 14
- **Python version** — `python --version`
- **Ollama version** — `ollama --version`
- **Conda environment** — output of `conda list | grep -E "torch|chromadb|langchain|sentence"`

For security vulnerabilities, do **not** open a public issue. See [SECURITY.md](SECURITY.md).

---

## Do Not Contribute

To keep the repository clean and safe, **never commit the following**:

- **API keys, tokens, or secrets** of any kind (OpenAI, HuggingFace, etc.)
- **Model weights** — `.bin`, `.safetensors`, `.gguf`, or any checkpoint files
- **ChromaDB data directories** — the vector store is built locally from source data
- **Bible JSON source files** — download these via the provided script
- **Personal files** — notes, scratch pads, IDE workspace files (`.vscode/`, `.idea/`)
- **`.env` files** — use `.env.example` as a template; never commit the real `.env`
- **Large binaries** — audio files, video files, datasets over a few KB

If you accidentally stage a secret or large file, remove it before committing and consider rotating the secret immediately.
