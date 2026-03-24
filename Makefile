# ==============================================================================
# Bible AI Assistant — project shortcuts
# Usage:  make <target>
# Requires: GNU make (Git Bash, WSL, or `winget install GnuWin32.Make`)
# ==============================================================================

.DEFAULT_GOAL := help

DOCKER_COMP := docker compose

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Demo (Docker) ─────────────────────────────────────────────────────────────
.PHONY: demo
demo: ## Start full stack via Docker Compose (RAG server + Gradio UI)
	$(DOCKER_COMP) up

.PHONY: demo-build
demo-build: ## Rebuild Docker images then start (use after code changes)
	$(DOCKER_COMP) up --build

.PHONY: down
down: ## Stop and remove all containers
	$(DOCKER_COMP) down

.PHONY: logs
logs: ## Stream logs from all running containers
	$(DOCKER_COMP) logs -f

# ── Model / Data ──────────────────────────────────────────────────────────────
.PHONY: model
model: ## Pull and verify the Ollama model (bible-assistant-orpo)
	ollama pull bible-assistant-orpo

.PHONY: index
index: ## Build the ChromaDB vector index (activate conda env first)
	build-index

# ── Dev ───────────────────────────────────────────────────────────────────────
.PHONY: install
install: ## Install all dependencies into the active environment
	pip install -e ".[rag,ui,dev]"

.PHONY: test
test: ## Run the full test suite with coverage report
	pytest tests/ --cov=rag --cov-report=term-missing

.PHONY: lint
lint: ## Lint and auto-fix with Ruff (check + format)
	ruff check . --fix && ruff format .

.PHONY: security
security: ## CVE scan (pip-audit) + SAST (bandit)
	pip-audit
	bandit -r rag/ training/ scripts/

.PHONY: ci
ci: lint test security ## Run full local CI pipeline (lint → test → security)
