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
	$(info )
	$(info Bible AI Assistant - available commands)
	$(info ----------------------------------------)
	$(info   demo           Start full stack via Docker Compose)
	$(info   demo-build     Rebuild Docker images then start)
	$(info   down           Stop and remove all containers)
	$(info   logs           Stream logs from all running containers)
	$(info   model          Pull the Ollama model)
	$(info   index          Build the ChromaDB vector index)
	$(info   install        Install all dependencies into active env)
	$(info   test           Run the full test suite with coverage)
	$(info   lint           Lint and auto-fix with Ruff)
	$(info   security       CVE scan + SAST)
	$(info   ci             Full local CI pipeline)
	$(info )
	@:

# ── Demo (Docker) ─────────────────────────────────────────────────────────────
.PHONY: check-docker
check-docker:
	@docker info > /dev/null 2>&1 || (echo "ERROR: Docker is not running. Start Docker Desktop first." && exit 1)

.PHONY: demo
demo: check-docker ## Start full stack via Docker Compose (RAG server + Gradio UI)
	$(DOCKER_COMP) up

.PHONY: demo-build
demo-build: check-docker ## Rebuild Docker images then start (use after code changes)
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
