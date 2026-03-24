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
	$(info   demo           Start full stack - auto-starts Ollama if needed)
	$(info   demo-build     Rebuild Docker images then start)
	$(info   ollama         Start Ollama manually in this terminal)
	$(info   down           Stop and remove all containers)
	$(info   logs           Stream logs from all running containers)
	$(info   status         Show health of all running services)
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
.PHONY: _ollama-start
_ollama-start:
	@powershell -Command "\
		try { Invoke-RestMethod http://localhost:11434/api/tags | Out-Null; Write-Host 'Ollama already running' } \
		catch { Write-Host 'Starting Ollama...'; Start-Process ollama -ArgumentList serve -WindowStyle Hidden; Start-Sleep 3 }"

.PHONY: demo
demo: _ollama-start ## Start full stack — auto-starts Ollama if not already running
	$(DOCKER_COMP) up

.PHONY: demo-build
demo-build: _ollama-start ## Rebuild Docker images then start — auto-starts Ollama if needed
	$(DOCKER_COMP) up --build

.PHONY: down
down: ## Stop and remove all containers
	$(DOCKER_COMP) down

.PHONY: logs
logs: ## Stream logs from all running containers
	$(DOCKER_COMP) logs -f

# ── Ollama ────────────────────────────────────────────────────────────────────
.PHONY: ollama
ollama: ## Start the Ollama inference server (keep this terminal open)
	ollama serve

.PHONY: model
model: ## Pull the Ollama model (bible-assistant-orpo)
	ollama pull bible-assistant-orpo

# ── Model / Data ──────────────────────────────────────────────────────────────
.PHONY: index
index: ## Build the ChromaDB vector index (activate conda env first)
	build-index

# ── Status ────────────────────────────────────────────────────────────────────
.PHONY: status
status: ## Show health of all running services
	@docker compose ps

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
