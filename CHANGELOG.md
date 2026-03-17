# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for milestone releases.

## [Unreleased]

### Changed

- Step 11 (WALKTHROUGH): Default to v3 model paths (qwen3-4b-bible-John-v3-merged, v3-f16.gguf, v3-q4_k_m.gguf); add `--outtype f16` to convert command; fix quantize command to use v3 paths; add troubleshooting for `llama-quantize` location (`build\bin\Release\` vs `build\Release\` on Windows)
- `generate_modelfile.py`: Default GGUF path set to `qwen3-4b-bible-John-v3-q4_k_m.gguf`

## [0.1.0] - YYYY-MM-DD

### Added

- Project scaffold and repository structure
- Biblical Constitution (CONSTITUTION.md) and system prompt
- .gitignore, .env.example, requirements.txt
- README and docs/architecture.md
- Placeholder directories and READMEs for data, training, rag, voice, deployment, ui
- Development workflow guide (docs/DEVELOPMENT_WORKFLOW.md)

### Notes

- Base model download and environment setup are the next steps (Section 6–7 of the guide).
