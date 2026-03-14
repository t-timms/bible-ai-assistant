#!/usr/bin/env python3
"""
Build Bible Q&A dataset for fine-tuning.
Reads from data/raw/, outputs data/processed/train.json in Qwen3 chat format.
See data/README.md and guide Section 8 for schema and data sources.
"""
from pathlib import Path

# TODO: Implement loading of raw Bible JSON/CSV (e.g. scrollmapper format).
# TODO: Generate Q&A pairs: verse lookup, cross-refs, theology, character studies,
#       constitution-testing, uncertainty examples. Inject system prompt into each example.
# TODO: Save as JSONL or JSON with "messages" array per example.
# Target: 30k–50k examples for first run.

def main() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
    # Placeholder: add real implementation
    raise NotImplementedError(
        "Dataset builder not yet implemented. Add raw data to data/raw/ and implement "
        "loading + Q&A generation. See data/README.md and guide Section 8."
    )


if __name__ == "__main__":
    main()
