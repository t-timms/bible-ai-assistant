#!/usr/bin/env python3
"""
Generate deployment/pc/Modelfile from prompts/system_prompt.txt and the GGUF path.
Run from project root: python deployment/pc/generate_modelfile.py
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
# Use absolute path so Ollama finds the local GGUF (relative FROM can be treated as a model name to pull)
GGUF_PATH = (PROJECT_ROOT / "models" / "qwen3-4b-bible-John-q4_k_m.gguf").resolve().as_posix()
OUTPUT_PATH = Path(__file__).resolve().parent / "Modelfile"


def main() -> None:
    system_text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    content = f'''# Generated from prompts/system_prompt.txt — do not edit by hand; re-run generate_modelfile.py
FROM {GGUF_PATH}

SYSTEM """{system_text}"""

PARAMETER temperature 0.2
PARAMETER num_ctx 2048
PARAMETER num_predict 256
'''
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
