#!/usr/bin/env python3
"""
Generate deployment/pc/Modelfile from prompts/system_prompt.txt and the GGUF path.
Run from project root (default GGUF is Q4 for faster Ollama import; use --gguf for F16 A/B):
  python deployment/pc/generate_modelfile.py
  python deployment/pc/generate_modelfile.py --gguf qwen3.5-4b-bible-John-v8-orpo-f16.gguf
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
# Default: Q4 ORPO GGUF — smaller/faster `ollama create` vs F16 (~8 GB gather step).
# For max quality / A/B, pass: --gguf qwen3.5-4b-bible-John-v8-orpo-f16.gguf
_DEFAULT_GGUF = "qwen3.5-4b-bible-John-v8-orpo-q4_k_m.gguf"
OUTPUT_PATH = Path(__file__).resolve().parent / "Modelfile"


def _resolve_gguf_arg(project_root: Path, gguf: str) -> Path:
    p = Path(gguf)
    if not p.is_absolute():
        p = (project_root / "models" / gguf).resolve()
    else:
        p = p.resolve()
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Write deployment/pc/Modelfile for Ollama.")
    parser.add_argument(
        "--gguf",
        type=str,
        default=_DEFAULT_GGUF,
        help=f"GGUF file name under models/ or absolute path (default: {_DEFAULT_GGUF})",
    )
    args = parser.parse_args()
    gguf_path = _resolve_gguf_arg(PROJECT_ROOT, args.gguf)
    if not gguf_path.is_file():
        raise FileNotFoundError(
            f"GGUF not found: {gguf_path}\n"
            f"Convert first: python ../llama.cpp/convert_hf_to_gguf.py models/...-merged --outfile models/{args.gguf}"
        )
    gguf_posix = gguf_path.as_posix()

    if not SYSTEM_PROMPT_PATH.is_file():
        raise FileNotFoundError(
            f"System prompt not found: {SYSTEM_PROMPT_PATH}\n"
            "Ensure prompts/system_prompt.txt exists in the project root."
        )
    system_text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    content = f'''# Generated from prompts/system_prompt.txt — do not edit by hand; re-run generate_modelfile.py
FROM {gguf_posix}

# ChatML template for Qwen; newline after assistant prompt per Qwen spec
TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """{system_text}"""

PARAMETER temperature 0.2
PARAMETER num_ctx 2048
PARAMETER num_predict 256
PARAMETER repeat_penalty 2.0
PARAMETER repeat_last_n 32
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "Q:"
PARAMETER stop "###"
'''
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
