#!/usr/bin/env python3
"""
Export a merged HF model directory to GGUF (F16 + quantized) via llama.cpp.

Pipeline (audit T8 - previously this stage was manual/untracked):
  1. convert_hf_to_gguf.py   -> <model-dir-name>.f16.gguf
  2. llama-quantize          -> <model-dir-name>.<quant>.gguf
  3. sha256 of both artifacts printed (paste into release notes)
  4. Modelfile regenerated through deployment/pc/generate_modelfile.py

llama.cpp resolution order: $LLAMA_CPP_DIR, then PATH lookup of
llama-quantize walking up to a checkout containing convert_hf_to_gguf.py.
The repo does NOT vendor llama.cpp; when missing we print exact clone/build
commands instead of failing cryptically mid-export.

Usage:
  python training/export_gguf.py --model-dir models/qwen3.5-4b-bible-John-v8-merged
  python training/export_gguf.py --model-dir ... --quant q4_k_m --skip-modelfile
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONVERT_SCRIPT_NAME = "convert_hf_to_gguf.py"
_QUANTIZE_SEARCH_PATHS = (
    "build/bin/Release/llama-quantize.exe",
    "build/bin/llama-quantize.exe",
    "build/bin/llama-quantize",
    "llama-quantize.exe",
    "llama-quantize",
)

_LLAMA_CPP_SETUP_MSG = (
    "llama.cpp not found. The repo does not vendor it. Either set LLAMA_CPP_DIR "
    "to an existing checkout containing convert_hf_to_gguf.py, or create one:\n"
    "  git clone https://github.com/ggerganov/llama.cpp\n"
    "  cd llama.cpp\n"
    "  cmake -B build\n"
    "  cmake --build build --config Release\n"
    "then re-run with LLAMA_CPP_DIR pointing at the clone."
)


def _find_quantize_binary(root: Path) -> Path | None:
    for rel in _QUANTIZE_SEARCH_PATHS:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def find_llama_cpp(
    environ: Mapping[str, str] | None = None, which: Callable[[str], str | None] = shutil.which
) -> Path | None:
    """Locate a llama.cpp checkout (dir containing convert_hf_to_gguf.py), or None.

    Order: $LLAMA_CPP_DIR verbatim, then the parent dirs of a PATH-resolved
    llama-quantize binary (up to 3 levels).
    """
    environ = os.environ if environ is None else environ
    declared = (environ.get("LLAMA_CPP_DIR") or "").strip()
    if declared:
        root = Path(declared).expanduser()
        if (root / CONVERT_SCRIPT_NAME).is_file():
            return root
    hit = which("llama-quantize")
    if hit:
        binary = Path(hit).resolve()
        for ancestor in [binary.parent, *binary.parents][1:4]:
            if (ancestor / CONVERT_SCRIPT_NAME).is_file():
                return ancestor
    return None


def build_convert_command(
    python_exe: str, convert_script: Path, model_dir: Path, out_f16: Path
) -> list[str]:
    return [
        python_exe,
        str(convert_script),
        str(model_dir),
        "--outfile",
        str(out_f16),
        "--outtype",
        "f16",
    ]


def build_quantize_command(
    quantize_bin: Path, f16_path: Path, out_quantized: Path, quant: str = "q4_k_m"
) -> list[str]:
    return [str(quantize_bin), str(f16_path), str(out_quantized), quant]


def export_gguf(
    model_dir: Path,
    out_dir: Path,
    quant: str = "q4_k_m",
    python_exe: str | None = None,
    runner: Callable[..., object] = subprocess.run,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[Path, Path]:
    """Convert + quantize ``model_dir``; returns (f16_path, quantized_path).

    ``runner`` defaults to subprocess.run(check=True); tests inject a stub.
    """
    llama_cpp = find_llama_cpp(environ=environ, which=which)
    if llama_cpp is None:
        raise FileNotFoundError(_LLAMA_CPP_SETUP_MSG)
    convert_script = llama_cpp / CONVERT_SCRIPT_NAME
    quantize_bin = _find_quantize_binary(llama_cpp)
    if quantize_bin is None:
        raise FileNotFoundError(
            f"llama-quantize binary not found under {llama_cpp}. Build it first:\n"
            "  cmake -B build && cmake --build build --config Release"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = model_dir.name
    f16_path = out_dir / f"{stem}.f16.gguf"
    quantized_path = out_dir / f"{stem}.{quant}.gguf"
    exe = python_exe or sys.executable

    runner(
        build_convert_command(exe, convert_script, model_dir, f16_path),
        check=True,
    )
    print(f"Converted {model_dir} -> {f16_path}")
    runner(
        build_quantize_command(quantize_bin, f16_path, quantized_path, quant),
        check=True,
    )
    print(f"Quantized -> {quantized_path} ({quant})")

    from training.train_unsloth import sha256_file

    for artifact in (f16_path, quantized_path):
        print(f"sha256({artifact.name}) = {sha256_file(artifact)}")
    return f16_path, quantized_path


def regenerate_modelfile(gguf_path: Path) -> None:
    """Re-run deployment/pc/generate_modelfile.py against the new GGUF.

    Read-only reuse of its argparse main() keeps Modelfile generation in one
    place; we only patch sys.argv for the duration of the call.
    """
    try:
        from deployment.pc.generate_modelfile import main as generate_modelfile_main
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        from deployment.pc.generate_modelfile import main as generate_modelfile_main

    original_argv = sys.argv
    try:
        sys.argv = ["generate_modelfile", "--gguf", str(gguf_path)]
        generate_modelfile_main()
    finally:
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export merged HF model dir to GGUF (F16 + quantized) via llama.cpp."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Merged HF model directory (e.g. models/qwen3.5-4b-bible-John-v8-merged)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory for GGUF artifacts (default: models/)",
    )
    parser.add_argument("--quant", default="q4_k_m", help="llama-quantize scheme (default q4_k_m)")
    parser.add_argument(
        "--skip-modelfile",
        action="store_true",
        help="Do not regenerate the Modelfile after export",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model dir not found: {model_dir}. Merge adapters first (training/merge_adapters.py)."
        )

    _, quantized_path = export_gguf(model_dir, args.out_dir, args.quant)

    if not args.skip_modelfile:
        regenerate_modelfile(quantized_path)
        print(f"Modelfile regenerated for {quantized_path.name}")


if __name__ == "__main__":
    main()
