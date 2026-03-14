#!/bin/bash
# Jetson Orin Nano Super: install llama.cpp and build with CUDA.
# Run on the Jetson. Assumes JetPack 6.2+ and model + chroma_db already transferred.

set -e
sudo apt update && sudo apt install -y cmake git build-essential
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
# Then install systemd service (llama_server.service) and Tailscale per deploy.md
