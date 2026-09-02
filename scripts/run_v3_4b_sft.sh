#!/usr/bin/env bash
# Full 1-epoch v3 SFT for Qwen3.5-4B (bf16 LoRA) on data/processed/train_v3.json.
# Launch detached:  tmux new-session -d -s v3sft 'bash ~/bible-ai-assistant/scripts/run_v3_4b_sft.sh'
# Watch:            tail -f ~/bible-ai-assistant/logs/v3sft_*.log
set -uo pipefail
cd "$HOME/bible-ai-assistant"
mkdir -p logs
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/v3sft_${TS}.log"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate bible-orpo
export WANDB_MODE=offline
# Reduce allocator fragmentation from the optimizer/eval allocations near the 16 GB ceiling.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== v3 4B SFT start $(date -Is) -> $LOG ===" | tee "$LOG"
python training/train_unsloth.py \
  --config training/config.v3-4b.yaml \
  --run-name qwen3.5-4b-bible-v3-sft \
  --no-wandb >> "$LOG" 2>&1
RC=$?
echo "=== v3 4B SFT done rc=$RC $(date -Is) ===" | tee -a "$LOG"
echo "EXIT_${RC}" >> "$LOG"
