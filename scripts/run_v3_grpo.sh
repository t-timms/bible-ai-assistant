#!/bin/bash
# v3 GRPO light probe — 150 steps, 1500 prompts, citation reward, from the v3 SFT adapter.
# Launch:  tmux new-session -d -s v3grpo 'bash ~/bible-ai-assistant/scripts/run_v3_grpo.sh'
# Watch:   tail -f ~/bible-ai-assistant/logs/v3grpo_*.log
set -uo pipefail
cd "$HOME/bible-ai-assistant"
mkdir -p logs
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/v3grpo_${TS}.log"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate bible-orpo
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== v3 GRPO start $(date -Is) -> $LOG ===" | tee "$LOG"
PYTHONPATH=. python training/train_grpo.py \
  --policy-path models/qwen3.5-4b-bible-v3-sft \
  --config training/config.v2.yaml \
  --data data/processed/train_v3.json \
  --corpus data/raw/bible_web.json \
  --run-name qwen3.5-4b-bible-v3-grpo \
  --max-steps 150 --limit-prompts 1500 --no-wandb >> "$LOG" 2>&1
RC=$?
echo "=== v3 GRPO done rc=$RC $(date -Is) ===" | tee -a "$LOG"
echo "EXIT_${RC}" >> "$LOG"
