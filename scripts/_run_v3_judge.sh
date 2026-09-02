#!/usr/bin/env bash
# Protocol-v3 LLM-as-judge eval for v2-4b and v3-sft (v3-grpo is byte-identical
# to v3-sft on keyword scoring, so skipped). Judge = qwen3.5:27b via Ollama :11434.
# Results -> docs/benchmark_runs/*_judge.json
#
#   tmux new-session -d -s v3judge 'bash ~/bible-ai-assistant/scripts/_run_v3_judge.sh'
#   tail -f ~/bible-ai-assistant/logs/v3judge_*.log
set -uo pipefail
cd "$HOME/bible-ai-assistant"
mkdir -p logs
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/v3judge_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate bible-orpo

# label | merged model dir | served name
ROWS=(
  "v2-4b|models/qwen3.5-4b-bible-v2-merged|bible-v2-4b"
  "v3-sft|models/qwen3.5-4b-bible-v3-merged|bible-v3-sft"
)

echo "=== judge model check ==="
ollama list | grep -q 'qwen3.5:27b' || { echo "qwen3.5:27b not in ollama — abort"; exit 1; }

for row in "${ROWS[@]}"; do
  IFS='|' read -r LABEL MDIR SERVED <<< "$row"
  echo "======== $(date -Is)  $LABEL  ($MDIR) ========"
  [ -f "$MDIR/model.safetensors" ] || { echo "MISSING $MDIR — skipping"; continue; }

  TFLOG="logs/tfserver_judge_${LABEL}_${TS}.log"
  MODEL_PATH="$MDIR" PORT=8001 SERVED_NAME="$SERVED" \
    nohup python scripts/_tf_openai_server.py > "$TFLOG" 2>&1 &
  TF_PID=$!
  echo "tf-server pid=$TF_PID log=$TFLOG"

  ok=0
  for i in $(seq 1 120); do
    if grep -q "ready on :8001" "$TFLOG" 2>/dev/null; then ok=1; echo "tf ready after ${i}s"; break; fi
    kill -0 "$TF_PID" 2>/dev/null || { echo "tf-server died early"; tail -30 "$TFLOG"; break; }
    sleep 2
  done
  [ "$ok" -eq 1 ] || { echo "tf-server never ready for $LABEL — skipping"; kill "$TF_PID" 2>/dev/null; sleep 3; continue; }

  LABEL="$LABEL" SERVED="$SERVED" JUDGE=1 bash scripts/_run_v3_eval.sh || echo "judge eval rc=$? for $LABEL (continuing)"

  kill "$TF_PID" 2>/dev/null
  wait "$TF_PID" 2>/dev/null
  sleep 5
  echo "======== $(date -Is)  $LABEL done ========"
done

echo "=== v3 judge evals done $(date -Is) ==="
echo "compare: python scripts/compare_benchmark_runs.py docs/benchmark_runs/*_v2-4b_judge.json docs/benchmark_runs/*_v3-sft_judge.json"
echo "JUDGE_ALL_DONE"
