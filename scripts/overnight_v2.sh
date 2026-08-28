#!/usr/bin/env bash
# Overnight v2 driver — GPU-gated, fail-fast, measurement-first.
#
# It does NOT launch a full multi-day SFT. It runs, in order:
#   1. preflight (no GPU)            — data/config/disk sanity, abort on any failure
#   2. wait for the GPU to go idle   — so it can be launched while the box is gaming
#   3. 9B SFT smoke (--max-steps 2)  — proves the exact invocation loads + steps
#   4. 9B SFT PROBE (--max-steps N)  — a short run whose only job is to be evaluated
#                                      for reasoning retention before a full run is booked
#   5. merge probe adapter -> GGUF Q4 — ready for `ollama create` + eval in the morning
#
# Launch (survives the terminal closing; .wslconfig already has vmIdleTimeout=-1):
#   tmux new-session -d -s v2 '~/bible-ai-assistant/scripts/overnight_v2.sh'
#   tmux attach -t v2        # to watch
#
# Env: conda bible-orpo. Logs: ~/bible-ai-assistant/logs/overnight_<ts>.log
set -uo pipefail

REPO="$HOME/bible-ai-assistant"
cd "$REPO"
TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs checkpoints_v2_9b models
LOG="$REPO/logs/overnight_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== overnight_v2 @ $(date -Is) — log $LOG ==="

CONFIG="training/config.v2-9b.yaml"
RUN_SMOKE="qwen3.5-9b-bible-v2-smoke"
RUN_PROBE="qwen3.5-9b-bible-v2-probe"
PROBE_STEPS="${PROBE_STEPS:-600}"        # ~0.15 epoch of 61k @ effective-batch 16
GPU_IDLE_UTIL="${GPU_IDLE_UTIL:-15}"     # %
GPU_IDLE_MEM_MIB="${GPU_IDLE_MEM_MIB:-2000}"
GPU_WAIT_MAX_MIN="${GPU_WAIT_MAX_MIN:-720}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate bible-orpo

die() { echo "ABORT: $*" ; exit 1 ; }

# ---------------------------------------------------------------- 1. preflight
echo "--- preflight ---"
python - <<'PY' || exit 1
import json, sys, shutil, pathlib
root = pathlib.Path.home() / "bible-ai-assistant"
tv = root / "data/processed/train_v2.json"
assert tv.exists(), f"missing {tv} — run training/build_dataset_v2.py"
d = json.load(open(tv))
assert isinstance(d, list) and len(d) > 50_000, f"train_v2.json wrong shape/size: {type(d).__name__} {len(d) if hasattr(d,'__len__') else '?'}"
assert all(k in d[0] for k in ("messages", "category")), "train_v2.json rows missing messages/category"
free_gb = shutil.disk_usage(root).free / 2**30
assert free_gb > 60, f"only {free_gb:.0f} GB free — need headroom for a 9B checkpoint + GGUF"
import yaml
cfg = yaml.safe_load((root / "training/config.v2-9b.yaml").read_text())
assert cfg["model"]["name"] == "Qwen/Qwen3.5-9B", cfg["model"]["name"]
print(f"  train_v2.json: {len(d)} rows | free: {free_gb:.0f} GB | base: {cfg['model']['name']}  OK")
PY
[ $? -eq 0 ] || die "preflight failed"

python -c "import torch; assert torch.cuda.is_available(); assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list(); print('  torch', torch.__version__, 'sees', torch.cuda.get_device_name(0))" || die "torch/CUDA/sm_120 check failed"

# ---------------------------------------------------------------- 2. GPU gate
echo "--- waiting for GPU to go idle (util<${GPU_IDLE_UTIL}% & mem<${GPU_IDLE_MEM_MIB}MiB), max ${GPU_WAIT_MAX_MIN} min ---"
deadline=$(( $(date +%s) + GPU_WAIT_MAX_MIN*60 ))
while :; do
  read -r U M <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')"
  if [ "${U:-100}" -lt "$GPU_IDLE_UTIL" ] && [ "${M:-99999}" -lt "$GPU_IDLE_MEM_MIB" ]; then
    echo "  GPU idle (util ${U}%, mem ${M}MiB) — proceeding at $(date -Is)"; break
  fi
  [ "$(date +%s)" -ge "$deadline" ] && die "GPU still busy after ${GPU_WAIT_MAX_MIN} min (util ${U}%, mem ${M}MiB)"
  sleep 120
done

export WANDB_MODE="${WANDB_MODE:-offline}"   # no interactive login overnight
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---------------------------------------------------------------- 3. SFT smoke
echo "--- 9B SFT smoke (--max-steps 2) @ $(date -Is) ---"
PYTHONPATH=. python training/train_unsloth.py \
  --config "$CONFIG" --run-name "$RUN_SMOKE" --no-wandb --max-steps 2 \
  2>&1 | tee logs/smoke_${TS}.log | tail -40
SMOKE_RC=$?
grep -qiE "traceback|CUDA out of memory|could not|no module named" logs/smoke_${TS}.log && die "smoke run reported errors — see logs/smoke_${TS}.log"
[ $SMOKE_RC -eq 0 ] || die "smoke run rc=$SMOKE_RC"
echo "  smoke OK — the 9B invocation loads, tokenizes and steps"

# ---------------------------------------------------------------- 4. SFT probe
# Bounded on purpose: the v2 corpus is 99.9% verse-recall (see the mix warning in
# V2_EXECUTION_PLAN.md). This probe exists to be EVALUATED for reasoning retention
# before a full multi-hour SFT is booked. It is NOT the final model.
echo "--- 9B SFT probe (--max-steps $PROBE_STEPS) @ $(date -Is) ---"
PYTHONPATH=. python training/train_unsloth.py \
  --config "$CONFIG" --run-name "$RUN_PROBE" --max-steps "$PROBE_STEPS" \
  2>&1 | tee logs/probe_${TS}.log | tail -60
PROBE_RC=$?
echo "  probe rc=$PROBE_RC"
[ $PROBE_RC -eq 0 ] || die "probe run failed — see logs/probe_${TS}.log"

ADAPTER="models/${RUN_PROBE}"
[ -d "$ADAPTER" ] || ADAPTER="checkpoints_v2_9b"
echo "--- merge adapter ($ADAPTER) -> full model @ $(date -Is) ---"
PYTHONPATH=. python training/merge_adapters.py --lora-path "$ADAPTER" 2>&1 | tail -20 || die "merge failed"

MERGED="$(ls -dt models/*merged* 2>/dev/null | head -1)"
echo "--- GGUF Q4 from $MERGED @ $(date -Is) ---"
LC="$HOME/wsl41361/llama.cpp"
if [ -f "$LC/convert_hf_to_gguf.py" ] && [ -n "$MERGED" ]; then
  python "$LC/convert_hf_to_gguf.py" "$MERGED" --outfile "models/${RUN_PROBE}-f16.gguf" --outtype f16 2>&1 | tail -10
  if [ -x "$LC/build/bin/llama-quantize" ]; then
    "$LC/build/bin/llama-quantize" "models/${RUN_PROBE}-f16.gguf" "models/${RUN_PROBE}-q4_k_m.gguf" Q4_K_M 2>&1 | tail -5
  else
    echo "  (llama-quantize not built — F16 GGUF only; build it or quantize in the morning)"
  fi
else
  echo "  (llama.cpp convert script or merged model missing — do GGUF conversion manually)"
fi

echo "=== overnight_v2 DONE @ $(date -Is) ==="
echo "next: ollama create ... -f <Modelfile>  then  python scripts/run_benchmark.py --ollama-model <name> --judge"
