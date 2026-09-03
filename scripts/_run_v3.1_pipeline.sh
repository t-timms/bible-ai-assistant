#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# V3.1 end-to-end pipeline — distill thematic_qa -> assemble -> SFT -> merge ->
# eval -> v4 rescore. Unattended, ~10-13 h GPU. Every stage is guarded and
# logs to logs/v3.1_pipeline_<ts>.log; a stage failure aborts with a nonzero
# EXIT_<n> marker so a morning check is unambiguous.
#
#   SMOKE:  SMOKE=1 bash scripts/_run_v3.1_pipeline.sh   # distill 20, skip SFT/eval — ~10 min, validates wiring
#   FULL :  tmux new-session -d -s v31 'bash ~/bible-ai-assistant/scripts/_run_v3.1_pipeline.sh'
#           tail -f ~/bible-ai-assistant/logs/v3.1_pipeline_*.log
#
# Preserves all v3 artifacts — writes only *.v3.1 / *-v3.1-* names.
# Prereqs (checked below): data/raw_v3/{distill_out,thematic_inputs}.jsonl,
#   ~/models/qwen3-14b-gguf/Qwen3-14B-Q5_K_M.gguf, ~/llama.cpp-full/build/bin/llama-server,
#   conda env bible-orpo, .venv-rag, rag/chroma_db/, Windows rebooted (7 h SFT).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$HOME/bible-ai-assistant" || exit 1
mkdir -p logs data/raw_v3 data/processed

SMOKE="${SMOKE:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/v3.1_pipeline_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

DATE_TAG="$(date +%Y%m%d)"
TEACHER="$HOME/models/qwen3-14b-gguf/Qwen3-14B-Q5_K_M.gguf"
LLAMA_SERVER="$HOME/llama.cpp-full/build/bin/llama-server"
THEMATIC_IN="data/raw_v3/thematic_inputs.jsonl"
THEMATIC_OUT="data/raw_v3/thematic_out.jsonl"
DISTILL_OUT="data/raw_v3/distill_out.jsonl"          # v3 regen categories — reused as-is
TRAIN_V31="data/processed/train_v3.1.json"
SFT_CONFIG="training/config.v3.1-4b.yaml"
RUN_NAME="qwen3.5-4b-bible-v3.1-sft"
ADAPTER_DIR="models/${RUN_NAME}"
MERGED_DIR="models/qwen3.5-4b-bible-v3.1-merged"
PARALLEL=4

die() { echo "!!! STAGE FAILED: $1"; echo "EXIT_${2:-1}"; exit "${2:-1}"; }
banner() { echo; echo "======== $(date -Is)  $1 ========"; }

# ── Preflight ───────────────────────────────────────────────────────────────
banner "preflight (SMOKE=$SMOKE)"
[ -f "$THEMATIC_IN" ]  || die "missing $THEMATIC_IN (run training/build_v3_thematic.py)" 10
[ -f "$DISTILL_OUT" ]  || die "missing $DISTILL_OUT (v3 regen distill output)" 10
[ -f "$TEACHER" ]      || die "missing teacher $TEACHER" 10
[ -x "$LLAMA_SERVER" ] || die "missing $LLAMA_SERVER" 10
[ -f "$SFT_CONFIG" ]   || die "missing $SFT_CONFIG" 10
[ -d rag/chroma_db ]   || die "missing rag/chroma_db" 10
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
df -h "$HOME" | tail -1
wc -l "$THEMATIC_IN"
for d in "$TRAIN_V31" "$ADAPTER_DIR" "$MERGED_DIR" "checkpoints_v3.1_4b"; do
  [ -e "$d" ] && echo "NOTE: $d already exists — will be overwritten by this run"
done

source ~/miniforge3/etc/profile.d/conda.sh

# ── Stage 1: teacher server ─────────────────────────────────────────────────
banner "stage 1 — llama-server (Qwen3-14B Q5_K_M) on :8001"
LLAMALOG="logs/v3.1_llama_${TS}.log"
"$LLAMA_SERVER" -m "$TEACHER" --host 127.0.0.1 --port 8001 \
  -ngl 99 -c 8192 -fa on --parallel "$PARALLEL" --no-webui > "$LLAMALOG" 2>&1 &
LLAMA_PID=$!
echo "llama-server pid=$LLAMA_PID log=$LLAMALOG"
ok=0
for i in $(seq 1 90); do
  curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1 && { ok=1; echo "teacher ready after ${i}s"; break; }
  kill -0 "$LLAMA_PID" 2>/dev/null || { echo "llama-server died:"; tail -30 "$LLAMALOG"; break; }
  sleep 2
done
[ "$ok" = 1 ] || die "teacher never came up" 11

# ── Stage 2: distill thematic_qa ───────────────────────────────────────────
banner "stage 2 — distill thematic_qa"
conda activate bible-orpo
DLIMIT=0; [ "$SMOKE" = 1 ] && DLIMIT=20
python training/distill_answers.py \
  --backend vllm --vllm-url http://127.0.0.1:8001/v1 --model qwen3-14b-q5km \
  --in "$THEMATIC_IN" --out "$THEMATIC_OUT" \
  --concurrency "$PARALLEL" --max-retries 2 --limit "$DLIMIT"
DRC=$?
kill "$LLAMA_PID" 2>/dev/null; wait "$LLAMA_PID" 2>/dev/null; sleep 3
[ "$DRC" = 0 ] || die "distill_answers rc=$DRC" 12
OKN=$(grep -c '"status": "ok"' "$THEMATIC_OUT" 2>/dev/null || echo 0)
echo "thematic_out: $(wc -l < "$THEMATIC_OUT") rows, $OKN ok"
[ "$OKN" -ge 10 ] || die "too few ok rows ($OKN)" 12

# ── Stage 3: assemble train_v3.1.json ──────────────────────────────────────
banner "stage 3 — assemble ${TRAIN_V31}"
python training/assemble_v3.py \
  --distilled "$DISTILL_OUT" --thematic "$THEMATIC_OUT" --out "$TRAIN_V31" || die "assemble_v3 failed" 13
python -c "import json,sys; d=json.load(open('$TRAIN_V31')); print('train_v3.1:', len(d), 'examples')" || die "train_v3.1 unreadable" 13

if [ "$SMOKE" = 1 ]; then
  banner "SMOKE done — distill+assemble wiring OK. Skipping SFT/merge/eval."
  echo "EXIT_0"; exit 0
fi

# ── Stage 4: SFT ──────────────────────────────────────────────────────────
banner "stage 4 — SFT (${SFT_CONFIG}) ~7 h"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python training/train_unsloth.py --config "$SFT_CONFIG" --run-name "$RUN_NAME" --no-wandb || die "SFT failed" 14
[ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || die "no adapter at $ADAPTER_DIR" 14
echo "adapter: $(du -sh "$ADAPTER_DIR" | cut -f1)"

# ── Stage 5: merge ────────────────────────────────────────────────────────
banner "stage 5 — merge adapter -> ${MERGED_DIR}"
python training/merge_adapters.py \
  --lora-path "$ADAPTER_DIR" --base-model Qwen/Qwen3.5-4B --output "$MERGED_DIR" || die "merge failed" 15
[ -f "$MERGED_DIR/model.safetensors" ] || die "no merged model.safetensors" 15

# ── Stage 6: coherence check ─────────────────────────────────────────────
banner "stage 6 — coherence check"
python scripts/_coherence_check.py "$MERGED_DIR" || die "coherence check FAILED — merged model is garbage" 16

# ── Stage 7: eval (protocol v4 keyword) ─────────────────────────────────
banner "stage 7 — eval v3.1 through the RAG stack"
TFLOG="logs/v3.1_tfserver_${TS}.log"
MODEL_PATH="$MERGED_DIR" PORT=8001 SERVED_NAME="bible-v3.1" \
  nohup python scripts/_tf_openai_server.py > "$TFLOG" 2>&1 &
TF_PID=$!
ok=0
for i in $(seq 1 120); do
  grep -q "ready on :8001" "$TFLOG" 2>/dev/null && { ok=1; echo "tf ready ${i}s"; break; }
  kill -0 "$TF_PID" 2>/dev/null || { echo "tf died:"; tail -30 "$TFLOG"; break; }
  sleep 2
done
[ "$ok" = 1 ] || { kill "$TF_PID" 2>/dev/null; die "tf-server never ready" 17; }
LABEL="v3.1" SERVED="bible-v3.1" JUDGE=0 bash scripts/_run_v3_eval.sh
ERC=$?
kill "$TF_PID" 2>/dev/null; wait "$TF_PID" 2>/dev/null; sleep 3
[ "$ERC" = 0 ] || die "eval rc=$ERC" 17
RESULT="$(ls -t docs/benchmark_runs/*_v3.1_keyword.json 2>/dev/null | head -1)"
[ -n "$RESULT" ] && [ -f "$RESULT" ] || die "no *_v3.1_keyword.json result written" 17
echo "result: $RESULT"

# ── Stage 8: v4 rescore + verdict ──────────────────────────────────────
banner "stage 8 — v4 rescore + compare"
python - <<PY
import json, re, statistics
d = json.load(open("$RESULT")); items = d.get("results") or []
EXPO = re.compile(r"(teach|about)\?\s*$", re.I)
def m(cat_filter):
    xs = [it["verse_accuracy_fuzzy"] for it in items
          if it.get("verse_accuracy_fuzzy") is not None and it.get("category") != "refusal" and cat_filter(it)]
    return statistics.mean(xs), len(xs)
allm, n1 = m(lambda it: True)
expo_excl, n2 = m(lambda it: not (it["category"]=="verse_lookup" and EXPO.search(it.get("question",""))))
hall = sum(1 for it in items if it.get("hallucination_detected"))
import collections
cats = collections.defaultdict(list)
for it in items:
    if it.get("verse_accuracy_fuzzy") is not None: cats[it["category"]].append(it["verse_accuracy_fuzzy"])
print(f"v3.1 fuzzy mean all-in {allm:.3f} (n={n1}) | expo-excl {expo_excl:.3f} (n={n2}) | halluc {hall}/{len(items)}")
for c in ("verse_lookup","character","context","cross_reference","topical","theological_reliability"):
    if cats[c]: print(f"  {c:<24} {statistics.mean(cats[c]):.3f}  (n={len(cats[c])})")
gate = expo_excl >= 0.52
syn_ok = all(statistics.mean(cats[c]) >= 0.50 for c in ("character","context","cross_reference","topical") if cats[c])
print()
print("GATE  expo-excl>=0.52 :", "PASS" if gate else f"FAIL ({expo_excl:.3f})")
print("GATE  synth cats>=0.50:", "PASS" if syn_ok else "FAIL")
print("VERDICT:", "SHIP v3.1" if (gate and hall <= 3) else "HOLD — inspect per-category + hallucinations")
PY

banner "PIPELINE COMPLETE"
echo "artifacts: $ADAPTER_DIR | $MERGED_DIR | $RESULT | $TRAIN_V31"
echo "next (CPU, needs HF token): convert_hf_to_gguf --no-mtp -> llama-quantize ladder -> publish -v3.1 + -v3.1-GGUF"
echo "EXIT_0"
