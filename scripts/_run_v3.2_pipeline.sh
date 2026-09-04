#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# V3.2 continued-fine-tune pipeline (DMT-style stage 2) — regenerate
# thematic_qa with the RAFT distractor-fix -> short continued-FT from the v3.1
# adapter -> merge -> coherence -> eval -> verdict. ~2-3 h GPU, not ~10 h: no
# fresh full SFT, no re-streaming smoltalk2 (no general_blend needed here).
#
#   SMOKE:  SMOKE=1 bash scripts/_run_v3.2_pipeline.sh   # small thematic batch, skip training
#   FULL :  tmux new-session -d -s v32 'bash ~/bible-ai-assistant/scripts/_run_v3.2_pipeline.sh'
#           tail -f ~/bible-ai-assistant/logs/v3.2_pipeline_*.log
#
# Preserves v3.1 and all earlier artifacts — writes only *v3.2* names.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$HOME/bible-ai-assistant" || exit 1
mkdir -p logs data/raw_v3 data/processed

SMOKE="${SMOKE:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/v3.2_pipeline_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

TEACHER="$HOME/models/qwen3-14b-gguf/Qwen3-14B-Q5_K_M.gguf"
LLAMA_SERVER="$HOME/llama.cpp-full/build/bin/llama-server"
THEMATIC_IN="data/raw_v3/thematic_inputs_v2.jsonl"
THEMATIC_OUT="data/raw_v3/thematic_out_v2.jsonl"
V31_DATASET="data/processed/train_v3.1.json"          # rehearsal source
CONTINUED_SET="data/processed/train_v3.2-continued.json"
SFT_CONFIG="training/config.v3.2-continued-4b.yaml"
BASE_ADAPTER="models/qwen3.5-4b-bible-v3.1-sft"        # what we continue FROM
RUN_NAME="qwen3.5-4b-bible-v3.2-continued"
ADAPTER_DIR="models/${RUN_NAME}"
MERGED_DIR="models/qwen3.5-4b-bible-v3.2-merged"
PARALLEL=4

die() { echo "!!! STAGE FAILED: $1"; echo "EXIT_${2:-1}"; exit "${2:-1}"; }
banner() { echo; echo "======== $(date -Is)  $1 ========"; }

# ── Preflight ───────────────────────────────────────────────────────────────
banner "preflight (SMOKE=$SMOKE)"
[ -f "$TEACHER" ]      || die "missing teacher $TEACHER" 10
[ -x "$LLAMA_SERVER" ] || die "missing $LLAMA_SERVER" 10
[ -f "$SFT_CONFIG" ]   || die "missing $SFT_CONFIG" 10
[ -f "$V31_DATASET" ]  || die "missing $V31_DATASET (rehearsal source)" 10
[ -f "$BASE_ADAPTER/adapter_config.json" ] || die "missing $BASE_ADAPTER (the v3.1 adapter to continue)" 10
[ -d rag/chroma_db ]   || die "missing rag/chroma_db" 10
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
df -h "$HOME" | tail -1

source ~/miniforge3/etc/profile.d/conda.sh

# ── Stage 1: regenerate thematic_qa retrieval inputs (new top_k) ──────────
# .venv-rag only (chromadb + sentence-transformers for retrieval) -- NOT
# combined with conda, to avoid a stale PATH once we switch envs for stage 2.
banner "stage 1 — regenerate thematic inputs (build_v3_thematic.py, updated top_k)"
source .venv-rag/bin/activate
LIMIT_ARG=(); [ "$SMOKE" = 1 ] && LIMIT_ARG=(--limit 40)
python training/build_v3_thematic.py --out "$THEMATIC_IN" --seed 42 "${LIMIT_ARG[@]}" \
  || die "build_v3_thematic failed" 11
[ -s "$THEMATIC_IN" ] || die "no thematic inputs written" 11
# Cleanly leave .venv-rag before conda-activating bible-orpo for the rest of
# the pipeline -- conda activate does not run a venv's deactivate, so a stale
# .venv-rag/bin ahead of the conda env's bin in PATH could shadow the wrong
# python/pip for every stage after this one.
type deactivate >/dev/null 2>&1 && deactivate
wc -l "$THEMATIC_IN"

# ── Stage 2: teacher server + distill (RAFT distractor-fix prompt) ────────
banner "stage 2 — llama-server (Qwen3-14B Q5_K_M) + distill"
LLAMALOG="logs/v3.2_llama_${TS}.log"
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
[ "$ok" = 1 ] || die "teacher never came up" 12

conda activate bible-orpo
python training/distill_answers.py \
  --backend vllm --vllm-url http://127.0.0.1:8001/v1 --model qwen3-14b-q5km \
  --in "$THEMATIC_IN" --out "$THEMATIC_OUT" --concurrency "$PARALLEL" --max-retries 2
DRC=$?
kill "$LLAMA_PID" 2>/dev/null; wait "$LLAMA_PID" 2>/dev/null; sleep 3
[ "$DRC" = 0 ] || die "distill_answers rc=$DRC" 12
OKN=$(grep -c '"status": "ok"' "$THEMATIC_OUT" 2>/dev/null || echo 0)
echo "thematic_out_v2: $(wc -l < "$THEMATIC_OUT") rows, $OKN ok"
[ "$OKN" -ge 10 ] || die "too few ok rows ($OKN)" 12

# ── Stage 3: assemble the continued-FT set (thematic-heavy + rehearsal) ───
banner "stage 3 — assemble ${CONTINUED_SET}"
python training/build_continued_ft_set.py \
  --thematic "$THEMATIC_OUT" --base "$V31_DATASET" --out "$CONTINUED_SET" \
  || die "build_continued_ft_set failed" 13
python -c "import json; d=json.load(open('$CONTINUED_SET')); print('train_v3.2-continued:', len(d), 'examples')" \
  || die "continued set unreadable" 13

if [ "$SMOKE" = 1 ]; then
  banner "SMOKE done — regen+distill+assemble wiring OK. Skipping continued-FT/merge/eval."
  echo "EXIT_0"; exit 0
fi

# ── Stage 4: continued fine-tune from the v3.1 adapter ────────────────────
banner "stage 4 — continued-FT (${SFT_CONFIG}) from ${BASE_ADAPTER}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python training/train_unsloth.py --config "$SFT_CONFIG" --model-path "$BASE_ADAPTER" \
  --run-name "$RUN_NAME" --no-wandb || die "continued-FT failed" 14
[ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || die "no adapter at $ADAPTER_DIR" 14
echo "adapter: $(du -sh "$ADAPTER_DIR" | cut -f1)"

# ── Stage 5: merge ─────────────────────────────────────────────────────
banner "stage 5 — merge adapter -> ${MERGED_DIR}"
python training/merge_adapters.py \
  --lora-path "$ADAPTER_DIR" --base-model Qwen/Qwen3.5-4B --output "$MERGED_DIR" || die "merge failed" 15
[ -f "$MERGED_DIR/model.safetensors" ] || die "no merged model.safetensors" 15

# ── Stage 6: coherence check ─────────────────────────────────────────
banner "stage 6 — coherence check"
python scripts/_coherence_check.py "$MERGED_DIR" || die "coherence check FAILED — merged model is garbage" 16

# ── Stage 7: eval (protocol v4 keyword) ─────────────────────────────
banner "stage 7 — eval v3.2 through the RAG stack"
TFLOG="logs/v3.2_tfserver_${TS}.log"
MODEL_PATH="$MERGED_DIR" PORT=8001 SERVED_NAME="bible-v3.2" \
  nohup python scripts/_tf_openai_server.py > "$TFLOG" 2>&1 &
TF_PID=$!
ok=0
for i in $(seq 1 120); do
  grep -q "ready on :8001" "$TFLOG" 2>/dev/null && { ok=1; echo "tf ready ${i}s"; break; }
  kill -0 "$TF_PID" 2>/dev/null || { echo "tf died:"; tail -30 "$TFLOG"; break; }
  sleep 2
done
[ "$ok" = 1 ] || { kill "$TF_PID" 2>/dev/null; die "tf-server never ready" 17; }
LABEL="v3.2" SERVED="bible-v3.2" JUDGE=0 bash scripts/_run_v3_eval.sh
ERC=$?
kill "$TF_PID" 2>/dev/null; wait "$TF_PID" 2>/dev/null; sleep 3
[ "$ERC" = 0 ] || die "eval rc=$ERC" 17
RESULT="$(ls -t docs/benchmark_runs/*_v3.2_keyword.json 2>/dev/null | head -1)"
[ -n "$RESULT" ] && [ -f "$RESULT" ] || die "no *_v3.2_keyword.json result written" 17
echo "result: $RESULT"

# ── Stage 8: v4 rescore + verdict, and a direct v3.1-vs-v3.2 diff ────────
banner "stage 8 — v4 rescore + compare to v3.1"
python - <<PY
import json, re, statistics
d = json.load(open("$RESULT")); items = d.get("results") or []
prev = json.load(open("docs/benchmark_runs/20260904_v3.1_keyword.json")).get("results") or []
prev_by_q = {it["question"]: it for it in prev}
EXPO = re.compile(r"(teach|about)\?\s*$", re.I)

def m(items, cat_filter):
    xs = [it["verse_accuracy_fuzzy"] for it in items
          if it.get("verse_accuracy_fuzzy") is not None and it.get("category") != "refusal" and cat_filter(it)]
    return statistics.mean(xs), len(xs)

allm, n1 = m(items, lambda it: True)
expo_excl, n2 = m(items, lambda it: not (it["category"]=="verse_lookup" and EXPO.search(it.get("question",""))))
hall = sum(1 for it in items if it.get("hallucination_detected"))

import collections
cats = collections.defaultdict(list)
for it in items:
    if it.get("verse_accuracy_fuzzy") is not None: cats[it["category"]].append(it["verse_accuracy_fuzzy"])

print(f"v3.2 fuzzy mean all-in {allm:.3f} (n={n1}) | expo-excl {expo_excl:.3f} (n={n2}) | halluc {hall}/{len(items)}")
for c in ("verse_lookup","character","context","cross_reference","topical","theological_reliability"):
    if cats[c]:
        v32 = statistics.mean(cats[c])
        prevvals = [prev_by_q[it["question"]]["verse_accuracy_fuzzy"] for it in items
                    if it["category"]==c and it["question"] in prev_by_q
                    and prev_by_q[it["question"]].get("verse_accuracy_fuzzy") is not None]
        v31 = statistics.mean(prevvals) if prevvals else float("nan")
        print(f"  {c:<24} v3.2={v32:.3f}  v3.1={v31:.3f}  delta={v32-v31:+.3f}  (n={len(cats[c])})")

gate = expo_excl >= 0.52
syn_ok = all(statistics.mean(cats[c]) >= 0.50 for c in ("character","context","cross_reference","topical") if cats[c])
print()
print("GATE  expo-excl>=0.52 :", "PASS" if gate else f"FAIL ({expo_excl:.3f})")
print("GATE  synth cats>=0.50:", "PASS" if syn_ok else "FAIL")
print("VERDICT:", "SHIP v3.2" if (gate and hall <= 3) else "HOLD — inspect per-category + hallucinations")
PY

banner "PIPELINE COMPLETE"
echo "artifacts: $ADAPTER_DIR | $MERGED_DIR | $RESULT | $CONTINUED_SET"
echo "next (CPU, needs HF token): convert_hf_to_gguf --no-mtp -> llama-quantize ladder -> publish -v3.2 + -v3.2-GGUF"
echo "EXIT_0"
