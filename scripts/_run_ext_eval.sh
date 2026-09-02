#!/usr/bin/env bash
# Protocol-v4 keyword benchmark for ONE model already served by Ollama on :11434.
# Called by run_external_baselines.sh. Starts the RAG server (its own venv)
# pointed at Ollama, waits for health, runs scripts/run_benchmark.py against
# benchmarks/manifest.v4.yaml, tears the RAG server down.
#
#   LABEL=ext-foo SERVED=some-ollama-tag bash scripts/_run_ext_eval.sh
set -uo pipefail
cd "$HOME/bible-ai-assistant"
LABEL="${LABEL:?set LABEL}"
SERVED="${SERVED:?set SERVED}"          # ollama tag (must be in `ollama list`)
TS="$(date +%Y%m%d-%H%M%S)"
RAGLOG="/tmp/rag_${LABEL}_${TS}.log"
BENCHLOG="/tmp/bench_${LABEL}_${TS}.log"

ollama list | awk '{print $1}' | grep -qx "$SERVED" || {
  echo "SERVED tag '$SERVED' not in ollama list — abort"; exit 1; }

# RAG server -> Ollama (NOT the tf-server used for the hybrid-arch bible models)
source .venv-rag/bin/activate
OLLAMA_URL="http://127.0.0.1:11434" OLLAMA_MODEL="$SERVED" CITATION_VERIFICATION_ENABLED=true \
  nohup uvicorn rag.rag_server:app --host 127.0.0.1 --port 8081 > "$RAGLOG" 2>&1 &
RAG_PID=$!
echo "rag_server pid=$RAG_PID log=$RAGLOG  (model=$SERVED)"

for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1 && { echo "rag healthy after ${i}s"; break; }
  sleep 2
  [ "$i" -eq 60 ] && { echo "RAG server never came up"; tail -30 "$RAGLOG"; kill $RAG_PID 2>/dev/null; exit 1; }
done

CMD=(python scripts/run_benchmark.py
     --label "$LABEL" --model-tag "$LABEL"
     --ollama-model "$SERVED"
     --manifest benchmarks/manifest.v4.yaml
     --rag-url http://127.0.0.1:8081/v1/chat/completions)
echo "RUN: ${CMD[*]}" | tee "$BENCHLOG"
"${CMD[@]}" >> "$BENCHLOG" 2>&1
RC=$?
echo "BENCH_EXIT_${RC}" | tee -a "$BENCHLOG"

kill $RAG_PID 2>/dev/null
sleep 3
echo "=== done: $BENCHLOG  (rc=$RC) ==="
tail -25 "$BENCHLOG"
exit $RC
