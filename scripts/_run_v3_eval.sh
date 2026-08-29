#!/usr/bin/env bash
# Protocol-v3 benchmark for a model already being served by _tf_openai_server.py on :8001.
#   LABEL=v2-4b  bash scripts/_run_v3_eval.sh
# Assumes: tf-server (bible-orpo env) live on 127.0.0.1:8001 serving SERVED name = $SERVED.
set -uo pipefail
cd "$HOME/bible-ai-assistant"
LABEL="${LABEL:?set LABEL}"
SERVED="${SERVED:-bible-v2-4b}"
JUDGE="${JUDGE:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
RAGLOG="/tmp/rag_${LABEL}_${TS}.log"
BENCHLOG="/tmp/bench_${LABEL}_${TS}.log"

# 1. RAG server (its own venv) pointed at the tf-server
source .venv-rag/bin/activate
OLLAMA_URL="http://127.0.0.1:8001" OLLAMA_MODEL="$SERVED" CITATION_VERIFICATION_ENABLED=true \
  nohup uvicorn rag.rag_server:app --host 127.0.0.1 --port 8081 > "$RAGLOG" 2>&1 &
RAG_PID=$!
echo "rag_server pid=$RAG_PID log=$RAGLOG"

# 2. wait for health
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1 && { echo "rag healthy after ${i}s"; break; }
  sleep 2
  [ "$i" -eq 60 ] && { echo "RAG server never came up"; tail -30 "$RAGLOG"; kill $RAG_PID 2>/dev/null; exit 1; }
done

# 3. run the benchmark
CMD=(python scripts/run_benchmark.py --label "$LABEL" --ollama-model "$SERVED"
     --rag-url http://127.0.0.1:8081/v1/chat/completions)
[ "$JUDGE" = "1" ] && CMD+=(--judge)
echo "RUN: ${CMD[*]}" | tee "$BENCHLOG"
"${CMD[@]}" >> "$BENCHLOG" 2>&1
RC=$?
echo "BENCH_EXIT_${RC}" | tee -a "$BENCHLOG"

kill $RAG_PID 2>/dev/null
echo "=== done: $BENCHLOG ==="
tail -40 "$BENCHLOG"
