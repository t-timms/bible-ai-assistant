#!/usr/bin/env bash
# Run the external comparators (benchmarks/external_comparators.yaml) through the
# UNCHANGED RAG stack on the protocol-v4 282-question suite. One JSON per model:
#   docs/benchmark_runs/<UTCdate>_ext-<key>_keyword.json      (run_benchmark.py naming)
# Then: python scripts/sota_scoreboard.py   -> docs/SOTA_EVAL.md
#
# GPU-INTENSIVE. Do NOT launch while gaming. Run  --only <key> --smoke-first  once
# to validate a single model + chat template before the full sweep.
#
#   tmux new-session -d -s extbase 'bash ~/bible-ai-assistant/scripts/run_external_baselines.sh'
#   tail -f ~/bible-ai-assistant/logs/extbase_*.log        # ends EXTBASE_ALL_DONE
#
# ───────────────────────── ETA / AUDIT (read before running) ─────────────────────────
# Proxy: our own v4 keyword run = 282 Qs, keyword-only (NO judge) ≈ 12–18 min on-GPU.
#   on-GPU comparators (fits_16gb:true)      ~15–20 min each × 7  ≈ 2.0–2.5 h
#   qwen3:32b (CPU-offload, fits_16gb:false) ~60–120 min               (optional; last)
#   one-time GGUF downloads (5 × ~5–8 GB Q4) ≈ 30–45 GB, ~20–40 min
#   Realistic unattended: ~3–4 h for the 7 fast models + downloads.
# WRITES (all NEW paths — nothing overwritten):
#   ~/.ollama/models/…                            pulled tags + `ollama create` models
#   models/ext_gguf/*.gguf , models/ext_gguf/Modelfile.*
#   docs/benchmark_runs/<date>_ext-*_keyword.json
#   prompts/evaluation_questions.json  <-- PROMOTED to the v4 split suite (see below)
#   logs/extbase_*.log , /tmp/rag_ext-*_*.log , /tmp/bench_ext-*_*.log
# SUITE PROMOTE: evaluate.py reads prompts/evaluation_questions.json (not the manifest
#   snapshot). This script copies benchmarks/suites/evaluation_questions.v3.json over it
#   after backing up the current file to prompts/evaluation_questions.pre-v4.json, and
#   hard-verifies the normalized sha256 == manifest.v4 suite_sha256 before any run.
#   Our own models' v4 numbers come from scripts/rescore_v4.py (same questions, same
#   responses, deterministic re-bucket) so the board stays apples-to-apples.
# KNOWN FAILURE MODES: wrong chat template -> fuzzy≈0 everywhere (--smoke-first catches it);
#   GGUF filename drift -> `ollama create` fails (fix gguf_file in the yaml); only ONE
#   model resident at a time; `ollama pull` registry throttling -> 3× retry.
# ─────────────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$HOME/bible-ai-assistant"

SMOKE_FIRST=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke-first) SMOKE_FIRST=1 ;;   # run one model, then stop for you to inspect
    --only) ONLY="${2:?}"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac; shift
done

mkdir -p logs models/ext_gguf
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/extbase_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

YAML="benchmarks/external_comparators.yaml"
MANIFEST="benchmarks/manifest.v4.yaml"
SNAP="benchmarks/suites/evaluation_questions.v3.json"
LIVE="prompts/evaluation_questions.json"
GGUF_DIR="models/ext_gguf"

echo "=== preflight $(date -Is) ==="
command -v ollama >/dev/null || { echo "ollama missing"; exit 1; }
python3 -c "import yaml" 2>/dev/null || { echo "pyyaml missing"; exit 1; }
[ -f "$MANIFEST" ] && [ -f "$SNAP" ] || { echo "v4 manifest/snapshot missing — run scripts/make_v4_suite.py"; exit 1; }

FREE_GB=$(df -PBG "$HOME" | awk 'NR==2{gsub("G","",$4); print $4+0}')
echo "disk free: ${FREE_GB} GB"
[ "${FREE_GB:-0}" -lt 60 ] && echo "WARNING: <60 GB free — GGUF downloads may not all fit"

# ---- promote the v4 split suite into the live file evaluate.py reads, with a hard check ----
PIN=$(python3 -c "import yaml;print(yaml.safe_load(open('$MANIFEST'))['suite_sha256'])")
snap_hash() { python3 - "$1" <<'PY'
import sys,hashlib
print(hashlib.sha256(open(sys.argv[1],"rb").read().replace(b"\r\n",b"\n")).hexdigest())
PY
}
[ "$(snap_hash "$SNAP")" = "$PIN" ] || { echo "SNAP sha256 != manifest pin — abort"; exit 1; }
if [ ! -f prompts/evaluation_questions.pre-v4.json ]; then
  cp "$LIVE" prompts/evaluation_questions.pre-v4.json
  echo "backed up $LIVE -> prompts/evaluation_questions.pre-v4.json"
fi
cp "$SNAP" "$LIVE"
[ "$(snap_hash "$LIVE")" = "$PIN" ] || { echo "LIVE sha256 != manifest pin after copy — abort"; exit 1; }
echo "promoted v4 suite into $LIVE  (sha256 $PIN OK)"

mapfile -t ROWS < <(python3 - "$YAML" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for c in d["comparators"]:
    print("\t".join([c["key"], c["serving"], c.get("ollama_tag",""),
                     c.get("gguf",""), c.get("gguf_file",""), c.get("template",""), c.get("group","")]))
PY
)
echo "comparators: ${#ROWS[@]}"

pull_retry() { for i in 1 2 3; do ollama pull "$1" && return 0; echo "pull retry $i…"; sleep 15; done; return 1; }

_tmpl() {
  case "$1" in
    llama3)  printf '<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' ;;
    mistral) printf '[INST] {{ .Prompt }} [/INST]' ;;
    phi3)    printf '<|user|>\n{{ .Prompt }}<|end|>\n<|assistant|>\n' ;;
    gemma)   printf '<start_of_turn>user\n{{ .Prompt }}<end_of_turn>\n<start_of_turn>model\n' ;;
    *)       printf '{{ .Prompt }}' ;;
  esac
}

run_one() {
  local key="$1" serving="$2" otag="$3" gurl="$4" gfile="$5" tmpl="$6"
  local served
  echo; echo "================ $(date -Is)  $key  ($serving) ================"
  if [ "$serving" = "ollama_tag" ]; then
    served="$otag"
    pull_retry "$otag" || { echo "PULL FAILED $otag — skip"; return 1; }
  else
    served="ext-${key}"
    local path="$GGUF_DIR/$gfile" repo
    repo="$(echo "$gurl" | sed 's#https://hf.co/##')"
    if [ ! -f "$path" ]; then
      echo "download: $repo :: $gfile"
      huggingface-cli download "$repo" "$gfile" --local-dir "$GGUF_DIR" \
        --local-dir-use-symlinks False || { echo "DOWNLOAD FAILED — skip"; return 1; }
    fi
    { printf 'FROM %s\n' "$path"; printf 'TEMPLATE """%s"""\n' "$(_tmpl "$tmpl")"; } > "$GGUF_DIR/Modelfile.$key"
    ollama create "$served" -f "$GGUF_DIR/Modelfile.$key" || { echo "ollama create FAILED — skip"; return 1; }
  fi

  LABEL="ext-$key" SERVED="$served" bash scripts/_run_ext_eval.sh || echo "eval failed for $key (continuing)"
  ollama stop "$served" 2>/dev/null || true
  echo "================ $(date -Is)  $key done ================"
}

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r key serving otag gurl gfile tmpl group <<< "$row"
  [ -n "$ONLY" ] && [ "$ONLY" != "$key" ] && continue
  run_one "$key" "$serving" "$otag" "$gurl" "$gfile" "$tmpl"
  [ "$SMOKE_FIRST" = "1" ] && { echo; echo "--smoke-first: stopping after $key. Inspect /tmp/bench_ext-${key}_*.log then re-run without the flag."; break; }
done

echo; echo "=== external baselines done $(date -Is) ==="
echo "restore editing surface if desired: cp prompts/evaluation_questions.pre-v4.json prompts/evaluation_questions.json"
echo "next: python scripts/sota_scoreboard.py   (writes docs/SOTA_EVAL.md)"
echo "EXTBASE_ALL_DONE"
