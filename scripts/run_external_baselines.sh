#!/usr/bin/env bash
# Run the external comparators (benchmarks/external_comparators.yaml) through the
# UNCHANGED RAG stack on the protocol-v5 282-question suite (same suite/hash as v4;
# v5 only adds the semantic metric, backfilled separately -- see SUITE PROMOTE below).
# One JSON per model:
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
#   hard-verifies the normalized sha256 == manifest.v5 suite_sha256 before any run (v4
#   and v5 pin the identical suite/hash -- v5 only adds a metric, see manifest.v5.yaml).
#   Our own models' numbers come from scripts/rescore_v5.py (same questions, same
#   responses, deterministic v4 re-bucket + semantic score, no re-run) so the board
#   stays apples-to-apples. After this sweep: backfill semantic onto each new
#   ext-*_keyword.json with `scripts/rescore_v5.py --file ... --out ..._v5semantic.json`
#   before trusting the semantic column in `scripts/sota_scoreboard.py`'s output.
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
MANIFEST="benchmarks/manifest.v5.yaml"
SNAP="benchmarks/suites/evaluation_questions.v3.json"
LIVE="prompts/evaluation_questions.json"
# Absolute, not relative: the Modelfile written below does `FROM $GGUF_DIR/$gfile`,
# and ollama's FROM directive treats a relative path as a REGISTRY model
# reference to pull (not a local file), failing with a DNS-lookup error against
# a literal host named "models" -- never caught before because no hf_gguf
# comparator had reached this step until the download-path bugs above were fixed.
GGUF_DIR="$HOME/bible-ai-assistant/models/ext_gguf"
# `hf` (like huggingface-cli before it) is only installed inside .venv-rag, not
# on the base system PATH -- this script never activates that venv (only
# scripts/_run_ext_eval.sh does, for the RAG server). Call it by full path so
# the GGUF download step doesn't silently fail with "command not found" while
# every other preflight check (which uses bare `ollama`/`python3`) still passes.
HF_BIN="$HOME/bible-ai-assistant/.venv-rag/bin/hf"

echo "=== preflight $(date -Is) ==="
command -v ollama >/dev/null || { echo "ollama missing"; exit 1; }
python3 -c "import yaml" 2>/dev/null || { echo "pyyaml missing"; exit 1; }
[ -f "$MANIFEST" ] && [ -f "$SNAP" ] || { echo "v4 manifest/snapshot missing — run scripts/make_v4_suite.py"; exit 1; }
[ -x "$HF_BIN" ] || { echo "$HF_BIN missing/not executable — hf_gguf comparators cannot download"; exit 1; }

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

# Field separator is \x1f (ASCII unit separator), NOT a tab. bash's `read`
# treats tab as "IFS whitespace" and silently COLLAPSES runs of consecutive
# tabs into one delimiter (a POSIX quirk that applies even when tab is the
# only IFS character) -- with \t this shifted every field after two
# back-to-back empty ones (ollama_tag AND gguf both unset, e.g.
# bible-study-phi3-mini / rhema-bibleai-gemma, which ship their own GGUF and
# have no separate `gguf:` repo) by two positions, so `gfile`/`tmpl` silently
# took on `tmpl`/`group`'s values -- caught via a direct repro before this
# script would have wrongly reported those 2 comparators' downloads as
# "not found" against a garbage repo path. \x1f is not IFS-whitespace, so
# empty fields are preserved exactly.
mapfile -t ROWS < <(python3 - "$YAML" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for c in d["comparators"]:
    print("\x1f".join([c["key"], c["serving"], c.get("ollama_tag",""),
                       c.get("gguf",""), c.get("gguf_file",""), c.get("template",""),
                       c.get("group",""), c.get("hf","")]))
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
  local key="$1" serving="$2" otag="$3" gurl="$4" gfile="$5" tmpl="$6" hfurl="$7"
  local served
  echo; echo "================ $(date -Is)  $key  ($serving) ================"
  if [ "$serving" = "ollama_tag" ]; then
    served="$otag"
    pull_retry "$otag" || { echo "PULL FAILED $otag — skip"; return 1; }
  else
    served="ext-${key}"
    local path="$GGUF_DIR/$gfile" repo
    # `gguf:` names a separate quant repo (e.g. mradermacher/*-GGUF); when a
    # comparator ships its own GGUF in its main repo instead (no `gguf:` key --
    # bible-study-phi3-mini, rhema-bibleai-gemma), fall back to `hf:`. Without
    # this, repo ends up empty and `hf download` fails on an empty repo id.
    repo="$(echo "${gurl:-$hfurl}" | sed 's#https://hf.co/##')"
    if [ ! -f "$path" ]; then
      echo "download: $repo :: $gfile"
      # `huggingface-cli` is deprecated and non-functional as of huggingface_hub
      # 1.29+ (installed in .venv-rag) -- it exits nonzero immediately, no
      # download attempted. `hf download` is the replacement; it materializes
      # real files under --local-dir directly, so --local-dir-use-symlinks
      # (removed) is no longer needed.
      "$HF_BIN" download "$repo" "$gfile" --local-dir "$GGUF_DIR" \
        || { echo "DOWNLOAD FAILED — skip"; return 1; }
    fi
    { printf 'FROM %s\n' "$path"; printf 'TEMPLATE """%s"""\n' "$(_tmpl "$tmpl")"; } > "$GGUF_DIR/Modelfile.$key"
    ollama create "$served" -f "$GGUF_DIR/Modelfile.$key" || { echo "ollama create FAILED — skip"; return 1; }
  fi

  LABEL="ext-$key" SERVED="$served" bash scripts/_run_ext_eval.sh || echo "eval failed for $key (continuing)"
  ollama stop "$served" 2>/dev/null || true
  echo "================ $(date -Is)  $key done ================"
}

for row in "${ROWS[@]}"; do
  IFS=$'\x1f' read -r key serving otag gurl gfile tmpl group hfurl <<< "$row"
  [ -n "$ONLY" ] && [ "$ONLY" != "$key" ] && continue
  run_one "$key" "$serving" "$otag" "$gurl" "$gfile" "$tmpl" "$hfurl"
  [ "$SMOKE_FIRST" = "1" ] && { echo; echo "--smoke-first: stopping after $key. Inspect /tmp/bench_ext-${key}_*.log then re-run without the flag."; break; }
done

echo; echo "=== external baselines done $(date -Is) ==="
echo "restore editing surface if desired: cp prompts/evaluation_questions.pre-v4.json prompts/evaluation_questions.json"
echo "next: python scripts/sota_scoreboard.py   (writes docs/SOTA_EVAL.md)"
echo "EXTBASE_ALL_DONE"
