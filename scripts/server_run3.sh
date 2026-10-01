#!/usr/bin/env bash
# Run 3: our implementation of the 70.83 reference's design, trained on the VG-LLM data. Plan and reading guide:
# docs/SERVER_RUN3.md (the design: docs/REFERENCE_70_ANALYSIS.md §6-7 - nothing of the reference's code is used).
#
# ONE command, run to the end without anyone stepping in:
#   cd /group-volume/wooyeol/vlm-live && git pull && mkdir -p outputs/run3
#   nohup bash scripts/server_run3.sh > outputs/run3/master.$(hostname -s).log 2>&1 &
# As run 2: on one node everything runs in order; the SAME command on more nodes shares the roles (claims on the shared
# volume; a node whose heartbeat stops for 30 min loses its claims and the role resumes elsewhere). A node may have 8 or
# 4 GPUs: each node uses what it has, with the global batch kept at 64 (8 x accum 8, or 4 x accum 16).
#   tail -f outputs/run3/STATUS           # every node writes here, lines tagged with its host
#   bash scripts/server_run3.sh stop      # on a node: ends that node's run and everything it started
#
#   first      p0 environment on every node; once: data - the data pass (scripts/inspect_reference_data.py ->
#              reference_data/DATA.md) and the mix as JSONL + index (scripts/prepare_mix.py -> data/vgllm; marked
#              records kept when their marks can be drawn with certainty; VSI-Bench scenes left out); calib - 12 steps
#              of the gated arm at the production settings -> budget.env: the steps of one pass of the prepared mix
#              (every arm trains the same steps on the same data order; TRAIN_HOURS > 0 caps it, NO_GO below GO_EPOCHS)
#   gated      fusion gated_input, the reference-style start (configs/server_4b_v3.yaml) -> VSI plain / room facts /
#              video mode, learning curve, the lmms-eval gate, latency
#   plain      the same data, LoRA and resolution, no geometry (server_4b_v3_plain.yaml) -> VSI -> the gate smoke,
#              the gate's two base runs (every arm's verdict is taken against them) and its own gate
#   control    the gated arm with shuffled geometry -> VSI -> holdout ablation gated vs control
#   base       the untrained model at this resolution: VSI plain / room facts; run 2's results copied in for reference
#   gated_wn   fusion gated_input, the reference's wonorm variant (server_4b_v3_wn.yaml) -> VSI -> gate
# The queue runs in that order: on two nodes gated and plain start together (the key comparison comes first).
# ROLE=gated,plain,... runs just those roles, in that order, without claims.
# Resumable: a finished step leaves a marker and is skipped on a re-run; a training arm continues from resume.pt.
set -uo pipefail

# ------------------------------------------------------------------------------------------------ settings
ROOT="${ROOT:-/group-volume/wooyeol/vlm-live}"
CFG="${CFG:-configs/server_4b_v3.yaml}"               # gated_input, LayerNorm, g = 0.5 at step 0
CFG_WN="${CFG_WN:-configs/server_4b_v3_wn.yaml}"      # gated_input, the reference's wonorm variant
CFG_PLAIN="${CFG_PLAIN:-configs/server_4b_v3_plain.yaml}"   # fusion none
MIX="${MIX:-live}"                                    # configs/data/vgllm.yaml: the 70.83 run's sources without
                                                      # its six CoT variants (live: no reasoning is trained)
REGISTRY="${REGISTRY:-configs/data/vgllm.yaml}"
PREP="${PREP:-data/vgllm}"                            # scripts/prepare_mix.py output (JSONL + index per source)
OUT="${OUT:-outputs/run3}"
R2="${R2:-outputs/run2}"                              # run 2's results, copied in for the comparison tables
WK="${WK:-outputs/weekend}"                           # the weekend's p2_videomme marker says VideoMME is downloaded
export HF_HOME="${HF_HOME:-/group-volume/wooyeol/hf_cache}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Budget: one pass of the prepared mix (the reference: one epoch at batch 64). TRAIN_HOURS > 0 caps it at the
# calibrated speed of the gated arm (the slowest: it runs CUT3R); 0 = no cap (the nodes have no time limit).
TRAIN_HOURS="${TRAIN_HOURS:-0}"
GO_EPOCHS="${GO_EPOCHS:-0.2}"       # with a cap: below this fraction of a pass the training roles are skipped (NO_GO)
EFF_BATCH=64                        # the reference's global batch; micro-steps per GPU = 64 / GPUs on the node
LR=1e-5; LORA_LR=5e-5; WD=0.01      # merger + gate / LoRA (the reference's rates); cosine, warmup 3 %
BETA2=0.999                         # AdamW (0.9, 0.999) as the reference
DONOR_BUFFER="${DONOR_BUFFER:-64}"
PROBE_SRC="${PROBE_SRC:-vlm3r_scannet}"   # the probe's holdout: VSI-style QA on ScanNet training scenes
VSI=(--videos 288 --seed 1)         # all of VSI-Bench: 288 videos, 5,130 questions
VSI_HOURS="${VSI_HOURS:-12}"        # per shard; one shard per GPU
ROUTED_ROOM=(--scene-map --no-map-image --route-facts)

cd "$ROOT" || { echo "no $ROOT"; exit 1; }
mkdir -p "$OUT/vsi/shards" "$OUT/gate"
echo run3 > "$OUT/RUN"
HOST=$(hostname -s 2> /dev/null || hostname)

if [[ "${1:-}" == stop ]]; then
  pid=$(cat "$OUT/PID.$HOST" 2> /dev/null) || { echo "no $OUT/PID.$HOST - nothing running from this node"; exit 1; }
  kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill -TERM "$1" 2> /dev/null; }
  kill_tree "$pid"
  echo "$(date '+%F %T') [$HOST] stopped by hand (pid $pid)" >> "$OUT/STATUS"
  exit 0
fi
# the GPUs this node was given: the scheduler's CUDA_VISIBLE_DEVICES when set (a 4-GPU allocation on an 8-GPU machine
# may be devices 4-7), else every GPU nvidia-smi lists. Our GPU k below means the k-th of these.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi > /dev/null; then GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
else GPUS=($(seq 0 $(( ${NGPU:-0} - 1 )))); fi                   # no nvidia-smi: NGPU= (dry runs)
NGPU=${#GPUS[@]}
[[ "$NGPU" == 4 || "$NGPU" == 8 ]] || { echo "this plan runs on 4 or 8 GPUs per node (found $NGPU: ${GPUS[*]})"; exit 1; }
ALLG=$(IFS=,; echo "${GPUS[*]}"); ACCUM=$(( EFF_BATCH / NGPU )); SHARDS=$NGPU
ROLE="${ROLE:-auto}"
ALL_ROLES=(gated plain base control gated_wn)
if [[ "$ROLE" == auto ]]; then
  RTAG=$HOST; QUEUE=("${ALL_ROLES[@]}")
else
  RTAG=${ROLE//,/+}; QUEUE=(${ROLE//,/ })
fi
echo $$ > "$OUT/PID.$HOST"

# ----------------------------------------------------------------------------------------------- helpers
log()      { local m; m="$(date '+%F %T') [$RTAG] $*"; echo "$m" >> "$OUT/STATUS"; echo "$m" >&2; }
fail()     { log "FAILED $*"; }
is_done()  { [[ -f "$OUT/$1.done" ]]; }
mark()     { touch "$OUT/$1.done"; log "$1 done"; }
have()     { [[ -f "$1" ]] || { log "$2 skipped: $1 missing"; return 1; }; }
on_gpu()   { local g=$1; shift; CUDA_VISIBLE_DEVICES=${GPUS[$g]} "$@"; }
wait_all() { local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }
hours_s()  { awk -v h="$1" -v f="$2" -v a="$3" 'BEGIN { printf "%d", h * 3600 * f + a }'; }
sps_of()   { grep -ao '[0-9.]* samp/s' "$1" 2> /dev/null | tail -n "${2:-4}" \
               | awk '{ s += $1; n++ } END { if (n) printf "%.3f", s / n }'; }
report()   { local t; t=$(mktemp "$OUT/REPORT.md.XXXXXX") || return 1
             python scripts/weekend_report.py "$OUT" > "$t" 2> "$OUT/report.$RTAG.err" && mv "$t" "$OUT/REPORT.md" || rm -f "$t"; }
base_model() { python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['base_model'])" "$1"; }
media_of() { python -c "import sys; sys.path.insert(0, 'src'); from live3r.data.mixture import load_registry
print(load_registry(sys.argv[1]).datasets[sys.argv[2]].media_root)" "$REGISTRY" "$1"; }
wait_marker() {   # marker, hours before giving up, [the role that produces it: stop waiting once it failed/skipped]
  local n=$1 limit role=${3:-}; limit=$(hours_s "${2:-1}" 1 0); local t=0
  while ! is_done "$n"; do
    if [[ -n "$role" && ( -f "$OUT/role.$role.failed" || -f "$OUT/role.$role.skipped" ) ]]; then
      log "$n will not come: role $role $( [[ -f "$OUT/role.$role.failed" ]] && echo failed || echo was skipped)"; return 1
    fi
    (( t >= limit )) && { fail "waited ${2:-1} h for $n - not produced (is the node that makes it running?)"; return 1; }
    sleep 60; t=$(( t + 60 ))
  done
}
with_lock() {     # a shared-filesystem lock for one-time work any node may do (mkdir is atomic on NFS too)
  local name=$1 t=0 max=${LOCK_WAIT_S:-21600}; shift
  until mkdir "$OUT/lock.$name" 2> /dev/null; do
    (( t >= max )) && { fail "lock.$name held for $(( max / 3600 )) h - if that node died, rmdir $OUT/lock.$name"; return 1; }
    sleep 15; t=$(( t + 15 ))
  done
  echo "$HOST $$ $(date '+%F %T')" > "$OUT/lock.$name/owner"
  "$@"; local rc=$?
  rm -rf "$OUT/lock.$name"
  return $rc
}
alive() { [[ -n "$(find "$OUT/alive.$1" -mmin -30 2> /dev/null)" ]]; }   # heartbeat within 30 min
_claim() {   # role: 0 if this node may run it now
  local r=$1 d="$OUT/claim.$1" owner
  if mkdir "$d" 2> /dev/null; then echo "$HOST" > "$d/owner"; return 0; fi
  owner=$(cat "$d/owner" 2> /dev/null)
  [[ "$owner" == "$HOST" ]] && return 0
  if [[ -n "$owner" ]] && ! alive "$owner"; then
    echo "$HOST" > "$d/owner"; log "took over role $r from $owner (no heartbeat for 30 min)"; return 0
  fi
  return 1
}
claim() { with_lock claims _claim "$1"; }

# ============================================================================================ p0 environment
p0_local() {
  local L="$OUT/p0_$RTAG.log" MODEL
  local CHECK="import torch, transformers, peft, lmms_eval, roma, omegaconf, scipy, yaml
import transformers.models.qwen3_5
print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)
n = torch.cuda.device_count(); print(n, 'GPUs'); assert n in (4, 8), f'this plan runs on 4 or 8 GPUs per node, found {n}'"
  if ! python -c "$CHECK" >> "$L" 2>&1; then
    log "p0: packages missing on $HOST - installing requirements (without flash-attn)"
    grep -v '^flash-attn' requirements.txt > "$OUT/requirements.noflash.txt"
    with_lock pip pip install -r "$OUT/requirements.noflash.txt" >> "$L" 2>&1
    python -c "$CHECK" >> "$L" 2>&1 || { fail "p0: environment on $HOST - $L"; return 1; }
  fi
  # the fused Gated DeltaNet (run 2's d0 compared it with the torch fallback; a mismatch there bans it here too)
  [[ -f "$R2/FLA_BAN" ]] && touch "$OUT/FLA_BAN"
  if [[ ! -f "$OUT/FLA_BAN" ]] && ! python -c "import fla" >> "$L" 2>&1; then
    log "p0: installing flash-linear-attention on $HOST (--no-deps)"
    { with_lock pip pip install -q --no-deps flash-linear-attention && { python -c "import einops" 2> /dev/null \
        || with_lock pip pip install -q einops; }; } >> "$L" 2>&1 \
      || log "p0 warning: flash-linear-attention did not install - the Gated DeltaNet stays on the torch fallback"
  fi
  if [[ -f "$OUT/FLA_BAN" ]] && python -c "import fla" 2> /dev/null; then
    with_lock pip pip uninstall -y flash-linear-attention >> "$L" 2>&1
  fi
  with_lock pip pip install -e . --no-deps -q >> "$L" 2>&1 || { fail "p0: pip install -e . on $HOST - $L"; return 1; }
  MODEL=$(base_model "$CFG") || { fail "p0: cannot read base_model from $CFG"; return 1; }
  [[ -f "$MODEL/config.json" ]] || { fail "p0: $MODEL missing"; return 1; }
  [[ -f data/eval/vsibench/test.jsonl ]] || { fail "p0: data/eval/vsibench missing"; return 1; }
  python -c "import sys; sys.path.insert(0, 'src'); from live3r.data.mixture import load_registry, resolve_mix
bad = [s.name for s in resolve_mix(sys.argv[1], load_registry(sys.argv[2])) if not s.annotation.is_file()]
assert not bad, f'annotation files missing: {bad}'" "$MIX" "$REGISTRY" >> "$L" 2>&1 \
    || { fail "p0: the $MIX annotation files are not all readable from $HOST - $L"; return 1; }
  # the suite on the shared checkout: one node at a time, nothing written into it (two nodes rewriting the same
  # __pycache__ at once is the prime suspect for the 09-30 attempt's collection error in tests/test_sim3.py)
  if ! with_lock tests env PYTHONDONTWRITEBYTECODE=1 LIVE3R_TOKENIZER="$MODEL" \
       python -m pytest tests -q -p no:cacheprovider --import-mode=prepend >> "$L" 2>&1; then
    fail "p0: tests on $HOST - $(grep -aE '^(ERROR|FAILED|E  )' "$L" | head -4 | tr '\n' ' ' | cut -c1-400) (all: $L)"
    return 1
  fi
}

# ============================================================================================ data + calib (once)
n_train() {   # samples in one pass of the prepared mix (the same count MixtureDataset builds)
  python - "$MIX" "$REGISTRY" "$PREP" << 'EOF'
import math, sys
sys.path.insert(0, "src")
import numpy as np
from live3r.data.mixture import load_registry, resolve_mix
n = 0
for s in resolve_mix(sys.argv[1], load_registry(sys.argv[2])):
    p = s.annotation_for_training(sys.argv[3])
    k = len(np.load(str(p) + ".idx.npy", mmap_mode="r")) if str(p).endswith(".jsonl") else (s.records or 0)
    n += int(math.floor(s.fraction)) * k + int(round((s.fraction - math.floor(s.fraction)) * k))
print(n)
EOF
}
_data() {
  is_done data && return 0
  log "data: the data pass (scripts/inspect_reference_data.py, CPU) on $HOST"
  timeout 4h python scripts/inspect_reference_data.py --registry "$REGISTRY" --out "$OUT/reference_data" \
    > "$OUT/data_pass.log" 2>&1 || log "data pass failed - $OUT/data_pass.log (not fatal: preparation checks the files)"
  log "data: preparing $MIX -> $PREP (JSONL + index per source; marks drawn where certain; VSI-Bench scenes out)"
  local marks; marks=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['data'].get('marks', 'skip'))" "$CFG")
  [[ "$marks" == ignore ]] && marks=keep
  timeout 8h python scripts/prepare_mix.py --mix "$MIX" --registry "$REGISTRY" --out "$PREP" --marks "$marks" \
    --exclude-scenes configs/data/vsibench_scenes.txt > "$OUT/prepare_mix.log" 2>&1 \
    || { fail "data: preparation - $OUT/prepare_mix.log"; return 1; }
  cp "$PREP/PREPARED.md" "$OUT/PREPARED.md"
  local n; n=$(n_train) || { fail "data: cannot count the prepared mix"; return 1; }
  log "data: $n samples in one pass of $MIX"
  mark data
}
plan() {   # budget.env from the calibrated speed of the gated arm
  python - "$(sps_of "$OUT/calib.log")" "$TRAIN_HOURS" "$EFF_BATCH" "$(n_train)" > "$OUT/budget.env.tmp" << 'EOF' || return 1
import math, sys
sps = float(sys.argv[1]) if sys.argv[1] else 0.0
hours, eff, n = float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
target = n if (sps <= 0 or hours <= 0) else min(n, hours * 3600 * sps * 0.95)
st = max(50, int(target // eff))
print(f"N_TRAIN={n}\nEFF_BATCH={eff}\nCALIB_SPS={sps:.3f}\nSTEPS={st}\nEPOCHS={max(1, math.ceil(st * eff / n))}"
      f"\nSAVE={max(10, math.ceil(st / 4))}\nPROBE={max(5, math.ceil(st / 8))}\nRESUME_EVERY={max(10, min(200, math.ceil(st / 20)))}")
EOF
  mv "$OUT/budget.env.tmp" "$OUT/budget.env"
}
train_cmd() {   # the trainer's common arguments
  echo -m live3r.train.train --mix "$MIX" --registry "$REGISTRY" --prepared-dir "$PREP" --grad-checkpointing \
    --stage sft --grad-accum "$ACCUM" --lr "$LR" --lora-lr "$LORA_LR" --weight-decay "$WD" --warmup-ratio 0.03 \
    --adam-beta2 "$BETA2"
}
_calib() {
  is_done calib && return 0
  log "calib: 12 steps of the gated arm at the production settings ($NGPU GPUs x accum $ACCUM) on $HOST"
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$ALLG timeout 4h torchrun --nproc_per_node "$NGPU" --master_port 29531 $(train_cmd) \
    --config "$CFG" --output "$OUT/calib" \
    --max-steps 12 --save-every 0 --log-every 2 --dump-samples 3 > "$OUT/calib.log" 2>&1 \
    || { fail "calib - $OUT/calib.log"; return 1; }
  grep -aq "sync OK" "$OUT/calib.log" || { fail "calib: ranks not in sync"; return 1; }
  plan || { fail "calib: budget could not be written"; return 1; }
  # shellcheck disable=SC1090
  source "$OUT/budget.env"
  log "calib: $CALIB_SPS samples/s per node (the reference: 6.48) -> $STEPS steps x $EFF_BATCH =" \
      "$(awk -v s="$STEPS" -v e="$EFF_BATCH" -v n="$N_TRAIN" 'BEGIN { printf "%.2f", s * e / n }') of a pass of $N_TRAIN," \
      "$(awk -v s="$STEPS" -v e="$EFF_BATCH" -v r="$CALIB_SPS" 'BEGIN { printf "%.0f", (r > 0 ? s * e / r / 3600 : 0) }') h per arm"
  # d1: where one training step's time goes (one GPU, a few minutes)
  # shellcheck disable=SC2046
  on_gpu 0 timeout 2h python $(train_cmd | sed 's/--grad-accum [0-9]*//') --config "$CFG" --grad-accum 6 \
    --max-steps 1 --profile 4 --save-every 0 --log-every 1 --output "$OUT/d1_profile" > "$OUT/d1_profile.log" 2>&1 \
    && mark d1 || log "d1 profile failed - $OUT/d1_profile.log (not fatal)"
  if awk -v h="$TRAIN_HOURS" -v st="$STEPS" -v e="$EFF_BATCH" -v n="$N_TRAIN" -v g="$GO_EPOCHS" \
       'BEGIN { exit !(h > 0 && st * e / n < g) }'; then
    printf 'at the calibrated %s samples/s per node, TRAIN_HOURS %s covers %s of a pass of %s (< GO_EPOCHS %s).\n' \
      "$CALIB_SPS" "$TRAIN_HOURS" "$(awk -v st="$STEPS" -v e="$EFF_BATCH" -v n="$N_TRAIN" 'BEGIN { printf "%.2f", st * e / n }')" \
      "$N_TRAIN" "$GO_EPOCHS" > "$OUT/NO_GO"
    log "NO-GO: $(cat "$OUT/NO_GO") Training roles are skipped; base and the diagnostics still run."
  fi
  mark calib
}
load_budget() { wait_marker calib 12 || return 1; source "$OUT/budget.env"; }   # shellcheck disable=SC1090

# ============================================================================================ training
train_arm() {   # name config args...
  local name=$1 cfg=$2; shift 2
  is_done "$name" && return 0
  rm -f "$OUT/$name/final.pt"
  [[ -f "$OUT/$name/resume.pt" ]] && log "$name: resuming from $OUT/$name/resume.pt"
  log "$name started on $HOST ($cfg, $STEPS steps)"
  local cap=()
  if awk -v h="$TRAIN_HOURS" 'BEGIN { exit !(h > 0) }'; then
    cap=(timeout --kill-after=10m "$(hours_s "$TRAIN_HOURS" 1.25 3600)")
  fi
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$ALLG ${cap[@]+"${cap[@]}"} \
    torchrun --nproc_per_node "$NGPU" --master_port 29512 $(train_cmd) --config "$cfg" --output "$OUT/$name" --resume \
      --epochs "$EPOCHS" --max-steps "$STEPS" --max-hours "$(awk -v h="$TRAIN_HOURS" 'BEGIN { print h * 1.25 }')" \
      --log-every 20 --save-every "$SAVE" --resume-every "$RESUME_EVERY" \
      --probe-ann "$PREP/$PROBE_SRC.holdout.jsonl" --probe-media-root "$(media_of "$PROBE_SRC")" \
      --probe-every "$PROBE" --probe-n 32 --probe-gen 6 "$@" >> "$OUT/$name.log" 2>&1
  [[ -f "$OUT/$name/final.pt" ]] || { fail "$name - $OUT/$name.log"; return 1; }
  mark "$name"
}

# ============================================================================================ evaluation
vsi_sharded() {   # name args... -> $OUT/vsi/<name>.json: 8 shards on this node's GPUs, merged
  local name=$1 k pids=() files=(); shift
  [[ -f "$OUT/vsi/$name.json" ]] && return 0
  log "VSI $name started on $HOST ($SHARDS shards)"
  for (( k = 0; k < SHARDS; k++ )); do
    files+=("$OUT/vsi/shards/$name.$k.json")
    [[ -f "$OUT/vsi/shards/$name.$k.json" ]] && continue
    on_gpu "$k" timeout "${VSI_HOURS}h" python scripts/eval_vsi_local.py "$@" "${VSI[@]}" --shard "$k" \
      --num-shards "$SHARDS" --out "$OUT/vsi/shards/$name.$k.json" > "$OUT/vsi/shards/$name.$k.log" 2>&1 &
    pids+=($!)
  done
  wait_all ${pids[@]+"${pids[@]}"} || { fail "VSI $name - $OUT/vsi/shards/$name.*.log"; return 1; }
  python scripts/merge_vsi_shards.py "$OUT/vsi/$name.json" "${files[@]}" > "$OUT/vsi/$name.log" 2>&1 \
    || { fail "VSI $name merge - $OUT/vsi/$name.log"; return 1; }
  log "VSI $name done: $(tail -n 1 "$OUT/vsi/$name.log" | sed 's/.*·//')"
}
eval_arm() {   # name cfg: the live path (32 uniform keyframes as images) plain and with the room facts
  local name=$1 cfg=$2 w="$OUT/$1/final.pt" rc=0
  have "$w" "VSI $name" || return 1
  vsi_sharded "$name" --config "$cfg" --weights "$w" --modes oracle-image || rc=1
  vsi_sharded "${name}_routed_room" --config "$cfg" --weights "$w" "${ROUTED_ROOM[@]}" --modes oracle-image || rc=1
  return $rc
}
curve() {   # VSI at the step checkpoints (every quarter)
  local arm=$1 cfg=$2 f n rc=0
  for f in "$OUT/$arm"/step*.pt; do
    [[ -f "$f" ]] || continue
    n=${f##*/step}; n=${n%.pt}
    [[ "$n" == "$STEPS" ]] && continue
    vsi_sharded "${arm}_step$n" --config "$cfg" --weights "$f" --modes oracle-image || rc=1
  done
  return $rc
}
gate_results() { find "$1" -name "*results*.json" 2> /dev/null | grep -q .; }
gate_env() { if [[ -f "$WK/p2_videomme.done" ]]; then echo GENERAL=videomme; else echo GENERAL=mmstar REFERENCE=; fi; }
gate_base() {   # gpu step(base|base_video): the untrained model at this resolution, no geometry
  local g=$1 s=$2 genv; read -r -a genv <<< "$(gate_env)"
  [[ "$s" == base_video ]] && genv=()
  is_done "gate_$s" && return 0
  log "gate $s started on $HOST GPU $g"
  on_gpu "$g" env ${genv[@]+"${genv[@]}"} ONLY="$s" timeout 20h bash scripts/run_gate.sh "$CFG_PLAIN" none "$OUT/gate" \
    > "$OUT/gate_$s.log" 2>&1
  gate_results "$OUT/gate/$s" || { fail "gate $s: no results written - $OUT/gate_$s.log"; return 1; }
  mark "gate_$s"
}
need_base() {   # gpu step: the base run exists, or this node makes it (a node already on it holds the lock)
  is_done "gate_$2" || LOCK_WAIT_S=90000 with_lock "gate_$2" gate_base "$1" "$2"
}
gate_arm() {    # gpu arm cfg: the trained arm, then its verdict against the two base runs
  local g=$1 arm=$2 cfg=$3 d="$OUT/gate_$2" genv; read -r -a genv <<< "$(gate_env)"
  is_done "gate_$arm" && return 0
  have "$OUT/$arm/final.pt" "gate $arm" || return 1
  mkdir -p "$d"; ln -sfn ../gate/base "$d/base"; ln -sfn ../gate/base_video "$d/base_video"
  if ! gate_results "$d/trained"; then
    log "gate $arm started on $HOST GPU $g"
    on_gpu "$g" env "${genv[@]}" ONLY=trained timeout 20h bash scripts/run_gate.sh "$cfg" "$OUT/$arm/final.pt" "$d" \
      > "$OUT/gate_$arm.log" 2>&1
    gate_results "$d/trained" || { fail "gate $arm: no results written - $OUT/gate_$arm.log"; return 1; }
  fi
  { need_base "$g" base && need_base "$g" base_video; } || return 1
  env "${genv[@]}" ONLY=check bash scripts/run_gate.sh "$cfg" "$OUT/$arm/final.pt" "$d" > "$OUT/gate_check_$arm.txt" 2>&1
  mark "gate_$arm"
}
ablation() {  # gpu name real control cfg: holdout loss with real / shuffled / no geometry, real vs control weights
  local g=$1 name=$2 real=$3 ctrl=$4 cfg=$5
  [[ -f "$OUT/$name.json" ]] && return 0
  { have "$real" "$name" && have "$ctrl" "$name"; } || return 1
  on_gpu "$g" timeout 4h python scripts/eval_geometry_ablation.py --config "$cfg" --stage sft --weights "$real" \
    --control-weights "$ctrl" --ann "$PREP/$PROBE_SRC.holdout.jsonl" --media-root "$(media_of "$PROBE_SRC")" --n 300 \
    --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || { fail "$name - $OUT/$name.log"; return 1; }
}

# ============================================================================================ roles
role_gated() {
  local rc=0
  train_arm gated "$CFG" || return 1
  eval_arm gated "$CFG" || rc=1
  vsi_sharded gated_video --config "$CFG" --weights "$OUT/gated/final.pt" --modes oracle-video || rc=1
  report
  gate_arm $(( NGPU - 1 )) gated "$CFG" & local g=$!     # the last GPU, alongside the curve on the others
  SHARDS=$(( NGPU - 1 )) curve gated "$CFG" || rc=1
  wait $g || rc=1
  if [[ ! -f "$OUT/latency_stream.json" ]]; then
    on_gpu 0 timeout 1h python scripts/bench_latency.py --config "$CFG" --frames 512 --offline --height 288 --width 512 \
      --out "$OUT/latency_stream.json" > "$OUT/latency_stream.log" 2>&1 || fail "latency (stream) - $OUT/latency_stream.log"
  fi
  grep -aqs "TTFT" "$OUT/latency_live.log" || on_gpu 0 timeout 1h python scripts/bench_latency.py --config "$CFG" \
    --mode deferred --scene-map --frames 512 --height 288 --width 512 > "$OUT/latency_live.log" 2>&1 \
    || fail "latency (live) - $OUT/latency_live.log"
  return $rc
}
role_plain() {
  local rc=0 a b c
  train_arm plain "$CFG_PLAIN" || return 1
  eval_arm plain "$CFG_PLAIN" || rc=1
  report
  # the lmms-eval gate: a 4-sample debug smoke first (lmms-eval swallows exceptions below DEBUG), then the two base
  # runs every arm's verdict is taken against (each trained arm runs its own gate when its role gets there)
  if ! is_done gate_smoke; then
    on_gpu 0 env GENERAL=mmstar REFERENCE= ONLY=trained LMMS_EXTRA="--limit 4 --verbosity DEBUG" timeout 2h \
      bash scripts/run_gate.sh "$CFG_PLAIN" "$OUT/plain/final.pt" "$OUT/gate_smoke" > "$OUT/gate_smoke.log" 2>&1
    if gate_results "$OUT/gate_smoke/trained"; then mark gate_smoke
    else fail "gate smoke: lmms-eval wrote no results - $OUT/gate_smoke.log has the exception; gate skipped"; return 1; fi
  fi
  need_base 0 base & a=$!
  need_base 1 base_video & b=$!
  sleep 5                                # the two base runs take their locks before the arm's check waits on them
  gate_arm 2 plain "$CFG_PLAIN" & c=$!
  wait_all $a $b $c || rc=1
  return $rc
}
role_control() {
  local rc=0
  train_arm control "$CFG" --geom-control shuffled --donor-buffer "$DONOR_BUFFER" || return 1
  eval_arm control "$CFG" || rc=1
  { wait_marker gated 400 gated && ablation 0 ablation_gated "$OUT/gated/final.pt" "$OUT/control/final.pt" "$CFG"; } || rc=1
  report
  return $rc
}
role_base() {
  local rc=0 r
  vsi_sharded base --config "$CFG_PLAIN" --modes oracle-image || rc=1
  vsi_sharded base_routed_room --config "$CFG_PLAIN" "${ROUTED_ROOM[@]}" --modes oracle-image || rc=1
  for r in s2_real s2_sft s2_control base_hint base_plain; do    # run 2 (128 tokens per frame, Sensenova only)
    [[ -f "$R2/vsi/$r.json" && ! -f "$OUT/vsi/r2_$r.json" ]] && cp "$R2/vsi/$r.json" "$OUT/vsi/r2_$r.json"
  done
  report
  return $rc
}
role_gated_wn() {
  local rc=0
  train_arm gated_wn "$CFG_WN" || return 1
  eval_arm gated_wn "$CFG_WN" || rc=1
  report
  gate_arm 0 gated_wn "$CFG_WN" || rc=1
  return $rc
}

# ============================================================================================ run
log "started on $HOST (GPUs $ALLG, accum $ACCUM) at $(git log --oneline -1 2> /dev/null) - roles: ${QUEUE[*]}"
touch "$OUT/alive.$HOST"
( n=0; while true; do touch "$OUT/alive.$HOST"; (( n % 6 == 5 )) && report; n=$(( n + 1 )); sleep 300; done ) &
REFRESH_PID=$!
trap 'kill "$REFRESH_PID" 2> /dev/null' EXIT
if ! is_done "p0.$HOST"; then
  log "p0 on $HOST"
  if p0_local; then mark "p0.$HOST"; else exit 1; fi
fi
{ is_done data || with_lock data _data; } && wait_marker data 12 || exit 1
{ is_done calib || with_lock calib _calib; } && load_budget || exit 1
report
for r in "${QUEUE[@]}"; do
  case " ${ALL_ROLES[*]} " in *" $r "*) ;; *) fail "unknown role '$r'"; continue ;; esac
  is_done "role.$r" && continue
  [[ -f "$OUT/role.$r.skipped" ]] && continue
  if [[ "$ROLE" == auto ]]; then claim "$r" || continue; fi
  if [[ "$r" != base && -f "$OUT/NO_GO" ]]; then
    log "role $r skipped: NO_GO - $(cat "$OUT/NO_GO")"; touch "$OUT/role.$r.skipped"; continue
  fi
  log "role $r starts on $HOST"
  rm -f "$OUT/role.$r.failed"
  if "role_$r"; then mark "role.$r"
  else touch "$OUT/role.$r.failed"; fail "role $r ended with failures (see the lines above)"; fi
  report
done
report && log "REPORT: $OUT/REPORT.md"
left=(); for r in "${ALL_ROLES[@]}"; do is_done "role.$r" || [[ -f "$OUT/role.$r.skipped" ]] || left+=("$r"); done
log "finished on $HOST$([[ ${#left[@]} -gt 0 ]] && echo " - still running or failed elsewhere: ${left[*]}")"
