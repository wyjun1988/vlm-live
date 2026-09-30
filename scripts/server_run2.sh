#!/usr/bin/env bash
# Run 2: one training round, one epoch per arm, as the new reference - and every weekend checkpoint re-evaluated on
# the same stack. Why and how to read it: docs/SERVER_RUN2.md (the analysis behind it: docs/SERVER_RESULTS_20260929.md).
#
# ONE command, run to the end without anyone stepping in:
#   cd /group-volume/wooyeol/vlm-live && git pull && mkdir -p outputs/run2
#   nohup bash scripts/server_run2.sh > outputs/run2/master.$(hostname -s).log 2>&1 &
# On one node (8 H100) it does everything in order (~3 days if the throughput fixes work). The SAME command on more
# nodes shares the work: each node claims the next unfinished role from the queue below (a claim lives on the shared
# volume; a node whose heartbeat stops for 30 min loses its claims, and the role resumes on the next node free).
#   tail -f outputs/run2/STATUS           # every node writes here, lines tagged with its host
#   bash scripts/server_run2.sh stop      # on a node: ends that node's run and everything it started
#
#   first        p0 environment on every node (installs flash-linear-attention if missing); once: d0 diagnostic
#                (scripts/diagnose_throughput.py, ~10 min: patch embed, fla vs torch), calib (12 S1 + 12 S2 steps at
#                the production settings -> the measured speed sizes every arm, budget.env) and d1 (a torch.profiler
#                table of one training step). If at the measured speed S2_HOURS covers less than GO_EPOCHS of an epoch,
#                NO_GO is written: the training roles are skipped with the reason, the rest still runs.
#   real         S1 real (short) -> S2 real (1 epoch) -> VSI plain / hint / room facts, S1, learning curve
#   control      the same with shuffled geometry (control) -> VSI -> holdout ablations S1 + S2 (real vs control)
#   sft          plain SFT, no geometry (1 epoch) -> VSI -> latency -> lmms-eval gate (debug smoke first)
#   recover      the weekend: base runs + every checkpoint re-evaluated on this stack (wk_*), ablations
#   joint        S2 joint (no S1, 1 epoch) -> VSI
# ROLE=real,control,... runs just those roles, in that order, without claims (the old one-role-per-node mode).
# Resumable: a finished step leaves a marker and is skipped on a re-run; a training arm continues from resume.pt.
set -uo pipefail

# ------------------------------------------------------------------------------------------------ settings
ROOT="${ROOT:-/group-volume/wooyeol/vlm-live}"
CFG="${CFG:-configs/server_4b_v2.yaml}"         # linear patch embed; everything in this run uses it
CFG_2B="${CFG_2B:-configs/server_2b_v2.yaml}"
MEDIA="${MEDIA:-data/sensenova_media}"
OUT="${OUT:-outputs/run2}"
WK="${WK:-outputs/weekend}"                     # the weekend's checkpoints and results (re-evaluated, not modified)
export HF_HOME="${HF_HOME:-/group-volume/wooyeol/hf_cache}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Budget: S1 is short (alignment only - the weekend S1 learned the answer format), S2 one epoch; both capped by
# hours at the calibrated speed. Pairs share one budget file, so real and control see the same steps.
S1_FRAC="${S1_FRAC:-0.05}"          # S1 samples as a fraction of an epoch (0.05 = 41.5k; the weekend had 13.6k)
S2_EPOCHS_WANT="${S2_EPOCHS_WANT:-1}"
S1_HOURS="${S1_HOURS:-6}"; S2_HOURS="${S2_HOURS:-72}"; JOINT_HOURS="${JOINT_HOURS:-$S2_HOURS}"
JOINT="${JOINT:-1}"                 # the joint role (S2 without S1) is in the queue
# go / no-go, decided by the script: if at the calibrated speed the S2 cap covers less than this many epochs, a
# "1 epoch" arm would silently become another small-fraction run (the weekend's 2 %) - the training roles are skipped.
# 0.5 epoch in 72 h = 1.6 samples/s per node (the weekend stack ran at 0.5).
GO_EPOCHS="${GO_EPOCHS:-0.5}"
EFF_BATCH=128; ACCUM=16             # 8 GPUs x 16 micro-steps
DONOR_BUFFER="${DONOR_BUFFER:-64}"  # control arm: donors kept (the weekend's 8 gave a same-shape donor 60 % of the time)
LR=3e-5; LORA_LR=1e-4
VSI=(--videos 288 --seed 1)         # all of VSI-Bench: 288 videos, 5,130 questions
SHARDS=8; VSI_HOURS="${VSI_HOURS:-12}"   # one VSI run = 8 shards on the node's 8 GPUs, then a merge
# the fact set for trained models (SERVER_RESULTS §4): room facts only - distance facts are now below the trained
# models, direction facts wait for the left/right mirror fix. Carries the format hint (route_facts adds it).
ROUTED_ROOM=(--scene-map --no-map-image --route-facts)
ALL8=0,1,2,3,4,5,6,7

cd "$ROOT" || { echo "no $ROOT"; exit 1; }
mkdir -p "$OUT/vsi/shards" "$OUT/gate"
echo run2 > "$OUT/RUN"
HOST=$(hostname -s 2> /dev/null || hostname)

if [[ "${1:-}" == stop ]]; then
  pid=$(cat "$OUT/PID.$HOST" 2> /dev/null) || { echo "no $OUT/PID.$HOST - nothing running from this node"; exit 1; }
  kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill -TERM "$1" 2> /dev/null; }
  kill_tree "$pid"
  echo "$(date '+%F %T') [$HOST] stopped by hand (pid $pid)" >> "$OUT/STATUS"
  exit 0
fi
ROLE="${ROLE:-auto}"
if [[ "$ROLE" == auto ]]; then
  RTAG=$HOST
  QUEUE=(real control sft recover); [[ "$JOINT" == 1 ]] && QUEUE+=(joint)
else
  RTAG=${ROLE//,/+}
  QUEUE=(${ROLE//,/ })
fi
echo $$ > "$OUT/PID.$HOST"

# ----------------------------------------------------------------------------------------------- helpers
log()      { local m; m="$(date '+%F %T') [$RTAG] $*"; echo "$m" >> "$OUT/STATUS"; echo "$m" >&2; }
fail()     { log "FAILED $*"; }
is_done()  { [[ -f "$OUT/$1.done" ]]; }
mark()     { touch "$OUT/$1.done"; log "$1 done"; }
have()     { [[ -f "$1" ]] || { log "$2 skipped: $1 missing"; return 1; }; }
on_gpu()   { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g "$@"; }
wait_all() { local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }
hours_s()  { awk -v h="$1" -v f="$2" -v a="$3" 'BEGIN { printf "%d", h * 3600 * f + a }'; }
sps_of()   { grep -ao '[0-9.]* samp/s' "$1" 2> /dev/null | tail -n "${2:-4}" \
               | awk '{ s += $1; n++ } END { if (n) printf "%.3f", s / n }'; }
report()   { local t; t=$(mktemp "$OUT/REPORT.md.XXXXXX") || return 1
             python scripts/weekend_report.py "$OUT" > "$t" 2> "$OUT/report.$RTAG.err" && mv "$t" "$OUT/REPORT.md" || rm -f "$t"; }
base_model() { python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['base_model'])" "$1"; }
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
  local name=$1 t=0; shift
  until mkdir "$OUT/lock.$name" 2> /dev/null; do
    (( t >= 21600 )) && { fail "lock.$name held for 6 h - if that node died, rmdir $OUT/lock.$name"; return 1; }
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
  [[ "$owner" == "$HOST" ]] && return 0                           # our own claim: a restart on the same node
  if [[ -n "$owner" ]] && ! alive "$owner"; then                   # the claimant stopped: take the role over
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
n = torch.cuda.device_count(); print(n, 'GPUs'); assert n >= 8, f'this plan needs 8 GPUs per node, found {n}'"
  if ! python -c "$CHECK" >> "$L" 2>&1; then
    log "p0: packages missing on $HOST - installing requirements (without flash-attn)"
    grep -v '^flash-attn' requirements.txt > "$OUT/requirements.noflash.txt"
    with_lock pip pip install -r "$OUT/requirements.noflash.txt" >> "$L" 2>&1
    python -c "$CHECK" >> "$L" 2>&1 || { fail "p0: environment on $HOST - $L"; return 1; }
  fi
  # F2: the fused Gated DeltaNet. transformers falls back to pure torch silently when fla is missing (the weekend
  # most likely ran the fallback: ~67k extra kernel launches per sample). d0 compares the two paths once.
  if [[ ! -f "$OUT/FLA_BAN" ]] && ! python -c "import fla" >> "$L" 2>&1; then
    log "p0: installing flash-linear-attention on $HOST (--no-deps: it must not move torch or transformers)"
    { with_lock pip pip install -q --no-deps flash-linear-attention && { python -c "import einops" 2> /dev/null \
        || with_lock pip pip install -q einops; }; } >> "$L" 2>&1 \
      || log "p0 warning: flash-linear-attention did not install - the Gated DeltaNet stays on the torch fallback"
  fi
  with_lock pip pip install -e . --no-deps -q >> "$L" 2>&1 || { fail "p0: pip install -e . on $HOST - $L"; return 1; }
  MODEL=$(base_model "$CFG") || { fail "p0: cannot read base_model from $CFG"; return 1; }
  [[ -f "$MODEL/config.json" ]] || { fail "p0: $MODEL missing (the weekend downloaded it)"; return 1; }
  [[ -f data/sensenova.jsonl && -f data/sensenova.holdout.jsonl ]] || { fail "p0: data/sensenova*.jsonl missing"; return 1; }
  [[ -f data/eval/vsibench/test.jsonl ]] || { fail "p0: data/eval/vsibench missing (weekend p2)"; return 1; }
  LIVE3R_TOKENIZER="$MODEL" python -m pytest tests -q -p no:cacheprovider >> "$L" 2>&1 \
    || { fail "p0: tests on $HOST - the tail of $L is what to send"; return 1; }
}
ban_fla_here() {  # after d0 found a mismatch: take fla out of this node's environment
  [[ -f "$OUT/FLA_BAN" ]] || return 0
  python -c "import fla" 2> /dev/null || return 0
  log "d0 banned fla - uninstalling it on $HOST"
  with_lock pip pip uninstall -y flash-linear-attention >> "$OUT/p0_$RTAG.log" 2>&1
}

# ============================================================================================ d0 + calib (once)
_d0() {
  is_done d0 && return 0
  local MODEL; MODEL=$(base_model "$CFG")
  log "d0: throughput diagnostic on $HOST GPU 0 (~10 min)"
  on_gpu 0 timeout 1h python scripts/diagnose_throughput.py --model "$MODEL" --fla-check > "$OUT/d0_throughput.txt" 2>&1 \
    || { fail "d0 diagnostic - $OUT/d0_throughput.txt"; return 1; }
  grep -a "F1 NEEDED\|Conv3d is fine\|vision tower\|F2 NEEDED\|fused kernels\|fla check" "$OUT/d0_throughput.txt" \
    | while read -r ln; do log "d0: $ln"; done
  if grep -aq "FLA MISMATCH" "$OUT/d0_throughput.txt"; then
    touch "$OUT/FLA_BAN"; log "d0: fla does not match the torch path - banned on every node"
  fi
  mark d0
}
plan() {   # budget.env from the calibrated speeds
  python - "$(sps_of "$OUT/calib_s1.log")" "$(sps_of "$OUT/calib_s2.log")" "$S1_FRAC" "$S2_EPOCHS_WANT" "$S1_HOURS" \
    "$S2_HOURS" "$EFF_BATCH" "$(wc -l < data/sensenova.jsonl)" > "$OUT/budget.env.tmp" << 'EOF' || return 1
import math, sys
s1, s2 = (float(x) if x else 0.0 for x in sys.argv[1:3])
frac, want, h1, h2 = (float(x) for x in sys.argv[3:7])
eff, n = int(sys.argv[7]), int(sys.argv[8])
def fit(target, hours, sps):
    s = target if sps <= 0 else min(target, hours * 3600 * sps * 0.95)
    return int(s // eff)
st1 = max(20, fit(frac * n, h1, s1))
st2 = max(50, fit(want * n, h2, s2))
print(f"N_TRAIN={n}\nCALIB_SPS_S1={s1:.3f}\nCALIB_SPS_S2={s2:.3f}\nS1_STEPS={st1}\nS1_EPOCHS={max(1, math.ceil(st1 * eff / n))}"
      f"\nS2_STEPS={st2}\nS2_EPOCHS={max(1, math.ceil(st2 * eff / n))}\nS1_SAVE={max(10, math.ceil(st1 / 2))}"
      f"\nS1_PROBE={max(5, math.ceil(st1 / 4))}\nS2_SAVE={max(10, math.ceil(st2 / 4))}\nS2_PROBE={max(5, math.ceil(st2 / 8))}"
      f"\nRESUME_EVERY={max(10, min(200, math.ceil(st2 / 20)))}")
EOF
  mv "$OUT/budget.env.tmp" "$OUT/budget.env"
}
_calib() {
  is_done calib && return 0
  local T=(-m live3r.train.train --config "$CFG" --ann data/sensenova.jsonl --media-root "$MEDIA" --grad-checkpointing
           --save-every 0 --log-every 2 --lr "$LR" --grad-accum "$ACCUM")
  log "calib: S1, 12 steps at the production settings (8 GPUs x accum $ACCUM) on $HOST"
  timeout 4h torchrun --nproc_per_node 8 --master_port 29530 "${T[@]}" --stage align --output "$OUT/calib_s1" \
    --max-steps 12 > "$OUT/calib_s1.log" 2>&1 || { fail "calib S1 - $OUT/calib_s1.log"; return 1; }
  grep -aq "sync OK" "$OUT/calib_s1.log" || { fail "calib S1: ranks not in sync"; return 1; }
  log "calib: S2 (LoRA), 12 steps"
  timeout 4h torchrun --nproc_per_node 8 --master_port 29530 "${T[@]}" --stage sft --output "$OUT/calib_s2" \
    --init-from "$OUT/calib_s1/final.pt" --lora-lr "$LORA_LR" --max-steps 12 > "$OUT/calib_s2.log" 2>&1 \
    || { fail "calib S2 - $OUT/calib_s2.log"; return 1; }
  grep -aq "sync OK" "$OUT/calib_s2.log" || { fail "calib S2: ranks not in sync"; return 1; }
  plan || { fail "calib: budget could not be written"; return 1; }
  # shellcheck disable=SC1090
  source "$OUT/budget.env"
  log "calib: S1 $CALIB_SPS_S1 / S2 $CALIB_SPS_S2 samples/s per node (weekend: 0.5) -> S1 $S1_STEPS steps" \
      "($(awk -v s="$S1_STEPS" -v n="$N_TRAIN" 'BEGIN { printf "%.3f", s * 128 / n }') ep), S2 $S2_STEPS steps" \
      "($(awk -v s="$S2_STEPS" -v n="$N_TRAIN" 'BEGIN { printf "%.2f", s * 128 / n }') ep)"
  # d1: where one training step's time goes (one GPU, a few minutes) - read with d0 whatever the verdict
  on_gpu 0 timeout 2h python -m live3r.train.train --config "$CFG" --ann data/sensenova.jsonl --media-root "$MEDIA" \
    --grad-checkpointing --stage sft --init-from "$OUT/calib_s1/final.pt" --lora-lr "$LORA_LR" --lr "$LR" \
    --max-steps 1 --grad-accum 6 --profile 4 --save-every 0 --log-every 1 --output "$OUT/d1_profile" \
    > "$OUT/d1_profile.log" 2>&1 && mark d1 || log "d1 profile failed - $OUT/d1_profile.log (not fatal)"
  if awk -v st="$S2_STEPS" -v n="$N_TRAIN" -v g="$GO_EPOCHS" 'BEGIN { exit !(st * 128 / n < g) }'; then
    printf 'at the calibrated %s samples/s per node one epoch takes %s h; the S2 cap of %s h covers %s epochs (< GO_EPOCHS %s).\n' \
      "$CALIB_SPS_S2" "$(awk -v n="$N_TRAIN" -v s="$CALIB_SPS_S2" 'BEGIN { printf "%.0f", n / (s > 0 ? s : 1e-9) / 3600 }')" \
      "$S2_HOURS" "$(awk -v st="$S2_STEPS" -v n="$N_TRAIN" 'BEGIN { printf "%.2f", st * 128 / n }')" "$GO_EPOCHS" > "$OUT/NO_GO"
    log "NO-GO: $(cat "$OUT/NO_GO") Training roles are skipped; the weekend re-evaluation and the diagnostics" \
        "(d0_throughput.txt, d1_profile/profile.txt) still run - send them."
  fi
  mark calib
}
load_budget() { wait_marker calib 8 || return 1; source "$OUT/budget.env"; }   # shellcheck disable=SC1090

# ============================================================================================ training
train_arm() {   # name hours epochs save probe args...
  local name=$1 hours=$2 epochs=$3 save=$4 probe=$5; shift 5
  is_done "$name" && return 0
  rm -f "$OUT/$name/final.pt"
  [[ -f "$OUT/$name/resume.pt" ]] && log "$name: resuming from $OUT/$name/resume.pt"
  log "$name started on $HOST"
  CUDA_VISIBLE_DEVICES=$ALL8 timeout --kill-after=10m "$(hours_s "$hours" 1.25 3600)" \
    torchrun --nproc_per_node 8 --master_port 29511 -m live3r.train.train \
      --config "$CFG" --ann data/sensenova.jsonl --media-root "$MEDIA" --grad-checkpointing \
      --output "$OUT/$name" --resume --epochs "$epochs" \
      --max-hours "$(awk -v h="$hours" 'BEGIN { print h * 1.25 }')" \
      --grad-accum "$ACCUM" --log-every 20 --save-every "$save" --resume-every "$RESUME_EVERY" --lr "$LR" \
      --probe-ann data/sensenova.holdout.jsonl --probe-every "$probe" --probe-n 32 --probe-gen 6 \
      "$@" >> "$OUT/$name.log" 2>&1
  [[ -f "$OUT/$name/final.pt" ]] || { fail "$name - $OUT/$name.log"; return 1; }
  mark "$name"
}
chain() {   # real|control: S1 (short) -> S2 (one epoch), same seed and data order for both arms
  local arm=$1 ctl=()
  [[ "$arm" == control ]] && ctl=(--geom-control shuffled --donor-buffer "$DONOR_BUFFER")
  train_arm "s1_$arm" "$S1_HOURS" "$S1_EPOCHS" "$S1_SAVE" "$S1_PROBE" --stage align --max-steps "$S1_STEPS" \
    ${ctl[@]+"${ctl[@]}"} || return 1
  train_arm "s2_$arm" "$S2_HOURS" "$S2_EPOCHS" "$S2_SAVE" "$S2_PROBE" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" --init-from "$OUT/s1_$arm/final.pt" ${ctl[@]+"${ctl[@]}"}
}

# ============================================================================================ evaluation
vsi_sharded() {   # name args... -> $OUT/vsi/<name>.json: 8 shards on this node's GPUs, merged
  local name=$1 k pids=() files=(); shift
  [[ -f "$OUT/vsi/$name.json" ]] && return 0
  log "VSI $name started on $HOST ($SHARDS shards)"
  for (( k = 0; k < SHARDS; k++ )); do
    files+=("$OUT/vsi/shards/$name.$k.json")
    [[ -f "$OUT/vsi/shards/$name.$k.json" ]] && continue      # a shard that finished before an interruption
    on_gpu "$k" timeout "${VSI_HOURS}h" python scripts/eval_vsi_local.py "$@" "${VSI[@]}" --shard "$k" \
      --num-shards "$SHARDS" --out "$OUT/vsi/shards/$name.$k.json" > "$OUT/vsi/shards/$name.$k.log" 2>&1 &
    pids+=($!)
  done
  wait_all ${pids[@]+"${pids[@]}"} || { fail "VSI $name - $OUT/vsi/shards/$name.*.log"; return 1; }
  python scripts/merge_vsi_shards.py "$OUT/vsi/$name.json" "${files[@]}" > "$OUT/vsi/$name.log" 2>&1 \
    || { fail "VSI $name merge - $OUT/vsi/$name.log"; return 1; }
  log "VSI $name done: $(tail -n 1 "$OUT/vsi/$name.log" | sed 's/.*·//')"
}
eval_arm() {   # name weights cfg: the three views of a trained arm, each with a same-prompt partner elsewhere
  local name=$1 w=$2 cfg=$3 rc=0
  have "$w" "VSI $name" || return 1
  vsi_sharded "$name" --config "$cfg" --weights "$w" --modes oracle-image || rc=1
  vsi_sharded "${name}_hint" --config "$cfg" --weights "$w" --format-hint --modes oracle-image || rc=1
  vsi_sharded "${name}_routed_room" --config "$cfg" --weights "$w" "${ROUTED_ROOM[@]}" --modes oracle-image || rc=1
  return $rc
}
curve() {   # VSI at the saved S2 checkpoints (every quarter of the plan)
  local arm=$1 f n rc=0
  for f in "$OUT/$arm"/step*.pt; do
    [[ -f "$f" ]] || continue
    n=${f##*/step}; n=${n%.pt}
    [[ "$n" == "$S2_STEPS" ]] && continue          # the last one is final.pt, evaluated as the arm itself
    vsi_sharded "${arm}_step$n" --config "$CFG" --weights "$f" --modes oracle-image || rc=1
  done
  return $rc
}
ablation() {  # gpu name stage real control cfg
  local g=$1 name=$2 stage=$3 real=$4 ctrl=$5 cfg=$6
  [[ -f "$OUT/$name.json" ]] && return 0
  { have "$real" "$name" && have "$ctrl" "$name"; } || return 1
  on_gpu "$g" timeout 4h python scripts/eval_geometry_ablation.py --config "$cfg" --stage "$stage" --weights "$real" \
    --control-weights "$ctrl" --ann data/sensenova.holdout.jsonl --media-root "$MEDIA" --n 500 \
    --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || { fail "$name - $OUT/$name.log"; return 1; }
}
gate_results() { find "$1" -name "*results*.json" 2> /dev/null | grep -q .; }
gate_job() {  # gpu step weights - marked done only when lmms-eval actually wrote results (the weekend's did not)
  local g=$1 s=$2 w=$3 genv=(GENERAL=mmstar REFERENCE=)
  is_done "gate_$s" && return 0
  [[ -f "$WK/p2_videomme.done" ]] && genv=(GENERAL=videomme)
  [[ "$s" == base_video ]] && genv=()
  log "gate $s started on $HOST GPU $g"
  on_gpu "$g" env ${genv[@]+"${genv[@]}"} ONLY="$s" timeout 20h bash scripts/run_gate.sh "$CFG" "$w" "$OUT/gate" \
    > "$OUT/gate_$s.log" 2>&1
  gate_results "$OUT/gate/$s" || { fail "gate $s: no results written - $OUT/gate_$s.log (grep 'Error during')"; return 1; }
  mark "gate_$s"
}

# ============================================================================================ roles
role_real() {
  local rc=0
  chain real || return 1
  eval_arm s2_real "$OUT/s2_real/final.pt" "$CFG" || rc=1
  vsi_sharded s1_real --config "$CFG" --weights "$OUT/s1_real/final.pt" --modes oracle-image || rc=1
  report
  curve s2_real || rc=1
  return $rc
}
role_control() {
  local rc=0 a b
  chain control || return 1
  eval_arm s2_control "$OUT/s2_control/final.pt" "$CFG" || rc=1
  vsi_sharded s1_control --config "$CFG" --weights "$OUT/s1_control/final.pt" --modes oracle-image || rc=1
  { wait_marker s1_real 48 real && ablation 0 p5_ablation_s1 align "$OUT/s1_real/final.pt" "$OUT/s1_control/final.pt" "$CFG"; } & a=$!
  { wait_marker s2_real 48 real && ablation 1 p7_ablation_s2 sft "$OUT/s2_real/final.pt" "$OUT/s2_control/final.pt" "$CFG"; } & b=$!
  wait_all $a $b || rc=1
  report
  return $rc
}
role_sft() {
  local rc=0 a b c
  # plain SFT: no geometry at all. real - this = the value of the whole geometry path.
  train_arm s2_sft "$S2_HOURS" "$S2_EPOCHS" "$S2_SAVE" "$S2_PROBE" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" --no-geometry || return 1
  eval_arm s2_sft "$OUT/s2_sft/final.pt" "$CFG" || rc=1
  report
  # latency on an idle node (gate 3), then the lmms-eval gate: a 4-sample debug smoke first, since the weekend's
  # gate runs exited 0 without results
  if [[ ! -f "$OUT/latency_stream.json" ]]; then
    on_gpu 0 timeout 1h python scripts/bench_latency.py --config "$CFG" --frames 512 --offline \
      --out "$OUT/latency_stream.json" > "$OUT/latency_stream.log" 2>&1 || fail "latency (stream) - $OUT/latency_stream.log"
  fi
  grep -aqs "TTFT" "$OUT/latency_live.log" || on_gpu 0 timeout 1h python scripts/bench_latency.py --config "$CFG" \
    --mode deferred --scene-map --frames 512 > "$OUT/latency_live.log" 2>&1 || fail "latency (live) - $OUT/latency_live.log"
  if ! is_done gate_smoke; then
    on_gpu 0 env GENERAL=mmstar REFERENCE= ONLY=base LMMS_EXTRA="--limit 4 --verbosity DEBUG" timeout 2h \
      bash scripts/run_gate.sh "$CFG" "$OUT/s2_sft/final.pt" "$OUT/gate_smoke" > "$OUT/gate_smoke.log" 2>&1
    if gate_results "$OUT/gate_smoke/base"; then mark gate_smoke
    else fail "gate smoke: lmms-eval wrote no results - $OUT/gate_smoke.log has the exception (DEBUG); gate skipped"; return 1; fi
  fi
  gate_job 0 base "$OUT/s2_sft/final.pt" & a=$!
  gate_job 1 base_video "$OUT/s2_sft/final.pt" & b=$!
  { wait_marker s2_real 48 real && gate_job 2 trained "$OUT/s2_real/final.pt"; } & c=$!
  wait_all $a $b $c || rc=1
  if is_done gate_base && is_done gate_base_video && is_done gate_trained; then
    local genv=(GENERAL=mmstar REFERENCE=)
    [[ -f "$WK/p2_videomme.done" ]] && genv=(GENERAL=videomme)
    env "${genv[@]}" ONLY=check bash scripts/run_gate.sh "$CFG" "$OUT/s2_real/final.pt" "$OUT/gate" \
      > "$OUT/gate_check.txt" 2>&1 && mark gate_check || { fail "gate check - $OUT/gate_check.txt"; rc=1; }
  fi
  return $rc
}
role_recover() {
  local rc=0 arm MODEL M2B a b c d
  MODEL=$(base_model "$CFG"); M2B=$(base_model "$CFG_2B")
  if ! is_done recover_vsi; then
    # the old-stack result of the same weights, for the stack check (new stack - old stack should be ~0)
    [[ -f "$WK/vsi/s2_real.json" && ! -f "$OUT/vsi/old_s2_real.json" ]] && cp "$WK/vsi/s2_real.json" "$OUT/vsi/old_s2_real.json"
    # in order of what the reading needs: rule 1 and its partners first
    for arm in s2_real s2_control s2_sft; do
      [[ -f "$WK/$arm/final.pt" ]] && { vsi_sharded "wk_$arm" --config "$CFG" --weights "$WK/$arm/final.pt" --modes oracle-image || rc=1; }
    done
    vsi_sharded base_hint --model "$MODEL" --patch-embed linear --format-hint --modes oracle-image || rc=1
    for arm in s2_real_seed1 s2_control_seed1 s2_joint s1_real s1_control; do
      [[ -f "$WK/$arm/final.pt" ]] && { vsi_sharded "wk_$arm" --config "$CFG" --weights "$WK/$arm/final.pt" --modes oracle-image || rc=1; }
    done
    for arm in s2_real s2_control s2_sft; do
      [[ -f "$WK/$arm/final.pt" ]] && { vsi_sharded "wk_${arm}_hint" --config "$CFG" --weights "$WK/$arm/final.pt" --format-hint --modes oracle-image || rc=1; }
    done
    vsi_sharded base_plain --model "$MODEL" --patch-embed linear --modes oracle-image,oracle-video || rc=1
    vsi_sharded base_routed_room --config "$CFG" "${ROUTED_ROOM[@]}" --modes oracle-image || rc=1
    vsi_sharded base_hint_2b --model "$M2B" --patch-embed linear --format-hint --modes oracle-image || rc=1
    vsi_sharded base_plain_2b --model "$M2B" --patch-embed linear --modes oracle-image,oracle-video || rc=1
    for arm in s2_real_2b s2_control_2b; do
      [[ -f "$WK/$arm/final.pt" ]] || continue
      vsi_sharded "wk_$arm" --config "$CFG_2B" --weights "$WK/$arm/final.pt" --modes oracle-image || rc=1
      vsi_sharded "wk_${arm}_hint" --config "$CFG_2B" --weights "$WK/$arm/final.pt" --format-hint --modes oracle-image || rc=1
    done
    (( rc == 0 )) && mark recover_vsi
    report
  fi
  if ! is_done recover_abl; then
    ablation 0 wk_p5_ablation_s1 align "$WK/s1_real/final.pt" "$WK/s1_control/final.pt" "$CFG" & a=$!
    ablation 1 wk_p7_ablation_s2 sft "$WK/s2_real/final.pt" "$WK/s2_control/final.pt" "$CFG" & b=$!
    ablation 2 wk_p7_ablation_s2_seed1 sft "$WK/s2_real_seed1/final.pt" "$WK/s2_control_seed1/final.pt" "$CFG" & c=$!
    ablation 3 wk_p7_ablation_2b sft "$WK/s2_real_2b/final.pt" "$WK/s2_control_2b/final.pt" "$CFG_2B" & d=$!
    if wait_all $a $b $c $d; then mark recover_abl; else rc=1; fi
    report
  fi
  return $rc
}
role_joint() {   # projector + LoRA from scratch with real geometry, no S1: is the alignment stage needed?
  local rc=0
  train_arm s2_joint "$JOINT_HOURS" "$S2_EPOCHS" "$S2_SAVE" "$S2_PROBE" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" || return 1
  eval_arm s2_joint "$OUT/s2_joint/final.pt" "$CFG" || rc=1
  return $rc
}

# ============================================================================================ run
log "started on $HOST at $(git log --oneline -1 2> /dev/null) - roles: ${QUEUE[*]}"
touch "$OUT/alive.$HOST"
# heartbeat every 5 min (claims), REPORT.md every 30 min through the long phases
( n=0; while true; do touch "$OUT/alive.$HOST"; (( n % 6 == 5 )) && report; n=$(( n + 1 )); sleep 300; done ) &
REFRESH_PID=$!
trap 'kill "$REFRESH_PID" 2> /dev/null' EXIT
if ! is_done "p0.$HOST"; then
  log "p0 on $HOST"
  if p0_local; then mark "p0.$HOST"; else exit 1; fi
fi
is_done d0 || with_lock d0 _d0 || exit 1
ban_fla_here
{ is_done calib || with_lock calib _calib; } && load_budget || exit 1
report
for r in "${QUEUE[@]}"; do
  case "$r" in real|control|sft|recover|joint) ;; *) fail "unknown role '$r'"; continue ;; esac
  is_done "role.$r" && continue
  [[ -f "$OUT/role.$r.skipped" ]] && continue
  if [[ "$ROLE" == auto ]]; then claim "$r" || continue; fi
  if [[ "$r" != recover && -f "$OUT/NO_GO" ]]; then     # the speed check decided against training
    log "role $r skipped: NO_GO - $(cat "$OUT/NO_GO")"; touch "$OUT/role.$r.skipped"; continue
  fi
  log "role $r starts on $HOST"
  rm -f "$OUT/role.$r.failed"
  if "role_$r"; then mark "role.$r"
  else touch "$OUT/role.$r.failed"; fail "role $r ended with failures (see the lines above)"; fi
  report
done
report && log "REPORT: $OUT/REPORT.md"
left=(); for r in real control sft recover joint; do
  is_done "role.$r" || [[ -f "$OUT/role.$r.skipped" ]] || left+=("$r"); done
log "finished on $HOST$([[ ${#left[@]} -gt 0 ]] && echo " - still running or failed elsewhere: ${left[*]}")"
