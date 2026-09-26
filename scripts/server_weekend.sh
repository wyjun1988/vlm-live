#!/usr/bin/env bash
# Four-day run on up to 4 nodes x 8 H100: Sensenova only, to set the baseline. One ROLE per node; the nodes
# coordinate through the shared outputs directory (markers), never through the network - there is no multi-node
# job. What each role answers, what to watch and what to send back: docs/SERVER_WEEKEND.md.
#
#   on every node:  cd /group-volume/wooyeol/vlm-live && mkdir -p outputs/weekend
#   node 1:  ROLE=real     nohup bash scripts/server_weekend.sh > outputs/weekend/master.real.log 2>&1 &
#   node 2:  ROLE=control  nohup bash scripts/server_weekend.sh > outputs/weekend/master.control.log 2>&1 &
#   node 3:  ROLE=sft      nohup bash scripts/server_weekend.sh > outputs/weekend/master.sft.log 2>&1 &
#   node 4:  ROLE=small    nohup bash scripts/server_weekend.sh > outputs/weekend/master.small.log 2>&1 &
#   tail -f outputs/weekend/STATUS            # every node writes here, lines are tagged [role]
#   bash scripts/server_weekend.sh stop       # on a node: ends that node's run and everything it started
#
#   real     S1 real -> S2 real -> VSI (plain, +hint, routed, learning curve, S1) -> the same again with seed 1
#   control  S1 control -> S2 control -> VSI + the gate's base measurements -> the same again with seed 1
#   sft      eval-data downloads; zero-shot VSI baseline (full); plain SFT (no geometry); S2 joint (no S1);
#            gate trained + verdict; S2 ablation; latency
#   small    the 2B model: S1 real || control (4 + 4 GPUs) -> S2 real || control -> VSI, ablation
# Fewer nodes: a node can take roles in order, e.g. ROLE=control,sft. real + control are the core.
#
# Resumable: every finished step leaves a marker and is skipped on a re-run; a training arm that was cut off
# continues from its last checkpoint (every SAVE_EVERY steps). A step whose inputs are missing waits for them
# (another node may still be producing them) and gives up after a deadline with a line in STATUS.
set -uo pipefail

# ------------------------------------------------------------------------------------------------ settings
ROOT="${ROOT:-/group-volume/wooyeol/vlm-live}"
CFG="${CFG:-configs/server_4b.yaml}"
CFG_2B="${CFG_2B:-configs/server_2b.yaml}"
ANN_RAW="${ANN_RAW:-data/sensenova_si_800k.json}"
MEDIA="${MEDIA:-data/sensenova_media}"
OUT="${OUT:-outputs/weekend}"
export HF_HOME="${HF_HOME:-/group-volume/wooyeol/hf_cache}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Budgets per arm: samples ("epoch" = every record of data/sensenova.jsonl once) with an hours cap; if the speed
# measured in p1 says an epoch does not fit, the budget is cut to fit. Both arms of a pair get the same steps.
# 4B arms run on 8 GPUs (grad-accum 16); the 2B pair shares a node (4 + 4 GPUs, grad-accum 32).
SAMPLES="${SAMPLES:-epoch}"
S1_HOURS="${S1_HOURS:-14}";  S2_HOURS="${S2_HOURS:-18}"          # 4B, 8 GPUs (an epoch ~12 h / ~15 h)
S1_HOURS_2B="${S1_HOURS_2B:-16}"; S2_HOURS_2B="${S2_HOURS_2B:-20}"  # 2B, 4 GPUs per arm
SFT_HOURS="${SFT_HOURS:-18}"                                      # plain SFT and S2 joint, 8 GPUs
SEEDS="${SEEDS:-2}"            # training seeds for the 4B real/control pair (seed 0, then 1, ...)
EFF_BATCH=128
SAVE_EVERY="${SAVE_EVERY:-500}"  # steps between checkpoints: resume point + learning-curve weights
LR=3e-5                        # projector (M2: 1e-3 made the injection 15-30x the vision signal)
LORA_LR=1e-4
VSI=(--videos 288 --seed 1)    # all of VSI-Bench: 288 videos, 5,130 questions
# Best zero-shot prompting from M2 at the time this run was defined (docs/BRAINSTORM.md I-27 + I-29): 48.6 -> 56.0
# on 60 videos, no training. Kept as is for the whole run so p3's base_routed and p7's *_routed compare like with
# like. The NEXT run adds `--direction-facts --direction-min-frames 1` (I-33, 2026-09-26: 56.0 -> 57.8).
ROUTED=(--scene-map --no-map-image --object-map --detector owlv2 --detector-device cuda
        --route-facts --route-objects --min-frames 3 --detect-every 1)
ALL8=0,1,2,3,4,5,6,7

cd "$ROOT" || { echo "no $ROOT"; exit 1; }
mkdir -p "$OUT/vsi" "$OUT/gate"
HOST=$(hostname -s 2> /dev/null || hostname)

if [[ "${1:-}" == stop ]]; then
  pid=$(cat "$OUT/PID.$HOST" 2> /dev/null) || { echo "no $OUT/PID.$HOST - nothing running from this node"; exit 1; }
  kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill -TERM "$1" 2> /dev/null; }
  kill_tree "$pid"
  echo "$(date '+%F %T') [$HOST] stopped by hand (pid $pid)" >> "$OUT/STATUS"
  exit 0
fi
ROLE="${ROLE:?set ROLE=real|control|sft|small (or a comma list, run in order) - see docs/SERVER_WEEKEND.md}"
RTAG=${ROLE//,/+}
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
sps_of()   { grep -ao '[0-9.]* samp/s' "$1" 2> /dev/null | tail -n "${2:-6}" \
               | awk '{ s += $1; n++ } END { if (n) printf "%.3f", s / n }'; }
report()   { local t; t=$(mktemp "$OUT/REPORT.md.XXXXXX") || return 1
             python scripts/weekend_report.py "$OUT" > "$t" 2> "$OUT/report.$RTAG.err" && mv "$t" "$OUT/REPORT.md" || rm -f "$t"; }
base_model() { python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['base_model'])" "$1"; }
# wait for a marker another node (or this one, earlier) produces; $2 = hours before giving up
wait_marker() {
  local n=$1 limit; limit=$(hours_s "${2:-1}" 1 0); local t=0
  while ! is_done "$n"; do
    (( t >= limit )) && { fail "waited ${2:-1} h for $n - not produced (is the node that makes it running?)"; return 1; }
    sleep 60; t=$(( t + 60 ))
  done
}
# a shared-filesystem lock for one-time work any node may do (mkdir is atomic on NFS too)
with_lock() {
  local name=$1 t=0; shift
  until mkdir "$OUT/lock.$name" 2> /dev/null; do
    (( t >= 10800 )) && { fail "lock.$name held for 3 h - if that node died, rmdir $OUT/lock.$name"; return 1; }
    sleep 15; t=$(( t + 15 ))
  done
  echo "$HOST $$ $(date '+%F %T')" > "$OUT/lock.$name/owner"
  "$@"; local rc=$?
  rm -rf "$OUT/lock.$name"
  return $rc
}
cfg_of() { [[ "$1" == 2b ]] && echo "$CFG_2B" || echo "$CFG"; }

# ============================================================================================ p0 preflight
p0_local() {   # this node's environment
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
  with_lock pip pip install -e . --no-deps -q >> "$L" 2>&1 || { fail "p0: pip install -e . on $HOST - $L"; return 1; }
  MODEL=$(base_model "$CFG") || { fail "p0: cannot read base_model from $CFG"; return 1; }
  if [[ ! -f "$MODEL/config.json" ]]; then
    with_lock model4b python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-4B', local_dir='$MODEL')" \
      >> "$L" 2>&1 || { fail "p0: model download - $L"; return 1; }
  fi
  LIVE3R_TOKENIZER="$MODEL" python -m pytest tests -q -p no:cacheprovider >> "$L" 2>&1 \
    || { fail "p0: tests on $HOST - the tail of $L is what to send"; return 1; }
}
_p0_shared() {  # assets and data on the shared volume - done by whichever node gets here first
  local L="$OUT/p0_shared.log" M2
  is_done p0 && return 0
  if [[ ! -d third_party/CUT3R ]]; then
    git clone https://github.com/CUT3R/CUT3R third_party/CUT3R >> "$L" 2>&1 || { fail "p0: CUT3R clone"; return 1; }
  fi
  if ! ls third_party/CUT3R/src/croco/models/curope/*.so > /dev/null 2>&1; then
    (cd third_party/CUT3R/src/croco/models/curope && python setup.py build_ext --inplace) >> "$L" 2>&1 \
      || log "p0 warning: curope did not build - CUT3R falls back to the slow PyTorch RoPE"
  fi
  if [[ ! -f checkpoints/cut3r_512_dpt_4_64.pth ]]; then
    mkdir -p checkpoints
    gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O checkpoints/cut3r_512_dpt_4_64.pth >> "$L" 2>&1 \
      || { fail "p0: CUT3R checkpoint (docs/NEXT_STEPS_SERVER.md step 5)"; return 1; }
  fi
  if [[ ! -f checkpoints/owlv2-base-patch16-ensemble/config.json ]]; then
    python -c "from huggingface_hub import snapshot_download as s
s('google/owlv2-base-patch16-ensemble', local_dir='checkpoints/owlv2-base-patch16-ensemble')" >> "$L" 2>&1 \
      || log "p0 warning: OWLv2 download failed - the routed VSI runs will fail"
  fi
  M2=$(base_model "$CFG_2B")
  if [[ ! -f "$M2/config.json" ]]; then
    python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-2B', local_dir='$M2')" >> "$L" 2>&1 \
      || log "p0 warning: Qwen3.5-2B download failed - the small role will fail"
  fi
  [[ -e "$ANN_RAW" && -e "$MEDIA" ]] || { fail "p0: $ANN_RAW or $MEDIA missing"; return 1; }
  if [[ ! -f data/sensenova.jsonl || ! -f data/sensenova.holdout.jsonl ]]; then
    log "p0: preparing Sensenova annotations (10-20 min)"
    python scripts/prepare_annotations.py "$ANN_RAW" data/sensenova.jsonl --media-root "$MEDIA" \
      --check-files 5000 > "$OUT/p0_prepare.log" 2>&1 \
      || { fail "p0: prepare_annotations - $OUT/p0_prepare.log"; return 1; }
  fi
  log "p0: $(wc -l < data/sensenova.jsonl) training records, $(wc -l < data/sensenova.holdout.jsonl) held out"
  mark p0
}
p0_shared() { is_done p0 || with_lock p0 _p0_shared; }

# ============================================================================================ p2 eval data
start_downloads() {   # the sft node; every node waits on the markers
  local t pending=()
  for t in vsibench mmstar videomme; do is_done "p2_$t" || pending+=("$t"); done
  [[ ${#pending[@]} -eq 0 ]] && return 0
  log "p2 downloads started in the background: ${pending[*]}"
  (
    for t in "${pending[@]}"; do
      if python scripts/fetch_eval_data.py "$t" >> "$OUT/p2_$t.log" 2>&1; then mark "p2_$t"
      else fail "p2_$t - $OUT/p2_$t.log"; fi
    done
  ) &
}

# ============================================================================================ p1 smoke
p1() {   # this node: data + LLM path, CUT3R, 8-GPU S1 and S2 (LoRA) - also the speed measurement
  local which=$1 cfg; cfg=$(cfg_of "$which")
  local pre="$OUT/p1_${which}_${RTAG}" T=(-m live3r.train.train --config "$cfg" --ann data/sensenova.jsonl
      --media-root "$MEDIA" --grad-checkpointing --save-every 0 --log-every 5 --lr "$LR")
  log "p1 ($which) smoke A: data + LLM path, dummy geometry (1 GPU)"
  on_gpu 0 timeout 1h python "${T[@]}" --geometry dummy --stage align --output "${pre}_smoke_a" --max-steps 30 \
    --grad-accum 1 --dump-samples 3 > "${pre}_smoke_a.log" 2>&1 || { fail "p1 smoke A - ${pre}_smoke_a.log"; return 1; }
  log "p1 ($which) smoke C: S1 on 8 GPUs, 60 steps"
  timeout 1h torchrun --nproc_per_node 8 --master_port 29510 "${T[@]}" --stage align --output "${pre}_smoke_c" \
    --max-steps 60 --grad-accum 2 > "${pre}_smoke_c.log" 2>&1 || { fail "p1 smoke C - ${pre}_smoke_c.log"; return 1; }
  grep -aq "sync OK" "${pre}_smoke_c.log" || { fail "p1 smoke C: ranks not in sync"; return 1; }
  log "p1 ($which) smoke D: S2 (LoRA) on 8 GPUs, 60 steps"
  timeout 1h torchrun --nproc_per_node 8 --master_port 29510 "${T[@]}" --stage sft --output "${pre}_smoke_d" \
    --init-from "${pre}_smoke_c/final.pt" --lora-lr "$LORA_LR" --max-steps 60 --grad-accum 2 \
    > "${pre}_smoke_d.log" 2>&1 || { fail "p1 smoke D - ${pre}_smoke_d.log"; return 1; }
  grep -aq "sync OK" "${pre}_smoke_d.log" || { fail "p1 smoke D: ranks not in sync"; return 1; }
  log "p1 ($which) throughput on $HOST (8 GPUs): S1 $(sps_of "${pre}_smoke_c.log") / S2 $(sps_of "${pre}_smoke_d.log") samples/s"
}

# ============================================================================================ budget
# One file per plan, written once (lock) from the first node that needs it and read by every other one - so the
# real and control arms of a pair always get the same steps. $1 = file, $2 = which (4b|2b), $3 = fraction of the
# 8-GPU smoke speed one arm gets (0.9 alone on a node; 0.425 = half a node next to the other arm).
plan() {
  local f="$OUT/$1" which=$2 frac=$3
  [[ -f "$f" ]] || with_lock "plan_$1" bash -c '[[ -f "$1" ]] || { python - "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" > "$1.tmp" && mv "$1.tmp" "$1"; }' _ \
      "$f" "$(sps_of "$OUT/p1_${which}_${RTAG}_smoke_c.log")" "$(sps_of "$OUT/p1_${which}_${RTAG}_smoke_d.log")" \
      "$SAMPLES" "$4" "$5" "$EFF_BATCH" "$(wc -l < data/sensenova.jsonl)" "$frac" << 'EOF'
import math, sys
c, d = (float(x) if x else 0.0 for x in sys.argv[1:3])
want, h1, h2, eff, n, frac = sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8])
w = n if want == "epoch" else float(want)
def fit(hours, sps):
    s = w if sps <= 0 else min(w, hours * 3600 * sps * frac)
    steps = max(50, int(s // eff))
    return steps, max(1, math.ceil(steps * eff / n))
st1, ep1 = fit(h1, c)
st2, ep2 = fit(h2, d)
print(f"N_TRAIN={n}\nSMOKE_SPS_S1={c:.2f}\nSMOKE_SPS_S2={d:.2f}\nS1_STEPS={st1}\nS1_EPOCHS={ep1}\nS2_STEPS={st2}\nS2_EPOCHS={ep2}")
EOF
  [[ -f "$f" ]] || { fail "budget $1 could not be written"; return 1; }
  # shellcheck disable=SC1090
  source "$f"
}

# ============================================================================================ training
# --resume continues from <arm>/resume.pt when a previous attempt was cut off. --max-hours is a soft stop that
# still saves final.pt; `timeout` is the hard stop for a hang. A finished arm leaves <name>.done. Logs append.
train_arm() {   # nproc accum gpus port cfg name hours epochs args...
  local nproc=$1 accum=$2 gpus=$3 port=$4 cfg=$5 name=$6 hours=$7 epochs=$8; shift 8
  is_done "$name" && return 0
  rm -f "$OUT/$name/final.pt"
  [[ -f "$OUT/$name/resume.pt" ]] && log "$name: resuming from $OUT/$name/resume.pt"
  log "$name started on $HOST GPUs $gpus"
  CUDA_VISIBLE_DEVICES=$gpus timeout --kill-after=10m "$(hours_s "$hours" 1.25 3600)" \
    torchrun --nproc_per_node "$nproc" --master_port "$port" -m live3r.train.train \
      --config "$cfg" --ann data/sensenova.jsonl --media-root "$MEDIA" --grad-checkpointing \
      --output "$OUT/$name" --resume --epochs "$epochs" \
      --max-hours "$(awk -v h="$hours" 'BEGIN { print h * 1.25 }')" \
      --grad-accum "$accum" --log-every 20 --save-every "$SAVE_EVERY" --lr "$LR" "$@" >> "$OUT/$name.log" 2>&1
  [[ -f "$OUT/$name/final.pt" ]] || { fail "$name - $OUT/$name.log"; return 1; }
  mark "$name"
}
# the 4B S1 -> S2 chain of one arm on this whole node; $1 = real|control, $2 = seed (0 = the main pair)
chain_4b() {
  local arm=$1 seed=$2 sfx="" ctl=()
  (( seed > 0 )) && sfx="_seed$seed"
  [[ "$arm" == control ]] && ctl=(--geom-control shuffled)
  plan budget.env 4b 0.9 "$S1_HOURS" "$S2_HOURS" || return 1
  log "S1/S2 plan: $S1_STEPS / $S2_STEPS steps x $EFF_BATCH = $S1_EPOCHS / $S2_EPOCHS epoch(s) of $N_TRAIN"
  train_arm 8 16 "$ALL8" 29511 "$CFG" "s1_$arm$sfx" "$S1_HOURS" "$S1_EPOCHS" --stage align --max-steps "$S1_STEPS" \
    --seed "$seed" ${ctl[@]+"${ctl[@]}"} || return 1
  train_arm 8 16 "$ALL8" 29513 "$CFG" "s2_$arm$sfx" "$S2_HOURS" "$S2_EPOCHS" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" --init-from "$OUT/s1_$arm$sfx/final.pt" --seed "$seed" ${ctl[@]+"${ctl[@]}"}
}

# ============================================================================================ evaluation
vsi_job() {   # gpu name args...  -> $OUT/vsi/<name>.json (skipped when it exists)
  local g=$1 name=$2; shift 2
  [[ -f "$OUT/vsi/$name.json" ]] && return 0
  log "VSI $name started on $HOST GPU $g"
  on_gpu "$g" timeout 6h python scripts/eval_vsi_local.py "$@" "${VSI[@]}" --out "$OUT/vsi/$name.json" \
    > "$OUT/vsi/$name.log" 2>&1 || { fail "VSI $name - $OUT/vsi/$name.log"; return 1; }
  log "VSI $name done"
}
vsi_set() {   # gpu arm-name cfg [routed]: plain, and routed when asked - the two views of a trained arm
  local g=$1 name=$2 cfg=$3 rc=0
  vsi_job "$g" "$name" --config "$cfg" --weights "$OUT/$name/final.pt" --modes oracle-image || rc=1
  [[ "${4:-}" == routed ]] && { vsi_job "$g" "${name}_routed" --config "$cfg" --weights "$OUT/$name/final.pt" \
      "${ROUTED[@]}" --modes oracle-image || rc=1; }
  return $rc
}
curve_steps() {   # S2 learning curve: the saved checkpoints nearest 25 / 50 / 75 % of the S2 plan
  local total f
  total=$(sed -n 's/^S2_STEPS=//p' "$OUT/budget.env" 2> /dev/null) || return 0
  [[ -n "$total" ]] || return 0
  for f in 0.25 0.5 0.75; do
    awk -v t="$total" -v f="$f" -v e="$SAVE_EVERY" 'BEGIN { n = int(t * f / e + 0.5) * e; if (n > 0 && n < t) print n }'
  done | sort -un
}
p3() {   # zero-shot baseline on the full benchmark, 3 GPUs
  local MODEL; MODEL=$(base_model "$CFG")
  vsi_job 0 base_plain --model "$MODEL" --modes oracle-image,oracle-video &
  local a=$!
  vsi_job 1 base_hint --model "$MODEL" --format-hint --modes oracle-image &
  local b=$!
  vsi_job 2 base_routed --config "$CFG" "${ROUTED[@]}" --modes oracle-image &
  wait_all $a $b $!
}
# The gate (scripts/run_gate.sh) in three measurements: base + base_video on the control node, trained + the
# verdict on the sft node. The general task is VideoMME; if its 101 GB never arrived, MMStar - decided once.
gate_env() {
  local f="$OUT/gate/general.env"
  if [[ ! -f "$f" ]]; then
    if wait_marker p2_videomme 12; then echo "GENERAL=videomme" > "$f"
    else log "gate: VideoMME not available - the general gate falls back to MMStar"; printf 'GENERAL=mmstar\nREFERENCE=\n' > "$f"; fi
  fi
  cat "$f"
}
gate_job() {  # gpu step
  local g=$1 s=$2 genv=()
  is_done "gate_$s" && return 0
  { wait_marker p2_vsibench 6 && wait_marker p2_mmstar 6; } || { fail "gate $s: eval data not downloaded"; return 1; }
  [[ "$s" == base_video ]] || genv=($(gate_env))     # base_video measures the spatial task only
  log "gate $s started on $HOST GPU $g"
  on_gpu "$g" env ${genv[@]+"${genv[@]}"} ONLY="$s" timeout 14h bash scripts/run_gate.sh "$CFG" "$OUT/s2_real/final.pt" "$OUT/gate" \
    > "$OUT/gate_$s.log" 2>&1 || { fail "gate $s - $OUT/gate_$s.log"; return 1; }
  mark "gate_$s"
}
ablation() {  # name stage real control cfg
  local name=$1 stage=$2 real=$3 ctrl=$4 cfg=$5
  [[ -f "$OUT/$name.json" ]] && return 0
  on_gpu 0 timeout 3h python scripts/eval_geometry_ablation.py --config "$cfg" --stage "$stage" --weights "$real" \
    --control-weights "$ctrl" --ann data/sensenova.holdout.jsonl --media-root "$MEDIA" --n 500 \
    --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || { fail "$name - $OUT/$name.log"; return 1; }
}

# ============================================================================================ roles
role_real() {
  local seed rc=0 n
  for seed in $(seq 0 $(( SEEDS - 1 ))); do
    local sfx=""; (( seed > 0 )) && sfx="_seed$seed"
    chain_4b real "$seed" || { rc=1; continue; }
    wait_marker p2_vsibench 6 || { rc=1; continue; }
    if (( seed == 0 )); then
      vsi_set 0 s2_real "$CFG" routed & local a=$!
      vsi_job 1 s2_real_hint --config "$CFG" --weights "$OUT/s2_real/final.pt" --format-hint --modes oracle-image & local b=$!
      vsi_set 2 s1_real "$CFG" & local c=$!
      (  # learning curve: does more Sensenova keep helping, or does the numeric prior take over?
        for n in $(curve_steps); do
          have "$OUT/s2_real/step$n.pt" "VSI s2_real_step$n" && vsi_job 3 "s2_real_step$n" --config "$CFG" \
            --weights "$OUT/s2_real/step$n.pt" --modes oracle-image
        done
      ) & local d=$!
      wait_all $a $b $c $d || rc=1
    else
      vsi_set 0 "s2_real$sfx" "$CFG" || rc=1
    fi
    report
  done
  return $rc
}
role_control() {
  local seed rc=0
  for seed in $(seq 0 $(( SEEDS - 1 ))); do
    local sfx=""; (( seed > 0 )) && sfx="_seed$seed"
    chain_4b control "$seed" || { rc=1; continue; }
    wait_marker p2_vsibench 6 || { rc=1; continue; }
    if (( seed == 0 )); then
      vsi_set 0 s2_control "$CFG" routed & local a=$!
      vsi_set 1 s1_control "$CFG" & local b=$!
      gate_job 2 base & local c=$!
      gate_job 3 base_video & local d=$!
      wait_all $a $b $c $d || rc=1
    else
      vsi_set 0 "s2_control$sfx" "$CFG" || rc=1
    fi
    report
  done
  return $rc
}
role_sft() {
  local rc=0 genv=()
  wait_marker p2_vsibench 6 || return 1
  if ! is_done p3; then
    log "p3 zero-shot VSI baseline on the full benchmark (3 GPUs)"
    p3 && mark p3 || rc=1
    report
  fi
  plan budget.sft.env 4b 0.9 "$SFT_HOURS" "$SFT_HOURS" || return 1
  # plain SFT: no geometry at all. real - this = the value of the whole geometry path.
  train_arm 8 16 "$ALL8" 29515 "$CFG" s2_sft "$SFT_HOURS" "$S2_EPOCHS" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" --no-geometry || rc=1
  # joint: projector + LoRA from scratch with real geometry, no S1. Is the alignment stage needed?
  train_arm 8 16 "$ALL8" 29516 "$CFG" s2_joint "$SFT_HOURS" "$S2_EPOCHS" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" || rc=1
  { is_done s2_sft && vsi_set 0 s2_sft "$CFG" && vsi_job 0 s2_sft_hint --config "$CFG" \
      --weights "$OUT/s2_sft/final.pt" --format-hint --modes oracle-image; } & local a=$!
  { is_done s2_joint && vsi_set 1 s2_joint "$CFG" routed; } & local b=$!
  { wait_marker s2_real 48 && gate_job 2 trained; } & local c=$!
  { wait_marker s2_real 48 && wait_marker s2_control 48 && ablation p7_ablation_s2 sft "$OUT/s2_real/final.pt" \
      "$OUT/s2_control/final.pt" "$CFG"; } & local d=$!
  {
    [[ -f "$OUT/latency_stream.json" ]] || on_gpu 4 timeout 1h python scripts/bench_latency.py --config "$CFG" --frames 512 \
      --offline --out "$OUT/latency_stream.json" > "$OUT/latency_stream.log" 2>&1 || fail "latency (stream) - $OUT/latency_stream.log"
    grep -aqs "TTFT" "$OUT/latency_live.log" || on_gpu 4 timeout 1h python scripts/bench_latency.py --config "$CFG" \
      --mode deferred --scene-map --frames 512 > "$OUT/latency_live.log" 2>&1 || fail "latency (live) - $OUT/latency_live.log"
  } & local e=$!
  wait_all $a $b $c $d $e || rc=1
  if wait_marker gate_base 48 && wait_marker gate_base_video 48 && is_done gate_trained; then
    genv=($(gate_env))
    env ${genv[@]+"${genv[@]}"} ONLY=check bash scripts/run_gate.sh "$CFG" "$OUT/s2_real/final.pt" "$OUT/gate" \
      > "$OUT/gate_check.txt" 2>&1 && mark gate_check || { fail "gate check - $OUT/gate_check.txt"; rc=1; }
  else
    rc=1
  fi
  return $rc
}
role_small() {   # the 2B model: the real/control pair shares this node, 4 GPUs each
  local rc=0 M2
  plan budget.2b.env 2b 0.425 "$S1_HOURS_2B" "$S2_HOURS_2B" || return 1
  log "2B S1/S2 plan: $S1_STEPS / $S2_STEPS steps x $EFF_BATCH = $S1_EPOCHS / $S2_EPOCHS epoch(s)"
  train_arm 4 32 0,1,2,3 29521 "$CFG_2B" s1_real_2b "$S1_HOURS_2B" "$S1_EPOCHS" --stage align --max-steps "$S1_STEPS" & local a=$!
  train_arm 4 32 4,5,6,7 29522 "$CFG_2B" s1_control_2b "$S1_HOURS_2B" "$S1_EPOCHS" --stage align --max-steps "$S1_STEPS" \
    --geom-control shuffled & local b=$!
  wait_all $a $b || return 1
  train_arm 4 32 0,1,2,3 29523 "$CFG_2B" s2_real_2b "$S2_HOURS_2B" "$S2_EPOCHS" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" --init-from "$OUT/s1_real_2b/final.pt" & a=$!
  train_arm 4 32 4,5,6,7 29524 "$CFG_2B" s2_control_2b "$S2_HOURS_2B" "$S2_EPOCHS" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" --init-from "$OUT/s1_control_2b/final.pt" --geom-control shuffled & b=$!
  wait_all $a $b || rc=1
  wait_marker p2_vsibench 6 || return 1
  M2=$(base_model "$CFG_2B")
  vsi_job 0 base_hint_2b --model "$M2" --format-hint --modes oracle-image & a=$!
  vsi_job 1 base_plain_2b --model "$M2" --modes oracle-image,oracle-video & b=$!
  { is_done s2_real_2b && vsi_set 2 s2_real_2b "$CFG_2B" routed; } & local c=$!
  { is_done s2_control_2b && vsi_set 3 s2_control_2b "$CFG_2B" routed; } & local d=$!
  { is_done s2_real_2b && is_done s2_control_2b && ablation p7_ablation_2b sft "$OUT/s2_real_2b/final.pt" \
      "$OUT/s2_control_2b/final.pt" "$CFG_2B"; } & local e=$!
  wait_all $a $b $c $d $e || rc=1
  return $rc
}

# ============================================================================================ run
log "started on $HOST at $(git log --oneline -1 2> /dev/null)"
roles=(${ROLE//,/ })
if ! is_done "p0.$RTAG"; then
  log "p0 preflight on $HOST"
  if p0_local; then mark "p0.$RTAG"; else exit 1; fi
fi
p0_shared || exit 1
wait_marker p0 3 || exit 1
for r in "${roles[@]}"; do [[ "$r" == sft ]] && start_downloads; done
( while true; do sleep 1800; report; done ) &   # REPORT.md stays current through the long training phases
REFRESH_PID=$!
trap 'kill "$REFRESH_PID" 2> /dev/null' EXIT
which=4b; for r in "${roles[@]}"; do [[ "$r" == small ]] && which=2b; done
if ! is_done "p1.$which.$RTAG"; then
  if p1 "$which"; then mark "p1.$which.$RTAG"; else report; exit 1; fi
fi
report
for r in "${roles[@]}"; do
  case "$r" in real|control|sft|small) ;; *) fail "unknown role '$r'"; continue ;; esac
  is_done "role.$r" && { log "role $r already finished"; continue; }
  log "role $r starts"
  if "role_$r"; then mark "role.$r"; else fail "role $r ended with failures (see the lines above)"; fi
  report
done
report && log "REPORT: $OUT/REPORT.md"
log "finished on $HOST"
