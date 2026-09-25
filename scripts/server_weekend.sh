#!/usr/bin/env bash
# Multi-day run on 8x H100: Sensenova only, to set the baseline. What each phase answers, what to watch and what to
# send back: docs/SERVER_WEEKEND.md. Sized for 4 days: one epoch for S1 and one for S2, each with a control arm.
#
#   cd /group-volume/wooyeol/vlm-live
#   mkdir -p outputs/weekend && nohup bash scripts/server_weekend.sh > outputs/weekend/master.log 2>&1 &
#   tail -f outputs/weekend/STATUS          # one line per step; stay until "p1 done" (~45 min)
#   bash scripts/server_weekend.sh stop     # ends the run and everything it started
#
# Resumable: a phase or job that succeeded leaves a marker and is skipped on a re-run, so after a crash start the
# same command again. A training arm that was cut off continues from its last checkpoint (every SAVE_EVERY steps:
# weights, optimizer, schedule and data position). A step whose inputs are missing is skipped with a line in
# STATUS, never run on garbage; independent steps still run. outputs/weekend/REPORT.md is refreshed every 30 min.
#
#   p0  preflight   imports, tests against the real tokenizer, model / CUT3R / OWLv2 present, Sensenova prepared
#   p1  smoke       training: A dummy geometry, B CUT3R, C 8-GPU S1, D 8-GPU S2 (LoRA) - sizes the budget
#                   (p1e) evaluation paths on a few items: ablation, VSI local plain + routed, lmms-eval
#   p2  eval data   background downloads: VSI-Bench, MMStar, VideoMME, unpacked once (fetch_eval_data.py)
#   p3  baseline    zero-shot VSI, all 288 videos: base plain (image + video mode) / + format hint / routed facts
#   p4  S1          real geometry || control (each sample gets another sample's geometry), 4 GPUs each
#   p5  ablation    S1 holdout loss: real / shuffled / none geometry, and the control projector
#   p6  S2 (LoRA)   real (from S1 real) || control (from S1 control)
#   p7  evaluation  8 GPUs at once: the pre-registered gate (lmms-eval: VSI, VideoMME, MMStar), VSI of every arm
#                   (live path: plain, + format hint, routed), the S2 learning curve, S2 ablation, latency
#   p8  report      outputs/weekend/REPORT.md
set -uo pipefail

# ------------------------------------------------------------------------------------------------ settings
ROOT="${ROOT:-/group-volume/wooyeol/vlm-live}"
CFG="${CFG:-configs/server_4b.yaml}"         # base model + CUT3R paths: one file for every tool
ANN_RAW="${ANN_RAW:-data/sensenova_si_800k.json}"
MEDIA="${MEDIA:-data/sensenova_media}"
OUT="${OUT:-outputs/weekend}"
export HF_HOME="${HF_HOME:-/group-volume/wooyeol/hf_cache}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Training budget per arm, in samples ("epoch" = every record of data/sensenova.jsonl once) with an hours cap.
# If the throughput measured in p1 says a stage would overrun its hours, its budget is cut to fit. Both arms of
# a stage always get the same number of steps - that is what makes them comparable. Four days: S1 <= 26 h and
# S2 <= 32 h planned, at most 1.25x that each if the estimate was optimistic, plus ~10 h of everything else.
S1_SAMPLES="${S1_SAMPLES:-epoch}"; S1_HOURS="${S1_HOURS:-26}"
S2_SAMPLES="${S2_SAMPLES:-epoch}"; S2_HOURS="${S2_HOURS:-32}"
EFF_BATCH=128                  # per arm: 4 GPUs x grad-accum 32 (= 8 x 16 in docs/NEXT_STEPS_SERVER.md)
SAVE_EVERY="${SAVE_EVERY:-500}"  # steps between checkpoints (~2 h per arm): resume point + learning-curve weights
LR=3e-5                        # projector (M2: 1e-3 made the injection 15-30x the vision signal)
LORA_LR=1e-4
VSI=(--videos 288 --seed 1)    # all of VSI-Bench: 288 videos, 5,130 questions
# Best zero-shot prompting from M2 (docs/BRAINSTORM.md I-27 + I-29): 48.6 -> 56.0 on 60 videos, no training.
ROUTED=(--scene-map --no-map-image --object-map --detector owlv2 --detector-device cuda
        --route-facts --route-objects --min-frames 3 --detect-every 1)

cd "$ROOT" || { echo "no $ROOT"; exit 1; }
mkdir -p "$OUT/vsi" "$OUT/gate"

# `bash scripts/server_weekend.sh stop` ends a run: the script and everything it started (torchrun, evaluations,
# downloads), children first. Works however the run was started (a process-group kill could take the caller along).
if [[ "${1:-}" == stop ]]; then
  pid=$(cat "$OUT/PID" 2> /dev/null) || { echo "no $OUT/PID - nothing to stop"; exit 1; }
  kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill -TERM "$1" 2> /dev/null; }
  kill_tree "$pid"
  echo "$(date '+%F %T') stopped by hand (pid $pid)" >> "$OUT/STATUS"
  exit 0
fi
echo $$ > "$OUT/PID"

# ----------------------------------------------------------------------------------------------- helpers
log()      { local m; m="$(date '+%F %T') $*"; echo "$m" >> "$OUT/STATUS"; echo "$m" >&2; }
fail()     { log "FAILED $*"; }
is_done()  { [[ -f "$OUT/$1.done" ]]; }
mark()     { touch "$OUT/$1.done"; log "$1 done"; }
need()     { local p=$1 d; shift; for d in "$@"; do is_done "$d" || { log "$p skipped: $d not done"; return 1; }; done; }
have()     { [[ -f "$1" ]] || { log "$2 skipped: $1 missing"; return 1; }; }
on_gpu()   { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g "$@"; }
wait_all() { local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }
hours_s()  { awk -v h="$1" -v f="$2" -v a="$3" 'BEGIN { printf "%d", h * 3600 * f + a }'; }
# mean of the last N "x samp/s" values in a training log (the first ones include start-up)
sps_of()   { grep -ao '[0-9.]* samp/s' "$1" 2> /dev/null | tail -n "${2:-6}" \
               | awk '{ s += $1; n++ } END { if (n) printf "%.3f", s / n }'; }
report()   { local t; t=$(mktemp "$OUT/REPORT.md.XXXXXX") || return 1   # written whole, then renamed into place
             python scripts/weekend_report.py "$OUT" > "$t" 2> "$OUT/report.err" && mv "$t" "$OUT/REPORT.md" || rm -f "$t"; }
base_model() { python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['base_model'])" "$CFG"; }

# ============================================================================================ p0 preflight
p0() {
  local L="$OUT/p0_preflight.log" MODEL
  local CHECK="import torch, transformers, peft, lmms_eval, roma, omegaconf, scipy, yaml
import transformers.models.qwen3_5
print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)
n = torch.cuda.device_count(); print(n, 'GPUs'); assert n >= 8, f'this plan needs 8 GPUs, found {n}'"
  if ! python -c "$CHECK" >> "$L" 2>&1; then
    log "p0: packages missing - installing requirements (without flash-attn)"
    grep -v '^flash-attn' requirements.txt > "$OUT/requirements.noflash.txt"
    pip install -r "$OUT/requirements.noflash.txt" >> "$L" 2>&1
    python -c "$CHECK" >> "$L" 2>&1 || { fail "p0: environment - $L"; return 1; }
  fi
  # registers `--model live3r` with lmms-eval (a package entry point); --no-deps leaves the environment alone
  pip install -e . --no-deps -q >> "$L" 2>&1 || { fail "p0: pip install -e . - $L"; return 1; }
  MODEL=$(base_model) || { fail "p0: cannot read base_model from $CFG"; return 1; }
  if [[ ! -f "$MODEL/config.json" ]]; then
    log "p0: downloading Qwen/Qwen3.5-4B to $MODEL"
    python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-4B', local_dir='$MODEL')" \
      >> "$L" 2>&1 || { fail "p0: model download - $L"; return 1; }
  fi
  LIVE3R_TOKENIZER="$MODEL" python -m pytest tests -q -p no:cacheprovider >> "$L" 2>&1 \
    || { fail "p0: tests - the tail of $L is what to send"; return 1; }
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
  [[ -e "$ANN_RAW" && -e "$MEDIA" ]] || { fail "p0: $ANN_RAW or $MEDIA missing"; return 1; }
  if [[ ! -f data/sensenova.jsonl || ! -f data/sensenova.holdout.jsonl ]]; then
    log "p0: preparing Sensenova annotations (10-20 min)"
    python scripts/prepare_annotations.py "$ANN_RAW" data/sensenova.jsonl --media-root "$MEDIA" \
      --check-files 5000 > "$OUT/p0_prepare.log" 2>&1 \
      || { fail "p0: prepare_annotations - $OUT/p0_prepare.log"; return 1; }
  fi
  log "p0: $(wc -l < data/sensenova.jsonl) training records, $(wc -l < data/sensenova.holdout.jsonl) held out"
}

# ============================================================================================ p2 eval data
start_downloads() {
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
  DL_PID=$!
}
wait_for() {   # a download marker; waits while the download job is alive
  while ! is_done "$1"; do
    if [[ -z "${DL_PID:-}" ]] || ! kill -0 "$DL_PID" 2> /dev/null; then is_done "$1"; return; fi
    sleep 60
  done
}

# ============================================================================================ p1 smoke
p1() {
  local T=(-m live3r.train.train --config "$CFG" --ann data/sensenova.jsonl --media-root "$MEDIA"
           --grad-checkpointing --save-every 0 --log-every 5 --lr "$LR")
  log "p1 smoke A: data + LLM path, dummy geometry (1 GPU)"
  on_gpu 0 timeout 1h python "${T[@]}" --geometry dummy --stage align --output "$OUT/smoke_a" --max-steps 30 \
    --grad-accum 1 --dump-samples 3 > "$OUT/p1_smoke_a.log" 2>&1 || { fail "p1 smoke A - $OUT/p1_smoke_a.log"; return 1; }
  log "p1 smoke B: CUT3R (1 GPU)"
  on_gpu 0 timeout 1h python "${T[@]}" --stage align --output "$OUT/smoke_b" --max-steps 30 --grad-accum 1 \
    > "$OUT/p1_smoke_b.log" 2>&1 || { fail "p1 smoke B - $OUT/p1_smoke_b.log"; return 1; }
  log "p1 smoke C: S1 on 8 GPUs (60 steps - also the speed measurement)"
  timeout 1h torchrun --nproc_per_node 8 --master_port 29510 "${T[@]}" --stage align --output "$OUT/smoke_c" \
    --max-steps 60 --grad-accum 2 > "$OUT/p1_smoke_c.log" 2>&1 || { fail "p1 smoke C - $OUT/p1_smoke_c.log"; return 1; }
  grep -aq "sync OK" "$OUT/p1_smoke_c.log" || { fail "p1 smoke C: ranks not in sync"; return 1; }
  log "p1 smoke D: S2 (LoRA) on 8 GPUs, from smoke C"
  timeout 1h torchrun --nproc_per_node 8 --master_port 29510 "${T[@]}" --stage sft --output "$OUT/smoke_d" \
    --init-from "$OUT/smoke_c/final.pt" --lora-lr "$LORA_LR" --max-steps 60 --grad-accum 2 \
    > "$OUT/p1_smoke_d.log" 2>&1 || { fail "p1 smoke D - $OUT/p1_smoke_d.log"; return 1; }
  grep -aq "sync OK" "$OUT/p1_smoke_d.log" || { fail "p1 smoke D: ranks not in sync"; return 1; }
  log "p1 throughput (8 GPUs): S1 $(sps_of "$OUT/p1_smoke_c.log") / S2 $(sps_of "$OUT/p1_smoke_d.log") samples/s"
}

# Evaluation paths on a few items, so a broken eval shows up now and not on day 3. A failure here does not stop
# training (the weights can be evaluated later) - it is a line in STATUS to act on.
p1e() {
  local W1="$OUT/smoke_c/final.pt" W2="$OUT/smoke_d/final.pt" j=() rc=0
  wait_for p2_vsibench || { fail "p1e: VSI-Bench not downloaded - $OUT/p2_vsibench.log"; return 1; }
  on_gpu 0 timeout 1h python scripts/eval_geometry_ablation.py --config "$CFG" --stage align --weights "$W1" \
    --control-weights "$W1" --ann data/sensenova.holdout.jsonl --media-root "$MEDIA" --n 8 \
    --out "$OUT/p1e_ablation.json" > "$OUT/p1e_ablation.log" 2>&1 &
  j+=($!)
  on_gpu 1 timeout 1h python scripts/eval_vsi_local.py --config "$CFG" --weights "$W2" --modes oracle-image \
    --videos 2 --first-videos 1 --seed 1 --out "$OUT/p1e_vsi.json" > "$OUT/p1e_vsi.log" 2>&1 &
  j+=($!)
  on_gpu 2 timeout 1h python scripts/eval_vsi_local.py --config "$CFG" "${ROUTED[@]}" --modes oracle-image \
    --videos 2 --first-videos 1 --seed 1 --out "$OUT/p1e_vsi_routed.json" > "$OUT/p1e_vsi_routed.log" 2>&1 &
  j+=($!)
  # 2 items per task: VSI's aggregation needs every question type and fails on so few, so success here is
  # "every request was answered" (the wrapper's progress bar reached 100%), not the exit code
  on_gpu 3 timeout 1h python -m lmms_eval --model live3r \
    --model_args "config=$CFG,weights=$W2,keyframe_budget=32,eval_path=live3r,enable_thinking=False" \
    --tasks vsibench,mmstar --limit 2 --batch_size 1 --output_path "$OUT/p1e_lmms" > "$OUT/p1e_lmms.log" 2>&1 &
  j+=($!)
  wait "${j[0]}" || { fail "p1e ablation - $OUT/p1e_ablation.log"; rc=1; }
  wait "${j[1]}" || { fail "p1e VSI local - $OUT/p1e_vsi.log"; rc=1; }
  wait "${j[2]}" || { fail "p1e VSI routed - $OUT/p1e_vsi_routed.log"; rc=1; }
  wait "${j[3]}"
  grep -aq "Live3R: 100%" "$OUT/p1e_lmms.log" || { fail "p1e lmms-eval - $OUT/p1e_lmms.log"; rc=1; }
  return $rc
}

# ============================================================================================ budget
# Steps per arm from the sample budget, the hours cap and the speed measured in p1 (8-GPU smoke -> one 4-GPU arm
# with the other arm beside it: half, less 15% for the shared CPU and disk). Written once; a re-run reuses it.
plan_s1() {
  [[ -f "$OUT/budget.env" ]] || python - "$(sps_of "$OUT/p1_smoke_c.log")" "$(sps_of "$OUT/p1_smoke_d.log")" \
      "$S1_SAMPLES" "$S1_HOURS" "$S2_SAMPLES" "$S2_HOURS" "$EFF_BATCH" "$(wc -l < data/sensenova.jsonl)" \
      > "$OUT/budget.env" << 'EOF'
import math, sys
c, d = (float(x) if x else 0.0 for x in sys.argv[1:3])
s1, h1, s2, h2, eff, n = sys.argv[3:9]
h1, h2, eff, n = float(h1), float(h2), int(eff), int(n)
want = lambda s: n if s == "epoch" else float(s)
arm1, arm2 = 0.5 * 0.85 * c, 0.5 * 0.85 * d
def plan(w, hours, sps):
    fit = w if sps <= 0 else min(w, hours * 3600 * sps)
    steps = max(50, int(fit // eff))
    return steps, max(1, math.ceil(steps * eff / n))
st1, ep1 = plan(want(s1), h1, arm1)
st2, ep2 = plan(want(s2), h2, arm2)
print(f"N_TRAIN={n}\nSMOKE_SPS_S1={c:.2f}\nSMOKE_SPS_S2={d:.2f}\nS1_STEPS={st1}\nS1_EPOCHS={ep1}\nS2_STEPS={st2}\nS2_EPOCHS={ep2}")
EOF
  # shellcheck disable=SC1091
  source "$OUT/budget.env"
}
# S2 is re-planned from what S1 actually ran at (4 GPUs, next to the other arm), scaled by what LoRA cost in the
# smokes. Written once, so a re-run uses the same number.
plan_s2() {
  [[ -f "$OUT/budget_s2.env" ]] || python - "$(sps_of "$OUT/s1_real.log" 20)" "$SMOKE_SPS_S1" "$SMOKE_SPS_S2" \
      "$S2_SAMPLES" "$S2_HOURS" "$EFF_BATCH" "$S2_STEPS" "$N_TRAIN" > "$OUT/budget_s2.env" << 'EOF'
import math, sys
arm, c, d = (float(x) if x else 0.0 for x in sys.argv[1:4])
want = sys.argv[4]; hours, eff, fallback, n = float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
w = n if want == "epoch" else float(want)
sps = arm * min(1.0, d / c) * 0.9 if (arm and c and d) else 0.0
steps = max(50, int(min(w, hours * 3600 * sps) // eff)) if sps else fallback
print(f"S1_ARM_SPS={arm:.2f}\nS2_STEPS={steps}\nS2_EPOCHS={max(1, math.ceil(steps * eff / n))}")
EOF
  # shellcheck disable=SC1091
  source "$OUT/budget_s2.env"
}

# ============================================================================================ training
# One arm: 4 GPUs, effective batch 128. --resume continues from <arm>/resume.pt (written every SAVE_EVERY steps)
# when a previous attempt was cut off - same data order and steps as if nothing had happened. --max-hours is a soft
# stop that still saves final.pt; `timeout` is the hard stop for a hang (NCCL, I/O). A finished arm leaves
# <name>.done and is not touched on a re-run. The log is appended to, so every attempt is in it.
train_arm() {   # gpus port name hours epochs args...
  local gpus=$1 port=$2 name=$3 hours=$4 epochs=$5; shift 5
  is_done "$name" && return 0
  rm -f "$OUT/$name/final.pt"
  [[ -f "$OUT/$name/resume.pt" ]] && log "$name: resuming from $OUT/$name/resume.pt"
  log "$name started on GPUs $gpus"
  CUDA_VISIBLE_DEVICES=$gpus timeout --kill-after=10m "$(hours_s "$hours" 1.25 3600)" \
    torchrun --nproc_per_node 4 --master_port "$port" -m live3r.train.train \
      --config "$CFG" --ann data/sensenova.jsonl --media-root "$MEDIA" --grad-checkpointing \
      --output "$OUT/$name" --resume --epochs "$epochs" \
      --max-hours "$(awk -v h="$hours" 'BEGIN { print h * 1.25 }')" \
      --grad-accum 32 --log-every 20 --save-every "$SAVE_EVERY" --lr "$LR" "$@" >> "$OUT/$name.log" 2>&1
  [[ -f "$OUT/$name/final.pt" ]] || { fail "$name - $OUT/$name.log"; return 1; }
  mark "$name"
}

p4() {
  plan_s1
  log "p4 S1: $S1_STEPS steps per arm = $((S1_STEPS * EFF_BATCH)) samples, $S1_EPOCHS epoch(s) of $N_TRAIN (smoke: $SMOKE_SPS_S1 samples/s on 8 GPUs)"
  train_arm 0,1,2,3 29511 s1_real "$S1_HOURS" "$S1_EPOCHS" --stage align --max-steps "$S1_STEPS" &
  local a=$!
  train_arm 4,5,6,7 29512 s1_control "$S1_HOURS" "$S1_EPOCHS" --stage align --max-steps "$S1_STEPS" \
    --geom-control shuffled &
  wait_all $a $!
}

p5() {
  on_gpu 0 timeout 3h python scripts/eval_geometry_ablation.py --config "$CFG" --stage align \
    --weights "$OUT/s1_real/final.pt" --control-weights "$OUT/s1_control/final.pt" \
    --ann data/sensenova.holdout.jsonl --media-root "$MEDIA" --n 500 --out "$OUT/p5_ablation_s1.json" \
    > "$OUT/p5_ablation_s1.log" 2>&1 || { fail "p5 - $OUT/p5_ablation_s1.log"; return 1; }
}

# The control S2 starts from the control S1: the control is the whole two-stage recipe with every sample's geometry
# swapped for another sample's, so real - control is what the geometry's content is worth to the final model.
p6() {
  plan_s1
  plan_s2
  log "p6 S2: $S2_STEPS steps per arm = $((S2_STEPS * EFF_BATCH)) samples, $S2_EPOCHS epoch(s) (S1 ran at $S1_ARM_SPS samples/s per arm)"
  train_arm 0,1,2,3 29513 s2_real "$S2_HOURS" "$S2_EPOCHS" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" --init-from "$OUT/s1_real/final.pt" &
  local a=$!
  train_arm 4,5,6,7 29514 s2_control "$S2_HOURS" "$S2_EPOCHS" --stage sft --max-steps "$S2_STEPS" \
    --lora-lr "$LORA_LR" --init-from "$OUT/s1_control/final.pt" --geom-control shuffled &
  wait_all $a $!
}

# ============================================================================================ evaluation
vsi_job() {   # gpu name args...  -> $OUT/vsi/<name>.json (skipped when it exists)
  local g=$1 name=$2; shift 2
  [[ -f "$OUT/vsi/$name.json" ]] && return 0
  log "VSI $name started on GPU $g"
  on_gpu "$g" timeout 6h python scripts/eval_vsi_local.py "$@" "${VSI[@]}" --out "$OUT/vsi/$name.json" \
    > "$OUT/vsi/$name.log" 2>&1 || { fail "VSI $name - $OUT/vsi/$name.log"; return 1; }
  log "VSI $name done"
}

p3() {
  local MODEL
  MODEL=$(base_model)
  wait_for p2_vsibench || { fail "p3: VSI-Bench not downloaded - $OUT/p2_vsibench.log"; return 1; }
  vsi_job 0 base_plain --model "$MODEL" --modes oracle-image,oracle-video &
  local a=$!
  vsi_job 1 base_hint --model "$MODEL" --format-hint --modes oracle-image &
  local b=$!
  vsi_job 2 base_routed --config "$CFG" "${ROUTED[@]}" --modes oracle-image &
  wait_all $a $b $!
}

# The general gate is VideoMME. If its 101 GB never arrived, the gate falls back to MMStar rather than not run at
# all - decided once, so the three measurements and the check agree. The report says which it was.
gate_env() {
  local f="$OUT/gate/general.env"
  if [[ ! -f "$f" ]]; then
    if wait_for p2_videomme; then echo "GENERAL=videomme" > "$f"
    else log "gate: VideoMME not available - the general gate falls back to MMStar"; printf 'GENERAL=mmstar\nREFERENCE=\n' > "$f"; fi
  fi
  cat "$f"
}
gate_job() {  # gpu step - one of scripts/run_gate.sh's three measurements (the pre-registered gate)
  local g=$1 s=$2 genv=()
  [[ -f "$OUT/gate/$s.ok" ]] && return 0
  { wait_for p2_vsibench && wait_for p2_mmstar; } || { fail "gate $s: eval data not downloaded"; return 1; }
  [[ "$s" == base_video ]] || genv=($(gate_env))     # base_video measures the spatial task only
  log "gate $s started on GPU $g"
  on_gpu "$g" env ${genv[@]+"${genv[@]}"} ONLY="$s" timeout 14h bash scripts/run_gate.sh "$CFG" "$OUT/s2_real/final.pt" "$OUT/gate" \
    > "$OUT/gate_$s.log" 2>&1 || { fail "gate $s - $OUT/gate_$s.log"; return 1; }
  touch "$OUT/gate/$s.ok"
  log "gate $s done"
}
# S2 learning curve: the saved checkpoints nearest 25 / 50 / 75 % of the S2 plan
curve_steps() {
  [[ -f "$OUT/budget_s2.env" ]] || return 0
  local total f
  total=$(sed -n 's/^S2_STEPS=//p' "$OUT/budget_s2.env")
  for f in 0.25 0.5 0.75; do
    awk -v t="$total" -v f="$f" -v e="$SAVE_EVERY" 'BEGIN { n = int(t * f / e + 0.5) * e; if (n > 0 && n < t) print n }'
  done | sort -un
}

# p7: one chain of jobs per GPU. The base measurements do not need training, so they run even if it failed.
R1="$OUT/s1_real/final.pt"; C1="$OUT/s1_control/final.pt"; R2="$OUT/s2_real/final.pt"; C2="$OUT/s2_control/final.pt"
gpu0() { gate_job 0 base; }
gpu1() { have "$R2" "gate trained" && gate_job 1 trained; }
gpu2() {
  local rc=0
  gate_job 2 base_video || rc=1
  { have "$R1" "VSI s1_real" && vsi_job 2 s1_real --config "$CFG" --weights "$R1" --modes oracle-image; } || rc=1
  return $rc
}
gpu3() {
  have "$R2" "VSI s2_real" || return 1
  local rc=0
  vsi_job 3 s2_real --config "$CFG" --weights "$R2" --modes oracle-image || rc=1
  vsi_job 3 s2_real_hint --config "$CFG" --weights "$R2" --format-hint --modes oracle-image || rc=1
  return $rc
}
gpu4() {
  local rc=0
  { have "$C2" "VSI s2_control" && vsi_job 4 s2_control --config "$CFG" --weights "$C2" --modes oracle-image; } || rc=1
  { have "$C1" "VSI s1_control" && vsi_job 4 s1_control --config "$CFG" --weights "$C1" --modes oracle-image; } || rc=1
  return $rc
}
gpu5() {
  local rc=0 n
  { have "$R2" "VSI s2_real_routed" && vsi_job 5 s2_real_routed --config "$CFG" --weights "$R2" "${ROUTED[@]}" \
      --modes oracle-image; } || rc=1
  for n in $(curve_steps); do   # learning curve: does more Sensenova keep helping, or does the numeric prior take over?
    have "$OUT/s2_real/step$n.pt" "VSI s2_real_step$n" && vsi_job 5 "s2_real_step$n" --config "$CFG" \
      --weights "$OUT/s2_real/step$n.pt" --modes oracle-image || rc=1
  done
  return $rc
}
gpu6() { have "$C2" "VSI s2_control_routed" && vsi_job 6 s2_control_routed --config "$CFG" --weights "$C2" "${ROUTED[@]}" --modes oracle-image; }
gpu7() {  # does the LoRA model read the geometry's content (holdout loss), then latency (gate 3)
  local rc=0
  if [[ ! -f "$OUT/p7_ablation_s2.json" ]]; then
    if have "$R2" "S2 ablation" && have "$C2" "S2 ablation"; then
      on_gpu 7 timeout 3h python scripts/eval_geometry_ablation.py --config "$CFG" --stage sft --weights "$R2" \
        --control-weights "$C2" --ann data/sensenova.holdout.jsonl --media-root "$MEDIA" --n 500 \
        --out "$OUT/p7_ablation_s2.json" > "$OUT/p7_ablation_s2.log" 2>&1 \
        || { fail "S2 ablation - $OUT/p7_ablation_s2.log"; rc=1; }
    else rc=1; fi
  fi
  if [[ ! -f "$OUT/latency_stream.json" ]]; then
    on_gpu 7 timeout 1h python scripts/bench_latency.py --config "$CFG" --frames 512 --offline \
      --out "$OUT/latency_stream.json" > "$OUT/latency_stream.log" 2>&1 || { fail "latency (stream) - $OUT/latency_stream.log"; rc=1; }
  fi
  if ! grep -aqs "TTFT" "$OUT/latency_live.log"; then
    on_gpu 7 timeout 1h python scripts/bench_latency.py --config "$CFG" --mode deferred --scene-map --frames 512 \
      > "$OUT/latency_live.log" 2>&1 || { fail "latency (live) - $OUT/latency_live.log"; rc=1; }
  fi
  return $rc
}

p7() {
  local j=() g rc genv=()
  for g in 0 1 2 3 4 5 6 7; do "gpu$g" & j+=($!); done
  wait_all "${j[@]}"
  rc=$?
  if [[ -f "$OUT/gate/base.ok" && -f "$OUT/gate/trained.ok" && -f "$OUT/gate/base_video.ok" ]]; then
    genv=($(gate_env))
    env ${genv[@]+"${genv[@]}"} ONLY=check bash scripts/run_gate.sh "$CFG" "$R2" "$OUT/gate" > "$OUT/gate_check.txt" 2>&1 \
      || { fail "gate check - $OUT/gate_check.txt"; rc=1; }
  else
    rc=1
  fi
  return $rc
}

# ============================================================================================ run
log "run (re)started at $(git log --oneline -1 2> /dev/null)"
if ! is_done p0; then
  log "p0 preflight"
  if p0; then mark p0; else exit 1; fi
fi
start_downloads
( while true; do sleep 1800; report; done ) &   # REPORT.md stays current through the long training phases
REFRESH_PID=$!
trap 'kill "$REFRESH_PID" 2> /dev/null' EXIT
if ! is_done p1; then
  if p1; then mark p1; else report; exit 1; fi
fi
report
if ! is_done p1e; then
  log "p1e evaluation paths"
  if p1e; then mark p1e; else log "p1e: an evaluation path is broken - training goes on; fix it before p7"; fi
fi
if ! is_done p3; then
  log "p3 zero-shot VSI baseline (3 GPUs)"
  p3 && mark p3
  report
fi
if ! is_done p4; then
  p4 && mark p4
  report
fi
if ! is_done p5 && need p5 p4; then
  log "p5 S1 geometry ablation"
  p5 && mark p5
  report
fi
if ! is_done p6 && need p6 p4; then
  p6 && mark p6
  report
fi
if ! is_done p7; then
  log "p7 evaluation (8 GPUs)"
  p7 && mark p7
fi
if report; then log "REPORT: $OUT/REPORT.md"; else fail "report - $OUT/report.err"; fi
log "run finished"
