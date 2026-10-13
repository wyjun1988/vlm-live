#!/usr/bin/env bash
# Run 4 preparation (2026-10-13; docs/SERVER_RUN4_PREP.md): what the next training needs, on one node.
#   1. gdn: which Gated DeltaNet a training step runs, and fla's fused one against transformers' torch path (a LoRA
#      forward + backward at 4k and 12k tokens, loss / gradients / ms; plus the vision tower and a text prefill):
#      scripts/diagnose_throughput.py --gdn-check on one GPU, ~15-30 min -> gdn_check.txt
#   2. vsi590k: VSI-590K's images for the registered annotation (~96.5 GB from Hugging Face), unpacked into DEST and
#      checked against the annotation -> VSI590K.md (hours; network and disk only)
# Then prep_to_send.tar.gz with both reports. It can run next to the run-3 evaluations (one GPU for a while).
#
#   cd /group-volume/wooyeol/vlm-live && git pull && mkdir -p outputs/run4_prep
#   nohup bash scripts/server_prep_run4.sh > outputs/run4_prep/master.$(hostname -s).log 2>&1 &
set -uo pipefail

ROOT="${ROOT:-/group-volume/wooyeol/vlm-live}"
OUT="${OUT:-outputs/run4_prep}"
MODEL="${MODEL:-/group-volume/wooyeol/models/Qwen3.5-4B}"
DEST="${DEST:-/group-volume/wooyeol/data/vsi590k_media}"          # = the vsi590k media root in configs/data/vgllm.yaml
STAGING="${STAGING:-/group-volume/wooyeol/data/vsi590k_archives}"
export HF_HOME="${HF_HOME:-/group-volume/wooyeol/hf_cache}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT" || { echo "no $ROOT"; exit 1; }
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') [$(hostname -s 2> /dev/null || hostname)] $*" | tee -a "$OUT/STATUS"; }
GPU=${CUDA_VISIBLE_DEVICES:-0}; GPU=${GPU%%,*}

log "gdn check on GPU $GPU"
if ! python -c "import fla" 2> /dev/null; then
  { pip install -q --no-deps flash-linear-attention && { python -c "import einops" 2> /dev/null || pip install -q einops; }; } \
    >> "$OUT/pip.log" 2>&1 || log "warning: flash-linear-attention did not install ($OUT/pip.log)"
fi
CUDA_VISIBLE_DEVICES=$GPU timeout 2h python scripts/diagnose_throughput.py --model "$MODEL" --gdn-check \
  > "$OUT/gdn_check.txt" 2>&1 || log "FAILED gdn check - $OUT/gdn_check.txt"
log "gdn: $(grep -a 'GDN \|gdn check' "$OUT/gdn_check.txt" | tail -n 1)"

log "vsi590k: download to $DEST (archives via $STAGING)"
timeout 48h python scripts/fetch_vsi590k.py --dest "$DEST" --staging "$STAGING" --report "$OUT/VSI590K.md" \
  > "$OUT/vsi590k.log" 2>&1 || log "vsi590k: not all resolved or failed - $OUT/vsi590k.log"
log "vsi590k: $(tail -n 1 "$OUT/vsi590k.log")"

files=(STATUS); for f in gdn_check.txt VSI590K.md vsi590k.log pip.log; do [[ -f "$OUT/$f" ]] && files+=("$f"); done
tar -czf "$OUT/prep_to_send.tar.gz" -C "$OUT" "${files[@]}" && log "to send: $OUT/prep_to_send.tar.gz"
