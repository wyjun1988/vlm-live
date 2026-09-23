#!/usr/bin/env bash
# 채택 게이트 측정 — 같은 평가 경로(eval_path=live3r)에서 가중치만 바꿔 두 번 잰다.
#
#   BASE_MODEL=/path/Qwen3.5-4B bash scripts/run_gate.sh configs/live3r_4b.yaml outputs/4b_s1/final.pt outputs/gate_s1
#
# 사전 등록 게이트 (docs/BENCHMARKS.md §4):
#   공간(spatial) vsibench  Δ ≥ +1.0      (5.7GB)
#   일반(general) videomme  Δ ≥ −1.0      (101GB — 서버 여유 1TB 확인, 2026-09-23)
#   참고(reference) mmstar  판정 안 함     (0.1GB — 이미지 일반 능력, 망각이 먼저 드러날 수 있는 곳)
set -euo pipefail

CONFIG="${1:?config yaml}"
WEIGHTS="${2:?학습 가중치 (.pt)}"
OUT="${3:-outputs/gate}"
SPATIAL="${SPATIAL:-vsibench}"
GENERAL="${GENERAL:-videomme}"
REFERENCE="${REFERENCE-mmstar}"
BUDGET="${BUDGET:-32}"
TASKS="${SPATIAL},${GENERAL}${REFERENCE:+,${REFERENCE}}"

COMMON="config=${CONFIG},keyframe_budget=${BUDGET},eval_path=live3r,enable_thinking=False"
[[ -n "${BASE_MODEL:-}" ]] && COMMON="pretrained=${BASE_MODEL},${COMMON}"

echo "== [1/2] 베이스 (weights 없음 = zero-init = 베이스 VLM 과 같은 출력, 같은 경로) =="
python -m lmms_eval --model live3r --model_args "${COMMON}" \
  --tasks "$TASKS" --batch_size 1 --log_samples --output_path "$OUT/base"

echo "== [2/2] 학습 결과 =="
python -m lmms_eval --model live3r --model_args "${COMMON},weights=${WEIGHTS}" \
  --tasks "$TASKS" --batch_size 1 --log_samples --output_path "$OUT/trained"

echo
python scripts/check_gate.py "$OUT/base" "$OUT/trained" \
  --spatial "$SPATIAL" --general "$GENERAL" --reference "${REFERENCE}"
