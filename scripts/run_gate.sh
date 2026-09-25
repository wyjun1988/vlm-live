#!/usr/bin/env bash
# 채택 게이트 측정 — 같은 평가 경로(eval_path=live3r)에서 가중치만 바꿔 잰다.
#
#   BASE_MODEL=/path/Qwen3.5-4B bash scripts/run_gate.sh configs/live3r_4b.yaml outputs/4b_s1/final.pt outputs/gate_s1
#
# 사전 등록 게이트 (docs/BENCHMARKS.md §4):
#   공간(spatial) vsibench  Δ ≥ +1.0      (5.7GB)
#       기준선 = 베이스의 **최선 형식** — 이미지 모드·비디오 모드 중 높은 쪽 (2026-09-24 결정).
#       베이스는 키프레임을 이미지로 받으면 VSI 답 형식을 자주 어긴다 (M2 실측 4B: 이미지 24.7 vs
#       비디오 48.6, 선택형 전부 0점) → 같은 경로 비교만 하면 학습이 "짧게 답하는 법"만 배워도 통과한다.
#       학습 결과는 이미지 모드(라이브 경로)로만 잰다.
#   일반(general) videomme  Δ ≥ −1.0      (101GB — 서버 여유 1TB 확인, 2026-09-23)
#   참고(reference) mmstar  판정 안 함     (0.1GB — 이미지 일반 능력, 망각이 먼저 드러날 수 있는 곳)
#
# 베이스 실행은 use_geometry=False — weights 가 없으면 프로젝터가 zero-init 이라 주입량이 정확히 0 이다
# (실제 4B 에서 로짓 Δ 0.0 확인). 출력은 같고 CUT3R 을 돌리지 않아 빠르다.
set -euo pipefail

CONFIG="${1:?config yaml}"
WEIGHTS="${2:?학습 가중치 (.pt)}"
OUT="${3:-outputs/gate}"
SPATIAL="${SPATIAL:-vsibench}"
GENERAL="${GENERAL:-videomme}"
REFERENCE="${REFERENCE-mmstar}"
BUDGET="${BUDGET:-32}"
BEST_FORMAT="${BEST_FORMAT:-1}"   # 1 = 게이트 1 기준선을 베이스의 최선 형식으로 (베이스를 비디오 모드로도 잰다)
# ONLY = one step: base | base_video | trained | check. scripts/server_weekend.sh runs the three measurements
# on separate GPUs at once, then 'check'. Default 'all' runs them in order as before.
ONLY="${ONLY:-all}"
step() { [[ "$ONLY" == all || "$ONLY" == "$1" ]]; }
TASKS="${SPATIAL},${GENERAL}${REFERENCE:+,${REFERENCE}}"

COMMON="config=${CONFIG},keyframe_budget=${BUDGET},eval_path=live3r,enable_thinking=False"
[[ -n "${BASE_MODEL:-}" ]] && COMMON="pretrained=${BASE_MODEL},${COMMON}"

if step base; then
echo "== [1/3] 베이스 — 이미지 모드 (학습·라이브와 같은 경로. weights 없음 = 베이스 VLM 과 같은 출력) =="
python -m lmms_eval --model live3r --model_args "${COMMON},use_geometry=False" \
  --tasks "$TASKS" --batch_size 1 --log_samples --output_path "$OUT/base"
fi

ALT=()
if [[ "$BEST_FORMAT" == "1" ]]; then
  if step base_video; then
  echo "== [2/3] 베이스 — 비디오 모드, 공간 태스크만 (게이트 1 최선 형식 기준선) =="
  python -m lmms_eval --model live3r --model_args "${COMMON},use_geometry=False,visual_mode=video" \
    --tasks "$SPATIAL" --batch_size 1 --log_samples --output_path "$OUT/base_video"
  fi
  ALT=(--base-alt "video=$OUT/base_video")
fi

if step trained; then
echo "== [3/3] 학습 결과 — 이미지 모드 =="
python -m lmms_eval --model live3r --model_args "${COMMON},weights=${WEIGHTS}" \
  --tasks "$TASKS" --batch_size 1 --log_samples --output_path "$OUT/trained"
fi

if step check; then
echo
python scripts/check_gate.py "$OUT/base" "$OUT/trained" ${ALT[@]+"${ALT[@]}"} \
  --spatial "$SPATIAL" --general "$GENERAL" --reference "${REFERENCE}"
fi
