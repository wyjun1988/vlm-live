#!/usr/bin/env bash
# 스트리밍 VSI 평가 — 라이브 제약 하에서 VSI-Bench 를 잰다.
#
#   bash scripts/run_streaming_eval.sh configs/live3r_4b.yaml "" outputs/stream_base
#   bash scripts/run_streaming_eval.sh configs/live3r_4b.yaml outputs/4b_s2/final.pt outputs/stream_s2
#
# 제약(하니스가 강제, 위반 시 예외):
#   시간순 1패스 · 상수 기하 상태 · 인과적 키프레임(총 길이 못 봄) · 토큰 예산 · 질문 후 재인코딩 금지
set -euo pipefail

CONFIG="${1:?config yaml}"
WEIGHTS="${2:-}"
OUT="${3:-outputs/stream}"

SELECTOR="${SELECTOR:-halving}"      # halving | reservoir | stride | uniform_oracle(라이브 아님)
BUDGET="${BUDGET:-32}"               # LLM 이 볼 키프레임 수
GEOM_STRIDE="${GEOM_STRIDE:-3}"      # 기하 인코더 인제스트 간격 (30fps ÷ 3 = 10fps)
TASKS="${TASKS:-vsibench}"

ARGS="config=${CONFIG},streaming=True,selector=${SELECTOR},keyframe_budget=${BUDGET},geom_stride=${GEOM_STRIDE},enable_thinking=False"
[[ -n "${BASE_MODEL:-}" ]] && ARGS="pretrained=${BASE_MODEL},${ARGS}"
[[ -n "$WEIGHTS" ]] && ARGS="${ARGS},weights=${WEIGHTS}"

mkdir -p "$OUT"
echo "== 스트리밍 VSI ==  (로컬 모델은 BASE_MODEL=/path 로)"
echo "   선택기=${SELECTOR}  키프레임=${BUDGET}  기하 stride=${GEOM_STRIDE}  thinking=off"
python -m lmms_eval --model live3r --model_args "$ARGS" \
  --tasks "$TASKS" --batch_size 1 --log_samples --output_path "$OUT"

echo
echo "비교용 오프라인 상한선을 보려면:"
echo "  SELECTOR=uniform_oracle bash $0 $CONFIG \"$WEIGHTS\" ${OUT}_oracle"
echo "  (⚠️ 총 길이를 보므로 라이브 점수가 아니다. 하니스가 결과에 표시한다)"
