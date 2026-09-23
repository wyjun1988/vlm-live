#!/usr/bin/env bash
# Live3R 평가 — 정확도(lmms-eval, 학습과 같은 입력 경로) + 지연(자체 하니스)
#
#   BASE_MODEL=/path/Qwen3.5-4B bash scripts/run_eval.sh configs/live3r_4b.yaml outputs/4b_s1/final.pt outputs/eval_4b
#
# weights 를 비우면 같은 경로의 베이스라인(zero-init = 베이스 VLM 과 같은 출력)이 측정된다.
# 채택 게이트 판정은 scripts/run_gate.sh (베이스·학습을 같은 경로로 재고 자동 판정).
#
# 평가 경로는 eval_path=live3r (학습과 같은 입력: 시스템 프롬프트 없음, 같은 해상도 규격,
# 영상은 균등 키프레임을 이미지 모드로). lmms-eval 부모 경로(eval_path=lmms)는 학습 분포와
# 달라서 참고용이다. 생성은 greedy.
set -euo pipefail

CONFIG="${1:?config yaml}"
WEIGHTS="${2:-}"
OUT="${3:-outputs/eval}"
# 용량: VSI-Bench 5.7GB · VideoMME 101GB · MMStar 0.1GB · CV-Bench·SPAR 소형 (서버 여유 1TB)
TASKS="${TASKS:-vsibench,vsibench_debiased,videomme,mmstar,cv_bench,sparbench}"
BUDGET="${BUDGET:-32}"

mkdir -p "$OUT"
ARGS="config=${CONFIG},keyframe_budget=${BUDGET},eval_path=live3r,enable_thinking=False"
[[ -n "${BASE_MODEL:-}" ]] && ARGS="pretrained=${BASE_MODEL},${ARGS}"
[[ -n "$WEIGHTS" ]] && ARGS="${ARGS},weights=${WEIGHTS}"

echo "== 정확도 (eval_path=live3r, 키프레임 ${BUDGET}) =="
python -m lmms_eval \
  --model live3r \
  --model_args "$ARGS" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUT"

echo
echo "== 지연 =="
PYTHONPATH=src python scripts/bench_latency.py \
  --config "$CONFIG" --frames 512 --offline --out "$OUT/latency.json"
