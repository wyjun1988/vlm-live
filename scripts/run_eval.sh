#!/usr/bin/env bash
# Live3R 평가 — 정확도(lmms-eval) + 지연(자체 하니스)
#
#   bash scripts/run_eval.sh configs/live3r_4b.yaml outputs/4b_s2/final.pt outputs/eval_4b
#
# weights 를 비우면 베이스라인(기하 주입 없음)이 측정된다.
set -euo pipefail

CONFIG="${1:?config yaml}"
WEIGHTS="${2:-}"
OUT="${3:-outputs/eval}"
TASKS="${TASKS:-vsibench,vsibench_debiased,mmsi_bench,cv_bench,sparbench,videomme}"

mkdir -p "$OUT"
ARGS="config=${CONFIG}"
[[ -n "$WEIGHTS" ]] && ARGS="${ARGS},weights=${WEIGHTS}"

echo "== 정확도 =="
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

echo
echo "== 채택 게이트 =="
echo "  1) vsibench 평균 >= 베이스라인 +1.0"
echo "  2) videomme 회귀 <= 1.0        ← 특화 역설 방어"
echo "  3) drift < 1.2 이고 frame_ingest p95 가 예산 내"
echo "  결과: $OUT"
