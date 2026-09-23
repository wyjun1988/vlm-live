#!/usr/bin/env bash
# Live3R 개발 환경 — 맥(MPS) 기준. GPU 머신은 requirements.txt 를 직접 쓴다.
set -euo pipefail

ENV_NAME="${1:-live3r}"
PY_VER="3.11"

if ! command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || {
    echo "conda 를 못 찾았다. miniforge 를 먼저 깔아라." >&2; exit 1; }
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env '$ENV_NAME' 이미 있음 — 재사용"
else
  conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi

conda activate "$ENV_NAME"
python -m pip install -U pip

# 맥에서는 flash-attn / decord 를 건너뛴다 (CUDA 전용 · 빌드 실패)
if [[ "$(uname -s)" == "Darwin" ]]; then
  grep -vE '^(flash-attn|decord)' requirements.txt > /tmp/live3r-req-mac.txt
  python -m pip install -r /tmp/live3r-req-mac.txt
else
  python -m pip install -r requirements.txt
fi
python -m pip install -r requirements-dev.txt

python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  mps={torch.backends.mps.is_available()}")
print(f"transformers {transformers.__version__}")
import transformers.models.qwen3_5  # noqa
print("qwen3_5 지원 OK")
PY
echo
echo "완료. 다음: conda activate $ENV_NAME && PYTHONPATH=src python scripts/smoke_test.py"
