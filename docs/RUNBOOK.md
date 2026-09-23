# GPU 머신 런북

> **2026-09-23 이후 서버 진행은 [`NEXT_STEPS_SERVER.md`](NEXT_STEPS_SERVER.md) 가 기준이다.**
> 이 런북은 공개 데이터(ScanNet 영상) 기준으로 쓴 것이라, Sensenova 로 진행하는 지금은
> 설치·CUT3R·평가 부분만 참고하면 된다.

**전제**: 이 코드는 맥에서 "가중치 없는 구조 검증"까지만 끝났다. 실모델·실데이터 경로는
**아직 한 번도 안 돌았다.** 그래서 아래 순서는 **싸게 실패하도록** 짜여 있다.
5단계(전체 학습)로 바로 뛰지 마라 — 앞 단계가 20분을 아껴주는 게 아니라 이틀을 아껴준다.

각 단계는 독립적으로 통과/실패가 난다. 실패하면 그 단계 메시지를 그대로 보내주면 된다.

---

## 0. 설치 (10분)

```bash
git clone git@github.com:wyjun1988/vlm-live.git && cd vlm-live
conda create -n live3r python=3.11 -y && conda activate live3r
pip install -r requirements.txt
pip install -e .                      # lmms-eval 에 live3r 모델 등록

# 기하 인코더
git clone https://github.com/CUT3R/CUT3R third_party/CUT3R
cd third_party/CUT3R/src/croco/models/curope && python setup.py build_ext --inplace && cd -
#   ⚠️ 이 컴파일은 선택이 아니다. CUT3R 포즈 토큰의 2D 위치가 -1 인데
#      순수 PyTorch RoPE2D 폴백은 cos/sin 테이블 F.embedding 룩업이라 음수에서 죽는다.
#      (live3r 이 폴백을 고쳐두긴 했지만 느리다)

mkdir -p checkpoints && cd checkpoints
pip install gdown && gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O cut3r_512_dpt_4_64.pth
cd -
```

**자체 점검** — 맥에서 통과한 것이 여기서도 통과하는지:
```bash
pytest tests -q                                  # 29개
PYTHONPATH=src python scripts/smoke_test.py      # 가중치 0
PYTHONPATH=src:third_party/CUT3R/src:third_party/CUT3R/src/croco \
  python scripts/verify_cut3r_adapter.py         # CUT3R 구조, 체크포인트 0
```

---

## 1. 기하 인코더 실측 (5분) — **여기서 설정값이 정해진다**

```bash
PYTHONPATH=src python scripts/verify_geometry_adapter.py \
  --name cut3r --checkpoint checkpoints/cut3r_512_dpt_4_64.pth \
  --tap-layers 6 9 12 --image-size 512 --repo-path third_party/CUT3R
```

보는 것:
- `hidden_size` → **configs/*.yaml 의 `geometry.hidden_size` 와 같아야 한다.** 다르면 고쳐라.
- `grid_hw` → 512 기준 격자. 프로젝터가 여기서 LLM 격자로 리샘플한다.
- `drift` ≈ 1.0, `상태 크기` 상수 → 아니면 스트리밍이 성립 안 한 것
- `프레임당 p50` → 30fps 예산 33ms 대비 어디쯤인지. **0.8B 판본의 운명이 여기서 갈린다.**

> `verify_geometry_adapter.py` 에 `--repo-path` 가 없다는 에러가 나면 그 인자만 빼고
> `PYTHONPATH` 에 `third_party/CUT3R/src:third_party/CUT3R/src/croco` 를 넣어라.

---

## 2. 베이스 VLM 로딩 + 1회 forward (10분)

먼저 **0.8B** 로 본다 — 다운로드가 작고, 구조 문제는 스케일과 무관하게 같이 터진다.

```bash
PYTHONPATH=src python - <<'PY'
import torch
from live3r.config import Live3RConfig
from live3r.model.live3r import Live3RModel

cfg = Live3RConfig.from_yaml("configs/live3r_08b.yaml")
cfg.geometry.name = "dummy"          # 1단계와 분리해서 본다
cfg.geometry.hidden_size = 768
m = Live3RModel.from_pretrained(cfg).cuda().eval()
print(m.trainable_parameter_summary())
print("d_llm", m.d_llm, "layers", m.n_layers, "patch", m.vision_patch,
      "merge", m.spatial_merge, "tpatch", m.temporal_patch)
PY
```

터지기 쉬운 곳: `transformers` 가 `qwen3_5` 를 모르면 버전을 올려라 (>=5.16).

---

## 3. 실모델 지연 (15분) — **"라이브"의 근거**

```bash
PYTHONPATH=src python scripts/bench_latency.py \
  --config configs/live3r_08b.yaml --device cuda --frames 512 --offline \
  --out outputs/latency_08b.json
PYTHONPATH=src python scripts/bench_latency.py \
  --config configs/live3r_4b.yaml --device cuda --frames 512 --offline \
  --out outputs/latency_4b.json
```

판정: `frame_ingest p95 < 33ms`, `TTFT < 500ms`, `drift < 1.2`.
**이 숫자가 프로젝트의 핵심 주장이다.** 학습보다 먼저 확보해라 — 학습해도 안 바뀐다.

---

## 4. 데이터 — 가장 싼 것부터 (30분)

```bash
pip install -U "huggingface_hub[cli]"
hf download Journey9ni/SpatialStackData --repo-type dataset --local-dir data/raw/spatialstack
PYTHONPATH=src python scripts/inspect_dataset.py data/raw/spatialstack
```

`inspect_dataset` 이 "로컬에 없는 영상 N개" 를 찍는다. ScanNet 영상이 필요하다.
**전체 1.3TB 를 받기 전에 100씬만 받아 파이프라인을 끝까지 통과시켜라.**
받는 법은 `docs/DATA.md` §2-2.

---

## 5. 학습 스모크 — 20스텝 (20분)

**전체 학습 전에 반드시.** 여기서 잡히는 건 형상·메모리·손실 스케일 문제다.

```bash
PYTHONPATH=src python -m live3r.train.train \
  --config configs/live3r_08b.yaml --stage align \
  --ann data/raw/spatialstack --video-root data/raw \
  --output outputs/smoke --max-steps 20 --grad-accum 4 --num-frames 8 \
  --log-every 1 --save-every 0 --grad-checkpointing
```

확인:
- 손실이 유한하고 **내려가기 시작**하는가 (20스텝이면 추세만)
- 로그의 `inj` (레이어별 ‖주입‖/‖비전 히든‖)가 0 에서 움직이는가 → 프로젝터가 실제로 학습되고 있다는 뜻.
  **끝까지 0 이면 프로젝터가 학습 안 되는 것**이다.
  (예전엔 `gate` 를 보라고 썼는데, 그때는 gate 가 초기화 교착 때문에 **영원히 0** 이었다 —
  진단 기준 자체가 버그를 가리고 있었다. 2026-09-23 수정.)
- GPU 메모리 여유 (4B 로 올릴 수 있는지 판단)

---

## 6. 본 학습

```bash
# S1 정렬 — 프로젝터만. LoRA 없이.
PYTHONPATH=src python -m live3r.train.train \
  --config configs/live3r_4b.yaml --stage align \
  --ann data/raw/vsi590k --video-root data/raw \
  --output outputs/4b_s1 --epochs 1 --lr 3e-5 --grad-accum 16 --grad-checkpointing   # 1e-3 은 주입 폭주 (M2 실측)

# S2 SFT — 프로젝터 + LoRA. S1 을 반드시 이어받는다.
PYTHONPATH=src python -m live3r.train.train \
  --config configs/live3r_4b.yaml --stage sft \
  --init-from outputs/4b_s1/final.pt \
  --ann data/raw/vsi590k --video-root data/raw \
  --output outputs/4b_s2 --epochs 1 --lr 2e-4 --grad-accum 16 --grad-checkpointing
```

> **S1 을 건너뛰지 마라.** 프로젝터가 zero-init 이라 시작 시 기하 기여가 정확히 0이다.
> LoRA 를 동시에 풀면 LLM 이 *기하 없이 푸는 지름길*을 먼저 배우고, 그 뒤엔 기하 토큰이
> 들어와도 무시한다. 정렬을 끝내고 LoRA 를 연다.

> **일반 데이터 리플레이 30% 는 아직 미구현이다.** 지금 S2 를 공간 데이터 100% 로 돌리면
> "특화 역설"(베이스보다 나빠짐)에 정면으로 걸릴 수 있다. 첫 사이클은 그대로 돌려
> VideoMME 회귀 폭을 **측정**하고, 그 수치를 보고 리플레이 비율을 정하는 게 낫다.

---

## 6.5 스트리밍 VSI 평가 — **지금 가장 먼저 재야 할 것**

학습 전에도 잰다. 학습 안 한 베이스라인 점수가 있어야 이후 변화가 해석된다.

```bash
# 라이브 제약 하에서 (기본: halving 선택기, 키프레임 32, 기하 10fps)
bash scripts/run_streaming_eval.sh configs/live3r_4b.yaml "" outputs/stream_base

# 오프라인 상한선 (⚠️ 총 길이를 보므로 라이브 점수가 아니다. 하니스가 표시한다)
SELECTOR=uniform_oracle bash scripts/run_streaming_eval.sh configs/live3r_4b.yaml "" outputs/stream_oracle

# 선택기 성질만 빠르게 (모델 불필요, 10초)
PYTHONPATH=src python scripts/bench_selectors.py
```

**두 숫자의 차이가 "미래를 보는 균등 샘플링을 잃은 비용"이다.** 3.3점 예산 중 손실 ②에 해당한다.
이게 크면 선택기를 손보고, 작으면 기하 인코더(손실 ①)에 예산을 몰아준다.

노브:
| 환경변수 | 기본 | 의미 |
|---|---|---|
| `SELECTOR` | `halving` | `halving` / `reservoir` / `stride` / `uniform_oracle`(라이브 아님) |
| `BUDGET` | 32 | LLM 이 볼 키프레임 수 |
| `GEOM_STRIDE` | 3 | 기하 인제스트 간격 (30fps ÷ 3 = 10fps) |

하니스가 강제하는 제약(위반 시 예외): 시간순 1패스 · 상수 기하 상태 ·
**총 길이 조회 금지** · 토큰 예산 · 질문 후 재인코딩 금지.

---

## 7. 평가

```bash
# 베이스라인 먼저 (weights 없이 = zero_init = 베이스 VLM 과 동일 출력)
bash scripts/run_eval.sh configs/live3r_4b.yaml "" outputs/eval_4b_base
# 학습 결과
bash scripts/run_eval.sh configs/live3r_4b.yaml outputs/4b_s2/final.pt outputs/eval_4b_s2
```

### 채택 게이트 (사전 등록 — 셋 다 만족해야 채택)
1. `vsibench` 평균 ≥ 베이스라인 **+1.0**
2. `videomme` 회귀 **≤ 1.0**  ← 특화 역설 방어
3. `drift < 1.2` 이고 `frame_ingest p95` 가 예산 내

---

## 보고할 것

각 단계에서 아래만 보내주면 다음 판단이 된다.

| 단계 | 보낼 것 |
|---|---|
| 1 | `verify_geometry_adapter` 출력 전체 (hidden_size·격자·프레임당 ms·drift) |
| 3 | `outputs/latency_*.json` 또는 콘솔 표 |
| 5 | 20스텝 로그 (loss 추세 + `gate` 값) + `nvidia-smi` 피크 |
| 7 | `*_results.json` 의 vsibench / videomme 수치 |
