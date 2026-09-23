# 서버 다음 진행사항 — 2026-09-23

**서버가 마지막으로 확인한 커밋은 `9a48137` 이다. 그 뒤의 변경 전체를 여기 요약한다.**
이 문서를 0단계부터 순서대로 따르면 된다. 단계마다 "보내줄 것"이 있다.

## 변경 요약 (9a48137 → 현재)

### A. 서버가 보고한 블로커 2건 — 해결
- **`live3r.data` 누락**: `.gitignore` 의 `data/` 가 `src/live3r/data/` 까지 무시했다
  (로컬엔 파일이 있어 테스트는 통과했다). 루트 고정 `/data/` 로 수정. 이제 깨끗한 클론에서 검증 후 푸시한다.
- **Sensenova 어댑터**: 이미지 1~28장 시퀀스를 이미지마다 별도 비전 블록으로. 문자열 `conversations`,
  멀티턴, 크기 제각각 이미지, placeholder 불일치·누락 파일 처리. 83만 건 지연 로딩(JSONL+오프셋 인덱스).

### B. 에러 없이 조용히 틀리던 버그 6건 — 수정
**이전 커밋으로 학습을 돌렸다면 결과를 버려야 한다** (1·2번 때문에 무엇도 학습되지 않았다).

| # | 버그 | 영향 |
|---|---|---|
| 1 | 패치 평탄화 순서가 공식과 다름 (래스터 vs 2×2 merge 블록) | **실가중치에서 모든 이미지가 뒤섞여 들어감** |
| 2 | 프로젝터 zero-init 교착 (fc2=0 과 gate=0 동시) | **모든 기울기가 정확히 0 — 프로젝터가 영원히 학습 안 됨** |
| 3 | LoRA 타깃명이 Qwen3-Next 기준 (`in_proj_qkvz`) | DeltaNet 24층 입력 프로젝션에 LoRA 누락 |
| 4 | 비디오 타임스탬프 `<0.0s>` → 공식 `<0.2 seconds>` | 사전학습 분포와 어긋남 |
| 5 | CUT3R 입력을 짧은 변 512 로 (공식은 긴 변) | 학습 해상도의 1.8배 입력 |
| 6 | gradient checkpointing 재계산 때 주입 훅 상태가 비어 있음 | 기울기 오류 또는 CheckpointError |

### C. 학습
- **torchrun DDP** (8 GPU): `no_sync` 누적, 로그마다 **랭크 간 파라미터 동기화 검사**(`sync OK`)
- 로그의 `inj` = 레이어별 ‖주입‖/‖비전 히든‖ — 프로젝터가 실제로 학습되는지의 직접 지표
- `--dump-samples N`: 학습 전에 프롬프트·라벨을 눈으로 확인

### D. 평가
- **홀드아웃**: `prepare_annotations.py` 가 무작위 2,000건을 학습에서 뺀다
- **기하 절제 평가** (`eval_geometry_ablation.py`): 진짜 기하 / 다른 샘플의 기하 / 기하 없음 손실 비교
  → "기하의 내용을 쓰나" 판정. S2 진행 여부를 정하는 관문
- **평가 입력을 학습과 통일** (`eval_path=live3r`, 기본값): lmms-eval 부모 경로는 학습과 세 군데가 달랐다
  (시스템 프롬프트 · 영상 비디오 모드 · 해상도 규칙). 생성은 **greedy** (qwen3_5 기본 temperature 0.7 이
  온도를 안 정한 태스크를 샘플링으로 돌리고 있었다)
- **게이트 자동 판정** (`run_gate.sh` + `check_gate.py`)

### E. 게이트 (사전 등록 — 사용자 결정)
| | 태스크 | 기준 | 용량 |
|---|---|---|---|
| 1. 공간 | `vsibench` | Δ ≥ +1.0 | 5.7 GB |
| 2. 일반 | **`videomme`** | Δ ≥ −1.0 | **101 GB** (서버 여유 1TB 확인 → 원래 게이트 유지) |
| 참고 | `mmstar` | 판정 안 함, 보고만 | 0.1 GB |
| 3. 지연 | `bench_latency.py` | drift < 1.2, ingest p95 예산 내 | — |

로컬 검증: 테스트 90개(공식 프로세서·실제 Qwen3.5 토크나이저 대조 포함), 가짜 Sensenova 로
S1 → S2(LoRA+checkpointing) → DDP 2프로세스 end-to-end, 실제 mp4 로 평가 3경로
(오프라인 키프레임·스트리밍·오라클), 깨끗한 클론에서 재현.

---

## 0. 서버에서 고친 코드 보존 → 최신 받기

서버에서 코드를 수정했다고 했으니, 덮어쓰기 전에 **반드시 남겨둔다.** 쓸만한 수정이면 반영하겠다.

```bash
cd /group-volume/wooyeol/vlm-live
git diff > /group-volume/wooyeol/server_changes_0923.patch
tar czf /group-volume/wooyeol/server_src_0923.tgz src scripts tests configs
git stash -u
git fetch origin
git reset --hard origin/main  # pull 대신 — 마지막 커밋을 덮어써서(amend) 올렸기 때문에
                              # 이미 받은 서버에서는 pull 이 '갈라진 이력'으로 멈출 수 있다
git log --oneline -1          # 가장 최근 커밋이어야 한다
ls src/live3r/data/           # __init__ collate datasets prompt vision
```

> `data/sensenova_*` 링크는 루트 `/data/` 라서 계속 무시된다 — 영향 없다.
> 예전 `.gitignore` 는 `data/` 로 써서 `src/live3r/data/` 까지 무시했다. 그게 누락 원인이었다.

**보내줄 것**: `server_changes_0923.patch` (비어 있지 않으면)

---

## 1. 환경 (10분)

```bash
pip install -r requirements.txt        # peft·scipy·roma·accelerate 가 새로 필요하다
pip install -e .                       # lmms-eval 에 --model live3r 등록 (평가 단계에서 필요)
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
python -c "import transformers.models.qwen3_5; print('qwen3_5 OK')"

# 모델 — 홈 디렉터리 말고 group-volume 에 (약 9GB)
export HF_HOME=/group-volume/wooyeol/hf_cache
huggingface-cli download Qwen/Qwen3.5-4B --local-dir /group-volume/wooyeol/models/Qwen3.5-4B
```

로컬 검증은 transformers **5.16.1** 에서 했다. 버전이 다르면 2단계의 공식 대조 테스트가 판정한다.

---

## 2. 테스트 — 실제 토크나이저·프로세서와 대조 (1분)

```bash
LIVE3R_TOKENIZER=/group-volume/wooyeol/models/Qwen3.5-4B python -m pytest tests -q
```

**기대: `78 passed`.** 특히 아래가 통과해야 한다 — 서버의 transformers 판이 우리와 같은 입력
형식을 만든다는 증거다:
- `test_vision_official.py` — 패치 평탄화가 공식과 비트 단위로 같다
- `test_prompt_builder.py::test_images_match_official_processor` / `test_video_matches_official_processor`
  — 프롬프트 토큰이 공식 템플릿+프로세서와 **토큰 단위로** 같다

**보내줄 것**: 실패가 있으면 그 출력 전체 (버전 차이면 여기서 드러난다)

---

## 3. Sensenova 어노테이션 검증·변환 (10~20분)

```bash
PYTHONPATH=src python scripts/prepare_annotations.py \
  data/sensenova_si_800k.json data/sensenova.jsonl \
  --media-root data/sensenova_media --check-files 5000
```

만드는 것: `data/sensenova.jsonl` + `.jsonl.idx.npy`(지연 로딩 인덱스) + `data/sensenova.report.json`
+ **`data/sensenova.holdout.jsonl`** (무작위 2,000건 — 학습에 절대 안 들어간다. 9단계에서 쓴다).
83만 건 JSON 배열을 DDP 8프로세스가 각자 통째로 파싱하면 메모리가 수십 GB 로 불어난다.
인덱스가 있으면 필요한 줄만 읽는다.

**보내줄 것**: 콘솔 마지막 요약 (총/유지/버린 사유 · 레코드당 이미지 · 턴 · 답변 유형 · 파일 누락)
— 특히 **placeholder 불일치 수**와 **파일 누락 비율**.

---

## 4. 스모크 A — 기하 인코더 없이 (1 GPU, 10분)

CUT3R 을 붙이기 전에 **데이터·LLM·학습 루프만** 따로 본다. 문제가 생기면 원인이 한 곳으로 좁혀진다.

```bash
PYTHONPATH=src python -m live3r.train.train \
  --config configs/live3r_4b.yaml --base-model /group-volume/wooyeol/models/Qwen3.5-4B \
  --geometry dummy --stage align \
  --ann data/sensenova.jsonl --media-root data/sensenova_media \
  --output outputs/smoke_dummy --max-steps 30 --grad-accum 1 --log-every 5 --save-every 0 \
  --dump-samples 3 --grad-checkpointing
```

확인:
1. **샘플 덤프** — 이미지 수만큼 `<|vision_start|><|image_pad|>×n<|vision_end|>` 가 있고,
   `⟦감독⟧` 이 **답변 + `<|im_end|>` 만**이어야 한다. 질문이나 이미지 토큰이 감독되면 안 된다.
2. **loss** 가 유한하고 내려가는 추세
3. **`inj`** (레이어별 ‖주입‖/‖비전 히든‖) — 0 에서 시작해 **움직여야 한다.**
   끝까지 0 이면 프로젝터가 학습 안 되는 것 (예전 교착 버그 재발)
4. `mem`, `samp/s`, `data/geom/fb ms` — 어디가 병목인지

---

## 5. CUT3R 준비 + 실측 (20분)

```bash
git clone https://github.com/CUT3R/CUT3R third_party/CUT3R
cd third_party/CUT3R/src/croco/models/curope && python setup.py build_ext --inplace && cd -
mkdir -p checkpoints && cd checkpoints
gdown --fuzzy 'https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/' && cd -

PYTHONPATH=src python scripts/verify_geometry_adapter.py --name cut3r \
  --checkpoint checkpoints/cut3r_512_dpt_4_64.pth --tap-layers 6 9 12 \
  --repo-path third_party/CUT3R --frames 200
```

> ⚠️ **curope 컴파일은 필수다.** CUT3R 포즈 토큰의 2D 위치가 `-1` 인데 순수 PyTorch RoPE2D
> 폴백은 음수에서 죽는다 (live3r 이 폴백을 고쳐두긴 했지만 느리다).

**보내줄 것**: 출력 전체 — `hidden_size`(768 이어야), `grid_hw`(512×384 → 24×32), 프레임당 ms, drift.

---

## 6. 스모크 B — CUT3R 포함 (1 GPU, 10분)

```bash
PYTHONPATH=src python -m live3r.train.train \
  --config configs/live3r_4b.yaml --base-model /group-volume/wooyeol/models/Qwen3.5-4B \
  --geometry-checkpoint checkpoints/cut3r_512_dpt_4_64.pth --cut3r-repo third_party/CUT3R \
  --stage align --ann data/sensenova.jsonl --media-root data/sensenova_media \
  --output outputs/smoke_cut3r --max-steps 30 --grad-accum 1 --log-every 5 --save-every 0 \
  --grad-checkpointing
```

4단계와 같은 항목 + `geom ms/샘플` (CUT3R 비용).

---

## 7. 스모크 C — 8 GPU DDP (10분)

```bash
PYTHONPATH=src torchrun --nproc_per_node 8 -m live3r.train.train \
  --config configs/live3r_4b.yaml --base-model /group-volume/wooyeol/models/Qwen3.5-4B \
  --geometry-checkpoint checkpoints/cut3r_512_dpt_4_64.pth --cut3r-repo third_party/CUT3R \
  --stage align --ann data/sensenova.jsonl --media-root data/sensenova_media \
  --output outputs/smoke_ddp --max-steps 20 --grad-accum 2 --log-every 5 --save-every 0 \
  --grad-checkpointing
```

확인: 로그 끝의 **`sync OK`** — 8개 랭크의 파라미터가 비트 단위로 같다는 뜻이다
(`DIFF` 가 뜨면 DDP 동기화 문제). 그리고 `samp/s` → **1에폭 소요시간 = 83만 ÷ samp/s**.

---

## 8. 본 학습 S1 (4~7 이 전부 초록일 때만)

```bash
PYTHONPATH=src torchrun --nproc_per_node 8 -m live3r.train.train \
  --config configs/live3r_4b.yaml --base-model /group-volume/wooyeol/models/Qwen3.5-4B \
  --geometry-checkpoint checkpoints/cut3r_512_dpt_4_64.pth --cut3r-repo third_party/CUT3R \
  --stage align --ann data/sensenova.jsonl --media-root data/sensenova_media \
  --output outputs/4b_s1 --epochs 1 --lr 1e-3 --grad-accum 16 --log-every 20 --save-every 500 \
  --grad-checkpointing 2>&1 | tee outputs/4b_s1.log
```

유효 배치 = 8 × 16 = 128. S1 은 프로젝터만 학습한다(LLM 동결). S2(LoRA)는 S1 결과를 보고 정한다.

---

## 9. S1 이 끝나면 — 기하가 실제로 쓰이는지 판정 (30분)

손실이 내려가도 그게 기하 덕인지는 모른다. 홀드아웃에서 **진짜 기하 / 다른 샘플의 기하 / 기하 없음**의
손실을 비교한다. 이게 S2(LoRA)로 넘어갈지 정하는 관문이다.

```bash
PYTHONPATH=src python scripts/eval_geometry_ablation.py \
  --config configs/live3r_4b.yaml --base-model /group-volume/wooyeol/models/Qwen3.5-4B \
  --geometry-checkpoint checkpoints/cut3r_512_dpt_4_64.pth --cut3r-repo third_party/CUT3R \
  --weights outputs/4b_s1/final.pt --stage align \
  --ann data/sensenova.holdout.jsonl --media-root data/sensenova_media --n 300 \
  --out outputs/4b_s1_ablation.json
```

- `shuffled − real > 0` (95% 하한이 양수) → 기하의 **내용**을 쓴다 → S2 진행
- `none − real > 0` 인데 shuffled 와 구분 안 됨 → 기하를 "신호"로만 쓴다 → S2 전에 원인을 본다
- 둘 다 ≈ 0 → 기하가 무시된다 → 멈추고 보고

**보내줄 것**: 콘솔 요약 전체 (특히 판정 줄과 "같은 격자 모양 비율")

---

## 10. 평가 — 베이스라인(학습 전)과 게이트(S1 후)

### 평가 데이터 용량 (HF 실측)

| 벤치 | 용량 | 용도 |
|---|---|---|
| **VSI-Bench** | **5.7 GB** | 주지표 · 게이트 1 |
| **VideoMME** | **101 GB** | 일반 능력 · 게이트 2 (서버 여유 1TB) |
| MMStar | 0.1 GB | 이미지 일반 능력 — 참고(판정 안 함). 규칙 채점, API 불필요 |
| MVBench | 17 GB | 영상 일반 능력 — 선택 |

lmms-eval 이 태스크 실행 때 HF 에서 받는다. 캐시 위치를 group-volume 으로:
```bash
export HF_HOME=/group-volume/wooyeol/hf_cache
```

### 평가 경로가 바뀌었다 (중요)

`--model live3r` 는 이제 **학습과 같은 입력**으로 평가한다 (`eval_path=live3r`, 기본값).
lmms-eval 부모 경로는 학습 분포와 세 군데가 달랐다 — 시스템 프롬프트("You are a helpful
assistant."), 영상 비디오 모드, 해상도 규칙. 그리고 qwen3_5 기본 temperature 0.7 때문에 온도를
안 정한 태스크(cv_bench)는 샘플링으로 돌았다 → 이제 **greedy**.

### 10-1. 베이스라인 — 학습 전에 지금 바로

VideoMME 는 zip 101GB 를 받고(`$HF_HOME/hub`) 다시 `$HF_HOME/videomme` 에 **압축을 푼다 → 약 200GB**.
첫 실행 때 오래 걸리니 학습(4~8단계)과 **병행해서** 먼저 받아두면 좋다.
저장소 ID 는 **태스크와 같아야** lmms-eval 이 캐시를 재사용한다 (다르면 101GB 를 두 번 받는다):
```bash
export HF_HOME=/group-volume/wooyeol/hf_cache      # 1단계와 같은 값
huggingface-cli download lmms-eval/Video-MME --repo-type dataset   # videomme.yaml 의 dataset_path
huggingface-cli download nyu-visionx/VSI-Bench --repo-type dataset
```

```bash
# 오프라인 (균등 키프레임 32장, 이미지 모드) — weights 없음 = 베이스 VLM 과 같은 출력
BASE_MODEL=/group-volume/wooyeol/models/Qwen3.5-4B TASKS=vsibench,videomme,mmstar \
  bash scripts/run_eval.sh configs/live3r_4b.yaml "" outputs/eval_base

# 스트리밍 (라이브 제약) vs 오프라인 오라클 — 이 둘의 차이가 '키프레임 선택 비용'
bash scripts/run_streaming_eval.sh configs/live3r_4b.yaml "" outputs/stream_base
SELECTOR=uniform_oracle bash scripts/run_streaming_eval.sh configs/live3r_4b.yaml "" outputs/stream_oracle
```

### 10-2. 게이트 — S1(또는 S2) 결과가 나오면

```bash
BASE_MODEL=/group-volume/wooyeol/models/Qwen3.5-4B \
  bash scripts/run_gate.sh configs/live3r_4b.yaml outputs/4b_s1/final.pt outputs/gate_s1
```

같은 경로에서 베이스(weights 없음)와 학습 결과를 한 번씩 재고 `check_gate.py` 가 판정한다:
공간(vsibench) Δ ≥ +1.0 · 일반(videomme) Δ ≥ −1.0. MMStar 는 같이 재서 보고만 한다.
대표 지표: vsibench → `vsibench_overall`, videomme → `videomme_perception_score`, mmstar → `average`.

**보내줄 것**: 10-1 의 VSI·VideoMME·MMStar 점수와 스트리밍/오라클 VSI, 10-2 의 판정 출력 전체.

---

## 보고 양식

```
[0] patch: 있음/없음
[1] torch / transformers / peft 버전:
[2] pytest: N passed / 실패 목록
[3] prepare: 총 / 유지 / 버린 사유 / 이미지 분포 / 누락 비율
[4] smoke A: loss 추세, inj 값, samp/s, mem, 덤프 이상 여부
[5] CUT3R: hidden_size, grid_hw, ms/frame, drift
[6] smoke B: loss, inj, geom ms
[7] DDP: sync OK?, samp/s → 1에폭 예상 시간
[9] (S1 후) 절제: real / shuffled / none 손실, 판정, 같은 격자 비율
[10-1] 베이스라인: VSI(오프라인) / VideoMME / MMStar / VSI(스트리밍) / VSI(오라클)
[10-2] 게이트: check_gate.py 출력
```
