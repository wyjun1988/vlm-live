# 서버 다음 진행사항 — 2026-09-23 (커밋 9edc999 이후)

서버 피드백 두 건(`live3r.data` 누락 · Sensenova 이미지 시퀀스 어댑터 없음)은 해결했다.
그 과정에서 Qwen3.5 공식 구현과 직접 대조해 **에러 없이 조용히 틀리던 버그 6건**을 찾아 고쳤다.
**이전 커밋으로 학습을 돌렸다면 결과를 버려야 한다** (아래 표 1·2번 때문에 무엇도 학습되지 않았다).

| # | 버그 | 영향 |
|---|---|---|
| 1 | 패치 평탄화 순서가 공식과 다름 (래스터 vs 2×2 merge 블록) | **실가중치에서 모든 이미지가 뒤섞여 들어감** |
| 2 | 프로젝터 zero-init 교착 (fc2=0 과 gate=0 동시) | **모든 기울기가 정확히 0 — 프로젝터가 영원히 학습 안 됨** |
| 3 | LoRA 타깃명이 Qwen3-Next 기준 (`in_proj_qkvz`) | DeltaNet 24층 입력 프로젝션에 LoRA 누락 |
| 4 | 비디오 타임스탬프 `<0.0s>` → 공식 `<0.2 seconds>` | 사전학습 분포와 어긋남 |
| 5 | CUT3R 입력을 짧은 변 512 로 (공식은 긴 변) | 학습 해상도의 1.8배 입력 |
| 6 | gradient checkpointing 재계산 때 주입 훅 상태가 비어 있음 | 기울기 오류 또는 CheckpointError |

로컬 검증: 테스트 78개(공식 프로세서·실제 Qwen3.5 토크나이저 대조 포함), 가짜 Sensenova 로
S1 → S2(LoRA+checkpointing) → DDP 2프로세스까지 end-to-end, 랭크 간 파라미터 동일 확인,
깨끗한 클론에서 재현.

---

## 0. 서버에서 고친 코드 보존 → 최신 받기

서버에서 코드를 수정했다고 했으니, 덮어쓰기 전에 **반드시 남겨둔다.** 쓸만한 수정이면 반영하겠다.

```bash
cd /group-volume/wooyeol/vlm-live
git diff > /group-volume/wooyeol/server_changes_0923.patch
tar czf /group-volume/wooyeol/server_src_0923.tgz src scripts tests configs
git stash -u
git pull
git log --oneline -1          # 9edc999 이후여야 한다
ls src/live3r/data/           # __init__ collate datasets prompt vision
```

> `data/sensenova_*` 링크는 루트 `/data/` 라서 계속 무시된다 — 영향 없다.
> 예전 `.gitignore` 는 `data/` 로 써서 `src/live3r/data/` 까지 무시했다. 그게 누락 원인이었다.

**보내줄 것**: `server_changes_0923.patch` (비어 있지 않으면)

---

## 1. 환경 (10분)

```bash
pip install -r requirements.txt        # peft·scipy·roma·accelerate 가 새로 필요하다
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

만드는 것: `data/sensenova.jsonl` + `.jsonl.idx.npy`(지연 로딩 인덱스) + `data/sensenova.report.json`.
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

## 병행 — VSI-Bench 확보 (가능하면)

주지표가 VSI-Bench 다. **학습 전 베이스라인 숫자**(스트리밍 vs 오프라인 오라클)가 있어야 이후
변화가 해석된다. 평가셋은 수 GB 수준이다 — 공간이 되면:

```bash
huggingface-cli download nyu-visionx/VSI-Bench --repo-type dataset --local-dir data/eval/vsibench
```

공간이 안 되면 알려달라. 부분집합(예: ScanNet 유래만)으로 시작하는 방법을 정리하겠다.

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
```
