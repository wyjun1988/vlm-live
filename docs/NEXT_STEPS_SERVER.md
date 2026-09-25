# 서버 다음 진행사항 — 2026-09-23 (09-24 M2 실측 반영)

> ## 2026-09-26 — start here: the four-day run on 4 nodes
> **Run [`docs/SERVER_WEEKEND.md`](SERVER_WEEKEND.md)**: `scripts/server_weekend.sh` with one `ROLE` per node
> (`real`, `control`, `sft`, `small`) does steps 1–10 of this document unattended (all but the streaming evaluation
> in 10-1), on SenseNova-SI only — one epoch for S1 and one for S2, real and control arms on separate nodes, plus a
> plain-SFT reference, an S2-without-S1 arm, a second seed and the 2B model — and writes `outputs/weekend/REPORT.md`.
> The nodes coordinate through markers on the shared volume; there is no multi-node job. Steps 0–7 below stay
> valid as the manual procedure and as reference; the script follows them. Changes since this document was last
> updated, all exercised by the script:
> - `train.py --no-geometry` — the plain-SFT reference arm (never runs the encoder, injects nothing).
> - `train.py --resume` — an interrupted arm continues from `resume.pt` (weights, optimizer, schedule, data
>   position) with the same data order as an uninterrupted run; a non-finite loss or gradient skips the sample or
>   the step instead of ending the run (three skipped steps in a row still stop it).
> - `configs/server_4b.yaml` — this server's paths (model, CUT3R) in one file for every tool.
> - `scripts/fetch_eval_data.py` — VSI-Bench / VideoMME / MMStar downloaded and unpacked once, so parallel
>   lmms-eval runs do not race; also lays out `data/eval/vsibench` for `scripts/eval_vsi_local.py`.
> - `scripts/run_gate.sh` — `ONLY=base|base_video|trained|check` runs one measurement (three GPUs at once).
> - S1 passes over text-only records (no gradient path through the projector) instead of stopping on one.
> - `train.py --max-hours` — a soft time cap that still saves `final.pt`.
> - `eval_vsi_local.py` — defaults to CUDA when present (was CPU off a Mac) and saves per-question scores.
> - `prepare_annotations.py` — the holdout rate uses the file's own size (a 3.6k-record file got 0 holdout).
> - M2 zero-shot results since 09-24 (format instruction, routed facts, 56.0 on 60 videos): `docs/BRAINSTORM.md` §0b.

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
| 1. 공간 | `vsibench` | Δ ≥ +1.0 — **기준선 = 베이스의 최선 형식** (이미지·비디오 모드 중 높은 쪽, 09-24 결정) | 5.7 GB |
| 2. 일반 | **`videomme`** | Δ ≥ −1.0 | **101 GB** (서버 여유 1TB 확인 → 원래 게이트 유지) |
| 참고 | `mmstar` | 판정 안 함, 보고만 | 0.1 GB |
| 3. 지연 | `bench_latency.py` | drift < 1.2, ingest p95 예산 내 | — |

### F. M2 실가중치 실측으로 찾은 것 (2026-09-24 — H100 이 막힌 동안)
실제 Qwen3.5 0.8B/4B · 실제 CUT3R 체크포인트 · 실제 Sensenova 1k(HF 미리보기)로 로컬에서 돌려 봤다.
상세 수치는 `docs/M2_LOCAL_20260924.md`.

| # | 발견 | 서버 영향 · 조치 |
|---|---|---|
| 1 | **S1 lr 1e-3 이면 주입이 비전 표현을 15~30배로 덮는다** (레이어 0, 10스텝 만에) | **8단계 lr 을 3e-5 로** 바꿨다. `inj` > 3 이면 학습 로그에 경고 |
| 2 | CUT3R 원본 `load_model()` 이 torch≥2.6 에서 실패 (`weights_only`, omegaconf) | 어댑터가 필요한 클래스만 허용해 안전하게 연다. `pip install omegaconf` 필요 (requirements 반영) |
| 3 | CUT3R 코드를 임포트하면 **루트 로거가 DEBUG** 가 된다 (accelerate `get_logger`) → PIL 이 이미지마다 DEBUG ~10줄 → 83만 건이면 로그 수 GB | 임포트 직후 레벨 복원 |
| 4 | `gdown --fuzzy` 가 최신 gdown 에 없다 | 5단계 명령을 파일 ID 형식으로 |
| 5 | Qwen3.5 템플릿 thinking 기본값이 크기마다 반대 (0.8B·2B off / **4B·9B on**) | 입력은 원래 빈 think 블록을 직접 넣어 무관. 기본값에 기대던 테스트 1개가 **서버의 4B 토크나이저로는 실패했을 것** → 수정 |
| 6 | `LazyJsonl` 이 한 번 쓰인 뒤 피클링 실패 (spawn 워커) | 수정 (리눅스 fork 에서는 안 터졌을 것) |
| 7 | **LM 헤드가 전 위치 로짓을 만든다** — 어휘 248k. 2.6k 토큰 샘플에서 순전파 추가 메모리 9.25GB (감독 토큰 61개). **4B · 8k 토큰이면 로짓·CE 만 ~28GB + 기울기** | 감독 위치 로짓만 계산 → 같은 샘플 1.17GB, 손실 동일(Δ 0). **서버 학습 메모리·속도에 직접 영향** — 7단계 `mem` 이 예전 예상보다 훨씬 낮게 나올 것 |
| 8 | **생성이 `<|im_end|>` 에서 안 멈춘다** — 로컬 Qwen3.5 체크포인트에 generation_config.json 이 없고 config eos 는 `<|endoftext|>` 뿐. LoRA 학습 후 답 뒤에 다음 턴을 이어 씀 (M2 18%) | `Live3RModel` 이 eos 에 `<|im_end|>` 추가. lmms-eval 게이트 경로는 원래 괜찮았다 (eos 를 따로 넘김) |
| 9 | **답이 아주 긴 샘플 하나가 OOM** — 1k 미리보기 id 464656 은 답이 4,860토큰 → 감독 위치 로짓만 4.5GB. M2 4B 학습이 여기서 죽었다 | `--max-length` 인자 추가 (M2 는 6144). 서버 H100 80GB 는 괜찮지만 83만 건에 더 긴 답이 있을 수 있다 — `mem` 을 봐 달라 |
| 10 | CUT3R 헤드를 부르면 `transpose_to_landscape` 래퍼가 `[B,2]` 텐서를 요구한다 | 수정. 우리 프레임 단위 CUT3R 재현이 원본 `forward_recurrent` 와 **점맵·포즈 Δ 0.0** 확인 (`scripts/check_cut3r_stream.py`) |

실가중치 결과: 우리 입력 경로 = 공식 경로 (4B bf16 로짓 Δ **0.0**) · zero-init 주입은 베이스와 로짓 Δ 0.0 ·
**VSI 키프레임 선택 비용 없음** (4B · 영상 60개 1,151문항: halving 49.9 vs 오라클 48.6, Δ 95% [−1.7, +4.9]) ·
**베이스 4B VSI ≈ 50** (비디오 모드, 부분집합 — 73.3 까지 ~23점) · 4B 지연 모드 TTFT 의 94% 가 시각 프리필
(질문만 TTFT 는 M2 에서도 0.41초) · **게이트 1 교란 발견 → 기준선을 베이스의 최선 형식으로** (10-2).

**S1 파일럿 (0.8B + 실제 CUT3R · Sensenova 1k 미리보기 905건 · 114스텝)**: 학습은 깨끗하게 돌았다
(주입 비율 < 1, 손실 1.7 → 0.7). 그런데 절제 판정은 **"기하를 신호로만 쓴다"** — 홀드아웃에서
none 2.49 → real 0.93 이지만 **shuffled 0.90 ≈ real**. 생성해 보면 프로젝터가 배운 건 Sensenova 의
**답 형식**(`C. the right`, 짧은 `yes`)과 답 분포였다 (베이스는 `A`, `Let's break this down…`).
주입 경로가 기하와 무관한 소프트 프롬프트처럼 쓰인 것. **대조 프로젝터**(같은 설정, 다른 샘플의 기하로 학습)도
학습 곡선이 겹치고 홀드아웃 0.887 (진짜 쪽 0.928) — **기하 내용의 가치 −0.04 ± 0.05, 구분 안 됨.**
서버 규모에서 내용이 쓰이기 시작할지는 8-2(대조 S1) · 9단계가 판정한다.

**S2 LoRA 파일럿 (4B · Sensenova 905건 · 1에폭, M2)**: VSI(영상 60개, 이미지 모드) 44.1, 대조(다른 샘플 기하) 42.3 —
진짜 − 대조 +1.9 [−0.9, +4.4]. 형식 실패는 0% 가 됐지만 수치 답이 데이터 분포로 쏠려(절대 거리 대부분 "1.1")
베이스 최선 형식(48.6)보다 낮다. **반면 베이스에 형식 지시 한 줄만 붙여도 53.7** — 작은 SFT 는 지시 한 줄만 못하다.
제로샷 CUT3R 장면 지도(지도 이미지 + 측정값)는 순효과 0 (방 크기 +13~19, 등장 순서·방향 −7~−10). 상세는 M2 문서 8·9절.

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

**기대: 전부 통과 (2026-09-24 기준 `100 passed`).** 특히 아래가 통과해야 한다 — 서버의 transformers 판이 우리와 같은 입력
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
gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O cut3r_512_dpt_4_64.pth && cd -

PYTHONPATH=src python scripts/verify_geometry_adapter.py --name cut3r \
  --checkpoint checkpoints/cut3r_512_dpt_4_64.pth --tap-layers 6 9 12 \
  --repo-path third_party/CUT3R --frames 200
```

> ⚠️ **curope 컴파일은 필수다.** CUT3R 포즈 토큰의 2D 위치가 `-1` 인데 순수 PyTorch RoPE2D
> 폴백은 음수에서 죽는다 (live3r 이 폴백을 고쳐두긴 했지만 느리다 — M2 MPS 폴백 실측 307ms/프레임).
>
> 체크포인트 로딩: CUT3R 원본 `load_model()` 은 torch≥2.6 에서 실패한다 (`weights_only` 기본값 변경,
> 체크포인트 안에 omegaconf 학습 설정이 들어 있다). live3r 어댑터는 필요한 클래스만 허용해서 안전하게
> 연다 — `pip install omegaconf` 만 되어 있으면 된다 (requirements 에 추가됨).
>
> M2 실측 기준값 (2026-09-24): 793M 파라미터 · 탭 `[1,768,768]`×3 · 포즈 `[1,1,768]` ·
> 512×384 입력 → 격자 24×32 · **재귀 상태 768 토큰 (7.88MB, 상수)** · drift 1.000.
> 서버 값이 이와 다르면 알려달라.

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
  --output outputs/4b_s1 --epochs 1 --lr 3e-5 --grad-accum 16 --log-every 20 --save-every 500 \
  --grad-checkpointing 2>&1 | tee outputs/4b_s1.log
```

유효 배치 = 8 × 16 = 128. S1 은 프로젝터만 학습한다(LLM 동결). S2(LoRA)는 S1 결과를 보고 정한다.

### 8-2. 대조 프로젝터 — 같은 S1 을 **다른 샘플의 기하**로 (09-24 결정)

M2 파일럿에서 진짜 기하로 학습한 프로젝터가 기하 내용이 아니라 **답 형식**을 배웠다 (F 절). 주입 경로 자체가
소프트 프롬프트처럼 학습되기 때문이다. 그래서 같은 데이터·스텝·시드로 **내용만 틀린 기하**를 주는 대조
프로젝터를 나란히 학습한다 — 기하 없이 배울 수 있는 것(형식·분포)은 대조군도 다 배운다. 9단계에서 둘의
홀드아웃 손실 차이가 **기하 내용의 가치**다.

```bash
PYTHONPATH=src torchrun --nproc_per_node 8 -m live3r.train.train \
  --config configs/live3r_4b.yaml --base-model /group-volume/wooyeol/models/Qwen3.5-4B \
  --geometry-checkpoint checkpoints/cut3r_512_dpt_4_64.pth --cut3r-repo third_party/CUT3R \
  --stage align --ann data/sensenova.jsonl --media-root data/sensenova_media \
  --output outputs/4b_s1_control --epochs 1 --lr 3e-5 --grad-accum 16 --log-every 20 --save-every 500 \
  --grad-checkpointing --geom-control shuffled 2>&1 | tee outputs/4b_s1_control.log
```

8단계와 **옵션이 `--geom-control shuffled` 와 출력 경로만 다르다** (시드·데이터·스텝 동일해야 비교가 된다).
로그 첫머리에 "대조 프로젝터 모드", 끝에 "같은 격자 기증자 N%" 가 찍힌다.
GPU 를 나눠 두 S1 을 동시에 돌려도 된다 — 4장씩이면 `--grad-accum 32` 로 유효 배치 128 을 맞춘다
(시드가 같으면 스텝마다 같은 샘플 묶음을 본다). S1 이 오래 걸리면 두 쪽 모두 같은 `--max-steps` 로 줄여라.

> ⚠️ **lr 은 3e-5** 다 (예전 문서의 1e-3 아님). M2 에서 실데이터로 재보니 lr 1e-3 은 10스텝 만에
> 레이어 0 의 주입 비율(`inj L0`)이 **15~30** 이 된다 — 기하 신호가 비전 표현을 수십 배로 덮는다.
> LLaVA 식 프로젝터(비전 토큰을 **대체**)의 lr 관례를 **더하는** 구조에 그대로 쓰면 안 된다.
> 로그의 `inj` 가 3 을 넘으면 경고가 뜬다 — 그러면 lr 을 더 낮춰라.

---

## 9. S1 이 끝나면 — 기하가 실제로 쓰이는지 판정 (30분)

손실이 내려가도 그게 기하 덕인지는 모른다. 홀드아웃에서 **진짜 기하 / 다른 샘플의 기하 / 기하 없음**의
손실을 비교한다. 이게 S2(LoRA)로 넘어갈지 정하는 관문이다.

```bash
PYTHONPATH=src python scripts/eval_geometry_ablation.py \
  --config configs/live3r_4b.yaml --base-model /group-volume/wooyeol/models/Qwen3.5-4B \
  --geometry-checkpoint checkpoints/cut3r_512_dpt_4_64.pth --cut3r-repo third_party/CUT3R \
  --weights outputs/4b_s1/final.pt --control-weights outputs/4b_s1_control/final.pt --stage align \
  --ann data/sensenova.holdout.jsonl --media-root data/sensenova_media --n 300 \
  --out outputs/4b_s1_ablation.json
```

`--control-weights` 를 주면 같은 샘플·같은 기증자로 대조 프로젝터도 잰다:
**기하 내용의 가치 = loss[대조·shuffled] − loss[real]** (95% 하한 > 0 이면 "가치 있다").
이게 S2 로 넘어갈지의 1순위 판정이다 — 아래 real/shuffled/none 비교는 진짜 기하 프로젝터 안에서의 의존도.

- `shuffled − real > 0` (95% 하한이 양수) → 기하의 **내용**을 쓴다 → S2 진행
- `none − real > 0` 인데 shuffled 와 구분 안 됨 → 기하를 "신호"로만 쓴다 → S2 전에 원인을 본다
- 둘 다 ≈ 0 → 기하가 무시된다 → 멈추고 보고

> M2 파일럿(0.8B · 905샘플)은 **두 번째 경우**였고, 원인은 **답 형식 학습**이었다 (`docs/M2_LOCAL_20260924.md` 6절).
> 서버 S1 도 같으면 S2 로 바로 넘어가지 말고 보고해 달라 — 형식을 기하 경로 밖에서 흡수시키는 설계를 같이 정한다.
> `none − real` 이 크다는 것만으로는 기하 효과가 아니다.

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

게이트 1 기준선용 **베이스 비디오 모드** 점수는 10-2 의 `run_gate.sh` 가 직접 잰다.

### 10-2. 게이트 — S1(또는 S2) 결과가 나오면

```bash
BASE_MODEL=/group-volume/wooyeol/models/Qwen3.5-4B \
  bash scripts/run_gate.sh configs/live3r_4b.yaml outputs/4b_s1/final.pt outputs/gate_s1
```

같은 경로에서 베이스(weights 없음)와 학습 결과를 한 번씩 재고 `check_gate.py` 가 판정한다:
공간(vsibench) Δ ≥ +1.0 · 일반(videomme) Δ ≥ −1.0. MMStar 는 같이 재서 보고만 한다.
대표 지표: vsibench → `vsibench_overall`, videomme → `videomme_perception_score`, mmstar → `average`.

> **게이트 1 기준선 = 베이스의 최선 형식** (09-24 결정): 베이스는 키프레임을 **이미지로** 받으면 VSI 답 형식을
> 자주 어긴다 — M2 4B 영상 60개: **이미지 모드 24.7 vs 비디오 모드 48.6**, 선택형 4종은 **전부 0점**
> (답이 전부 `"To determine..."`·`"Based on..."` 로 시작해 16토큰에서 잘린다). 같은 경로 비교만 하면
> 학습이 공간 이해 없이 **"짧게 답하는 법"만 배워도** +1.0 을 넘는다. 그래서 `run_gate.sh` 는 베이스를
> 3번 잰다: [1/3] 이미지 모드(전 태스크) · **[2/3] 비디오 모드(vsibench 만)** · [3/3] 학습 결과(이미지 모드).
> `check_gate.py --base-alt video=...` 가 공간 게이트의 기준선으로 **높은 쪽**을 쓰고 출력에
> `↳ 게이트1 기준선 = 베이스 최선 형식: 기본 X · video Y → ...` 로 표시한다. VideoMME 는 같은 경로 비교 그대로.
> 베이스 실행은 `use_geometry=False` (zero-init 이라 출력이 같고 CUT3R 을 안 돌려 빠르다).

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
[8] S1 · 대조 S1: 마지막 loss·inj, 대조 로그 끝의 '같은 격자 기증자' 비율
[9] (S1 후) 절제: real / shuffled / none / 대조·shuffled 손실, **기하 내용의 가치**, 두 판정, 같은 격자 비율
[10-1] 베이스라인: VSI(오프라인) / VideoMME / MMStar / VSI(스트리밍) / VSI(오라클)
[10-2] 게이트: check_gate.py 출력 전체 (기준선 줄 포함)
```
