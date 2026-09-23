# Live3R — 스트리밍 3D 공간이해 VLM

**스트리밍 3D 기하 인코더의 잠재 토큰을 Qwen3.5 소형 VLM 디코더에 다깊이 residual 로 주입해,
프레임당 상수 비용으로 공간을 이해하고 즉시 답하는 모델.**

타깃: VSI-Bench · MMSI-Bench (정확도) + OVO-S-Bench (스트리밍 공간지능) + 자체 지연 하니스.
스케일: Qwen3.5 **4B → 2B → 0.8B**.

---

## 왜 (포지셔닝)

| | 기하 인코더 | 융합 | 라이브 | VSI-Bench |
|---|---|---|---|---|
| VLM-3R (CVPR'26) | CUT3R (스트리밍) | late cross-attn 1회 | 미구현 | — |
| SpatialStack (CVPR'26) | VGGT (**오프라인 전프레임**) | deepstack 다깊이 | **불가** | 67.5 (오픈 1위) |
| **Live3R** | **CUT3R 계보 (스트리밍)** | **deepstack 다깊이** | **가능** | 목표 |

SpatialStack 이 품질 1위지만 VGGT 는 쿼리마다 전 프레임을 다시 본다 → O(N²), 매 쿼리 재계산.
VLM-3R 은 스트리밍 인코더를 쓰지만 융합이 약하고 베이스가 구형이다.
**Live3R = SpatialStack 의 융합 + VLM-3R 의 스트리밍 인코더 + Qwen3.5 소형화 + 라이브 런타임.**

측정된 구조적 이점 (초소형 모델, 128프레임, MPS):
```
오프라인(전 프레임 재인코딩) TTFT 6,100 ms  →  Live3R TTFT 46 ms   = 132× 
drift 1.08 (프레임당 비용이 스트림 길이에 거의 무관)
```

## 지금 상태

- [x] 설계 + 리서치 (`docs/DESIGN.md`, `docs/RESEARCH_NOTES.md`)
- [x] 코어 구현: `GeometryStream` 추상화 · `GeometryProjector` · `DeepStackInjector` · `LiveSession`
- [x] **맥에서 가중치 없이 도는 엔드투엔드 형상 검증** (`scripts/smoke_test.py`, 19 tests)
- [x] 지연 하니스 (`scripts/bench_latency.py`)
- [x] 스케일별 설정 (4B / 2B / 0.8B)
- [x] 데이터 로더 + 2단계 학습 스크립트 (`src/live3r/data`, `src/live3r/train`)
- [x] 평가 계획 (`docs/BENCHMARKS.md`) — 채택 게이트 사전 등록
- [x] **CUT3R 어댑터 완성** — 실제 저장소 코드로 재귀 루프·탭 토큰·포즈 토큰 검증
- [x] **lmms-eval 연동** — `--model live3r` 등록, 타깃 태스크 9종 존재 확인
- [x] **스트리밍 VSI 평가 하니스** — 제약을 기계가 강제, 인과적 키프레임 선택기 4종
- [x] **Sensenova(이미지 시퀀스) 어댑터 + torchrun DDP 학습** — 가짜 데이터로 S1→S2→DDP end-to-end 검증
- [ ] 서버 스모크 → S1 본 학습 (`docs/NEXT_STEPS_SERVER.md`)
- [ ] 기하 인코더 베이크오프
- [ ] 2B / 0.8B 스케일 다운 실측

## 빠른 시작 (맥, 다운로드 0)

```bash
bash scripts/setup_env.sh          # conda env 'live3r'
conda activate live3r
PYTHONPATH=src python scripts/smoke_test.py
PYTHONPATH=src python scripts/bench_latency.py --tiny --frames 128 --offline
python -m pytest tests -q
```

## GPU 머신에서

> **지금 서버가 할 일: [`docs/NEXT_STEPS_SERVER.md`](docs/NEXT_STEPS_SERVER.md)** (2026-09-23)
>
> **단계별 런북: [`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — 설치부터 학습·평가까지 복붙 가능한 순서.
> 실모델 경로는 아직 한 번도 안 돌았으니 싸게 실패하는 순서대로 가는 게 낫다.

```bash
pip install -r requirements.txt
# 1) 기하 인코더 어댑터를 실제 체크포인트로 검증 (형상·지연 실측)
PYTHONPATH=src python scripts/verify_geometry_adapter.py \
    --name cut3r --checkpoint checkpoints/cut3r_512_dpt_4_64.pth
# 2) 실모델 지연
PYTHONPATH=src python scripts/bench_latency.py --config configs/live3r_4b.yaml --device cuda --offline
```

데이터는 `docs/DATA.md` — **무엇을 받아야 하는지 우선순위와 용량이 정리돼 있다.**

## 구조

```
src/live3r/
  geometry/   GeometryStream 추상화 + 어댑터 (dummy/cut3r/anchor3r/lingbot/vggt_window)
  fusion/     GeometryProjector (격자 리샘플) + DeepStackInjector (훅 주입)
  model/      Live3RModel (Qwen3.5 래퍼), LoRA
  serve/      LiveSession (스트리밍 런타임 + M-RoPE 커서)
  eval/       지연 하니스
configs/      live3r_{4b,2b,08b,dummy}.yaml
scripts/      setup_env.sh  smoke_test.py  bench_latency.py  verify_geometry_adapter.py
```

### 에러 없이 조용히 틀리던 것들 (전부 테스트로 막아둠)

2026-09-23 에 Qwen3.5 공식 프로세서·템플릿과 **직접 대조**해서 찾은 것들이 가장 무거웠다.
랜덤 초소형 모델 테스트로는 하나도 안 잡힌다 — 전부 에러 없이 돈다.

| 버그 | 결과 | 막는 테스트 |
|---|---|---|
| 패치 평탄화가 래스터 순서 (공식은 2×2 merge 블록) | 실가중치에서 이미지가 뒤섞임 | `test_vision_official.py` |
| 프로젝터 fc2=0 · gate=0 동시 초기화 | 모든 기울기 0, 영원히 학습 안 됨 | `test_train_step.py` |
| LoRA 타깃 `in_proj_qkvz` (Qwen3-Next 이름) | DeltaNet 24층에 LoRA 누락 | `test_lora_targets.py` |
| 타임스탬프 `<0.0s>` (공식 `<0.2 seconds>`) | 사전학습 분포 이탈 | `test_prompt_builder.py` |
| CUT3R 입력 짧은 변 512 (공식 긴 변) | 1.8배 큰 입력 | `test_streaming_harness.py` |
| checkpointing 재계산 때 주입 상태 비어 있음 | 기울기 오류 | `test_grad_checkpointing.py` |
| `.gitignore` 의 `data/` 가 `src/live3r/data/` 까지 무시 | 패키지가 커밋에서 빠짐 | 깨끗한 클론 테스트 |

### 설계상 조용히 틀리기 쉬운 곳 (형상)

1. **격자 불일치** — 기하 인코더 patch14 vs Qwen3.5 patch16+merge2. 리샘플 없으면 엉뚱한
   위치에 더해지고 에러는 안 난다. → `GeometryProjector` 가 src/dst 격자를 모두 요구한다.
2. **시간축 불일치** — Qwen3.5 `temporal_patch=2`, 즉 원본 2프레임 = 비전 토큰 1블록.
   기하는 프레임마다 1출력 → 정확히 2배 어긋난다. → `_pool_frames_to_patches`.
3. **스트리밍 M-RoPE** — Qwen3.5 는 `rope_deltas` 를 첫 프리필에 고정한다. 그대로 두면
   두 번째 비전 블록부터 3D 위치가 텍스트처럼 붙어 공간 정보가 뭉개진다.
   → `LiveSession` 이 M-RoPE 커서를 직접 관리한다.
4. **비디오 프롬프트 규약** — Qwen3.5 는 프레임(temporal patch)마다 **별도 vision 세그먼트**를
   기대한다(`<0.2 seconds><|vision_start|>…<|vision_end|><1.2 seconds>…`). 한 덩어리로 이어붙이면
   `get_rope_index` 가 grid 를 프레임 수만큼 쪼개 소비하는 것과 어긋난다.
   학습 콜레이터와 `LiveSession` 이 같은 형식을 쓴다.

## CUT3R 연결 — GPU 머신에서 할 일

어댑터는 완성됐고 실제 CUT3R 코드로 구조 검증까지 끝났다
(`scripts/verify_cut3r_adapter.py`, 체크포인트 없이 소형 랜덤 가중치로).
GPU 머신에서 남은 건 **실가중치 실측** 뿐이다:

```bash
git clone https://github.com/CUT3R/CUT3R
cd CUT3R/src/croco/models/curope && python setup.py build_ext --inplace && cd -
gdown --fuzzy 'https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/'
PYTHONPATH=src python scripts/verify_geometry_adapter.py --name cut3r \
    --checkpoint cut3r_512_dpt_4_64.pth --tap-layers 6 9 12 --repo-path ./CUT3R
```

> ⚠️ **curope 컴파일은 선택이 아니다.** CUT3R 은 카메라/포즈 토큰의 2D 위치를 `-1` 로 준다.
> CUDA 커널은 `freq = pos * inv_freq` 로 직접 계산해 음수도 정상이지만,
> 순수 PyTorch 폴백은 cos/sin 테이블을 `F.embedding` 으로 **룩업**해서 음수에서 IndexError 가 난다.
> (`src/live3r/geometry/rope_patch.py` 가 폴백을 고쳐 CPU/MPS 에서도 돌게 해뒀지만, 속도는 컴파일판이 낫다.)

## 위험 — "특화 역설"

OVO-S-Bench 저자 보고: 스트리밍/공간 특화 변형 **15개 중 13개가 자기 베이스보다 낮았다**
(L4 allocentric 평균 −6.1). 공간 SFT 가 일반 능력을 갉아먹는다.
방어선: LLM 은 LoRA 만 · 백본 동결 · 프로젝터 zero-init · 일반 데이터 30% 리플레이 ·
**VideoMME 회귀 게이트**.

### 채택 게이트 (사전 등록)
변형 채택은 셋 다 만족해야 한다.
1. VSI-Bench 평균 ≥ 베이스라인 +1.0
2. VideoMME 회귀 ≤ 1.0
3. `drift < 1.2` 이고 `frame_ingest_ms p95` 가 예산 내

## 문서
- `docs/DESIGN.md` — 아키텍처 결정
- `docs/RESEARCH_NOTES.md` — 2026-09 시점 기술 조사 (모델·벤치·데이터 비교표 + 출처)
- `docs/DATA.md` — 무엇을 받아야 하나
- `docs/BENCHMARKS.md` — 벤치·지연 지표·채택 게이트
- `docs/RUNBOOK.md` — GPU 머신 단계별 실행
- `docs/NEXT_STEPS_SERVER.md` — **지금 서버가 할 일** (2026-09-23)
- `docs/REVIEW_20260923.md` — 방향·구현 검증 (증명된 것 / 미검증 / 방향 문제 7건)

## 라이선스
Apache-2.0. 참조 구현(SpatialStack, VLM-3R, CUT3R, Anchor3R, LingBot-Map, Qwen3.5)도 모두 호환.
