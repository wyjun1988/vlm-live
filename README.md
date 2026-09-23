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
- [ ] CUT3R 어댑터의 프레임 추론 연결 — **GPU 머신에서 30분 작업** (§막힌 곳)
- [ ] 데이터 로더 / 학습 스크립트
- [ ] 평가 (lmms-eval 연동)

## 빠른 시작 (맥, 다운로드 0)

```bash
bash scripts/setup_env.sh          # conda env 'live3r'
conda activate live3r
PYTHONPATH=src python scripts/smoke_test.py
PYTHONPATH=src python scripts/bench_latency.py --tiny --frames 128 --offline
python -m pytest tests -q
```

## GPU 머신에서

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

### 설계상 조용히 틀리기 쉬운 3곳 (전부 테스트로 막아둠)

1. **격자 불일치** — 기하 인코더 patch14 vs Qwen3.5 patch16+merge2. 리샘플 없으면 엉뚱한
   위치에 더해지고 에러는 안 난다. → `GeometryProjector` 가 src/dst 격자를 모두 요구한다.
2. **시간축 불일치** — Qwen3.5 `temporal_patch=2`, 즉 원본 2프레임 = 비전 토큰 1블록.
   기하는 프레임마다 1출력 → 정확히 2배 어긋난다. → `_pool_frames_to_patches`.
3. **스트리밍 M-RoPE** — Qwen3.5 는 `rope_deltas` 를 첫 프리필에 고정한다. 그대로 두면
   두 번째 비전 블록부터 3D 위치가 텍스트처럼 붙어 공간 정보가 뭉개진다.
   → `LiveSession` 이 M-RoPE 커서를 직접 관리한다.

## 막힌 곳 (GPU 머신에서 풀어야 함)

`src/live3r/geometry/cut3r.py::CUT3RStream.ingest` 가 `NotImplementedError` 다.
CUT3R 저장소의 프레임 단위 추론 진입점(리비전마다 이름이 다름)을 연결해야 한다.
탭 토큰 추출은 forward hook 으로 이미 걸려 있으므로, 추론 호출 한 줄과
`GeomOutput` 조립만 남았다. `scripts/verify_geometry_adapter.py` 가 검증한다.
그 전까지는 `geometry.name=dummy` 로 전체 파이프라인이 돈다.

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

## 라이선스
Apache-2.0. 참조 구현(SpatialStack, VLM-3R, CUT3R, Anchor3R, LingBot-Map, Qwen3.5)도 모두 호환.
