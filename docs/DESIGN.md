# Live3R — 설계

**한 줄**: 스트리밍 3D 기하 인코더의 잠재 토큰을 Qwen3.5 소형 VLM 의 디코더 여러 깊이에
residual 로 주입해, **프레임당 상수 비용**으로 공간을 이해하고 **즉시 답하는** VLM.

## 0. 왜 새로 만드나 (포지셔닝)

| | 기하 인코더 | 융합 | 라이브 가능? | VSI-Bench |
|---|---|---|---|---|
| VLM-3R | CUT3R (스트리밍) | late cross-attn 1회 | 구조적으로 가능하나 미구현 | — |
| SpatialStack | VGGT (**오프라인·전프레임 양방향**) | deepstack 다깊이 residual | **불가** | 67.5 (오픈 1위) |
| **Live3R (우리)** | **CUT3R 계보 (스트리밍)** | **deepstack 다깊이 residual** | **가능** | 목표 |

SpatialStack 이 품질 1위지만 VGGT 는 쿼리마다 전 프레임을 다시 본다 → N 프레임에서 O(N²),
매 쿼리 full recompute. 데모는 되지만 라이브는 안 된다.
VLM-3R 은 스트리밍 인코더를 쓰지만 융합이 약하고 베이스가 구형(LLaVA-NeXT-Video-7B).

**Live3R = SpatialStack 의 융합 레시피 + VLM-3R 의 스트리밍 인코더 + Qwen3.5 소형화 + 라이브 런타임.**
셋 다 Apache-2.0 계열이라 라이선스도 깨끗하다.

## 1. 아키텍처

```
              ┌──────────── 프레임 스트림 (t=1,2,3,...) ────────────┐
              │                                                      │
        ┌─────▼─────┐                                    ┌───────────▼──────────┐
        │ Qwen3.5   │  비전 토큰 V_t                     │  GeometryStream      │
        │ 비전 타워 │ ──────────────┐                    │  (CUT3R / Anchor3R / │
        └───────────┘               │                    │   LingBot-Map ...)   │
                                    │                    │  state_t = f(state_{t-1}, I_t)
                                    │                    └───────────┬──────────┘
                                    │                     G_t = {feat tokens, pose token}
                                    │                                │
                                    │                   ┌────────────▼────────────┐
                                    │                   │ GeometryProjector × K   │
                                    │                   │ (토큰 머지 + MLP → d_llm)│
                                    │                   └────────────┬────────────┘
                                    │                                │
                    ┌───────────────▼────────────────────────────────▼──────────┐
                    │  Qwen3.5 디코더                                            │
                    │   layer 0  : h += P_0(G[l_0])   ← deepstack residual add  │
                    │   layer 1  : h += P_1(G[l_1])                             │
                    │   layer 2  : h += P_2(G[l_2])                             │
                    │   layer 3.. : (LoRA 만)                                    │
                    └────────────────────────────┬──────────────────────────────┘
                                                 │ KV 캐시 = 증분 유지
                                            답변 토큰
```

### 1.1 GeometryStream 인터페이스 (핵심 추상화)

```python
class GeometryStream(Protocol):
    def reset(self) -> None: ...
    def ingest(self, frame: Tensor) -> GeomOutput: ...   # O(1) w.r.t. 스트림 길이
    @property
    def state_bytes(self) -> int: ...                    # 상수여야 한다 (테스트로 강제)
```

`GeomOutput` = 선택된 깊이별 latent token `dict[layer_idx -> Tensor[N_tok, C]]`
 + `pose_token Tensor[1, C]` + (옵션) `pointmap`, `conf`.

**불변식**: `ingest` 의 시간/메모리가 스트림 길이에 대해 상수. 테스트로 강제한다
(`tests/test_streaming_invariants.py`). 이걸 깨는 인코더는 어댑터로 감싸더라도 "오프라인"으로 태그.

구현체:
- `CUT3RStream` — 기본값. VLM-3R 이 쓰는 그 경로.
- `AnchorStream`, `LingBotStream` — 업그레이드 후보 (추론 전용 가중치).
- `VGGTWindowStream` — **오프라인 상한선(oracle)**. 슬라이딩 윈도우로 근사. 라이브 아님, 비교군 전용.
- `DummyStream` — 가중치 없이 형상만 맞춘 난수. **맥에서 CI/스모크 테스트용.**

### 1.2 융합 — DeepStack residual add

SpatialStack 설정을 그대로 기본값으로: 기하 레이어 `[11,17,23]` → LLM 레이어 `[0,1,2]`.
비전 토큰이 놓인 위치에만 더한다 (텍스트 토큰은 건드리지 않음).

```
h[:, vision_pos, :] += Proj_k( merge(G_t[geo_layer_k]) )
```

- `merge` = 공간 2×2 pixel-unshuffle 계열 경량 머저 (토큰 수 축소).
- `Proj_k` = LayerNorm → Linear(C → d_llm) → GELU → Linear, **zero-init 마지막 레이어**
  → 학습 초기에 베이스 VLM 동작을 정확히 보존 (특화 역설 방어의 1차 장치).
- 대안 `xattn` (VLM-3R 식) 은 ablation 플래그로 유지.

### 1.3 pose/전역 상태 토큰

OVO-S-Bench 의 L4(allocentric 매핑)가 전 모델 공통 병목이다. 스트리밍 인코더는 전역 상태를
이미 들고 있으므로, **camera pose token 과 인코더 상태 요약을 별도 토큰 슬롯으로 LLM 에 명시 노출**한다.
(VLM-3R 은 view token 을 쓰지만 concat 후 한 번만 본다.)

### 1.4 학습 대상

| 모듈 | 1단계(정렬) | 2단계(SFT) |
|---|---|---|
| 기하 인코더 | 동결 | 동결 |
| Qwen3.5 비전 타워 | 동결 | 동결 |
| GeometryProjector ×K | **학습** | **학습** |
| LLM | 동결 | **LoRA** (r=32, q/k/v/o + MLP) |

인코더를 영구 동결하는 이유: Anchor3R/LingBot-Map 은 학습 코드 미공개 + 스트리밍 인코더를
파인튜닝하면 기하 품질이 무너지기 쉽다. 동결이 어댑터 교체 자유도도 준다.

## 2. 라이브 런타임

```python
sess = LiveSession(model, fps_budget=30)
for frame in camera:        # 항상 도는 루프
    sess.ingest(frame)      # 기하 상태 갱신 + 비전 토큰 KV 프리필 (증분)
answer = sess.ask("내가 방금 지나친 소파에서 냉장고까지 몇 미터야?")   # 재계산 없음
```

핵심: **프레임 인제스트와 질의 응답의 분리.** 프레임은 들어오는 대로 KV 캐시에 누적하고,
질문은 캐시 뒤에 붙여 디코딩만 한다. 오프라인 VLM 은 질문이 올 때마다 전 프레임을 다시 인코딩한다.

프레임 예산 관리(스트림이 길어질 때):
- 기하 상태는 인코더가 상수로 관리 (CUT3R/Anchor3R 의 설계 특성).
- 비전 토큰 KV 는 선형 증가 → **토큰 예산 정책** 필요:
  `keyframe`(움직임 기반 샘플링) / `merge`(오래된 토큰 풀링) / `evict`(FIFO).
  Qwen3.5 의 Gated DeltaNet 레이어는 애초에 상수 상태라 압박이 어텐션 레이어에만 걸린다.

## 3. 스케일 다운 계획 (4B → 2B → 0.8B)

동일 코드·동일 데이터. 스케일마다 바뀌는 것:
1. `d_llm` (2560 → …), 주입 레이어 인덱스 (레이어 수에 비례해 재배치)
2. **기하 토큰 예산** — 0.8B 에서는 CUT3R 793M 가 LLM 보다 크다. 대응 축:
   - (a) 토큰 선택/머지 비율 상향 (Good Token Hunting 계열)
   - (b) 인코더 자체 증류 — 4B 판본의 프로젝터 출력을 타깃으로 작은 인코더 학습
   - (c) 인코더를 낮은 fps 로만 돌리고 사이 프레임은 상태 보간
3. LoRA rank (32 → 16 → 8)

각 스케일에서 **정확도 × 지연** 파레토를 기록한다. 0.8B 의 존재 이유는 정확도가 아니라
"30fps 인제스트 + TTFT<300ms 를 엣지에서" 이므로 판정 기준이 다르다.

## 4. 평가

### 정확도
`lmms-eval` 태스크로 통일: `vsibench`, `mmsibench`, `cvbench`, `blink_spatial`, `sparbench`
\+ **`videomme` (회귀 게이트)**, + OVO-S-Bench (스트리밍 prefix-only).

### 지연 — 자체 하니스 `src/live3r/eval/latency.py`
- `frame_ingest_ms` (p50/p95), `TTFT`, `TPOT`, `peak_mem_mb`
- **`drift`**: 프레임 1,000 째의 ingest 비용 / 10 째 비용. 1.0 에 가까워야 진짜 스트리밍.
- 목표선: TTFT < 500ms, ingest < 33ms/frame (30fps), drift < 1.2

### 채택 게이트 (사전 등록)
변형 하나를 채택하려면 **세 개 모두** 만족:
1. VSI-Bench 평균 ≥ 베이스라인 +1.0
2. VideoMME **회귀 ≤ 1.0** (특화 역설 방어)
3. `drift < 1.2` 이고 `frame_ingest_ms p95` 가 예산 내

## 5. 저장소 구조

```
src/live3r/
  geometry/   base.py registry.py dummy.py cut3r.py anchor3r.py lingbot.py vggt_window.py
  fusion/     projector.py deepstack.py xattn.py
  model/      live3r.py  lora.py  config.py
  data/       datasets.py collate.py
  train/      train.py
  eval/       latency.py runner.py
  serve/      session.py
configs/      live3r_4b.yaml  live3r_2b.yaml  live3r_08b.yaml
scripts/      setup_env.sh  smoke_test.py  fetch_samples.sh
tests/        (맥에서 GPU 없이 도는 것만)
```

## 6. 작업 순서

- [x] P0 리서치 + 설계
- [ ] P1 스캐폴드 + `DummyStream` 으로 **엔드투엔드 형상 검증** (맥 MPS, 가중치 없이)
- [ ] P2 Qwen3.5-4B 실제 로딩 + deepstack 후킹 + LoRA
- [ ] P3 CUT3R 어댑터 + 실제 기하 토큰
- [ ] P4 데이터 파이프라인 + 학습 스크립트 (원격 GPU 에서 실행)
- [ ] P5 평가 하니스 (정확도 + 지연)
- [ ] P6 2B / 0.8B 스케일 다운
