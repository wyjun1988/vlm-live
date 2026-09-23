# 평가 계획

## 1. 정확도

`lmms-eval` 로 통일한다 (SpatialStack 이 쓰는 것과 같은 태스크명 — 수치 비교가 바로 된다).
어댑터는 `src/live3r/eval/lmms_live3r.py` (`--model live3r`). lmms-eval 0.7.3 에서
아래 태스크가 전부 존재하는 것을 확인했다 (`tests/test_lmms_adapter.py` 가 회귀를 막는다).

```bash
pip install -e .            # entry point 로 live3r 모델이 등록된다
bash scripts/run_eval.sh configs/live3r_4b.yaml outputs/4b_s2/final.pt outputs/eval_4b
```

어댑터 설계: lmms-eval 의 `qwen3_5` 모델을 상속해 **로딩만** 바꾼다. 생성 경로는 손대지 않는다 —
`Live3RModel.enable_auto_geometry()` 가 `base.forward` 를 감싸 기하 인코딩·주입을 자동으로 하므로
상위 코드는 평범한 Qwen3.5 를 돌린다고 믿으면 된다. 업스트림이 바뀌어도 잘 안 깨진다.

| 태스크 | 역할 | 기준선 |
|---|---|---|
| `vsibench` | **주 지표**. 비디오 공간지능 8태스크 | SpatialStack-5B 67.5 (오픈소스 1위) |
| `vsibench_debiased` | 지름길 해법 제거판 — **같이 봐야 한다** | — |
| `mmsi_bench` | 멀티이미지 공간지능 (ICLR'26) | EASI 리더보드 |
| `mmsi_video` | 비디오판 | — |
| `vsisuper`, `revsi` | 장시간·반복 공간 | — |
| `cv_bench` | 2D/3D 일반 공간 | SpatialStack 85.5 (3D 92.2) |
| `blink`, `sparbench` | 보조 | — |
| **`videomme`** | **회귀 게이트** — 일반 비디오 능력 | 베이스 Qwen3.5-4B 자체 |

## 2. 스트리밍 공간지능 — OVO-S-Bench

`InternLM/OVO-S-Bench` (arXiv 2606.03890). 쿼리 시점 **이전 프레임만** 본다(prefix-only).
Live3R 의 존재 이유를 직접 재는 유일한 공개 벤치다.

| 레벨 | 내용 | 비고 |
|---|---|---|
| L1 | 즉각 자기중심 지각 (거리·스케일·깊이순서) | 근거 구간 중앙값 2.0초 |
| L2 | 시공간 문맥 추적 (시야에서 사라진 것) | 36.8초 |
| L3 | 생성적 공간 추론 (심적 회전·경로계획) | 2.0초 |
| **L4** | **allocentric 매핑** (전역 방향·위상·궤적) | 278.7초. **전 모델 공통 병목** |

현재 성적: Gemini-3.1-Pro 59.2 / Qwen3-VL-235B-A22B 53.6 / 텍스트온리 37.1 / 랜덤 31.3 / 인간 92.2.

**L4 가 우리 차별점이 될 자리다.** 스트리밍 인코더가 이미 전역 상태(카메라 포즈·씬 메모리)를
들고 있는데 기존 모델들은 그걸 LLM 에 노출하지 않는다. `GeometryConfig.expose_pose_token`.

> ⚠️ 데이터: OVO-S-Bench 는 **어노테이션만 배포**한다. 영상은 Ego4D / ARKitScenes / RoomTour3D /
> Sekai / OmniWorld / CODa / Honda HDD 등에서 각각 받아야 하고 라이선스가 소스마다 다르다.
> 초기에는 ARKitScenes·VSI-Bench 유래 부분집합만으로 시작하는 게 현실적이다.

## 3. 지연 — 자체 하니스

공개 스트리밍 벤치는 정확도 위주라 지연 표준이 없다. `src/live3r/eval/latency.py`.

```bash
PYTHONPATH=src python scripts/bench_latency.py --config configs/live3r_4b.yaml --device cuda --offline
```

| 지표 | 목표 | 의미 |
|---|---|---|
| `frame_ingest_ms` p95 | **< 33 ms** | 30fps 입력을 놓치지 않는다 |
| `ttft_ms` | **< 500 ms** | Gemini Live 체감선 |
| `drift` | **< 1.2** | 후반 프레임 비용 / 초반 비용. 1.0 근처여야 진짜 스트리밍 |
| `offline_speedup` | 보고용 | 같은 질문을 전 프레임 재인코딩으로 했을 때 대비 |

초소형 모델 + 더미 인코더(맥 MPS, 128프레임) 참고치: drift 1.08, 오프라인 대비 **215×**.
이건 구조적 이점 확인일 뿐 실모델 수치가 아니다.

## 4. 채택 게이트 (사전 등록)

변형 하나를 채택하려면 **셋 다** 만족해야 한다. 하나라도 못 맞추면 기각 또는 보류.

1. `vsibench` 평균 ≥ 베이스라인 **+1.0**
2. `videomme` 회귀 **≤ 1.0** ← 특화 역설 방어
3. `drift < 1.2` 이고 `frame_ingest_ms p95` 가 스케일별 예산 내

**왜 게이트를 미리 박아두나**: OVO-S-Bench 저자들이 공간/스트리밍 특화 변형 15개 중
13개가 자기 베이스 백본보다 낮았다고 보고했다(L4 평균 −6.1). 공간 점수만 보면
"개선됐다"고 착각하기 쉽다. 일반 능력 회귀를 같이 보지 않으면 잘못된 방향으로 몇 주를 태운다.

## 5. 스케일별 기대치

| 판본 | 존재 이유 | 판정 기준 |
|---|---|---|
| 4B | 품질 | vsibench 절대값 |
| 2B | 균형 | 품질/지연 파레토 |
| **0.8B** | **엣지 라이브** | **정확도가 아니라 30fps 인제스트 + TTFT<300ms 달성 여부** |

0.8B 를 4B 와 같은 잣대로 보면 안 된다. 이 판본은 "돌아가느냐"가 지표다.
CUT3R 793M 이 LLM 보다 무거운 구간이라 기하 인코더 증류가 별도 과제로 남는다.
