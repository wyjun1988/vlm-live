# 리서치 노트 — 2026-09-23 조사

이 프로젝트의 기술 선택 근거. 조사 시점 2026-09-23. 새 논문이 나오면 이 파일을 갱신한다.

## 1. 베이스 VLM — Qwen3.5 소형 시리즈

2026-03-02 공개. `0.8B / 2B / 4B / 9B`, Apache-2.0, **네이티브 멀티모달**(비전 인코더 내장),
262K 컨텍스트(YaRN 로 1.01M 확장).

Qwen3.5-4B 스펙 (HF 모델카드):
- hidden 2560, 32 layers, vocab 248,320
- **하이브리드 어텐션**: Gated Attention(Q16/KV4, head_dim 256) + **Gated DeltaNet**(V32/QK16, head_dim 128)
- sparse MoE
- Thinking(기본) / Instruct(non-thinking) 모드

> **이게 왜 중요한가**: Gated DeltaNet 계열 레이어는 **선형 어텐션 = 상태 크기 고정**이다.
> 프레임이 계속 들어오는 라이브 시나리오에서 KV 캐시가 선형으로 커지지 않는 레이어가 섞여 있다는 뜻.
> 즉 베이스 모델 자체가 스트리밍에 유리하다. 우리 "라이브" 테제와 정렬된다.

타깃 스케일 다운 경로: 4B → 2B → 0.8B. 같은 토크나이저/학습 레시피를 공유하므로
동일 코드로 3개 스케일을 돌릴 수 있다 (hidden/layer 수만 config 에서 갈린다).

## 2. 3D 기하 인코더 — CUT3R 계보

### 계보 정리
| 모델 | 시점 | 방식 | 스트리밍 | 가중치 | 비고 |
|---|---|---|---|---|---|
| DUSt3R / MASt3R | 2024 | pairwise | ✗ | ○ | 전역 정렬 최적화 필요 |
| **CUT3R** | 2025 | RNN(고정크기 implicit memory) | **○** | ○ | VLM-3R 이 쓰는 것. 793M |
| VGGT | 2025 | 전 프레임 양방향 어텐션 | ✗ | ○ | 품질 최고, **O(N²)·오프라인** |
| StreamVGGT / STream3R / TTT3R | 2025 | causal VGGT | ○ | ○ | KV 캐시가 선형 증가 |
| **Mem3R** | 2026-04 | 이중 메모리 + Test-Time Training | ○ | ? | 793M→**644M** 경량화, pose 를 MLP 암묵메모리로 분리 |
| **LingBot-Map** | 2026-04 | Geometric Context Transformer | ○ | ○ (Apache-2.0) | ECCV2026 oral/best paper 후보. **20.29 FPS @518×378**, 10K+ 프레임, paged-KV. 학습코드 ✗ |
| Spark3R | 2026-05 | asymmetric token reduction | **✗(오프라인)** | ? | 속도는 빠르나 배치 전체 필요 |
| **Anchor3R** | 2026-06 | transient anchor + 상대포즈 그래프 | ○ | ○ (Apache-2.0) | 48프레임 학습→10K+ 일반화, **GPU 8.2GB 상수**, 7Scenes CD 1위. 학습코드 ✗ |

### 선택
- **기본값 = CUT3R.** 이유: (a) VLM-3R 이 검증한 경로라 재현 기준선이 된다, (b) latent token
  (feature token + camera view token)을 꺼내 쓰는 인터페이스가 이미 알려져 있다, (c) 가중치 공개.
- **업그레이드 후보 = LingBot-Map / Anchor3R.** 둘 다 Apache-2.0 + 가중치 공개 + 상수 메모리.
  다만 **학습 코드 미공개**라 "동결 인코더 + 프로젝터만 학습" 형태로만 쓸 수 있다.
  우리 레시피는 애초에 인코더를 동결하므로 문제되지 않는다.
- **VGGT 는 의도적으로 배제.** SpatialStack 이 쓰지만 전 프레임 양방향이라 쿼리마다 전체 재계산 →
  라이브 불가. (단 **오프라인 상한선(oracle) 비교군**으로는 유지한다.)

→ 그래서 코드는 `GeometryStream` 인터페이스로 추상화하고 어댑터를 갈아끼운다.

### CUT3R 내부 (저장소 코드 직접 확인, 2026-09-23)
```
enc_embed_dim 1024 / enc_depth 24   dec_embed_dim 768 / dec_depth 12   patch 16
state_size 324 (또는 256), local_mem_size 256
_decoder() 가 돌려주는 dec = 길이 dec_depth+1 튜플
    dec[0]      [B, N, 1024]   투영 전 인코더 출력, 포즈 토큰 없음
    dec[1..12]  [B, 1+N, 768]  **index 0 이 카메라/포즈 토큰**
CUT3R 자신의 헤드가 dec[0], dec[6], dec[9], dec[12] 사용 → **우리 탭도 (6,9,12)**
재귀 상태 = (state_feat, state_pos, init_state_feat, mem, init_mem) — 전부 고정 크기
```

⚠️ **curope 컴파일 필수.** 포즈 토큰 위치가 `-1` 인데 순수 PyTorch RoPE2D 폴백은
cos/sin 테이블 `F.embedding` 룩업이라 음수에서 IndexError. CUDA 커널은 `freq = pos*inv_freq`
직접 계산이라 무사하다. 어디에도 안 적혀 있는 함정 — `src/live3r/geometry/rope_patch.py` 로 우회.
추가로 croco 는 `models.pos_embed` 와 `croco.models.pos_embed` 두 경로로 동시 로드되므로
패치는 **둘 다** 잡아야 한다.

## 3. 융합(fusion) 방식 — VLM-3R vs SpatialStack

### VLM-3R (CVPR 2026, VITA-Group)
- 베이스: LLaVA-NeXT-Video-7B-Qwen2
- CUT3R 의 feature token + camera view token 을 concat → `Z_3D`
- **Spatial-Visual-View Fusion**: VLM 의 비전 토큰이 `Z_3D` 에 cross-attention → residual → MLP
- 데이터: `Journey9ni/VLM-3R-DATA`, 200K+ QA (ScanNet/ScanNet++/ARKitScenes) + 4,225 route planning
- 한계: **LLM 입력 직전 1회 융합**(late fusion). LLM 내부 추론 단계에서 기하 정보를 재참조하지 못함.

### SpatialStack (CVPR 2026, `jzh15/SpatialStack`, Apache-2.0) — 현재 오픈소스 SOTA
- 베이스: **Qwen3.5-4B** (SpatialStack-5B 는 이걸 쓴 판본), 기하 인코더 = **VGGT-1B**
- **계층적 주입**: 기하 레이어 `[11,17,23]` → LLM 레이어 `[0,1,2]` 에 `deepstack_language_add`
  (경량 merger 로 압축 후 residual adapter 로 더함)
- VSI-Bench 오픈소스 1위 **67.5**, CV-Bench 85.5 (3D 92.2)
- 평가: `lmms-eval` (vsibench, cvbench, blink_spatial, sparbench, videomme, mmsibench)
- 학습: 8노드 × 8×H200, py3.12 / torch 2.10+cu129 / flash_attn 2.8.3

### 우리 선택
**SpatialStack 의 계층적 주입 + VLM-3R 의 스트리밍 인코더.**
- 주입은 deepstack(멀티 깊이 residual add)이 late fusion 보다 우월함이 이미 보여졌다 → 그쪽을 채택.
- 인코더는 VGGT(오프라인) 대신 CUT3R 계보(스트리밍) → **라이브 가능**.
- cross-attention(VLM-3R 식)은 **ablation 축**으로 남겨둔다.

## 4. 벤치마크

### 정확도
| 벤치 | 내용 | 규모 | 비고 |
|---|---|---|---|
| **VSI-Bench** | 비디오 기반 공간지능 8태스크 | 5K+ QA | ScanNet/ScanNet++/ARKitScenes. `nyu-visionx/VSI-Bench` |
| **MMSI-Bench** | 멀티이미지 공간지능 (ICLR 2026) | — | `InternRobotics/MMSI-Bench`, EASI 리더보드 |
| MMSI-Video-Bench | 비디오판 홀리스틱 공간지능 | — | 2512.10863 |
| VSTI-Bench | VLM-3R 자체 시공간 벤치 | 138,600 test QA | 시간축 포함 |
| CV-Bench / BLINK-spatial / SPAR-Bench | 보조 | — | lmms-eval 지원 |
| VideoMME | **일반 능력 회귀 감시용** | — | 필수 (아래 §5) |

### 라이브/지연 — 사용자 질문 "그걸 측정하는 벤치가 있는지"에 대한 답: **있다**
| 벤치 | 내용 |
|---|---|
| **OVO-S-Bench** (2606.03890, `InternLM/OVO-S-Bench`) | **스트리밍 × 공간지능**. 정확히 우리 타깃. 1,680문항/1,722쿼리, 348영상, 9개 소스(Ego4D, ARKitScenes, RoomTour3D, Sekai, CODa 등). 쿼리 시점 **이전 프레임만** 볼 수 있음(prefix-only). L1 즉각지각 / L2 시공간 기억 / L3 생성적 추론 / L4 **allocentric 매핑**. 평균 prefix 8.8분 |
| OVO-Bench | 온라인 비디오 이해 1,640문항 12태스크 |
| StreamingBench (ICASSP 2026) | 스트리밍 비디오 이해 |
| RTV-Bench | 실시간. "모델 크기·프레임레이트를 올려도 성능이 안 오른다" 보고 |

OVO-S-Bench 현재 성적: Gemini-3.1-Pro 59.2 / Qwen3-VL-235B-A22B 53.6(오픈소스 1위) /
텍스트온리 37.1 / 랜덤 31.3 / **인간 92.2**. L4(allocentric)가 38개 시스템 중 32개에서 최저.

> ⚠️ **"특화 역설"(Specialization Paradox)**: OVO-S-Bench 저자들이 보고하길, 스트리밍 특화·공간
> 파인튜닝된 변형 15개 중 **13개가 자기 베이스 백본보다 낮았다**(L4 평균 −6.1). 공간 SFT 가
> 일반 능력과 전역 매핑 능력을 갉아먹는다는 뜻. → 우리 설계에 직접 반영(§5).

### 지연 측정
공개 벤치들은 정확도 위주고 **지연 자체를 표준화해 재는 하니스는 빈약**하다. 그래서 자체 계측을 만든다:
- `TTFT` (쿼리 도착 → 첫 토큰), `TPOT`, **`frame_ingest_ms`**(프레임당 기하+비전 인코딩),
  `peak_mem`, `drift`(스트림 길이에 따른 프레임당 비용 증가율)
- Gemini Live 체감 기준 ≈ **TTFT < 500ms, 30fps 입력에서 ingest < 33ms/frame** 을 목표선으로 둔다.

## 5. 설계에 직접 반영할 위험

1. **특화 역설** → LLM 은 LoRA 만, 백본 동결, 일반 데이터 리플레이 혼합, VideoMME 회귀 게이트.
   "VSI 올랐는데 VideoMME 떨어짐"이면 채택 기각.
2. **0.8B 에서 인코더가 본체보다 크다.** CUT3R 793M / VGGT 1B 는 0.8B LLM 보다 무겁다.
   → 기하 토큰 예산과 인코더 자체를 스케일에 맞춰 줄이는 축(증류/토큰선택)이 필요.
   참고: "Good Token Hunting" (2605.23892) — visual geometry transformer 의 토큰 선택.
3. **L4(allocentric) 가 전 모델 공통 병목.** 스트리밍 인코더가 유지하는 전역 상태를
   그냥 흘려보내지 말고 LLM 에 명시적으로 노출하는 설계가 차별점이 될 수 있다.

## 6. 학습 데이터 후보

| 데이터셋 | 규모 | 비고 |
|---|---|---|
| `nyu-visionx/VSI-590K` | 590K QA | Cambrian-S 용. 10개 소스, 실사+시뮬+웹 의사라벨. VSI-Bench +30%p 기여 |
| `Journey9ni/VLM-3R-DATA` | 200K+ QA | VLM-3R. ScanNet/ScanNet++/ARKitScenes |
| `Journey9ni/SpatialStackData` | 51,779 샘플 (0.38GB, 어노테이션만) | ScanNet 영상 별도 필요 |
| ScanNet / ScanNet++ / ARKitScenes | 원본 비디오·3D | 위 어노테이션들의 실체. **용량 큼** |
| 일반 비디오 SFT (리플레이용) | — | LLaVA-Video-178K 등. 특화 역설 방어용 |

→ 상세 다운로드 계획은 `docs/DATA.md`.

## 출처
- Qwen3.5 small series: https://artificialanalysis.ai/articles/qwen3-5-small-models , https://huggingface.co/Qwen/Qwen3.5-4B
- VLM-3R: https://arxiv.org/abs/2505.20279 , https://github.com/VITA-Group/VLM-3R
- SpatialStack: https://spatial-stack.github.io/ , https://github.com/jzh15/SpatialStack
- Cambrian-S / VSI-590K: https://arxiv.org/abs/2511.04670 , https://huggingface.co/datasets/nyu-visionx/VSI-590K
- OVO-S-Bench: https://arxiv.org/html/2606.03890 , https://github.com/InternLM/OVO-S-Bench
- MMSI-Bench: https://arxiv.org/abs/2505.23764 , https://github.com/InternRobotics/MMSI-Bench
- Anchor3R: https://arxiv.org/abs/2606.05035 , https://github.com/polar-explorer/Anchor3R
- LingBot-Map: https://arxiv.org/html/2604.14141 , https://github.com/robbyant/lingbot-map
- Mem3R: https://arxiv.org/html/2604.07279
- Spark3R: https://arxiv.org/pdf/2605.06270
- Good Token Hunting: https://arxiv.org/pdf/2605.23892
