# 학습 데이터 계획

**원칙**: 전체를 다 받지 않는다. 어노테이션(작음) 먼저 → 형식 확인 → 필요한 원본 비디오만.
맥에서는 샘플만 받아 파이프라인을 검증하고, 대용량은 GPU 머신에서 받는다.

## 0. 한눈에 — 무엇을 받아야 하나

| 우선 | 이름 | 크기 | 무엇 | 왜 |
|---|---|---|---|---|
| **P0** | `nyu-visionx/VSI-590K` | 어노 수 GB | 590K 공간 QA | 메인 SFT. VSI-Bench +30%p 기여가 보고됨 |
| **P0** | ScanNet (v2) | ~1.3TB 전체 / **RGB만 ~250GB** | 실내 RGB-D 스캔 | 위 어노의 원본 영상. `.sens` 에서 RGB 추출 |
| **P1** | `Journey9ni/VLM-3R-DATA` | 수 GB | 200K QA + 4,225 route planning | VLM-3R 재현 기준선 |
| **P1** | ARKitScenes | ~600GB (부분 선택 가능) | 실내 스캔 | VSI-Bench 소스 중 하나 |
| **P1** | ScanNet++ | ~1TB (신청 필요) | 고품질 실내 | 동일 |
| **P2** | `Journey9ni/SpatialStackData` | **0.38GB** | 51,779 거리 QA (ScanNet 영상 참조) | SpatialStack 재현. 가장 싸게 시작 가능 |
| **P2** | 일반 비디오 SFT (LLaVA-Video-178K 등) | 수백 GB | 리플레이 | **특화 역설 방어** — 없으면 일반 능력이 무너진다 |
| **평가** | `nyu-visionx/VSI-Bench` | 작음 | 5K+ QA | 주 벤치 |
| **평가** | `InternRobotics/MMSI-Bench` | 작음 | 멀티이미지 공간 | 주 벤치 |
| **평가** | `JoeLeelyf/OVO-S-Bench` | **어노만** | 1,680문항 | **라이브 공간지능**. 영상은 9개 원본에서 따로 |

## 1. 지금 당장 (맥, 샘플 확인용 — 총 1GB 미만)

```bash
pip install -U "huggingface_hub[cli]"

# 어노테이션만. 비디오 없이 형식/스키마 확인
hf download Journey9ni/SpatialStackData --repo-type dataset --local-dir data/raw/spatialstack
hf download nyu-visionx/VSI-Bench       --repo-type dataset --local-dir data/raw/vsibench
hf download JoeLeelyf/OVO-S-Bench       --repo-type dataset --local-dir data/raw/ovo-s

# VSI-590K 는 크다 → 파일 하나만 맛보기
hf download nyu-visionx/VSI-590K --repo-type dataset --include "*.json" --local-dir data/raw/vsi590k
```

확인: `PYTHONPATH=src python scripts/inspect_dataset.py data/raw/spatialstack`

## 2. GPU 머신 (실학습)

### 2-1. 어노테이션
```bash
hf download nyu-visionx/VSI-590K      --repo-type dataset --local-dir data/raw/vsi590k
hf download Journey9ni/VLM-3R-DATA    --repo-type dataset --local-dir data/raw/vlm3r
hf download Journey9ni/SpatialStackData --repo-type dataset --local-dir data/raw/spatialstack
```

### 2-2. 원본 영상 — ScanNet 이 1순위
ScanNet 은 **신청서 제출 후 다운로드 스크립트**를 받는 구조다 (http://www.scan-net.org).
전체는 1.3TB 지만 우리가 필요한 건 RGB 프레임뿐이다:

```bash
# 신청 후 받은 download-scannet.py 사용. .sens 만 받아 RGB 추출
python download-scannet.py -o data/raw/scannet --type .sens
# SensReader 로 RGB 추출 후 mp4 인코딩 (어노가 scannet/videos/scene####_##.mp4 를 참조한다)
python scripts/scannet_to_mp4.py --src data/raw/scannet --dst data/raw/scannet/videos --fps 24
```

> **먼저 100씬만 받아라.** SpatialStackData 51K 샘플이 참조하는 씬 수가 그 정도면
> 파이프라인 전체를 검증할 수 있다. 1.3TB 를 받고 나서 형식이 안 맞는 걸 발견하면 최악이다.

### 2-3. 평가용
```bash
hf download nyu-visionx/VSI-Bench       --repo-type dataset --local-dir data/eval/vsibench
hf download InternRobotics/MMSI-Bench   --repo-type dataset --local-dir data/eval/mmsibench
hf download JoeLeelyf/OVO-S-Bench       --repo-type dataset --local-dir data/eval/ovo-s
```
OVO-S-Bench 는 **어노테이션만 배포**한다. 영상은 Ego4D / ARKitScenes / RoomTour3D / Sekai /
OmniWorld / CODa / Honda HDD 등에서 각각 받아야 하고 라이선스가 소스마다 다르다
(논문 Appendix E.2). → 초기에는 **ARKitScenes·VSI-Bench 유래 부분집합만**으로 시작하는 게 현실적이다.

## 3. 체크포인트

```bash
# 기하 인코더 (기본값 CUT3R)
mkdir -p checkpoints
gdown --fuzzy 'https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/'  # cut3r_512_dpt_4_64.pth

# 업그레이드 후보 (둘 다 Apache-2.0, 상수 메모리)
hf download polar-explorer/Anchor3R --local-dir checkpoints/anchor3r

# 베이스 VLM
hf download Qwen/Qwen3.5-4B   --local-dir checkpoints/qwen3.5-4b
hf download Qwen/Qwen3.5-2B   --local-dir checkpoints/qwen3.5-2b
hf download Qwen/Qwen3.5-0.8B --local-dir checkpoints/qwen3.5-0.8b
```

## 4. 학습 믹스 (제안)

| 단계 | 데이터 | 비율 | 학습 대상 | 목적 |
|---|---|---|---|---|
| S1 정렬 | VSI-590K 부분집합 200K | 100% | 프로젝터만 | 기하 토큰 ↔ LLM 공간 정렬 |
| S2 SFT | VSI-590K + VLM-3R-DATA | 70% | 프로젝터 + LoRA | 공간 추론 |
| S2 SFT | 일반 비디오 SFT | **30%** | 동일 | **특화 역설 방어 — 빼지 마라** |
| S3 스트리밍 | prefix-only 변환 샘플 | — | 동일 | OVO-S-Bench 류 대응 |

S1 을 따로 두는 이유: 프로젝터가 zero-init 이라 처음엔 기여가 0이다. LoRA 를 동시에 풀면
LLM 이 기하 없이 푸는 지름길을 먼저 배운다. 정렬을 먼저 끝내고 LoRA 를 연다.

## 5. 형식

우리 로더가 먹는 정규화 형식 (`src/live3r/data/datasets.py`):
```json
{
  "id": "...", "video": "scannet/videos/scene0191_00.mp4",
  "conversations": [{"from": "human", "value": "<video>\n질문"}, {"from": "gpt", "value": "2.3"}],
  "frames": [12, 24, 36],          // 선택: 미리 정한 샘플 인덱스
  "data_source": "spatialstack"
}
```
VSI-590K / VLM-3R-DATA / SpatialStackData 는 전부 LLaVA 스타일 `conversations` 라 어댑터가 얇다.
