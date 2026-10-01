# 기존 시스템(VGGT → CUT3R + 4B, 최고 70.2)에 물어볼 것

> **Status 2026-09-30 (English):** the reference team's reply was an answer-generation runbook (how to run their model
> on a benchmark and hand over prediction files), not answers to the questions below. What it did give:
> - VSI-Bench overall **70.83** (lmms-eval `vsibench`, thinking off; results
>   `logs/vsibench_qwen35_nothink/cot__qwen3.5-4b-allcot-sqa-gt-cut3r-lora2/20260728_140541_results.json`).
>   77.91 is SQA3D's `is` sub-score, not VSI.
> - Model: Qwen/Qwen3.5-4B + LoRA `/group-volume/jcho/VG-LLM/ckpt/cot/qwen3.5-4b-allcot-sqa-gt-cut3r-lora2`
>   (VG-LLM codebase), encoder CUT3R `/group-volume/jcho/CUT3R/src/cut3r_512_dpt_4_64.pth` (the same checkpoint name
>   as ours), geometry patch 16; evaluation `scripts/evaluation/eval_lora_multi.sh`.
> - Everything is on the same group volume as our server, so most questions were answered by reading their files
>   (two read-only passes, 2026-09-30; findings in `docs/REFERENCE_70_ANALYSIS.md`). **We do not take their code** -
>   the local copies of their scripts were deleted; we implement the design ourselves. Their training data may be used:
>   `configs/data/vgllm.yaml`, read by `scripts/inspect_reference_data.py` (data only).
> - **2026-10-01: the remaining questions (settings, input samples, other runs' results - no code) are in
>   [`docs/REFERENCE_70_REQUEST.md`](REFERENCE_70_REQUEST.md)**, the request to forward to the reference team.

코드를 가져올 수 없으므로, 아래 항목을 그쪽 담당자에게 물어 채워주면 Live3R 이 **같은 데이터·같은 운영점**에서
비교 가능해진다. 답이 없으면 추측으로 맞춰야 하고, 그러면 격차의 원인을 못 가린다.

**2026-09-28 갱신.** 우리 쪽 현재 위치: 제로샷(학습 없음) 57.8 / 서버 베이스라인(Sensenova 1에폭) 평가 진행 중.
목표는 70.2 를 라이브 제약(질문 후 1초 답변, 스트리밍 기하) 안에서 유지하는 것. 그래서 지금 가장 급한 것은
**P0-B 평가 프로토콜** — 같은 4B 라도 프레임을 이미지로 받는지 비디오로 받는지에 따라 24.7 vs 48.6 이 나온다.
프로토콜을 맞추기 전의 점수 비교는 의미가 없다.

우선순위 순. **P0 없이는 비교도 학습 확정도 못 한다.**

## P0-A — 체크포인트 (가장 확실한 길)

0. **70.2 모델의 체크포인트(또는 추론 코드+가중치)를 받을 수 있는가?** 우리 평가 하네스로 돌리면 같은 잣대의
   숫자가 바로 나오고, 유형별로 어디서 이기는지·라이브 제약에서 몇 점이 깎이는지가 측정된다. 이게 가능하면
   아래 P1 의 절반은 필요 없다.

## P0 — 데이터 (이게 제일 급하다)

1. 학습데이터의 **JSON 스키마 실제 예시 3개** (민감정보 제외).
   특히: 영상 경로 필드명 / 대화 포맷 / 프레임 인덱스를 미리 박아두는지 / 태스크 라벨
2. 총 샘플 수와 **구성비** (VSI-590K? 자체 생성? 일반 비디오 SFT 를 섞는가? 몇 %?)
3. 영상 소스와 로컬 경로 규약 (ScanNet/ScanNet++/ARKitScenes, mp4 인가 프레임 폴더인가)
4. **일반 데이터 리플레이를 섞는가?** 섞는다면 무엇을 몇 %로.
   (섞지 않고도 VideoMME 가 안 떨어졌다면 그 자체가 중요한 정보다)

## P0-B — 평가 프로토콜 (숫자 비교의 전제; 70.2 가 어떤 조건의 숫자인지)

5. VSI-Bench 평가에서 **프레임 몇 장**, 어떻게 고르는가(균등/fps/키프레임)? 영상 전체 길이를 알고 고르는가?
6. LLM 에 프레임을 **이미지 여러 장으로 넣는가, 비디오(타임스탬프 포함) 로 넣는가**? 해상도(픽셀 예산)와
   1문항당 **비전 토큰 수**.
7. **thinking on/off**, 시스템 프롬프트, 형식 지시 문장이 있는가. 채점 코드(lmms-eval 버전인가 자체인가).
8. **VSI 8개 유형별 점수** (전체 70.2 만으로는 어디서 이기는지 모른다) + 문항 수(5,130 전체인가 부분집합인가).

## P0-C — 인코더 운영점

9. CUT3R 에 **몇 프레임을 넣는가** (전부 / 키프레임 / N fps)? 상태를 리셋하는가? 리셋한다면 주기·방법.
   (우리 측정: 키프레임 32장만 넣으면 방 크기 1.03배, 1fps 면 1.23배, 3fps 면 1.39배 — 스케일 드리프트.
   그쪽은 어떻게 피했는가?)
10. CUT3R 입력 해상도, 어느 층의 토큰을 쓰는가, 포즈 토큰을 쓰는가.
11. **VGGT → CUT3R 전환의 이유와 점수 변화** (VGGT 였을 때 73.3 → CUT3R 70.2 로 기억한다. 맞는가?)

## P1 — 융합·학습 레시피

12. 기하 토큰을 LLM 에 **어떻게 넣는가**?
   (a) 입력 직전 1회 concat/cross-attn  (b) 디코더 여러 깊이 residual(deepstack)  (c) 기타
   (b) 라면 **CUT3R 어느 레이어 → LLM 어느 레이어**인지. 프로젝터 구조(MLP? merger? zero-init?), 토큰 수.
13. 학습 범위: **full FT / LoRA(rank·타깃) / 프로젝터만**. 단계 구성(정렬 단계를 따로 두는가), lr, 배치,
    스텝/에폭, 총 샘플 수. (우리 S1 프로젝터-만 단계는 Sensenova 에서 답 형식만 배웠다 — 그쪽은?)
14. 학습에 쓴 GPU·시간.

## P1 — 일반 능력·지연

15. VideoMME 등 **일반 벤치 점수** (우리 회귀 게이트의 기준선이 된다). 일반 데이터 리플레이 없이 유지됐는가.
16. 질문 1개에 걸리는 시간(프레임 인코딩 포함/제외), 프레임당 CUT3R 시간, GPU 메모리.

## P2 — 참고

17. 시도했다가 **안 됐던 것** — 이게 제일 값지다. 같은 벽에 두 번 부딪힐 이유가 없다.
    (우리가 이미 부딪힌 벽: 프로젝터 정렬이 형식만 배움 / 소규모 SFT 의 수치 답 붕괴("1.1") / 객체 인지맵
    텍스트·지도 이미지의 제로샷 무효 / 재귀 상태의 스케일 드리프트 — 같은 것을 봤는지 궁금하다.)
18. 학습데이터에 Sensenova 외에 무엇을 얼마나 섞었는가 (P0 데이터 항목의 답이 곧 이것).
