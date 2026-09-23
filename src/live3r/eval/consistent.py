"""학습과 같은 입력 경로로 평가한다 — 비스트리밍(오프라인) 평가용.

lmms-eval 의 부모(qwen3_5) 경로는 학습(SenseNova-SI) 분포와 **세 군데**가 다르다 (2026-09-23 확인):

  1. 시스템 프롬프트 "You are a helpful assistant." 를 붙인다 — 학습 프롬프트에는 없다
  2. 영상을 **비디오 모드**로 넣는다 (2프레임=1블록, 타임스탬프, 기하가 2장씩 평균) —
     프로젝터는 이미지 모드만 학습했다 → 처음 보는 입력
  3. 영상 해상도를 총 픽셀 예산(최대 768프레임)으로 프레임 수에 따라 바꾼다 —
     학습은 이미지당 고정 범위(VisionSpec)

그래서 VideoMME 회귀 게이트를 부모 경로로 재면 "일반 능력 회귀"가 아니라 "처음 보는 입력에 대한
반응"을 잰다. 이 모듈은 학습과 같은 형식으로 넣는다:

  * 이미지 → 이미지마다 별도 비전 블록 (학습과 같다)
  * 영상   → 균등 키프레임 N장을 **이미지로** (스트리밍 평가의 기본 visual_mode 와도 같다)
  * 시스템 프롬프트 없음, thinking off, 같은 VisionSpec
  * 기하는 이미지/키프레임 순서대로 스트리밍 인코더에 흘린다 (학습 때와 같은 방식)

오프라인 평가라 영상 총 길이를 알고 균등 추출한다 — **라이브 점수가 아니다** (그건 streaming.py).
게이트 비교는 항상 같은 경로에서 가중치만 바꿔서 한다 (weights 없음 = zero-init = 베이스와 동일 출력).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.prompt import PromptBuilder, PromptError
from ..data.vision import geometry_frame, llm_grid, prepare_image, tokens_per_step

_PH = re.compile(r"<image>|<video>")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".mpg", ".mpeg"}


@dataclass
class PreparedInput:
    input_ids: torch.Tensor          # [1, L]
    mm_token_type_ids: torch.Tensor  # [1, L]
    pixel_kwargs: dict               # pixel_values / image_grid_thw (이미지 모드뿐)
    geometry: object | None          # GeometryBundle
    n_images: int
    frame_indices: list[int] | None = None


# ------------------------------------------------------------------- 영상 읽기
def uniform_indices(total: int, n: int) -> list[int]:
    """공식 Qwen 비디오 프로세서와 같은 균등 추출 — `linspace(0, total-1, n).round()`."""
    if total <= 0:
        return []
    n = min(n, total)
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def is_video(v) -> bool:
    return isinstance(v, (str, Path)) and Path(str(v)).suffix.lower() in VIDEO_EXTS


def read_video_frames(path, n: int) -> tuple[np.ndarray, list[int]]:
    """영상에서 균등 N장을 읽는다 → ([N,H,W,3] uint8, 인덱스).

    decord 는 임의 접근이라 긴 영상(VideoMME 최대 1시간)에서도 필요한 프레임만 디코딩한다.
    PyAV 폴백은 처음부터 순차 디코딩하되 필요한 프레임만 보관한다 (느리지만 메모리는 N장).
    """
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(str(path))
        idx = uniform_indices(len(vr), n)
        return vr.get_batch(idx).asnumpy(), idx
    except ImportError:
        pass
    import av  # type: ignore

    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        total = stream.frames or 0
    if total <= 0:  # 컨테이너가 프레임 수를 안 알려주면 한 번 센다
        with av.open(str(path)) as c:
            total = sum(1 for _ in c.decode(video=0))
    idx = uniform_indices(total, n)
    want = set(idx)
    out = {}
    with av.open(str(path)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
            if len(out) == len(want):
                break
    got = [i for i in idx if i in out]
    return np.stack([out[i] for i in got]), got


def load_visual(v):
    """lmms-eval 의 visual 항목 하나 → PIL/ndarray. 경로면 연다."""
    if isinstance(v, (str, Path)):
        from PIL import Image

        with Image.open(v) as im:
            return im.convert("RGB")
    return v


# ------------------------------------------------------------------- 입력 구성
def _question_with_placeholders(question: str, n_images: int) -> str:
    """질문 속 placeholder 를 정리한다.

    개수가 이미지 수와 같으면 그 자리에 끼운다 (Sensenova 학습 형식과 같다).
    0개면 모든 이미지를 앞에 둔다 (부모 경로와 같은 배치). 개수가 어긋나면 지우고 앞에 둔다 —
    평가는 멈추면 안 되니까. (학습에서는 어긋난 레코드를 버린다.)
    """
    k = len(_PH.findall(question))
    if k in (0, n_images):
        return question.replace("<video>", "<image>")
    return _PH.sub("", question)


def prepare_inputs(
    model,
    prompt: PromptBuilder,
    question: str,
    visuals: list | None = None,
    frame_indices: list[int] | None = None,
    use_geometry: bool = True,
) -> PreparedInput:
    """이미지(또는 영상에서 뽑은 키프레임) 리스트 + 질문 → 모델 입력.

    Args:
        visuals: PIL / ndarray[H,W,3] / 텐서 이미지 리스트. 영상이면 이미 뽑은 키프레임들.
        use_geometry: False 면 기하 없이 (베이스라인 확인용)
    """
    visuals = [load_visual(v) for v in (visuals or [])]
    spec = model.spec
    if not visuals:
        ids = prompt.build_query(_PH.sub("", question), [])
        return PreparedInput(ids, torch.zeros_like(ids), {}, None, 0, frame_indices)

    vlm = [prepare_image(v, spec) for v in visuals]
    segments = [prompt.image_segment(tokens_per_step(g[0], spec)) for _, g in vlm]
    q = _question_with_placeholders(question, len(visuals))
    try:
        ids = prompt.build_query(q, segments)
    except PromptError:
        ids = prompt.build_query(_PH.sub("", question), segments)
    mm = torch.zeros_like(ids)
    mm[ids == model.image_token_id] = 1
    grids = torch.cat([g for _, g in vlm], 0)
    pixel_kwargs = {"pixel_values": torch.cat([pv for pv, _ in vlm], 0), "image_grid_thw": grids}

    geometry = None
    if use_geometry and model.injector is not None:
        unit = getattr(model.geometry, "patch_size", 16)
        geom = [geometry_frame(v, model.cfg.geometry.image_size, unit) for v in visuals]
        outs = model.run_geometry(geom)
        geometry = model.build_geometry_embeds(
            outs, [llm_grid(g, spec) for g in grids], pool_temporal=False
        )
    return PreparedInput(ids, mm, pixel_kwargs, geometry, len(visuals), frame_indices)


@torch.no_grad()
def generate(model, prepared: PreparedInput, tokenizer, **gen_kwargs) -> str:
    """준비된 입력으로 생성한다. 기하가 있으면 프리필에서만 주입된다 (디코딩 스텝은 건너뛴다)."""
    device = next(model.base.parameters()).device
    ids = prepared.input_ids.to(device)
    kw = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        mm_token_type_ids=prepared.mm_token_type_ids.to(device),
        **{k: v.to(device) for k, v in prepared.pixel_kwargs.items()},
        **gen_kwargs,
    )
    if prepared.geometry is not None:
        with model.injector.primed(prepared.geometry.embeds, model.visual_pos_mask(ids)):
            out = model.base.generate(**kw)
    else:
        out = model.base.generate(**kw)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
