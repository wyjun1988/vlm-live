"""Qwen3.5 비전 전처리 — 공식 프로세서와 **같은 텐서**를 만든다.

왜 직접 구현하나:
    공식 프로세서는 (a) 영상 프레임을 `linspace(0, total_frames-1, n)` 으로 샘플링하고
    (b) 이미지당 최대 16M 픽셀까지 허용한다. (a)는 총 길이를 아는 오프라인 동작이라
    스트리밍 제약과 정면으로 충돌하고, (b)는 토큰 수를 통제할 수 없다.
    그래서 프레임 선택·해상도는 우리가 정하고, **텐서 레이아웃만은 공식과 같게** 만든다.

왜 공식과 같아야 하나 (2026-09-23 실제 버그):
    Qwen 비전 타워의 patch merger 는 **연속된 4개 패치를 2×2 블록으로 가정**하고 합친다.
    그래서 공식 patchify 는 패치를 (gh/m, gw/m, m, m) — 즉 2×2 merge 블록 단위로 — 평탄화한다.
    래스터 순서(gh, gw)로 평탄화하면 merger 가 1×4 가로 띠를 합치고, 회전 위치 임베딩도
    내용과 어긋난다. **실가중치에서는 모든 이미지가 뒤섞여 들어가는데 에러는 안 난다.**
    랜덤 초소형 모델로는 절대 안 잡히므로 tests/test_vision_official.py 가 공식 구현과 직접 비교한다.

레이아웃 (transformers qwen3_vl VideoProcessor.patchify 와 동일):
    프레임 [T, C, H, W]  →  view (gt, tp, C, gh/m, m, p, gw/m, m, p)
                         →  permute (gt, gh/m, gw/m, m_h, m_w, C, tp, p_h, p_w)
                         →  [gt*gh*gw, C*tp*p*p]
    이미지는 프레임 1장을 tp 번 복제한 비디오와 정확히 같다 (공식 이미지 프로세서의 expand 와 동일).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Qwen3.5 비전 타워와 CUT3R 이 둘 다 이 정규화를 쓴다 (둘 다 저장소 코드로 확인)
MEAN = 0.5
STD = 0.5


@dataclass(frozen=True)
class VisionSpec:
    """Qwen3.5 비전 입력 규격."""

    patch: int = 16
    merge: int = 2
    temporal_patch: int = 2
    #: 이미지(또는 프레임) 1장당 픽셀 하한/상한. 공식 기본 상한은 16M 이라 토큰이 폭발한다.
    #: 기본값은 lmms-eval qwen3_5 와 같게 (64~128)×32×32 → 이미지당 LLM 토큰 64~128개.
    min_pixels: int = 64 * 32 * 32
    max_pixels: int = 128 * 32 * 32

    @property
    def factor(self) -> int:
        """해상도가 나누어떨어져야 하는 단위 = patch × merge (= 32)."""
        return self.patch * self.merge

    @classmethod
    def from_config(cls, vision_config, **overrides) -> "VisionSpec":
        kw = dict(
            patch=vision_config.patch_size,
            merge=vision_config.spatial_merge_size,
            temporal_patch=getattr(vision_config, "temporal_patch_size", 2),
        )
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """`transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize` 와 동일."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"종횡비가 200 을 넘는다: {height}x{width}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# --------------------------------------------------------------------- 텐서 변환
def to_float01(x) -> torch.Tensor:
    """uint8 [H,W,3] · [T,H,W,3] (numpy/torch) 또는 PIL 이미지 → float [T,3,H,W] ∈ [0,1]."""
    if hasattr(x, "convert") and hasattr(x, "size"):  # PIL
        x = np.array(x.convert("RGB"))  # np.asarray 는 PIL 버퍼를 읽기전용으로 공유한다 → 복사
    if isinstance(x, np.ndarray):
        if not x.flags.writeable:
            x = x.copy()
        x = torch.from_numpy(np.ascontiguousarray(x))
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    return x.float()


def resize01(frames01: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """[T,3,H,W] ∈ [0,1] → [T,3,h,w]. 공식은 PIL BICUBIC 이라 antialias bicubic 으로 맞춘다."""
    if tuple(frames01.shape[-2:]) == tuple(size):
        return frames01
    out = F.interpolate(frames01, size=size, mode="bicubic", align_corners=False, antialias=True)
    return out.clamp_(0.0, 1.0)


def normalize(frames01: torch.Tensor) -> torch.Tensor:
    return (frames01 - MEAN) / STD


# ----------------------------------------------------------------- patchify
def patchify(frames: torch.Tensor, spec: VisionSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """정규화된 [T,3,H,W] → (pixel_values [gt*gh*gw, 3*tp*p*p], grid_thw [1,3]).

    T 가 temporal_patch 로 안 나누어떨어지면 마지막 프레임을 복제한다 (공식과 동일).
    """
    t, c, h, w = frames.shape
    p, m, tp = spec.patch, spec.merge, spec.temporal_patch
    if h % (p * m) or w % (p * m):
        raise ValueError(
            f"해상도 {h}x{w} 가 {p * m}(=patch×merge) 의 배수가 아니다. "
            "smart_resize 를 거치지 않은 프레임이다."
        )
    if pad := (-t) % tp:
        frames = torch.cat([frames, frames[-1:].expand(pad, c, h, w)], dim=0)
        t += pad
    gt, gh, gw = t // tp, h // p, w // p
    x = frames.reshape(gt, tp, c, gh // m, m, p, gw // m, m, p)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)  # (gt, gh/m, gw/m, m_h, m_w, C, tp, p_h, p_w)
    flat = x.reshape(gt * gh * gw, c * tp * p * p)
    return flat, torch.tensor([[gt, gh, gw]], dtype=torch.long)


def unpatchify(flat: torch.Tensor, grid_thw: torch.Tensor, spec: VisionSpec) -> torch.Tensor:
    """patchify 의 역변환 → [gt*tp, 3, H, W]. 자동 기하 모드가 픽셀에서 프레임을 복원할 때 쓴다."""
    gt, gh, gw = (int(v) for v in grid_thw.reshape(-1, 3)[0])
    p, m, tp = spec.patch, spec.merge, spec.temporal_patch
    c = flat.shape[-1] // (tp * p * p)
    x = flat.reshape(gt, gh // m, gw // m, m, m, c, tp, p, p)
    x = x.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)  # (gt, tp, C, gh/m, m_h, p_h, gw/m, m_w, p_w)
    return x.reshape(gt * tp, c, gh * p, gw * p)


# ---------------------------------------------------------------- 고수준 API
def prepare_image(img, spec: VisionSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """이미지 1장 → (pixel_values, image_grid_thw[1,3]). 공식 이미지 프로세서와 같은 결과."""
    x = to_float01(img)
    _, _, h, w = x.shape
    th, tw = smart_resize(h, w, spec.factor, spec.min_pixels, spec.max_pixels)
    return patchify(normalize(resize01(x, (th, tw))), spec)


def prepare_video(frames, spec: VisionSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """프레임 시퀀스 → (pixel_values_videos, video_grid_thw[1,3]).

    해상도는 프레임마다 이미지와 같은 smart_resize 를 쓴다. (공식 비디오 프로세서는 총 픽셀
    예산을 프레임 수로 나누는데, 그건 프레임 수 = 미래 정보를 요구한다. 스트리밍에서는 못 쓴다.)
    """
    x = to_float01(frames)
    _, _, h, w = x.shape
    th, tw = smart_resize(h, w, spec.factor, spec.min_pixels, spec.max_pixels)
    return patchify(normalize(resize01(x, (th, tw))), spec)


def llm_grid(grid_row, spec: VisionSpec) -> tuple[int, int]:
    """grid_thw 한 행 → LLM 비전 토큰 격자 (H, W) — temporal patch 하나(=이미지 하나)당."""
    _, gh, gw = (int(v) for v in torch.as_tensor(grid_row).reshape(-1)[:3])
    return gh // spec.merge, gw // spec.merge


def tokens_per_step(grid_row, spec: VisionSpec) -> int:
    h, w = llm_grid(grid_row, spec)
    return h * w


def geometry_frame(img, long_side: int, unit: int) -> torch.Tensor:
    """기하 인코더 입력 [3,h,w] (정규화됨).

    CUT3R(512 판본)은 **긴 변 512** 로 학습됐다 (dust3r load_images: long edge = size).
    짧은 변으로 맞추면 4:3 영상이 683×512 로 들어가 학습 해상도보다 1.8배 커진다.
    원본 CUT3R 은 16 배수로 center-crop 하지만, 여기서는 **리사이즈로 맞춘다** —
    VLM 쪽 프레임과 시야(FOV)를 똑같이 유지해야 기하 격자와 비전 격자가 정렬되기 때문이다
    (종횡비 왜곡은 2% 미만).
    """
    x = to_float01(img)
    _, _, h, w = x.shape
    scale = long_side / max(h, w)
    th = max(unit, int(round(h * scale / unit)) * unit)
    tw = max(unit, int(round(w * scale / unit)) * unit)
    return normalize(resize01(x, (th, tw)))[0]


def frame_timestamps(indices: list[int], fps: float, temporal_patch: int) -> list[float]:
    """temporal patch 당 타임스탬프 — 공식 Qwen3VLProcessor._calculate_timestamps 와 동일.

    패치 안 첫·끝 프레임 시각의 평균. 프레임 수가 모자라면 마지막 인덱스를 복제한다.
    """
    idx = list(indices)
    if pad := (-len(idx)) % temporal_patch:
        idx += [idx[-1]] * pad
    ts = [i / fps for i in idx]
    return [(ts[i] + ts[i + temporal_patch - 1]) / 2 for i in range(0, len(ts), temporal_patch)]
