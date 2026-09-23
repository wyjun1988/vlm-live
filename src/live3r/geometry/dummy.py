"""DummyStream — 가중치 없이 형상만 맞추는 기하 인코더.

용도:
  * 맥/CPU 에서 엔드투엔드 파이프라인 형상 검증
  * CI 유닛테스트
  * 기하 정보가 실제로 기여하는지 보는 **음성 대조군**
    (학습 후 dummy 로 바꿔서 성능이 안 떨어지면 기하 토큰이 무시되고 있다는 뜻)
"""

from __future__ import annotations

import torch

from ..config import GeometryConfig
from .base import GeomOutput, GeometryStream
from .registry import register


class DummyStream(GeometryStream):
    is_streaming = True

    def __init__(
        self,
        tap_layers: tuple[int, ...] = (11, 17, 23),
        hidden_size: int = 1024,
        image_size: int = 518,
        patch_size: int = 14,
        seed: int = 0,
    ) -> None:
        super().__init__(tap_layers, hidden_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self._seed = seed
        # 고정 크기 "상태" — 진짜 스트리밍 인코더의 recurrent state 를 흉내낸다
        self.register_buffer("_state", torch.zeros(1, 256, hidden_size), persistent=False)
        self.reset()

    def reset(self) -> None:
        self._frame_index = 0
        self._state.zero_()

    @torch.no_grad()
    def ingest(self, frames: torch.Tensor) -> GeomOutput:
        b, _, h, w = frames.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        n = gh * gw
        g = torch.Generator(device="cpu").manual_seed(self._seed * 100003 + self._frame_index)
        tokens = {
            l: torch.randn(b, n, self.hidden_size, generator=g).to(frames.device, frames.dtype)
            for l in self.tap_layers
        }
        pose = torch.randn(b, 1, self.hidden_size, generator=g).to(frames.device, frames.dtype)
        # 상태는 갱신되지만 크기는 불변 — 상수 메모리 불변식.
        # 토큰 수 n 과 무관하게 [1,1,C] 로 줄여 브로드캐스트한다 (n<256 이어도 안전).
        upd = tokens[self.tap_layers[-1]][:1].mean(dim=1, keepdim=True).float()
        self._state.mul_(0.9).add_(0.1 * upd)
        out = GeomOutput(
            tokens=tokens, grid_hw=(gh, gw), pose_token=pose, frame_index=self._frame_index
        )
        self._frame_index += 1
        return out

    def state_bytes(self) -> int:
        return self._state.numel() * self._state.element_size()


@register("dummy")
def _build(cfg: GeometryConfig) -> DummyStream:
    return DummyStream(
        tap_layers=cfg.tap_layers,
        hidden_size=cfg.hidden_size,
        image_size=cfg.image_size,
    )
