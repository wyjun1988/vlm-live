"""GeometryStream — 스트리밍 3D 기하 인코더 추상화.

핵심 불변식:
    ingest() 의 시간·메모리가 스트림 길이에 대해 **상수**여야 한다.
    이걸 만족 못 하면 라이브가 아니다. tests/test_streaming_invariants.py 가 강제한다.

인코더가 바뀌어도 (CUT3R → Anchor3R → LingBot-Map) 모델 코드는 그대로여야 한다.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import torch


@dataclass
class GeomOutput:
    """한 프레임(또는 한 청크)을 먹인 뒤 인코더가 내놓는 것.

    Attributes:
        tokens: {tap_layer_idx: Tensor[B, N_tok, C]} — 인코더 중간 깊이별 잠재 토큰.
            N_tok 는 (H/patch)*(W/patch) 형태의 공간 토큰 수.
        pose_token: Tensor[B, 1, C] | None — 카메라/뷰 토큰. 전역 좌표계 정보를 담는다.
        grid_hw: (h, w) — tokens 의 공간 격자 크기. 머저가 필요로 한다.
        pointmap: Tensor[B, 3, H, W] | None — 선택. 디버그·시각화·보조손실용.
        conf: Tensor[B, 1, H, W] | None — 선택. pointmap 신뢰도.
        frame_index: 스트림 내 이 프레임의 인덱스.
    """

    tokens: dict[int, torch.Tensor]
    grid_hw: tuple[int, int]
    pose_token: torch.Tensor | None = None
    pointmap: torch.Tensor | None = None
    conf: torch.Tensor | None = None
    frame_index: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        h, w = self.grid_hw
        return h * w


class GeometryStream(abc.ABC, torch.nn.Module):
    """스트리밍 기하 인코더 어댑터의 베이스.

    구현체는 reset/ingest/hidden_size/state_bytes 를 채운다.
    `is_streaming = False` 인 구현체(VGGT 윈도우 등)는 비교군 전용이며,
    라이브 경로에서 쓰면 LiveSession 이 경고한다.
    """

    #: 스트림 길이에 대해 상수 비용인가. False 면 비교군(oracle) 전용.
    is_streaming: bool = True

    def __init__(self, tap_layers: tuple[int, ...], hidden_size: int) -> None:
        super().__init__()
        self.tap_layers = tuple(tap_layers)
        self._hidden_size = hidden_size
        self._frame_index = 0

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @abc.abstractmethod
    def reset(self) -> None:
        """새 스트림 시작. 내부 상태를 비운다."""
        self._frame_index = 0

    @abc.abstractmethod
    def ingest(self, frames: torch.Tensor) -> GeomOutput:
        """프레임을 먹이고 잠재 토큰을 돌려준다.

        Args:
            frames: Tensor[B, 3, H, W] — 정규화된 RGB. 한 번에 한 타임스텝.

        Returns:
            GeomOutput
        """

    @abc.abstractmethod
    def state_bytes(self) -> int:
        """현재 내부 상태가 차지하는 바이트. 스트림이 길어져도 상수여야 한다."""

    @torch.no_grad()
    def ingest_sequence(self, frames: torch.Tensor) -> list[GeomOutput]:
        """편의 함수: Tensor[T, 3, H, W] 를 순서대로 먹인다."""
        outs = []
        for t in range(frames.shape[0]):
            outs.append(self.ingest(frames[t : t + 1]))
        return outs
