"""GeometryProjector — 기하 잠재 토큰을 LLM 히든 공간으로 옮긴다.

**여기가 가장 조용히 틀리기 쉬운 지점이다.**
기하 인코더와 VLM 비전 타워는 패치 크기가 다르다:
  * Qwen3.5 비전: patch 16 + spatial_merge 2 → LLM 비전 토큰 1개 = 원본 32px
  * CUT3R/VGGT 계열: patch 14 @ 518 → 37×37 격자
그래서 기하 토큰을 LLM 비전 토큰 격자에 **리샘플**하지 않으면 엉뚱한 위치에 더해진다.
deepstack 은 위치별 elementwise add 라서 격자가 어긋나면 조용히 성능만 깎인다.
→ forward 가 src_grid / dst_grid 를 모두 받게 강제한다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GeometryProjector(nn.Module):
    """기하 토큰 [B,N,C_geo] → LLM 비전 토큰 격자에 맞춘 [B,H*W,d_llm].

    Args:
        c_geo: 기하 인코더 잠재 차원
        d_llm: LLM 히든 차원
        merge_size: 투영 전에 m×m 공간 머지 (채널 concat). 로컬 문맥 집약 + 연산 절감.
        mlp_ratio: 은닉 배율
        zero_init: 마지막 Linear 를 0 으로 초기화. True 면 학습 시작 시 주입량이 정확히 0 →
            베이스 VLM 동작 보존. 특화 역설(docs/RESEARCH_NOTES.md §5-1) 방어의 1차 장치.
    """

    def __init__(
        self,
        c_geo: int,
        d_llm: int,
        merge_size: int = 2,
        mlp_ratio: float = 2.0,
        zero_init: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.c_geo = c_geo
        self.d_llm = d_llm
        self.merge_size = max(1, int(merge_size))
        c_in = c_geo * self.merge_size * self.merge_size
        hidden = int(d_llm * mlp_ratio)

        self.norm = nn.LayerNorm(c_in)
        self.fc1 = nn.Linear(c_in, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, d_llm)
        # 주입 세기를 학습으로 조절 — 레이어별로 기하가 얼마나 필요한지 데이터가 정하게 둔다
        self.gate = nn.Parameter(torch.zeros(1))

        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        src_grid: tuple[int, int],
        dst_grid: tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, C_geo] (N == src_grid[0]*src_grid[1])
            src_grid: 기하 토큰의 (h, w)
            dst_grid: LLM 비전 토큰의 (H, W) — 프레임 1장 기준
        Returns:
            [B, H*W, d_llm]
        """
        b, n, c = tokens.shape
        sh, sw = src_grid
        if n != sh * sw:
            raise ValueError(f"토큰 수 {n} != src_grid {sh}x{sw}={sh * sw}")
        if c != self.c_geo:
            raise ValueError(f"기하 차원 {c} != 설정값 {self.c_geo}")

        x = tokens.transpose(1, 2).reshape(b, c, sh, sw)  # [B,C,h,w]

        m = self.merge_size
        if m > 1:
            ph, pw = (-sh) % m, (-sw) % m  # 나누어떨어지게 패딩
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            x = F.pixel_unshuffle(x, m)  # [B, C*m*m, h/m, w/m]

        dh, dw = dst_grid
        if x.shape[-2:] != (dh, dw):
            x = F.interpolate(x.float(), size=(dh, dw), mode="bilinear", align_corners=False).to(tokens.dtype)

        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C*m*m]
        x = self.norm(x)
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return x * self.gate


class PoseProjector(nn.Module):
    """카메라/포즈 토큰 [B,1,C] → [B,1,d_llm].

    OVO-S-Bench 의 L4(allocentric 매핑)가 전 모델 공통 병목이라, 스트리밍 인코더가
    이미 들고 있는 전역 자세 정보를 LLM 에 명시적으로 넘긴다.
    """

    def __init__(self, c_geo: int, d_llm: int, zero_init: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(c_geo),
            nn.Linear(c_geo, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )
        self.gate = nn.Parameter(torch.zeros(1))
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, pose_token: torch.Tensor) -> torch.Tensor:
        return self.net(pose_token) * self.gate
