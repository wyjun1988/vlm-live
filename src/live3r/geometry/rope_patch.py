"""CroCo RoPE2D 의 순수 PyTorch 폴백을 음수 위치에서도 돌게 고친다.

**왜 필요한가 (조용히 안 알려진 함정):**
CUT3R 은 카메라/포즈 토큰의 2D 위치를 **-1** 로 준다 (`pose_pos = -ones(B,1,2)`).
  * CUDA 커널(`croco/models/curope`)은 `freq = pos * inv_freq` 를 직접 계산해서 음수도 정상.
  * 순수 PyTorch 폴백(`croco/models/pos_embed.py`)은 미리 만든 cos/sin 테이블을
    `F.embedding(pos1d, cos)` 로 **룩업**한다 → 음수 인덱스에서 IndexError.

즉 **curope 를 컴파일하지 않으면 pose_head 가 있는 CUT3R 추론이 아예 실패한다.**
GPU 머신에서는 컴파일하는 게 맞다:
    cd <CUT3R>/src/croco/models/curope && python setup.py build_ext --inplace
이 패치는 (a) 맥/CPU 에서 구조 검증을 돌리려고, (b) curope 가 없는 환경에서도
동작은 하게 하려고 둔다. 수치는 CUDA 커널과 동일하다 (테이블 룩업 대신 직접 계산).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope1d(self, tokens: torch.Tensor, pos1d: torch.Tensor, cos, sin):
    """테이블 룩업 대신 직접 계산 — 음수·비정수 위치 모두 처리한다.

    tokens: [B, H, N, D]   pos1d: [B, N]
    """
    assert pos1d.ndim == 2
    d = tokens.shape[-1]
    inv_freq = 1.0 / (self.base ** (torch.arange(0, d, 2, device=tokens.device).float() / d))
    freqs = pos1d.float()[..., None] * inv_freq            # [B, N, D/2]
    freqs = torch.cat((freqs, freqs), dim=-1).to(tokens.dtype)  # [B, N, D]
    cos_, sin_ = freqs.cos()[:, None], freqs.sin()[:, None]     # [B, 1, N, D]
    return (tokens * cos_) + (_rotate_half(tokens) * sin_)


def _candidate_modules():
    """croco 는 `models.pos_embed` 와 `croco.models.pos_embed` 두 경로로 동시에 로드될 수 있다.

    CUT3R 이 sys.path 에 `<repo>/src` 와 `<repo>/src/croco` 를 둘 다 넣기 때문이다.
    실제로 쓰이는 쪽만 패치하면 조용히 안 먹는다 — **둘 다** 잡는다.
    """
    import importlib
    import sys

    seen = {}
    for name, mod in list(sys.modules.items()):
        if name.endswith("pos_embed") and hasattr(mod, "RoPE2D"):
            seen[id(mod)] = mod
    for name in ("croco.models.pos_embed", "models.pos_embed"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "RoPE2D"):
            seen[id(mod)] = mod
    return list(seen.values())


def patch_rope2d(force: bool = False) -> int:
    """croco 의 PyTorch RoPE2D 를 고친다. curope(CUDA) 판본이면 건드리지 않는다.

    Returns:
        실제로 패치한 클래스 수 (0 이면 불필요하거나 실패)
    """
    global _patched
    mods = _candidate_modules()
    if not mods:
        logger.debug("croco pos_embed 를 못 찾았다 — RoPE 패치 생략")
        return 0

    n = 0
    for mod in mods:
        rope_cls = getattr(mod, "RoPE2D", None)
        if rope_cls is None:
            continue
        # curope 판본은 apply_rope1d 자체가 없다 (CUDA 커널이 직접 처리) → 건드릴 것도 없다
        if not hasattr(rope_cls, "apply_rope1d"):
            continue
        if getattr(rope_cls, "_live3r_patched", False) and not force:
            n += 1
            continue
        rope_cls.apply_rope1d = _apply_rope1d
        rope_cls._live3r_patched = True
        n += 1

    if n:
        _patched = True
        logger.info("RoPE2D 폴백 %d개를 음수 위치 지원 판본으로 교체했다", n)
    return n
