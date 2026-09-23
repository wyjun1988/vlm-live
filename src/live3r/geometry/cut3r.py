"""CUT3R 어댑터 — 스트리밍 기하 인코더 기본값.

CUT3R(`ARCroco3DStereo`)은 고정 크기 implicit state 를 재귀 갱신하는 RNN 이라 프레임당 O(1) 이다.
VLM-3R 이 검증한 경로라 재현 기준선이 된다.

**구조 (저장소 코드 확인, 2026-09):**
    enc_embed_dim 1024 / enc_depth 24  ·  dec_embed_dim 768 / dec_depth 12  ·  patch 16
    `_decoder` 가 돌려주는 `dec` 는 길이 dec_depth+1 의 튜플:
        dec[0]      [B, N, 1024]    — 투영 전 인코더 출력. 포즈 토큰 없음
        dec[1..12]  [B, 1+N, 768]   — **index 0 이 카메라/포즈 토큰**, 1: 이 공간 토큰
    CUT3R 자신의 헤드가 dec[0], dec[6], dec[9], dec[12] 를 쓴다
    (= dec_depth*2//4, *3//4, dec_depth). 그래서 **탭 기본값도 (6, 9, 12)** 로 둔다 —
    SpatialStack 의 (11,17,23) 은 VGGT 24층 기준이라 그대로 쓰면 범위를 벗어난다.

**설치:**
    git clone https://github.com/CUT3R/CUT3R
    # geometry.options.repo_path 에 CUT3R 경로를 주거나 PYTHONPATH=<repo>/src
    gdown --fuzzy 'https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/'

GPU 머신에서 먼저 실측해라:
    PYTHONPATH=src python scripts/verify_geometry_adapter.py --name cut3r \
        --checkpoint checkpoints/cut3r_512_dpt_4_64.pth --tap-layers 6 9 12 \
        --repo-path /path/to/CUT3R
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from ..config import GeometryConfig
from .base import GeomOutput, GeometryStream
from .registry import register

logger = logging.getLogger(__name__)

DEFAULT_TAP_LAYERS = (6, 9, 12)


class CUT3RStream(GeometryStream):
    """CUT3R 을 GeometryStream 인터페이스에 맞춘 래퍼.

    `forward_recurrent` 는 전체 시퀀스를 한 번에 받는다. 우리는 한 프레임씩 받아야 하므로
    그 루프의 **몸통 한 스텝**을 여기서 재구성한다 (원본 코드를 고치지 않는다).
    """

    is_streaming = True

    def __init__(
        self,
        checkpoint: str | None = None,
        tap_layers: tuple[int, ...] = DEFAULT_TAP_LAYERS,
        image_size: int = 512,
        repo_path: str | None = None,
        device: str | torch.device = "cpu",
        net=None,
    ) -> None:
        """net 을 직접 주면 체크포인트 없이 쓴다 (구조 검증·테스트용)."""
        # CUT3R 포즈 토큰의 위치가 -1 이라 순수 PyTorch RoPE2D 폴백이 터진다.
        # curope(CUDA)가 있으면 아무것도 안 한다. 자세한 사정은 rope_patch 모듈 상단.
        from .rope_patch import patch_rope2d

        if net is None:
            if not checkpoint:
                raise ValueError("checkpoint 또는 net 중 하나는 있어야 한다")
            net = _load_cut3r(checkpoint, repo_path, device)
        patch_rope2d()  # 모델이 만들어진 뒤에 걸어야 두 모듈 인스턴스를 모두 잡는다
        dec_depth = int(getattr(net, "dec_depth"))
        bad = [t for t in tap_layers if not (1 <= t <= dec_depth)]
        if bad:
            raise ValueError(
                f"탭 레이어 {bad} 가 범위를 벗어난다. CUT3R dec_depth={dec_depth} 이므로 1..{dec_depth}. "
                f"권장값 {DEFAULT_TAP_LAYERS}. (dec[0] 은 인코더 차원이라 탭으로 쓰지 않는다)"
            )
        super().__init__(tap_layers, int(net.dec_embed_dim))
        self.net = net
        self.dec_depth = dec_depth
        self.image_size = image_size
        self.patch_size = int(getattr(net.patch_embed, "patch_size", (16, 16))[0]
                              if hasattr(getattr(net, "patch_embed", None), "patch_size")
                              else 16)
        if not getattr(net, "pose_head_flag", False):
            raise ValueError(
                "이 체크포인트는 pose_head 가 꺼져 있다. 카메라 토큰을 못 뽑으므로 "
                "cut3r_512_dpt_4_64.pth 같은 pose_head 가 있는 판본을 써라."
            )
        self._state = None  # (state_feat, state_pos, init_state_feat, mem, init_mem)
        self.reset()

    # ------------------------------------------------------------------ 상태 관리
    def reset(self) -> None:
        self._frame_index = 0
        self._state = None

    def state_bytes(self) -> int:
        if self._state is None:
            return 0
        return sum(
            t.numel() * t.element_size() for t in self._state if isinstance(t, torch.Tensor)
        )

    # ---------------------------------------------------------------------- 인제스트
    @torch.no_grad()
    def ingest(self, frames: torch.Tensor) -> GeomOutput:
        """프레임 1장 [B,3,H,W] → 잠재 토큰.

        `ARCroco3DStereo.forward_recurrent` 의 한 스텝과 동일한 순서로 돈다:
          인코딩 → (첫 프레임이면) 상태 초기화 → 포즈 토큰 조회 → 재귀 롤아웃 → 상태·메모리 갱신
        """
        net = self.net
        b, _, h, w = frames.shape
        shape = torch.tensor([h, w], device=frames.device).unsqueeze(0).repeat(b, 1)

        img_out, img_pos, _ = net._encode_image(frames, shape)
        feat_i = img_out[-1]

        if self._state is None:
            state_feat, state_pos = net._init_state(feat_i, img_pos)
            mem = net.pose_retriever.mem.expand(b, -1, -1)
            init_state_feat = state_feat.clone()
            init_mem = mem.clone()
        else:
            state_feat, state_pos, init_state_feat, mem, init_mem = self._state

        global_img_feat = net._get_img_level_feat(feat_i)
        if self._frame_index == 0:
            pose_feat = net.pose_token.expand(b, -1, -1)
        else:
            pose_feat = net.pose_retriever.inquire(global_img_feat, mem)
        pose_pos = -torch.ones(b, 1, 2, device=feat_i.device, dtype=img_pos.dtype)

        new_state_feat, dec = net._recurrent_rollout(
            state_feat, state_pos, feat_i, img_pos, pose_feat, pose_pos, init_state_feat,
            img_mask=None, reset_mask=None, update=None,
        )
        dec = list(dec)
        if len(dec) != self.dec_depth + 1:
            raise RuntimeError(
                f"디코더 출력 길이 {len(dec)} != dec_depth+1 ({self.dec_depth + 1}). "
                "CUT3R 리비전이 바뀐 듯하다 — 탭 인덱스 규약을 다시 확인해라."
            )

        out_pose_feat = dec[-1][:, 0:1]                       # 카메라/뷰 토큰
        self._state = (
            new_state_feat,
            state_pos,
            init_state_feat,
            net.pose_retriever.update_mem(mem, global_img_feat, out_pose_feat),
            init_mem,
        )

        gh, gw = h // self.patch_size, w // self.patch_size
        tokens = {t: dec[t][:, 1:] for t in self.tap_layers}  # index 0(포즈) 제거
        n = gh * gw
        got = next(iter(tokens.values())).shape[1]
        if got != n:
            raise RuntimeError(
                f"공간 토큰 {got}개 != 격자 {gh}x{gw}={n}. patch_size 추정({self.patch_size})이 틀렸다."
            )
        out = GeomOutput(
            tokens=tokens,
            grid_hw=(gh, gw),
            pose_token=out_pose_feat,
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return out


def _load_cut3r(checkpoint: str, repo_path: str | None, device):
    if repo_path:
        root = Path(repo_path)
        for cand in (root / "src", root):
            if (cand / "dust3r").is_dir():
                # croco 는 최상위 `models` 패키지로 임포트된다 → src/croco 도 넣어야 한다
                for pth in (str(cand), str(cand / "croco")):
                    if pth not in sys.path:
                        sys.path.insert(0, pth)
                break
    try:
        from dust3r.model import ARCroco3DStereo  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"CUT3R 코드를 못 찾았다 ({exc}).\n"
            "  git clone https://github.com/CUT3R/CUT3R\n"
            "  → geometry.options.repo_path 에 클론 경로를 주거나 PYTHONPATH=<repo>/src"
        ) from exc
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"CUT3R 체크포인트가 없다: {checkpoint}\n"
            "  gdown --fuzzy 'https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/'"
        )
    net = ARCroco3DStereo.from_pretrained(checkpoint)
    return net.to(device).eval()


@register("cut3r")
def _build(cfg: GeometryConfig) -> CUT3RStream:
    if not cfg.checkpoint:
        raise ValueError("geometry.checkpoint 가 필요하다 (cut3r_512_dpt_4_64.pth)")
    taps = cfg.tap_layers or DEFAULT_TAP_LAYERS
    return CUT3RStream(
        checkpoint=cfg.checkpoint,
        tap_layers=taps,
        image_size=cfg.image_size,
        repo_path=cfg.options.get("repo_path"),
    )
