"""VGGT on a sliding window of keyframes — the non-recurrent comparison encoder (registry name `vggt_window`).

VGGT (Wang et al., 2025) takes a *set* of images and returns, for all of them jointly, camera poses, world points
and confidence in the frame of the set's first image, at a normalised (not metric) scale. To use it as a stream:
each new frame is appended to a window of the last `window` frames, the model runs on the window, and the new
frame's outputs are carried into the global frame the first window established through the cameras the two
consecutive windows share (a similarity transform, `sim3.py`). Cost per frame is constant in the stream length
but `window` times a single-image model; `is_streaming = False` marks it as a comparison encoder.

What it is for: the routed-measurement harness (SceneMap / ObjectMap facts scored against VSI ground truth) is an
encoder bake-off that needs no training. Direction accuracy is scale-free; the metric facts (room area, distances)
come out in VGGT's normalised units and are compared after ratio normalisation.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import torch

from ..config import GeometryConfig
from .base import GeometryStream, GeomOutput
from .registry import register
from .sim3 import apply_sim3, apply_sim3_points, sim3_from_shared_cameras

logger = logging.getLogger(__name__)
PATCH = 14


class WindowChainer:
    """Keeps the global camera-to-world pose of every frame still inside the window and turns each new window's
    poses (in that window's own frame) into the global frame via the frames both windows contain."""

    def __init__(self) -> None:
        self.global_c2w: dict[int, np.ndarray] = {}
        self.last_sim3: tuple[float, np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self.global_c2w.clear()
        self.last_sim3 = None

    def update(self, frame_ids: list[int], c2w_window: np.ndarray) -> tuple[np.ndarray, tuple[float, np.ndarray, np.ndarray]]:
        """frame_ids: the window's frame indices in order; c2w_window: [S,4,4] in the window's frame.
        Returns the NEW (last) frame's global pose and the (s, R, t) that maps this window into the global frame."""
        shared = [i for i, f in enumerate(frame_ids[:-1]) if f in self.global_c2w]
        if not shared:                                   # the first window defines the global frame
            s, r, t = 1.0, np.eye(3), np.zeros(3)
        else:
            s, r, t = sim3_from_shared_cameras(np.stack([self.global_c2w[frame_ids[i]] for i in shared]),
                                               c2w_window[shared])
        new = apply_sim3(s, r, t, c2w_window[-1])
        # frames fixed earlier keep their pose (no re-fixing: the maps already hold points in it); prune the rest
        keep = set(frame_ids)
        self.global_c2w = {f: p for f, p in self.global_c2w.items() if f in keep}
        for i, f in enumerate(frame_ids):
            self.global_c2w.setdefault(f, apply_sim3(s, r, t, c2w_window[i]))
        self.global_c2w[frame_ids[-1]] = new
        self.last_sim3 = (s, r, t)
        return new, (s, r, t)


def extrinsics_to_c2w(extri: np.ndarray) -> np.ndarray:
    """VGGT extrinsics [S,3,4] (world -> camera) -> camera-to-world [S,4,4]."""
    out = np.tile(np.eye(4), (len(extri), 1, 1))
    rt = np.transpose(extri[:, :3, :3], (0, 2, 1))
    out[:, :3, :3] = rt
    out[:, :3, 3] = -np.einsum("nij,nj->ni", rt, extri[:, :3, 3])
    return out


class VGGTWindowStream(GeometryStream):
    is_streaming = False

    def __init__(self, model_id: str, tap_layers: tuple[int, ...], window: int = 8, hidden_size: int = 2048) -> None:
        super().__init__(tap_layers, hidden_size)
        from vggt.models.vggt import VGGT

        self.net = VGGT.from_pretrained(model_id).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.patch_size = PATCH
        self.window = int(window)
        self._frames: deque = deque(maxlen=self.window)
        self._ids: deque = deque(maxlen=self.window)
        self.chain = WindowChainer()
        self.decode_points = True
        logger.info("VGGT window encoder: %s, window %d, taps %s", model_id, self.window, tap_layers)

    def reset(self) -> None:
        super().reset()
        self._frames.clear()
        self._ids.clear()
        self.chain.reset()

    def state_bytes(self) -> int:
        return sum(f.numel() * f.element_size() for f in self._frames)

    @torch.no_grad()
    def ingest(self, frames: torch.Tensor) -> GeomOutput:
        """frames: [1,3,H,W], normalised like the other encoders ((x - 0.5) / 0.5); VGGT wants [0, 1]."""
        if frames.dim() != 4 or frames.shape[0] != 1:
            raise ValueError(f"one frame at a time, got {tuple(frames.shape)}")
        dev = next(self.net.parameters()).device
        img = (frames[0].float() * 0.5 + 0.5).clamp(0, 1).to(dev)
        _, h, w = img.shape
        if h % PATCH or w % PATCH:
            raise ValueError(f"frame {h}x{w} is not a multiple of {PATCH} - use geometry_frame(..., 518, 14)")
        self._frames.append(img)
        self._ids.append(self._frame_index)
        batch = torch.stack(list(self._frames))[None]                       # [1,S,3,H,W]

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        tokens, ps_idx = self.net.aggregator(batch)
        pose_enc = self.net.camera_head(tokens)[-1]                          # [1,S,9]
        extri, _ = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])  # [1,S,3,4]
        pts, conf = self.net.point_head(tokens, batch, ps_idx)              # [1,S,H,W,3], [1,S,H,W]

        c2w_window = extrinsics_to_c2w(extri[0].float().cpu().numpy())
        c2w_new, (s, r, t) = self.chain.update(list(self._ids), c2w_window)
        pts_new = apply_sim3_points(s, r, t, pts[0, -1].float().cpu().numpy())   # [H,W,3] in the global frame

        taps = {l: tokens[l][:, -1, ps_idx:, :].float() for l in self.tap_layers}   # [1,N,C] of the new frame
        pose_tok = tokens[self.tap_layers[0]][:, -1, 0:1, :].float()                # the camera token
        out = GeomOutput(
            tokens=taps, grid_hw=(h // PATCH, w // PATCH), pose_token=pose_tok,
            pointmap=torch.from_numpy(pts_new.astype(np.float32)).permute(2, 0, 1)[None].to(dev),   # MPS: no float64
            conf=conf[0, -1][None, None].float(), frame_index=self._frame_index,
            extra={"c2w": torch.from_numpy(c2w_new.astype(np.float32))[None], "sim3_scale": s},
        )
        self._frame_index += 1
        return out


@register("vggt_window")
def _build(cfg: GeometryConfig) -> VGGTWindowStream:
    return VGGTWindowStream(model_id=cfg.checkpoint or "facebook/VGGT-1B", tap_layers=cfg.tap_layers or (11, 17, 23),
                            window=int(cfg.options.get("window", 8)), hidden_size=cfg.hidden_size or 2048)
