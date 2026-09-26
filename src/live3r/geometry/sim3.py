"""Chaining the coordinate frames of overlapping windows (a windowed, non-recurrent encoder such as VGGT).

Every VGGT call expresses its poses and points in the frame of the window's first image, at its own scale. Two
consecutive windows share all but one frame; the shared cameras give the similarity transform (scale, rotation,
translation) that carries the new window into the global frame the first window established. Pure numpy.
"""

from __future__ import annotations

import numpy as np


def chordal_mean_rotation(rots: np.ndarray) -> np.ndarray:
    """Rotation closest (Frobenius) to the mean of the given rotations [N,3,3] — the chordal L2 mean."""
    u, _, vt = np.linalg.svd(rots.sum(0))
    r = u @ vt
    if np.linalg.det(r) < 0:           # keep a proper rotation
        u[:, -1] *= -1
        r = u @ vt
    return r


def sim3_from_shared_cameras(c2w_global: np.ndarray, c2w_window: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) with  X_global = s * R @ X_window + t, from the same cameras seen in both frames.

    c2w_global / c2w_window: [N,4,4] camera-to-world matrices of the N shared frames, in the global frame and in
    the new window's frame. Scale comes from the spread of the camera centres (needs N >= 2, else 1.0); rotation is
    the chordal mean of the per-camera relative rotations; translation aligns the centroids of the centres.
    """
    cg, cw = c2w_global[:, :3, 3], c2w_window[:, :3, 3]
    n = len(cg)
    if n >= 2:
        dg = np.linalg.norm(cg - cg.mean(0), axis=1).mean()
        dw = np.linalg.norm(cw - cw.mean(0), axis=1).mean()
        s = float(dg / dw) if dw > 1e-6 and dg > 1e-6 else 1.0
    else:
        s = 1.0
    rel = np.einsum("nij,nkj->nik", c2w_global[:, :3, :3], c2w_window[:, :3, :3])   # R_g @ R_w^T
    r = chordal_mean_rotation(rel)
    t = cg.mean(0) - s * (r @ cw.mean(0))
    return s, r, t


def apply_sim3(s: float, r: np.ndarray, t: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """A camera-to-world matrix [4,4] (or [N,4,4]) from the window frame into the global frame."""
    out = np.array(c2w, dtype=np.float64, copy=True)
    out[..., :3, :3] = np.einsum("ij,...jk->...ik", r, c2w[..., :3, :3])
    out[..., :3, 3] = s * np.einsum("ij,...j->...i", r, c2w[..., :3, 3]) + t
    return out


def apply_sim3_points(s: float, r: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points [..., 3] from the window frame into the global frame."""
    return s * np.einsum("ij,...j->...i", r, pts) + t
