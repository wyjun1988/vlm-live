"""The windowed encoder's bookkeeping (live3r.geometry.vggt_window) - without the model.

Each window comes with its own frame and scale; the chainer must recover one consistent global trajectory from the
frames consecutive windows share. The model itself is only exercised by the bake-off run.
"""

import numpy as np
import pytest

from live3r.geometry.vggt_window import WindowChainer, extrinsics_to_c2w
from tests.test_sim3 import rot, trajectory


def express_in_window(c2w_global: np.ndarray, s: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """X_w = R^T (X_g - t) / s applied to camera-to-world matrices."""
    w = c2w_global.copy()
    w[:, :3, :3] = np.einsum("ij,njk->nik", r.T, c2w_global[:, :3, :3])
    w[:, :3, 3] = np.einsum("ij,nj->ni", r.T, c2w_global[:, :3, 3] - t) / s
    return w


def test_chainer_recovers_the_trajectory_from_windows_with_arbitrary_frames_and_scales():
    rng = np.random.default_rng(0)
    truth = trajectory(20, rng)                       # the global frame = the first window's frame
    truth = np.einsum("ij,njk->nik", np.linalg.inv(truth[0]), truth)   # so frame 0 is the identity
    chain = WindowChainer()
    W = 6
    got = []
    for k in range(len(truth)):
        ids = list(range(max(0, k - W + 1), k + 1))
        if k == 0:
            s, r, t = 1.0, np.eye(3), np.zeros(3)      # VGGT: the first image of the set is the identity
        else:
            s, r, t = rng.uniform(0.5, 2.0), rot(rng.normal(size=3), rng.uniform(0, 180)), rng.normal(size=3)
        window = express_in_window(truth[ids], s, r, t)
        window = np.einsum("ij,njk->nik", np.linalg.inv(window[0]), window)   # VGGT anchors on the window's first frame
        new, _ = chain.update(ids, window)
        got.append(new)
    got = np.stack(got)
    # The global scale is unobservable until a window shares two frames with the past: the first two-frame window
    # defines it (VGGT's scale is normalised anyway). Compare after fixing that one global factor; rotations exact.
    k = np.linalg.norm(truth[1, :3, 3]) / np.linalg.norm(got[1, :3, 3])
    assert np.allclose(k * got[:, :3, 3], truth[:, :3, 3], atol=1e-6)
    assert np.allclose(got[:, :3, :3], truth[:, :3, :3], atol=1e-6)
    assert set(chain.global_c2w) == set(range(20 - W, 20)), "only the window's frames are kept"


def test_chainer_reset_forgets_the_frame():
    chain = WindowChainer()
    chain.update([0], np.eye(4)[None])
    chain.reset()
    assert chain.global_c2w == {} and chain.last_sim3 is None


def test_extrinsics_to_c2w_inverts_world_to_camera():
    r = rot([0.3, 1, 0.2], 50)
    t = np.array([1.0, 2.0, 3.0])
    extri = np.concatenate([r, t[:, None]], axis=1)[None]      # world -> camera
    c2w = extrinsics_to_c2w(extri)[0]
    x_w = np.array([0.5, -1.0, 2.0, 1.0])
    x_c = np.append(r @ x_w[:3] + t, 1.0)
    assert np.allclose(c2w @ x_c, x_w, atol=1e-9)
