"""Camera-height scale anchor (live3r.geometry.scale_anchor): a uniform inflation of the encoder's output is undone
at the output, angles untouched."""

import numpy as np
import pytest

from live3r.geometry.scale_anchor import CameraHeightAnchor


def frame(scale: float, cam_height: float = 1.5, rng=None):
    """A camera at `cam_height` above a floor, looking forward and slightly down; output inflated by `scale`.
    World: y is down (OpenCV, the first camera's frame), so the floor is at y = +cam_height."""
    rng = rng or np.random.default_rng(0)
    c2w = np.eye(4)
    c2w[:3, 1] = [0, 1, 0]                       # camera down = world down
    c2w[:3, 3] = [rng.normal(0, 0.05), 0.0, rng.uniform(0, 3)] if scale == 1.0 else [0.5, 0.0, 2.0]
    floor = np.stack([rng.uniform(-3, 3, 4000), np.full(4000, cam_height), rng.uniform(0, 6, 4000)], 1)
    wall = np.stack([rng.uniform(-3, 3, 1000), rng.uniform(-1.0, cam_height, 1000), np.full(1000, 6.0)], 1)
    pts = np.concatenate([floor, wall])
    c2w_s = c2w.copy()
    c2w_s[:3, 3] *= scale
    return c2w_s, pts * scale, np.ones(len(pts))


def test_reference_is_built_then_inflation_is_undone():
    a = CameraHeightAnchor(ref_frames=5, window=3)
    for _ in range(5):
        assert a.update(*frame(1.0)) == 1.0
    assert a.reference == pytest.approx(1.5, rel=0.02)
    # the encoder starts inflating: 1.2x, then 1.4x
    for s in (1.2, 1.2, 1.2):
        f = a.update(*frame(s))
    assert f == pytest.approx(1 / 1.2, rel=0.03)
    for s in (1.4, 1.4, 1.4):
        f = a.update(*frame(s))
    assert f == pytest.approx(1 / 1.4, rel=0.03)
    c2w, pts, _ = frame(1.4)
    c2w2, pts2 = a.apply(f, c2w, pts)
    assert np.allclose(c2w2[:3, :3], c2w[:3, :3])                 # rotation untouched
    assert c2w2[:3, 3] == pytest.approx(c2w[:3, 3] / 1.4, rel=0.03)
    assert np.percentile(pts2[:, 1], 98) - c2w2[1, 3] == pytest.approx(1.5, rel=0.05)   # y is down: the floor is the high end


def test_frames_without_a_floor_do_not_move_the_estimate():
    a = CameraHeightAnchor(ref_frames=3, window=3)
    for _ in range(3):
        a.update(*frame(1.0))
    f_before = a.update(*frame(1.3))
    # a frame with too few points (looking into a wall / dark): factor unchanged
    c2w, _, _ = frame(1.3)
    assert a.update(c2w, np.zeros((50, 3)), np.ones(50)) == f_before


def test_factor_is_clamped():
    a = CameraHeightAnchor(ref_frames=2, window=1, clamp=(0.5, 2.0))
    for _ in range(2):
        a.update(*frame(1.0))
    assert a.update(*frame(5.0)) == pytest.approx(0.5)
