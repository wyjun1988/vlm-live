"""Scale anchor (live3r.geometry.scale_anchor): a uniform inflation of the encoder's output is undone at the output,
angles untouched; the floor_level mode ignores the camera's own height; the gate ignores table tops."""

import numpy as np
import pytest

from live3r.geometry.scale_anchor import CameraHeightAnchor, ScaleAnchor, build_anchor


def frame(scale: float, cam_height: float = 1.5, rng=None, cam_y: float = 0.0, table: float | None = None):
    """A camera at height `cam_height` above a floor, looking forward and slightly down; output inflated by `scale`.
    World: y is down (OpenCV, the first camera's frame), so the floor is at y = cam_height (relative to the first
    camera at y = 0). cam_y moves this frame's camera down (crouching); table=t replaces the floor by a table top
    t metres above it (nothing lower in view)."""
    rng = rng or np.random.default_rng(0)
    c2w = np.eye(4)
    c2w[:3, 1] = [0, 1, 0]                       # camera down = world down
    c2w[:3, 3] = [rng.normal(0, 0.05), cam_y, rng.uniform(0, 3)] if scale == 1.0 else [0.5, cam_y, 2.0]
    y_low = cam_height - (table or 0.0)
    floor = np.stack([rng.uniform(-3, 3, 4000), np.full(4000, y_low), rng.uniform(0, 6, 4000)], 1)
    wall = np.stack([rng.uniform(-3, 3, 1000), rng.uniform(-1.0, y_low, 1000), np.full(1000, 6.0)], 1)
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


def test_floor_level_ignores_a_crouching_camera_where_camera_height_is_fooled():
    """The camera drops by 0.5 m (no scale change): the floor's depth below the origin is unchanged, the camera's
    height above the floor is not."""
    fl = ScaleAnchor(mode="floor_level", ref_frames=3, window=1)
    ch = ScaleAnchor(mode="camera_height", ref_frames=3, window=1)
    for _ in range(3):
        fl.update(*frame(1.0))
        ch.update(*frame(1.0))
    assert fl.reference == pytest.approx(1.5, rel=0.02)
    crouch = frame(1.0, cam_y=0.5)
    assert fl.update(*crouch) == pytest.approx(1.0, abs=0.03)
    assert ch.update(*crouch) == pytest.approx(1.5, rel=0.05)      # reads the lower camera as a 1/1.5 shrink
    # and floor_level still undoes a real inflation
    for _ in range(2):
        f = fl.update(*frame(1.3))
    assert f == pytest.approx(1 / 1.3, rel=0.03)


def test_gate_skips_a_table_top_frame():
    """Only a table top 0.7 m above the floor is in view: ungated, the anchor reads a lower camera and inflates;
    with a 25% gate the frame has no points near the predicted floor and is skipped."""
    plain = ScaleAnchor(mode="floor_level", ref_frames=3, window=1)
    gated = ScaleAnchor(mode="floor_level", ref_frames=3, window=1, gate=0.25, support=0.02)
    for _ in range(3):
        plain.update(*frame(1.0))
        gated.update(*frame(1.0))
    table = frame(1.0, table=0.7)
    assert plain.update(*table) == pytest.approx(1.5 / 0.8, rel=0.05)
    assert gated.update(*table) == 1.0
    assert gated.skipped == 1
    # a floor frame after the table: measured again
    assert gated.update(*frame(1.2)) == pytest.approx(1 / 1.2, rel=0.03)


def test_support_skips_a_frame_whose_low_tail_is_not_a_surface():
    a = ScaleAnchor(mode="camera_height", ref_frames=2, window=1, support=0.1)
    for _ in range(2):
        a.update(*frame(1.0))
    rng = np.random.default_rng(1)
    c2w = np.eye(4)
    c2w[:3, 1] = [0, 1, 0]
    # a wall only: y spread uniformly, no concentration at any level -> the 8% band holds far fewer than 10%
    wall = np.stack([rng.uniform(-3, 3, 3000), rng.uniform(-1.0, 1.5, 3000), np.full(3000, 6.0)], 1)
    assert a.update(c2w, wall, np.ones(len(wall))) == 1.0
    assert a.skipped == 1


def test_reference_too_small_disables_the_anchor():
    a = ScaleAnchor(mode="floor_level", ref_frames=2, window=1, min_ref=0.3)
    for _ in range(2):
        a.update(*frame(1.0, cam_height=0.1))
    assert a.disabled
    assert a.update(*frame(1.5, cam_height=0.1)) == 1.0


def test_build_anchor_from_config_values():
    assert build_anchor(None) is None
    assert build_anchor("none") is None
    assert build_anchor("floor_level").mode == "floor_level"
    a = build_anchor({"mode": "floor_level", "window": 9, "gate": 0.25})
    assert (a.window, a.gate) == (9, 0.25)
    with pytest.raises(ValueError):
        build_anchor("altitude")
    with pytest.raises(ValueError):
        build_anchor(3)


def test_global_factor_is_the_geometric_mean_of_the_per_frame_factors():
    a = ScaleAnchor(mode="floor_level", ref_frames=3, window=1)
    assert a.global_factor() == 1.0
    for _ in range(3):
        a.update(*frame(1.0))
    assert a.global_factor() == 1.0                              # no history yet, only the reference
    for s in (1.2, 1.5):
        a.update(*frame(s))
    assert a.history == pytest.approx([1 / 1.2, 1 / 1.5], rel=0.03)
    assert a.global_factor() == pytest.approx((1 / 1.2 * 1 / 1.5) ** 0.5, rel=0.03)
