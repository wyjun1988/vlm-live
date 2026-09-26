"""Chaining overlapping windows of a non-recurrent encoder into one frame (live3r.geometry.sim3)."""

import numpy as np
import pytest

from live3r.geometry.sim3 import apply_sim3, apply_sim3_points, chordal_mean_rotation, sim3_from_shared_cameras


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def trajectory(n, rng):
    """n cameras walking a bent path with varied orientations."""
    out = []
    for i in range(n):
        c2w = np.eye(4)
        c2w[:3, :3] = rot([0, 1, 0], 15 * i) @ rot([1, 0, 0], 5 * np.sin(i))
        c2w[:3, 3] = [0.6 * i, 0.1 * np.sin(i), 0.4 * np.cos(i)] + rng.normal(0, 0.02, 3)
        out.append(c2w)
    return np.stack(out)


def test_sim3_recovers_a_known_similarity_from_shared_cameras():
    rng = np.random.default_rng(0)
    g = trajectory(7, rng)
    s, r, t = 2.3, rot([0.2, 1.0, 0.3], 40), np.array([1.0, -2.0, 0.5])
    # the window expresses the same cameras in its own frame: X_w = R^T (X_g - t) / s
    w = g.copy()
    w[:, :3, :3] = np.einsum("ij,njk->nik", r.T, g[:, :3, :3])
    w[:, :3, 3] = np.einsum("ij,nj->ni", r.T, g[:, :3, 3] - t) / s
    s2, r2, t2 = sim3_from_shared_cameras(g, w)
    assert s2 == pytest.approx(s, rel=1e-6)
    assert np.allclose(r2, r, atol=1e-8) and np.allclose(t2, t, atol=1e-8)
    assert np.allclose(apply_sim3(s2, r2, t2, w), g, atol=1e-8)
    pts_w = rng.normal(size=(5, 3))
    assert np.allclose(apply_sim3_points(s2, r2, t2, pts_w), s * pts_w @ r.T + t)


def test_sim3_with_one_shared_camera_keeps_scale_and_uses_its_rotation():
    g = trajectory(1, np.random.default_rng(1))
    r = rot([0, 0, 1], 90)
    w = g.copy()
    w[:, :3, :3] = np.einsum("ij,njk->nik", r.T, g[:, :3, :3])
    w[:, :3, 3] = np.einsum("ij,nj->ni", r.T, g[:, :3, 3])
    s2, r2, t2 = sim3_from_shared_cameras(g, w)
    assert s2 == 1.0 and np.allclose(r2, r, atol=1e-8) and np.allclose(t2, 0, atol=1e-8)


def test_chordal_mean_is_a_proper_rotation_between_its_inputs():
    a, b = rot([0, 1, 0], 10), rot([0, 1, 0], 30)
    m = chordal_mean_rotation(np.stack([a, b]))
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-9) and np.linalg.det(m) == pytest.approx(1.0)
    assert np.allclose(m, rot([0, 1, 0], 20), atol=1e-9)
