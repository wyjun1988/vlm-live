"""Windowed CUT3R state (I-36): the reset bookkeeping keeps every window in one metric global frame."""

import numpy as np
import pytest
import torch

from live3r.geometry.base import GeomOutput
from live3r.geometry.cut3r import ResetChainer
from tests.test_sim3 import rot


def c2w(r, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


def test_reset_chainer_realigns_a_new_window_rigidly_and_transforms_outputs():
    ch = ResetChainer(reset_every=3, overlap=2)
    # three frames in the first window (= the global frame)
    truth = [c2w(rot([0, 1, 0], 10 * i), [0.5 * i, 0.0, 0.1 * i]) for i in range(6)]
    for i in range(3):
        out = GeomOutput(tokens={}, grid_hw=(1, 1), pointmap=torch.zeros(1, 3, 2, 2),
                         extra={"c2w": torch.from_numpy(truth[i]).float()[None]})
        assert not ch.due() or i == 3
        ch.apply(out)
        ch.remember(torch.full((1, 3, 4, 4), float(i)), out.extra["c2w"][0].numpy())
    assert ch.due() and len(ch.overlap_frames()) == 2 and float(ch.overlap_frames()[0][0, 0, 0, 0]) == 1.0
    # the new window expresses everything in a rotated, shifted frame - and 1.3x too big (a scale error that a
    # rigid alignment must NOT absorb: each window is metric on its own)
    r, t = rot([0.2, 1, 0], 35), np.array([2.0, -1.0, 0.5])
    to_window = lambda m: c2w(r.T @ m[:3, :3], r.T @ (m[:3, 3] - t))       # noqa: E731
    ch.rebase([to_window(truth[1]), to_window(truth[2])])
    assert ch.resets == 1 and ch.since_reset == 0
    new = GeomOutput(tokens={}, grid_hw=(1, 1), pointmap=torch.tensor(r.T @ (np.array([1.0, 2.0, 3.0]) - t),
                                                                       dtype=torch.float32).view(1, 3, 1, 1),
                     extra={"c2w": torch.from_numpy(to_window(truth[3])).float()[None]})
    ch.apply(new)
    assert np.allclose(new.extra["c2w"][0].numpy(), truth[3], atol=1e-5)
    assert np.allclose(new.pointmap.view(3).numpy(), [1.0, 2.0, 3.0], atol=1e-5)


def test_reset_chainer_is_not_due_before_the_window_fills_or_without_an_overlap_frame():
    ch = ResetChainer(reset_every=2, overlap=1)
    assert not ch.due()                     # nothing remembered yet
    out = GeomOutput(tokens={}, grid_hw=(1, 1), extra={"c2w": torch.eye(4)[None]})
    ch.apply(out)
    ch.remember(torch.zeros(1, 3, 2, 2), np.eye(4))
    assert not ch.due()
    ch.apply(GeomOutput(tokens={}, grid_hw=(1, 1), extra={"c2w": torch.eye(4)[None]}))
    assert ch.due()


def _stream(apply: str):
    """A CUT3RStream without its network: only the anchor bookkeeping of `_anchored` / `metric_scale` is under test."""
    from live3r.geometry.cut3r import CUT3RStream
    from live3r.geometry.scale_anchor import ScaleAnchor

    s = CUT3RStream.__new__(CUT3RStream)
    s.anchor = ScaleAnchor(mode="floor_level", ref_frames=2, window=1)
    s.anchor_apply = apply
    return s


def _floor_frame(scale: float) -> GeomOutput:
    rng = np.random.default_rng(0)
    floor = np.stack([rng.uniform(-3, 3, 4000), np.full(4000, 1.5), rng.uniform(0, 6, 4000)], 1)
    wall = np.stack([rng.uniform(-3, 3, 1000), rng.uniform(-1.0, 1.5, 1000), np.full(1000, 6.0)], 1)
    pts = np.concatenate([floor, wall]) * scale
    m = np.eye(4)
    m[:3, 3] = [0.5 * scale, 0.0, 2.0 * scale]
    return GeomOutput(tokens={}, grid_hw=(1, 1), pointmap=torch.from_numpy(pts.T.reshape(1, 3, 50, 100).astype(np.float32)),
                      conf=torch.ones(1, 1, 50, 100), extra={"c2w": torch.from_numpy(m.astype(np.float32))[None]})


def test_global_anchor_mode_leaves_the_geometry_alone_and_exposes_one_metric_scale():
    s = _stream("global")
    assert s.metric_scale == 1.0
    for _ in range(2):
        s._anchored(_floor_frame(1.0))
    out = s._anchored(_floor_frame(1.4))
    assert out.extra["scale_factor"] == pytest.approx(1 / 1.4, rel=0.03)
    assert out.extra["c2w"][0, 2, 3].item() == pytest.approx(2.0 * 1.4)          # untouched
    assert out.pointmap[0, 1].max().item() == pytest.approx(1.5 * 1.4, rel=0.01)  # untouched
    assert s.metric_scale == pytest.approx(1 / 1.4, rel=0.03) and out.extra["metric_scale"] == s.metric_scale


def test_per_frame_anchor_mode_rescales_the_output():
    s = _stream("per_frame")
    for _ in range(2):
        s._anchored(_floor_frame(1.0))
    out = s._anchored(_floor_frame(1.4))
    assert out.extra["c2w"][0, 2, 3].item() == pytest.approx(2.0, rel=0.03)
    assert out.pointmap[0, 1].max().item() == pytest.approx(1.5, rel=0.03)


def test_anchored_passes_the_self_view_points_to_an_estimator_that_reads_them():
    from live3r.geometry.cut3r import CUT3RStream
    from live3r.geometry.scale_anchor import SelfViewScale

    s = CUT3RStream.__new__(CUT3RStream)
    s.anchor, s.anchor_apply = SelfViewScale(relative=False, window=1), "global"
    out = _floor_frame(1.3)
    world = out.pointmap[0].permute(1, 2, 0).numpy().reshape(-1, 3).astype(np.float64)
    c2w = out.extra["c2w"][0].numpy().astype(np.float64)
    cam = (world - c2w[:3, 3]) @ c2w[:3, :3] / 1.3                     # the camera-frame points at true scale
    out.extra["pointmap_self"] = torch.from_numpy(cam.T.reshape(1, 3, 50, 100).astype(np.float32))
    out.extra["conf_self"] = torch.ones(1, 1, 50, 100)
    s._anchored(out)
    assert s.metric_scale == pytest.approx(1 / 1.3, rel=1e-3) and out.extra["metric_scale"] == s.metric_scale
    assert out.extra["c2w"][0, 2, 3].item() == pytest.approx(2.0 * 1.3)   # global mode: geometry untouched
