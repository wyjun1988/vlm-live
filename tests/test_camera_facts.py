"""Camera motion between two views as text (live3r.serve.camera_facts) and its routing in the lmms-eval wrapper."""

import math

import numpy as np

from live3r.serve.camera_facts import camera_motion_from_images, camera_motion_text, relative_pose


def _pose(t=(0, 0, 0), yaw_deg=0.0, pitch_deg=0.0):
    """A camera-to-world matrix: first turn by yaw about the world's y (down) axis, tilt by pitch, then translate."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])   # +pitch: forward tilts up (-y)
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    m[:3, 3] = t
    return m


def test_relative_pose_directions_and_turn():
    a = _pose()
    r = relative_pose(a, _pose(t=(1.0, 0, 0)))
    assert abs(r["right"] - 1.0) < 1e-9 and abs(r["forward"]) < 1e-9 and abs(r["yaw_deg"]) < 1e-9
    r = relative_pose(a, _pose(t=(0, 0, 2.0), yaw_deg=30))
    assert abs(r["forward"] - 2.0) < 1e-9 and abs(r["yaw_deg"] - 30) < 1e-6          # +yaw = turned right
    r = relative_pose(a, _pose(yaw_deg=-45))
    assert abs(r["yaw_deg"] + 45) < 1e-6 and r["distance"] < 1e-9
    r = relative_pose(a, _pose(pitch_deg=20))
    assert abs(r["pitch_deg"] - 20) < 1e-6                                           # +pitch = tilted up
    # the relation is relative: the same motion from a moved, turned camera reads the same
    a2 = _pose(t=(3, -1, 5), yaw_deg=120)
    b2 = a2 @ _pose(t=(1.0, 0, 2.0), yaw_deg=30)
    r = relative_pose(a2, b2)
    assert abs(r["right"] - 1.0) < 1e-9 and abs(r["forward"] - 2.0) < 1e-9 and abs(r["yaw_deg"] - 30) < 1e-6


def test_camera_motion_text_reads_the_dominant_directions():
    t = camera_motion_text(relative_pose(_pose(), _pose(t=(1.0, 0, 1.5), yaw_deg=35)))
    assert t.startswith("Measured from a 3D reconstruction of the two images: from the first view to the second, ")
    assert "moved to the right and forward (about 1.8 m)" in t and "turned right by about 35 degrees" in t
    t = camera_motion_text(relative_pose(_pose(), _pose(t=(-0.02, 0, 0.01), yaw_deg=-60)))
    assert "stayed in place" in t and "turned left by about 60 degrees" in t
    t = camera_motion_text(relative_pose(_pose(), _pose(t=(0, 0, -1.0), yaw_deg=3)))
    assert "moved backward" in t and "kept its heading" in t and "tilted" not in t
    t = camera_motion_text(relative_pose(_pose(), _pose(t=(0, -0.5, 0), pitch_deg=25)))
    assert "moved up" in t and "tilted up by about 25 degrees" in t


class _FakeGeom:
    """A geometry stream whose poses are the frames' order along x; decode_points must be on for poses."""

    def __init__(self):
        self.decode_points = False
        self.calls = 0

    def reset(self):
        self.calls = 0

    def ingest(self, frames):
        from types import SimpleNamespace

        import torch

        self.calls += 1
        extra = {"c2w": torch.from_numpy(_pose(t=(float(self.calls - 1), 0, 0)))[None]} if self.decode_points else {}
        return SimpleNamespace(extra=extra)


def test_camera_motion_from_images_runs_a_fresh_stream_and_restores_the_encoder():
    geom = _FakeGeom()
    imgs = [np.zeros((48, 64, 3), np.uint8), np.zeros((48, 64, 3), np.uint8), np.zeros((48, 64, 3), np.uint8)]
    rel = camera_motion_from_images(geom, imgs, image_size=64, unit=16, device="cpu")
    assert rel is not None and abs(rel["right"] - 2.0) < 1e-9                       # first to last
    assert geom.decode_points is False and geom.calls == 0                           # restored and reset


def test_wrapper_routes_camera_question_types_only():
    from live3r.eval.lmms_live3r import is_camera_question

    assert is_camera_question({"question_type": "Positional Relationship (Cam.–Cam.)"})
    assert is_camera_question({"question_type": "Motion (Cam.)"})
    assert not is_camera_question({"question_type": "Positional Relationship (Cam.–Obj.)"})
    assert not is_camera_question({"question_type": "Motion (Obj.)"})
    assert not is_camera_question({"question_type": "MSR"}) and not is_camera_question({})
