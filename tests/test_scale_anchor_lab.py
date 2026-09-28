"""scripts/scale_anchor_lab.py: replaying anchors on a captured raw stream reproduces the live pipeline's scene map
(none) and lets a variant be scored against the room area."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scale_anchor_lab as lab  # noqa: E402

from live3r.geometry.scale_anchor import ScaleAnchor  # noqa: E402


def synthetic_capture(n_frames: int = 24, drift: float = 1.4, seed: int = 0) -> dict:
    """A 4 x 6 m room, camera 1.5 m up, walking along the room; the output's scale grows linearly to `drift`."""
    rng = np.random.default_rng(seed)
    idx, key, c2w, pts, conf = [], [], [], [], []
    for f in range(n_frames):
        s = 1.0 + (drift - 1.0) * f / (n_frames - 1)
        m = np.eye(4)
        m[:3, 1] = [0, 1, 0]
        m[:3, 3] = np.array([0.0, 0.0, 6.0 * f / n_frames]) * s
        floor = np.stack([rng.uniform(-2, 2, 3000), np.full(3000, 1.5), rng.uniform(0, 6, 3000)], 1)
        walls = np.stack([rng.choice([-2.0, 2.0], 600), rng.uniform(-1.0, 1.5, 600), rng.uniform(0, 6, 600)], 1)
        p = np.concatenate([floor, walls]) * s
        idx.append(f * 30)
        key.append(f % 3 == 0)
        c2w.append(m.astype(np.float32))
        pts.append(p.astype(np.float32))
        conf.append(rng.uniform(1, 5, len(p)).astype(np.float32))
    return {"idx": np.array(idx), "key": np.array(key), "c2w": np.stack(c2w), "pts": np.stack(pts),
            "conf": np.stack(conf), "meta": {"dataset": "syn", "scene": "room"}}


def test_none_replay_matches_a_directly_fed_scene_map():
    cap = synthetic_capture(drift=1.0)
    facts, _ = lab.replay(cap, None)
    assert facts["area_m2"] == pytest.approx(4 * 6, rel=0.08)      # 2-98 percentiles of a uniform fill
    assert 2.0 < facts["height_m"] < 2.6                           # walls span 2.5 m; the 98th percentile sits below the top


def test_floor_level_anchor_removes_the_drift_in_replay():
    cap = synthetic_capture(drift=1.4)
    drifted, _ = lab.replay(cap, None)
    assert drifted["area_m2"] > 1.3 * 24                         # the inflated frames widen the map
    fixed, anchor = lab.replay(cap, ScaleAnchor(mode="floor_level", ref_frames=4, window=3))
    assert fixed["area_m2"] == pytest.approx(24, rel=0.12)
    assert anchor.factor == pytest.approx(1 / 1.4, rel=0.05)


def test_capture_round_trip_and_variant_parsing(tmp_path):
    cap = synthetic_capture(n_frames=6)
    np.savez_compressed(tmp_path / "syn__room.npz", idx=cap["idx"], key=cap["key"], c2w=cap["c2w"], pts=cap["pts"],
                        conf=cap["conf"], meta=json.dumps(cap["meta"]))
    back = lab.load_capture(tmp_path / "syn__room.npz")
    assert back["meta"]["scene"] == "room" and back["pts"].shape == cap["pts"].shape
    assert lab.parse_variant("mode=floor_level,window=9,support=0.05") == {"mode": "floor_level", "window": 9,
                                                                            "support": 0.05}
    assert lab.make_anchor(lab.variants("smoke")["v1 camera_height (chain 13)"]).min_ref == 0.0
    assert lab.make_anchor(None) is None
