"""Object-level cognitive map: grounding-output parsing, 3D lifting with a point map, cross-frame merging."""

import numpy as np
import torch

from live3r.geometry.base import GeomOutput
from live3r.serve.object_map import ObjectMap, detection_prompt, parse_detections


def test_parse_normal_truncated_reversed_and_whole_image():
    text = ('```json\n[\n {"bbox_2d": [100, 200, 300, 400], "label": "Chair"},\n'
            ' {"label": "tv", "bbox_2d": [500, 100, 700, 300]},\n'
            ' {"bbox_2d": [0, 0, 1000, 990], "label": "bed"},\n'         # whole image -> dropped
            ' {"bbox_2d": [10, 10, 60')                                   # truncated -> ignored
    dets = parse_detections(text)
    labels = sorted(d["label"] for d in dets)
    assert labels == ["chair", "tv"]
    chair = next(d for d in dets if d["label"] == "chair")
    assert chair["bbox"] == [0.1, 0.2, 0.3, 0.4]


def _geom(points_fn, h=40, w=40):
    """Point map where pixel (r, c) -> points_fn(r/h, c/w)."""
    rr, cc = np.meshgrid(np.arange(h) / h, np.arange(w) / w, indexing="ij")
    pm = np.stack(points_fn(rr, cc)).astype(np.float32)
    return GeomOutput(tokens={}, grid_hw=(1, 1), pointmap=torch.from_numpy(pm).unsqueeze(0),
                      conf=torch.ones(1, 1, h, w))


def test_lift_merge_and_text():
    # frame 1: the left half of the image is 1 m left, the right half 1 m right, both 2 m ahead
    g = _geom(lambda r, c: (np.where(c < 0.5, -1.0, 1.0), np.zeros_like(r), np.full_like(r, 2.0)))
    om = ObjectMap()
    assert om.add(1, [{"bbox": [0.05, 0.3, 0.4, 0.7], "label": "chair"},
                      {"bbox": [0.6, 0.3, 0.95, 0.7], "label": "tv"}], g) == 2
    # frame 3: the same chair again (merged), plus a second chair 3 m away (new instance)
    g2 = _geom(lambda r, c: (np.where(c < 0.5, -1.05, 2.0), np.zeros_like(r), np.where(c < 0.5, 2.0, 4.5)))
    om.add(3, [{"bbox": [0.05, 0.3, 0.4, 0.7], "label": "chair"},
               {"bbox": [0.6, 0.3, 0.95, 0.7], "label": "chair"}], g2)
    inst = om.instances()
    chairs = [i for i in inst if i["label"] == "chair"]
    assert len(chairs) == 2 and sorted(i["n_obs"] for i in chairs) == [1, 2]
    text = om.text(n_frames=32)
    assert "- chair: 2 found" in text and "- tv: 1 found" in text
    assert text.index("- chair") < text.index("- tv") or "first seen in frame 1" in text
    assert "chair-tv 2.0 m" in text          # nearest chair to the tv, 2 m apart (x: -1 -> +1)


def test_detection_prompt_lists_vocab():
    p = detection_prompt(["chair", "tv"])
    assert "chair, tv" in p and "bbox_2d" in p


def _pts_obs(om, label, frame, center, spread=0.1, n=50, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.asarray(center, np.float32) + rng.uniform(-spread, spread, (n, 3)).astype(np.float32)
    om._observe(label, np.median(pts, axis=0), frame, pts)


def test_lookup_synonyms_plural_and_min_frames():
    from live3r.serve.object_map import canon

    assert canon("Trash Can") == "trash bin" and canon("chairs") == "chair" and canon("glass") == "glass"
    om = ObjectMap()
    _pts_obs(om, canon("couch"), 1, (0, 0, 2))
    _pts_obs(om, canon("sofa"), 4, (0.1, 0, 2.1), seed=1)       # same sofa from another keyframe
    _pts_obs(om, "lamp", 2, (3, 0, 2))                          # seen once -> not trusted
    assert len(om.lookup("sofas")) == 1 and om.lookup("lamp") == [] and len(om.lookup("lamp", min_frames=1)) == 1


def test_closest_point_distance_and_absent_objects():
    om = ObjectMap()
    for f in (1, 2):
        _pts_obs(om, "table", f, (0, 0, 2), spread=0.5, seed=f)     # a 1 m wide table centred at x=0
        _pts_obs(om, "chair", f, (1.5, 0, 2), spread=0.2, seed=f + 5)
    d = om.pair_distance("table", "chair")
    assert 0.6 < d < 1.0, d          # closest points ~0.8 m apart (centres are 1.5 m apart)
    assert om.pair_distance("table", "piano") is None


def test_size_aware_merge_keeps_a_big_object_as_one_instance():
    om = ObjectMap(merge_dist=0.8)
    _pts_obs(om, "bed", 1, (0, 0, 2), spread=1.0, seed=1)      # 2 m wide bed seen whole
    _pts_obs(om, "bed", 2, (1.1, 0, 2), spread=0.3, seed=2)    # later a partial view, centroid 1.1 m away
    assert len(om.lookup("bed")) == 1


def test_size_from_full_box_not_inner_region():
    """I-30: extent uses the full box (the inner region clips it); percentiles reject background in the box."""
    om = ObjectMap()
    rng = np.random.default_rng(0)
    for f in (1, 2, 3):
        inner = np.stack([rng.uniform(-0.25, 0.25, 60), rng.uniform(-0.1, 0.1, 60), 2 + rng.uniform(-0.1, 0.1, 60)], 1)
        box = np.stack([rng.uniform(-1.0, 1.0, 120), rng.uniform(-0.2, 0.2, 120), 2 + rng.uniform(-0.2, 0.2, 120)], 1)
        far = np.array([[0.0, 0.0, 9.0]] * 2)                       # background seen through/behind the object
        om._observe("table", np.median(inner, 0), f, inner.astype(np.float32),
                    np.concatenate([box, far]).astype(np.float32))
    size = om.size_m("table", min_frames=3)
    assert 1.7 < size < 2.3, size            # true extent 2.0 m along x, not the 0.5 m inner region
    assert om.size_m("table", min_frames=5) is None



def test_view_gap_counts_only_time_separated_observations():
    """At ~1 fps three consecutive detection frames are one view, not three (2026-09-27)."""
    from live3r.serve.object_map import ObjectMap

    om = ObjectMap(view_gap=30)
    inst = {"label": "chair", "frames": {100, 101, 102, 140, 171, 172}}
    assert om.n_views(inst) == 3                         # 100 | 140 | 171 (101, 102, 172 are within the gap)
    assert ObjectMap(view_gap=0).n_views(inst) == 6      # the keyframe convention: every frame is a view
    om._inst.append(dict(inst, obs=[], pts=[], box_pts=[], first=100))
    assert om.lookup("chair", min_frames=3) and not om.lookup("chair", min_frames=4)


def test_observation_log_keeps_every_lifted_detection_with_its_score():
    """I-49: with log=True every lifted observation is kept as it came - label (canonical), frame, detector score,
    position, a point sample of at most 64 - merged or not; without it nothing is kept."""
    g = _geom(lambda r, c: (np.where(c < 0.5, -1.0, 1.0), np.zeros_like(r), np.full_like(r, 2.0)))
    dets = [{"bbox": [0.05, 0.3, 0.4, 0.7], "label": "couch", "score": 0.4},
            {"bbox": [0.6, 0.3, 0.95, 0.7], "label": "tv"}]
    om = ObjectMap(log=True)
    om.add(5, dets, g)
    om.add(9, dets[:1], g)                       # the same sofa again: merged, but logged
    assert [(o["label"], o["frame"]) for o in om.log] == [("sofa", 5), ("tv", 5), ("sofa", 9)]
    assert om.log[0]["score"] == 0.4 and np.isnan(om.log[1]["score"])
    assert np.allclose(om.log[0]["xyz"], [-1.0, 0.0, 2.0]) and 0 < len(om.log[0]["pts"]) <= 64
    assert len([i for i in om.instances() if i["label"] == "sofa"]) == 1
    assert ObjectMap().log is None


def test_pick_and_pair_distance_follow_the_instance_rule():
    """I-49: two tvs - one seen in three frames at score 0.3, one seen once at 0.9. most_views picks the first and
    pair_distance keeps the nearest pair; score_max picks the confident one for both."""
    om = {r: ObjectMap(instance_rule=r) for r in ("most_views", "score_max")}
    for m in om.values():
        for f in (1, 2, 3):
            m._observe("tv", np.array([0.0, 0.0, 0.0]), f, np.zeros((1, 3), np.float32), score=0.3)
        m._observe("tv", np.array([4.0, 0.0, 0.0]), 4, np.full((1, 3), [4.0, 0, 0], np.float32), score=0.9)
        m._observe("sofa", np.array([1.0, 0.0, 0.0]), 1, np.full((1, 3), [1.0, 0, 0], np.float32), score=0.5)
    assert np.allclose(np.median(om["most_views"].pick("tv", 1)["obs"], axis=0), [0, 0, 0])
    assert np.allclose(np.median(om["score_max"].pick("tv", 1)["obs"], axis=0), [4, 0, 0])
    assert abs(om["most_views"].pair_distance("tv", "sofa", 1) - 1.0) < 1e-6
    assert abs(om["score_max"].pair_distance("tv", "sofa", 1) - 3.0) < 1e-6
    assert om["score_max"].pick("lamp", 1) is None
