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
