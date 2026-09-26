"""Scoring of generated answers to the geometry questions (scripts/geomqa_generate_eval.py)."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("geomqa_generate_eval", Path(__file__).parent.parent / "scripts" /
                                              "geomqa_generate_eval.py")
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)


@pytest.mark.parametrize("kind,pred,truth,expected", [
    ("camera_turn", "left", "left", 1.0),
    ("camera_turn", "The camera turned left.", "left", 1.0),
    ("camera_turn", "left or right", "left", 0.0),            # hedging is not an answer
    ("camera_turn", "the same way", "the same way", 1.0),
    ("camera_turn", "right", "the same way", 0.0),
    ("closest_image", "Image 3", "image 3", 1.0),
    ("closest_image", "image 2", "image 3", 0.0),
    ("closest_image", "3", "image 3", 0.0),
    ("camera_displacement", "2.1", "2.0", 0.9),                # 5% off: 9 of 10 thresholds pass
    ("camera_displacement", "3.0 m", "2.0", 0.0),              # 50% off: every threshold fails
    ("room_area", "about 18 square meters", "20", 0.8),        # 10% off: 8 of 10 thresholds pass
    ("path_length", "I cannot tell", "4.5", 0.0),
])
def test_score(kind, pred, truth, expected):
    assert ge.score(kind, pred, truth) == pytest.approx(expected)


def test_mra_matches_vsi_definition():
    assert ge.mra(1.0, 1.0) == 1.0 and ge.mra(1.5, 1.0) == 0.0 and ge.mra(1.2, 1.0) == pytest.approx(0.6)
    assert ge.first_number("about 1,250 cm") == 1250.0 and ge.first_number("none") is None
