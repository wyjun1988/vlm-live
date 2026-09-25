"""Geometry-only training questions (I-14) — checked against a trajectory whose answers are known by hand.

The whole point of these questions is that the label cannot be produced without reading the geometry, so the
labels themselves have to be right. A synthetic camera path gives exact ground truth.
"""

import math
import re
import random

import numpy as np
import pytest

from live3r.data.geometry_qa import (
    MIN_CHOICE_MARGIN_M,
    balanced_sample,
    build_questions,
    camera_frame,
    closest_image_questions,
    displacement_questions,
    path_length_question,
    to_record,
    turn_questions,
    yaw_delta,
)


def pose(x=0.0, y=0.0, z=0.0, phi_deg=0.0):
    """OpenCV camera (looks along +z, +y down) at world (x, y, z), turned by phi about the world up axis.

    World frame = the first camera's frame: +x right, +y down, +z forward. Turning toward world +x (to the
    camera's right) is a positive phi.
    """
    p = math.radians(phi_deg)
    c2w = np.eye(4)
    c2w[:3, 0] = (math.cos(p), 0, -math.sin(p))   # camera right
    c2w[:3, 1] = (0, 1, 0)                        # camera down
    c2w[:3, 2] = (math.sin(p), 0, math.cos(p))    # camera forward
    c2w[:3, 3] = (x, y, z)
    return c2w


def test_camera_frame_is_gravity_aligned_and_preserves_distances():
    c2ws = np.stack([pose(0, 0, 0), pose(3, 0, 4), pose(0, -1.5, 0, phi_deg=90)])
    pos, yaw, R = camera_frame(c2ws)
    # a rotation cannot change distances
    assert np.isclose(np.linalg.norm(pos[1] - pos[0]), 5.0)
    # the world "up" (-y here) must become the map's +z
    assert np.isclose(pos[2, 2] - pos[0, 2], 1.5, atol=1e-6)
    # turning toward the camera's right decreases yaw
    assert yaw_delta(yaw[0], yaw[2]) < 0


@pytest.mark.parametrize("phi,expected", [(90, "right"), (-90, "left"), (5, "the same way")])
def test_turn_direction(phi, expected):
    c2ws = np.stack([pose(0, 0, 0), pose(0, 0, 1, phi_deg=phi)])
    pos, yaw, _ = camera_frame(c2ws)
    qs = turn_questions(pos, yaw, k=4, rng=random.Random(0))
    assert [q["answer"] for q in qs] == [expected]
    assert "turn left, turn right, or keep facing the same way" in qs[0]["question"]


def test_ambiguous_turn_is_dropped():
    """25 degrees is neither 'the same way' nor a clear turn — teaching it would teach noise."""
    c2ws = np.stack([pose(0, 0, 0), pose(0, 0, 1, phi_deg=25)])
    pos, yaw, _ = camera_frame(c2ws)
    assert turn_questions(pos, yaw, k=4, rng=random.Random(0)) == []


def test_displacement_answers_are_the_true_distances():
    c2ws = np.stack([pose(0, 0, 0), pose(3, 0, 4), pose(0, 0, 10)])
    pos, _, _ = camera_frame(c2ws)
    qs = displacement_questions(pos, k=3, rng=random.Random(0))
    got = {q["question"]: q["answer"] for q in qs}
    assert got["How far apart were the camera positions when image 1 and image 2 were taken, in meters?"] == "5.0"
    assert got["How far apart were the camera positions when image 1 and image 3 were taken, in meters?"] == "10.0"


def test_tiny_displacement_is_dropped():
    c2ws = np.stack([pose(0, 0, 0), pose(0.05, 0, 0)])
    pos, _, _ = camera_frame(c2ws)
    assert displacement_questions(pos, k=3, rng=random.Random(0)) == []


def test_balanced_sample_spreads_the_answers():
    """Without balancing the model can score by always answering the mode (the 905-sample S2 did exactly that)."""
    pairs = [(0, i, 1.0) for i in range(50)] + [(1, i, float(v)) for i, v in enumerate(range(2, 12))]
    got = [v for _, _, v in balanced_sample(pairs, k=8, rng=random.Random(0))]
    assert len(got) == 8
    assert len(set(got)) >= 5, got                 # not all the same value
    assert max(got) >= 8.0                          # the rare large values get picked


def test_closest_image_answer_is_really_the_closest():
    c2ws = np.stack([pose(0, 0, 0), pose(1, 0, 0), pose(9, 0, 0), pose(15, 0, 0), pose(21, 0, 0)])
    pos, _, _ = camera_frame(c2ws)
    qs = closest_image_questions(pos, k=5, rng=random.Random(0), n_options=4)
    assert qs
    for q in qs:
        anchor = int(re.search(r"closest to where image (\d+) was taken", q["question"]).group(1)) - 1
        opts = [int(m) - 1 for m in re.findall(r"image (\d+)", q["question"])][:-1]
        truth = min(opts, key=lambda i: np.linalg.norm(pos[i] - pos[anchor]))
        assert q["answer"] == f"image {truth + 1}", (q["question"], q["answer"])


def test_closest_image_rejects_a_near_tie():
    """From image 1 the other two are 5.00 m and 5.25 m away — too close to call, so no question about it.
    (Anchored on image 3 the same three cameras DO give a clear answer, and that one is still emitted.)"""
    tie = np.stack([pose(0, 0, 0), pose(5, 0, 0), pose(5 + MIN_CHOICE_MARGIN_M / 2, 0, 0)])
    pos, _, _ = camera_frame(tie)
    qs = closest_image_questions(pos, k=5, rng=random.Random(0), n_options=2)
    assert all("closest to where image 1 was taken" not in q["question"] for q in qs), [q["question"] for q in qs]
    assert qs, "the unambiguous anchors should still produce questions"


def test_path_length_follows_the_order():
    c2ws = np.stack([pose(0, 0, 0), pose(3, 0, 0), pose(3, 0, 4)])
    pos, _, _ = camera_frame(c2ws)
    assert path_length_question(pos)[0]["answer"] == "7.0"


def test_build_and_record_format():
    c2ws = np.stack([pose(0, 0, 0), pose(2, 0, 0), pose(2, 0, 3, phi_deg=90),
                     pose(0, 0, 3, phi_deg=180), pose(0, 0, 6, phi_deg=200)])
    qs = build_questions(c2ws, facts={"area_m2": 18.4, "height_m": 2.5}, per_kind=2, seed=0)
    kinds = {q["kind"] for q in qs}
    assert {"camera_displacement", "camera_turn", "path_length", "room_area", "room_height"} <= kinds
    assert next(q for q in qs if q["kind"] == "room_area")["answer"] == "18"

    rec = to_record("vid0_q3", ["a.jpg", "b.jpg"], qs[0])
    assert rec["conversations"][0]["value"].startswith("<image><image>")
    assert rec["conversations"][1]["value"] == qs[0]["answer"]
    assert rec["image"] == ["a.jpg", "b.jpg"] and rec["data_source"] == "geometry_qa"


def test_no_question_is_answerable_without_the_images():
    """Sanity: every question names specific images or this scene — none is answerable from a prior."""
    c2ws = np.stack([pose(0, 0, 0), pose(2, 0, 0), pose(2, 0, 3, phi_deg=90), pose(0, 0, 3, phi_deg=180),
                     pose(0, 0, 6, phi_deg=200)])
    for q in build_questions(c2ws, facts={"area_m2": 18.4, "height_m": 2.5}, seed=1):
        assert ("image" in q["question"]) or ("this room" in q["question"])


@pytest.mark.parametrize("h,emitted", [(2.6, True), (1.7, False), (6.0, False)])
def test_room_height_is_dropped_when_the_ceiling_was_not_captured(h, emitted):
    """Hand-held scans rarely see the ceiling; a 1.7 m 'room height' is the scan's span, not the room's."""
    from live3r.data.geometry_qa import room_questions

    kinds = {q["kind"] for q in room_questions({"area_m2": 18.0, "height_m": h})}
    assert ("room_height" in kinds) is emitted
    assert "room_area" in kinds          # the area is measured from the floor and stays
