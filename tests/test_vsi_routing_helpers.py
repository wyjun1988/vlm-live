"""Question-time routing helpers of scripts/eval_vsi_local.py (I-32 appearance order, counts) and the thinking
probe's answer split - pure functions, tested with a fake object map."""

import importlib.util
import os
from pathlib import Path

import pytest

from live3r.eval.prefix_cache import split_thinking

spec = importlib.util.spec_from_file_location("eval_vsi_local", Path(__file__).parent.parent / "scripts" /
                                              "eval_vsi_local.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class FakeMap:
    """label -> list of (first_frame, n_frames[, xyz]) instances."""

    def __init__(self, inst):
        self.inst = inst

    def lookup(self, name, min_frames=2):
        return [{"first": t[0], "frames": set(range(t[1])), "obs": [t[2] if len(t) > 2 else (0.0, 0.0, 0.0)]}
                for t in self.inst.get(name, []) if t[1] >= min_frames]

    def pair_distance(self, a, b, min_frames=2):
        return None


ORDER_Q = ("What will be the first-time appearance order of the following categories in the video: "
           "ceiling light, cup, heater, door?")


def test_order_facts_are_attached_only_when_every_object_was_seen_in_distinct_frames():
    om = FakeMap({"ceiling light": [(2, 5)], "cup": [(7, 3), (1, 1)], "heater": [(9, 4)], "door": [(4, 6)]})
    text = ev.object_facts_for(ORDER_Q, om, min_frames=3, order_facts=True)
    # the cup's frame-1 sighting was seen once and does not pass the filter; frame 7 does
    assert "appearance order is ceiling light, door, cup, heater" in text
    assert ev.object_facts_for(ORDER_Q, om, min_frames=3, order_facts=False) == ""
    # heater missing -> nothing attached (a partial order would mislead)
    om2 = FakeMap({"ceiling light": [(2, 5)], "cup": [(7, 3)], "door": [(4, 6)]})
    assert ev.object_facts_for(ORDER_Q, om2, min_frames=3, order_facts=True) == ""
    # a tie (same first frame) is not an order
    om3 = FakeMap({"ceiling light": [(2, 5)], "cup": [(2, 3)], "heater": [(9, 4)], "door": [(4, 6)]})
    assert ev.object_facts_for(ORDER_Q, om3, min_frames=3, order_facts=True) == ""


def test_order_and_count_questions_are_logged_at_every_threshold():
    om = FakeMap({"table": [(1, 6), (3, 2), (5, 1)]})
    log = []
    assert ev.object_facts_for("How many table(s) are in this room?", om, min_frames=3, log=log) == ""
    assert log[-1]["kind"] == "count" and (log[-1]["c1"], log[-1]["c2"], log[-1]["c3"]) == (3, 2, 1)
    assert ev.object_facts_for("How many table(s) are in this room?", om, min_frames=2, count_facts=True) \
        .startswith("From an object tracker run over the video: about 2 distinct table(s)")
    ev.object_facts_for(ORDER_Q, om, min_frames=3, log=log)
    assert log[-1]["kind"] == "order" and log[-1]["f1"]["cup"] is None


@pytest.mark.parametrize("raw,expected", [
    ("Let me look at the frames.\n</think>\n\nB", ("Let me look at the frames.", "B", True)),
    ("The camera moves left and", ("The camera moves left and", "", False)),
    ("</think>3.5", ("", "3.5", True)),
])
def test_split_thinking(raw, expected):
    assert split_thinking(raw) == expected


TOK = os.environ.get("LIVE3R_TOKENIZER")


@pytest.mark.skipif(not TOK, reason="LIVE3R_TOKENIZER 미설정")
def test_build_query_thinking_prefix_ends_in_an_open_think_block():
    from transformers import AutoTokenizer

    from live3r.data.prompt import ASSISTANT_PREFIX, THINK_PREFIX, PromptBuilder

    tok = AutoTokenizer.from_pretrained(TOK)
    pb = PromptBuilder(tok, 248056, 248057, 248053, 248054)
    off = tok.decode(pb.build_query("Q?", [pb.image_segment(4)])[0])
    on = tok.decode(pb.build_query("Q?", [pb.image_segment(4)], THINK_PREFIX)[0])
    assert off.endswith(ASSISTANT_PREFIX) and on.endswith("<|im_start|>assistant\n<think>\n")
    assert off.split("<|im_start|>assistant")[0] == on.split("<|im_start|>assistant")[0], "the cached prefix must not change"


@pytest.mark.parametrize("c,easy,medium,hard", [
    ((-1.0, 1.0), "left", "left", "front-left"),      # facing +y from the origin: -x is the left hand
    ((1.0, 1.0), "right", "right", "front-right"),
    ((-1.0, -1.0), "left", "back", "back-left"),       # 135 degrees behind: VSI calls it "back"
    ((0.5, -1.0), "right", "back", "back-right"),
    ((-1.0, -0.2), "left", "left", "back-left"),       # 101 degrees: not yet "back", but the back-left quadrant
])
def test_direction_label_conventions(c, easy, medium, hard):
    a, b = (0.0, 0.0), (0.0, 1.0)
    assert (ev.direction_label(a, b, c, "easy"), ev.direction_label(a, b, c, "medium"),
            ev.direction_label(a, b, c, "hard")) == (easy, medium, hard)


def test_direction_facts_use_the_most_observed_instance_and_need_all_three():
    om = FakeMap({"stove": [(1, 5, (0.0, 0.0, 0.0))], "sofa": [(2, 4, (0.0, 3.0, 0.0))],
                  "tv": [(3, 2, (5.0, 0.0, 0.0)), (3, 6, (-2.0, 2.0, 0.0))]})
    q = ("If I am standing by the stove and facing the sofa, is the tv to my left, right, or back?\n"
         "An object is to my back if I would have to turn at least 135 degrees in order to face it.")
    log = []
    text = ev.object_facts_for(q, om, min_frames=3, log=log, direction_facts=True)
    assert text.endswith("the tv is to your left.\n")              # the 6-view tv at (-2, 2), not the 2-view one
    assert log[-1]["kind"] == "direction" and log[-1]["difficulty"] == "medium"
    assert log[-1]["r3"] == "left" and log[-1]["r1"] == "left"
    # a separate, looser threshold for directions: the 2-view tv at (5, 0) is the only one below 3 views -> "right"
    om_loose = FakeMap({"stove": [(1, 5, (0.0, 0.0, 0.0))], "sofa": [(2, 4, (0.0, 3.0, 0.0))], "tv": [(3, 2, (5.0, 0.0, 0.0))]})
    assert ev.object_facts_for(q, om_loose, min_frames=3, direction_facts=True) == ""
    assert ev.object_facts_for(q, om_loose, min_frames=3, direction_facts=True, direction_min_frames=1).endswith("to your right.\n")
    om2 = FakeMap({"stove": [(1, 5, (0.0, 0.0, 0.0))], "sofa": [(2, 4, (0.0, 3.0, 0.0))]})
    assert ev.object_facts_for(q, om2, min_frames=3, direction_facts=True) == ""
    assert ev.object_facts_for(q, om, min_frames=3, direction_facts=False) == ""   # logged, not attached
