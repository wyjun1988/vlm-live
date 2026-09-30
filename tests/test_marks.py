"""Marks drawn onto the frames (data/marks.py), reasoning stripped from answers, and the preparation's policies.

The spar_info values below are records of the VG-LLM SPAR sets as the 2026-09-30 data pass read them."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from live3r.data import marks as mk
from live3r.data.datasets import normalize_record, strip_thinking

REPO = Path(__file__).resolve().parents[1]
# scene0424_01_12 (32 frames): "When the camera shifts to tv (in Frame-16, Object0) (green bbox) and points toward
# books (in Frame-21, Object1) (blue bbox), what becomes the relative placement of ... (red ...) ... (yellow ...)"
IMAGINATION = {"yellow_bbox": [[559, 330, 865, 511]], "green_bbox": [[728, 1, 1000, 263]], "red_bbox": [[460, 3, 998, 971]],
               "blue_bbox": [[229, 218, 453, 412]], "red_point": [[798, 330]], "blue_point": [[322, 309]],
               "yellow_point": [[722, 413]], "green_point": [[910, 114]], "point_img_idx": [[16, 21, 16, 22]],
               "bbox_img_idx": [[16, 21, 16, 22]], "type": "spatial_imagination_oo_video"}
Q_IMAGINATION = ("When the camera shifts to tv (in Frame-16, Object0) (green bbox) and points toward books (in "
                 "Frame-21, Object1) (blue bbox), what becomes the relative placement of the cabinet (red bbox) and the "
                 "shelf (yellow bbox)?")


def test_parse_keeps_only_records_with_something_to_draw():
    assert mk.parse(json.dumps(IMAGINATION))["type"] == "spatial_imagination_oo_video"
    assert mk.parse({"red_bbox": [[1, 2, 3, 4]]}) is not None
    for nothing in ('{"type": "obj_count"}', "{}", "", None, "not json", {"type": "view_change_infer"}, []):
        assert mk.parse(nothing) is None


def test_colored_marks_follow_the_order_the_question_names_them():
    plan = mk.plan(IMAGINATION, Q_IMAGINATION, 32)
    where = {(m.kind, m.color): m.image for m in plan}
    g, b, r, y = (mk.COLORS[c] for c in ("green", "blue", "red", "yellow"))
    assert where[("bbox", g)] == 16 and where[("bbox", b)] == 21 and where[("bbox", r)] == 16 and where[("bbox", y)] == 22
    assert where[("point", g)] == 16 and len(plan) == 8
    # scene0534_00_31: backpack (in Frame-30) (green bbox), map (in Frame-0) (blue bbox), board (in Frame-0) (red)
    info = {"green_bbox": [[892, 336, 1000, 599]], "red_bbox": [[311, 0, 677, 289]], "blue_bbox": [[0, 0, 198, 411]],
            "point_img_idx": [[30, 0, 0]], "bbox_img_idx": [[30, 0, 0]]}
    q = "After settling at backpack (in Frame-30, Object0) (green bbox) and facing map (in Frame-0, Object1)(blue bbox), where is bulletin board (in Frame-0, Object2) (red bbox)?"
    where = {m.color: m.image for m in mk.plan(info, q, 32)}
    assert where == {g: 30, b: 0, r: 0}


def test_ambiguous_placement_is_not_guessed():
    # scene0092_04_5: three colored marks on frames [[29, 27, 7]], but the question names no color
    info = {"green_bbox": [[244, 123, 529, 392]], "red_bbox": [[0, 314, 704, 998]], "blue_bbox": [[651, 433, 1000, 815]],
            "bbox_img_idx": [[29, 27, 7]]}
    assert mk.plan(info, "Where is blanket (Object2) once they settle at bed (Object0) and turn toward sink (Object1)?", 32) is None
    assert mk.plan({"red_point": [[1, 2]], "point_img_idx": [[5]]}, "the chair (red point)", 3) is None   # frame 5 of 3
    assert mk.plan({"red_point": [[1, 2]], "blue_point": [[3, 4]], "point_img_idx": [[0]]},
                   "the chair (red point) and the desk (blue point)", 2) is None                       # 2 marks, 1 index


def test_single_images_and_unindexed_marks_go_on_the_first_image():
    # depth_prediction_oc (one image): "Given that paper (red point) is at 2.7 meters, ... box (blue point)"
    info = {"red_bbox": [[208, 33, 226, 131]], "blue_bbox": [[646, 553, 668, 586]], "red_point": [[217, 82]],
            "blue_point": [[657, 569]]}
    assert {m.image for m in mk.plan(info, "paper (red point) ... box (blue point)", 1)} == {0}
    # spatial_volume_infer names no color, one image: nothing ambiguous
    assert [m.image for m in mk.plan({"red_bbox": [[749, 539, 897, 793]]}, "the recycling bin's size?", 1)] == [0]
    # position_matching: "shown with a red bbox in the first image", two images, no index
    assert [m.image for m in mk.plan({"red_bbox": [[248, 17, 582, 315]]}, "a red bbox in the first image", 2)] == [0]


def test_listed_objects_are_labelled():
    # spatial_imagination_map_mv: "Object0 (bbox0), ..., Object6 (bbox6)", object 5 on the second image
    info = {"point_list": [[[596, 674], [931, 904], [713, 486], [725, 675], [591, 520], [628, 582], [589, 866]]],
            "bbox_list": [[[546, 591, 647, 751], [903, 861, 960, 949], [675, 395, 750, 570], [651, 585, 800, 756],
                           [551, 438, 632, 594], [595, 538, 661, 629], [547, 787, 632, 956]]],
            "point_img_idx": [[0, 0, 0, 0, 0, 1, 0]], "bbox_img_idx": [[0, 0, 0, 0, 0, 1, 0]]}
    plan = mk.plan(info, "For 7 objects: Object0 (bbox0), ...", 3)
    assert [m.label for m in plan] == [f"bbox{i}" for i in range(7)] and all(m.kind == "bbox" for m in plan)
    assert [m.image for m in plan] == [0, 0, 0, 0, 0, 1, 0]


def test_drawing_puts_the_colors_where_the_marks_are():
    from PIL import Image

    imgs = [Image.new("RGB", (1296, 968), (128, 128, 128)) for _ in range(2)]
    plan = [mk.Mark("point", (500.0, 500.0), 1, mk.COLORS["red"]),
            mk.Mark("bbox", (100.0, 100.0, 300.0, 400.0), 0, mk.COLORS["blue"]),
            mk.Mark("bbox", (600.0, 600.0, 900.0, 900.0), 0, mk.COLORS["red"], label="bbox3")]
    out = mk.draw(imgs, plan)
    a0, a1 = np.array(out[0]), np.array(out[1])
    assert tuple(a1[484, 648]) == (255, 0, 0)                          # the disc's centre: x 0.5 * 1296, y 0.5 * 968
    assert tuple(a0[int(0.25 * 968), int(0.1 * 1296) + 1]) == (0, 0, 255)   # the left edge of the blue box
    assert tuple(a0[int(0.25 * 968), int(0.2 * 1296)]) == (128, 128, 128)   # the inside stays untouched
    label = a0[int(0.6 * 968) + 8:int(0.6 * 968) + 30, int(0.6 * 1296) + 8:int(0.6 * 1296) + 60]
    assert (label == [255, 255, 255]).all(-1).any() and (label == [255, 0, 0]).all(-1).any()   # white text on red
    assert np.array(imgs[0]).std() == 0, "the originals are not modified"


def test_reasoning_is_stripped_to_the_final_answer():
    assert strip_thinking("<think>Reference objects: a sofa.\nSo C.</think>\n\nC") == "C"
    assert strip_thinking("2.1") == "2.1"
    assert strip_thinking("<think>only reasoning</think>") == "<think>only reasoning</think>"
    conv = [{"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "<image>\nHow far?"}, {"from": "gpt", "value": "<think>...\n</think>\n\n2.1"}]
    rec = normalize_record({"images": ["a.jpg"], "conversations": conv, "spar_info": '{"type": "x"}'}, strip_reasoning=True)
    assert rec.turns == [("<image>\nHow far?", "2.1")] and rec.marks is None
    kept = normalize_record({"images": ["a.jpg"], "conversations": conv})       # off by default: earlier runs unchanged
    assert kept.turns[0][1].startswith("<think>")


@pytest.mark.skipif(not __import__("os").environ.get("LIVE3R_TOKENIZER"), reason="LIVE3R_TOKENIZER not set")
def test_dataset_draws_marks_on_what_both_paths_see_and_passes_over_uncertain_ones(tmp_path):
    import os

    from PIL import Image
    from transformers import AutoTokenizer

    from live3r.config import DataConfig
    from live3r.data.datasets import SpatialVQADataset
    from live3r.data.prompt import PromptBuilder
    from live3r.data.vision import VisionSpec

    (tmp_path / "m").mkdir()
    for k in range(2):
        Image.new("RGB", (256, 128), (128, 128, 128)).save(tmp_path / "m" / f"{k}.png")
    q = lambda t: [{"from": "human", "value": t}, {"from": "gpt", "value": "Yes"}]  # noqa: E731
    info = json.dumps({"red_bbox": [[100, 100, 400, 900]], "blue_bbox": [[500, 100, 900, 900]], "bbox_img_idx": [[1, 0]]})
    recs = [{"id": "sure", "images": ["0.png", "1.png"], "spar_info": info,
             "conversations": q("<image><image>Is the desk (red bbox) below the mug (blue bbox)?")},
            {"id": "unsure", "images": ["0.png", "1.png"], "spar_info": info,
             "conversations": q("<image><image>Is the desk below the mug?")}]
    (tmp_path / "a.json").write_text(json.dumps(recs))
    pb = PromptBuilder(AutoTokenizer.from_pretrained(os.environ["LIVE3R_TOKENIZER"]), 248056, 248057, 248053, 248054)

    def ds(policy):
        return SpatialVQADataset(tmp_path / "a.json", tmp_path / "m", pb, VisionSpec(long_side=256), geom_long_side=256,
                                 geom_unit=16, data_cfg=DataConfig(marks=policy), max_fail_rate=1.0)

    drawn, plain = ds("draw"), ds("ignore")
    a, b = drawn[0], plain[0]
    assert a["id"] == "sure" and not torch_equal(a["pixel_values"], b["pixel_values"])
    # the geometry path sees the marked frames too; "red" is named first -> the first index (1): red on frame 1
    g1 = (a["geom_frames"][1] - b["geom_frames"][1]).abs().sum(0)
    g0 = (a["geom_frames"][0] - b["geom_frames"][0]).abs().sum(0)
    assert g1.sum() > 0 and g0.sum() > 0
    red1 = (a["geom_frames"][1][0] > 0.9) & (a["geom_frames"][1][2] < -0.9)       # normalised (x - 0.5) / 0.5
    blue0 = (a["geom_frames"][0][2] > 0.9) & (a["geom_frames"][0][0] < -0.9)
    assert red1.any() and blue0.any()
    assert drawn[1]["id"] == "sure" and drawn.n_skip >= 1 and drawn.n_fail == 0   # "unsure" is passed over


def torch_equal(x, y) -> bool:
    import torch

    return x.shape == y.shape and bool(torch.equal(x, y))


@pytest.mark.parametrize("policy,kept", [("keep", 4), ("skip", 2), ("draw", 3)])
def test_preparation_policies_and_scene_exclusion(tmp_path, policy, kept):
    q = lambda t, a="1": [{"from": "human", "value": t}, {"from": "gpt", "value": a}]  # noqa: E731
    recs = [
        {"id": 0, "images": ["spar/scannet/scene0999_00/1.jpg"], "conversations": q("<image>chair (red point)?"),
         "spar_info": '{"red_point": [[1, 2]]}'},                                          # marked, drawable
        {"id": 1, "images": ["spar/scannet/scene0999_00/1.jpg", "spar/scannet/scene0999_00/2.jpg"],
         "conversations": q("<image><image>where?"), "spar_info": '{"red_point": [[1, 2]], "blue_point": [[3, 4]], '
                                                                '"point_img_idx": [[0, 1]]}'},   # marked, not certain
        {"id": 2, "images": ["spar/scannet/scene0999_00/3.jpg"], "conversations": q("<image>count?", "<think>x</think>\n\n3"),
         "spar_info": '{"type": "obj_count"}'},                                            # no marks; reasoning
        {"id": 3, "images": ["spar/scannet/scene0999_00/4.jpg"], "conversations": q("<image>size?")},
        {"id": 4, "images": ["spar/scannet/scene0685_00/1.jpg"], "conversations": q("<image>size?")},   # VSI-Bench
    ]
    (tmp_path / "a.json").write_text(json.dumps(recs))
    out = tmp_path / "a.jsonl"
    subprocess.run([sys.executable, str(REPO / "scripts/prepare_annotations.py"), str(tmp_path / "a.json"), str(out),
                    "--media-root", str(tmp_path), "--holdout", "0", "--check-files", "0", "--marks", policy,
                    "--exclude-scenes", str(REPO / "configs/data/vsibench_scenes.txt")], check=True, capture_output=True)
    rep = json.loads(out.with_suffix(".report.json").read_text())
    assert rep["kept"] == kept and rep["dropped"].get("VSI-Bench scene") == 1
    assert rep["marked"] >= 1 and rep["reasoning_answers_stripped"] == 1 and rep["marks"] == policy
