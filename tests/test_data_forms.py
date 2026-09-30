"""The record forms of the VG-LLM data: videos as frames given as images (files and frame directories), marks,
and the long-side sizing that makes the geometry and visual token grids coincide."""

import json
import os

import numpy as np
import pytest
import torch

from live3r.config import DataConfig, Live3RConfig
from live3r.data.datasets import SpatialVQADataset, frames_to_use, normalize_record
from live3r.data.vision import VisionSpec, geometry_frame, llm_grid, prepare_image, smart_resize, target_size

TOK = os.environ.get("LIVE3R_TOKENIZER")


@pytest.mark.parametrize("hw", [(1080, 1920), (480, 640), (968, 1296), (1440, 1920), (1920, 1440), (360, 640)])
def test_long_side_grids_coincide_with_the_geometry_grid(hw):
    """long_side 512: the VLM's 32-px token grid == CUT3R's patch-16 grid merged 2x2 (same rule, same rounding)."""
    spec = VisionSpec(long_side=512)
    h, w = target_size(*hw, spec)
    assert max(h, w) == 512 and h % 32 == 0 and w % 32 == 0
    img = torch.zeros(*hw, 3, dtype=torch.uint8).numpy()
    _, grid = prepare_image(img, spec)
    g = geometry_frame(img, 512, 16)
    assert llm_grid(grid[0], spec) == (g.shape[1] // 32, g.shape[2] // 32)


def test_long_side_token_counts_and_default_unchanged():
    spec = VisionSpec(long_side=512)
    assert target_size(1080, 1920, spec) == (288, 512)      # 9 x 16 = 144 tokens
    assert target_size(480, 640, spec) == (384, 512)        # 12 x 16 = 192 tokens
    d = VisionSpec()
    assert target_size(480, 640, d) == smart_resize(480, 640, 32, d.min_pixels, d.max_pixels) == (288, 416)


def test_data_config_rejects_unknown_marks_policy():
    with pytest.raises(ValueError, match="marks"):
        Live3RConfig.from_dict({"data": {"marks": "draw"}})


def test_frames_to_use():
    assert frames_to_use(300, 30.0, 0.5, 4, 8) == 5            # 10 s at one frame per 2 s
    assert frames_to_use(60, 30.0, 0.5, 4, 8) == 4             # 2 s -> the minimum
    assert frames_to_use(9000, 30.0, 0.5, 4, 8) == 8           # 5 min -> the maximum
    assert frames_to_use(3, 1.0, 0.5, 4, 8) == 3               # never more than the video has
    assert frames_to_use(0, 30.0, 0.5, 4, 8) == 0


def test_marks_are_read_and_empty_marks_are_none():
    conv = [{"from": "human", "value": "<image>Q"}, {"from": "gpt", "value": "A"}]
    assert normalize_record({"image": "a.jpg", "conversations": conv, "spar_info": '{"red_point": [[1, 2]]}'}).marks
    for empty in (None, "", "{}", {}, []):
        assert normalize_record({"image": "a.jpg", "conversations": conv, "spar_info": empty}).marks is None
    assert normalize_record({"images": ["a.jpg", "b.jpg"], "conversations": conv}).images == ["a.jpg", "b.jpg"]


def test_marked_records_are_skipped_not_failed(tmp_path):
    rec = lambda i, m=None: {"id": i, "image": ["x.jpg"], "spar_info": m,  # noqa: E731
                             "conversations": [{"from": "human", "value": "<image>q"}, {"from": "gpt", "value": "a"}]}
    ann = tmp_path / "a.json"
    ann.write_text(json.dumps([rec(0, {"red_point": [[1, 2]]}), rec(1), rec(2, {"blue_bbox": [[1, 2, 3, 4]]})]))
    ds = SpatialVQADataset(ann, tmp_path, prompt=None, spec=None, geom_long_side=512, geom_unit=16,
                           data_cfg=DataConfig(marks="skip"))
    ds._build_images = lambda r: {"id": r.id, "media_type": "images"}
    assert all(ds[i]["id"] == "1" for i in range(3))
    assert ds.n_fail == 0 and ds.n_skip >= 2
    ds.cfg = DataConfig(marks="ignore")
    assert ds[0]["id"] == "0"


def _video_dataset(tmp_path, recs, **cfg):
    from transformers import AutoTokenizer

    from live3r.data.prompt import PromptBuilder

    ann = tmp_path / "v.json"
    ann.write_text(json.dumps(recs))
    pb = PromptBuilder(AutoTokenizer.from_pretrained(TOK), 248056, 248057, 248053, 248054)
    spec = VisionSpec(long_side=128)
    return SpatialVQADataset(ann, tmp_path / "media", pb, spec, geom_long_side=128, geom_unit=16,
                             data_cfg=DataConfig(video_as_images=True, **cfg), max_fail_rate=1.0), spec


@pytest.mark.skipif(not TOK, reason="LIVE3R_TOKENIZER not set")
def test_frame_directory_video_becomes_images(tmp_path):
    from PIL import Image

    d = tmp_path / "media/hound/v1"
    d.mkdir(parents=True)
    for k in range(20):                                   # 20 frames at the assumed 1 fps = 20 s -> 10 at 0.5 fps -> 8
        Image.fromarray(np.full((96, 128, 3), k * 10, dtype=np.uint8)).save(d / f"{k:04d}.jpg")
    recs = [{"id": "v", "video": "hound/v1",
             "conversations": [{"from": "human", "value": "<video>\nWhat changes?"}, {"from": "gpt", "value": "It brightens."}]}]
    ds, spec = _video_dataset(tmp_path, recs, frames_dir_fps=1.0)
    item = ds[0]
    assert item["media_type"] == "images" and item["pool_temporal"] is False
    assert item["image_grid_thw"].shape[0] == 8 and len(item["geom_frames"]) == 8 and len(item["llm_grids"]) == 8
    n_img = int((item["mm_token_type_ids"] == 1).sum())
    assert n_img == 8 * 3 * 4                             # 96x128 at long side 128 -> 96x128 -> 3 x 4 tokens
    assert int((item["mm_token_type_ids"] == 2).sum()) == 0, "no video tokens: the frames are images"
    text = ds.prompt.tok.decode(item["input_ids"][0])
    assert "<video>" not in text and text.count("<|vision_start|>") == 8
    assert item["cache_key"].startswith("vimg|hound/v1|")


@pytest.mark.skipif(not TOK, reason="LIVE3R_TOKENIZER not set")
def test_video_file_becomes_images_without_placeholder(tmp_path):
    av = pytest.importorskip("av")
    (tmp_path / "media").mkdir()
    path = tmp_path / "media/clip.mp4"
    with av.open(str(path), "w") as c:                    # 6 s at 10 fps -> 3 frames at 0.5 fps -> the minimum 4
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for k in range(60):
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.full((48, 64, 3), k * 4, dtype=np.uint8), format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    recs = [{"id": "f", "video": "clip.mp4",
             "conversations": [{"from": "human", "value": "How long is it?"}, {"from": "gpt", "value": "6 s"}]}]
    ds, _ = _video_dataset(tmp_path, recs)
    item = ds[0]
    assert item["image_grid_thw"].shape[0] == 4 and len(item["geom_frames"]) == 4
    text = ds.prompt.tok.decode(item["input_ids"][0])
    assert text.index("<|vision_start|>") < text.index("How long")
