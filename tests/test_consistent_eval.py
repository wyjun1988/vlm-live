"""학습과 같은 입력 경로의 평가 (eval/consistent.py + lmms 어댑터 생성 루프).

부모 lmms-eval 경로는 학습과 세 군데가 다르다: 시스템 프롬프트 · 비디오 모드 · 총 픽셀 예산.
VideoMME 회귀 게이트를 이 경로로 재야 "일반 능력 회귀"를 잰다.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from live3r.eval.consistent import is_video, uniform_indices

TOK = os.environ.get("LIVE3R_TOKENIZER")
needs_tok = pytest.mark.skipif(not TOK, reason="LIVE3R_TOKENIZER 미설정")


def test_uniform_indices_match_official_formula():
    """공식 Qwen 비디오 프로세서: linspace(0, total-1, n).round()."""
    assert uniform_indices(100, 5) == np.linspace(0, 99, 5).round().astype(int).tolist()
    assert uniform_indices(3, 8) == [0, 1, 2]  # 프레임이 모자라면 전부
    assert uniform_indices(0, 4) == []


def test_is_video():
    assert is_video("a/b.MP4") and is_video("x.webm")
    assert not is_video("a.jpg") and not is_video(np.zeros((2, 2, 3)))


def _write_mp4(path, n=40, h=96, w=128):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            img = np.full((h, w, 3), i * 6 % 255, dtype=np.uint8)  # 프레임마다 밝기가 다르다
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_read_video_frames_uniform(tmp_path):
    from live3r.eval.consistent import read_video_frames

    vid = tmp_path / "v.mp4"
    _write_mp4(vid, n=40)
    frames, idx = read_video_frames(vid, 8)
    assert idx == uniform_indices(40, 8)
    assert frames.shape[0] == 8 and frames.dtype == np.uint8
    # 밝기가 인덱스를 따라 증가해야 한다 = 실제로 그 프레임을 읽었다
    means = frames.reshape(8, -1).mean(1)
    assert np.all(np.diff(means) > 0)


@pytest.fixture(scope="module")
def live():
    from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
    from live3r.model.live3r import Live3RModel

    cfg = Live3RConfig(
        base_model="tiny", dtype="fp32",
        geometry=GeometryConfig(name="dummy", tap_layers=(1, 2, 3), hidden_size=64, image_size=128),
        fusion=FusionConfig(inject_layers=(0, 1, 2), zero_init=False),
        lora=LoRAConfig(enabled=False),
    )
    return Live3RModel.from_pretrained(cfg, tokenizer_path=TOK).eval()


@needs_tok
def test_images_like_training_no_system_prompt(live):
    from live3r.data.prompt import PromptBuilder
    from live3r.eval.consistent import prepare_inputs

    pb = PromptBuilder.from_model(live)
    imgs = [np.random.default_rng(i).integers(0, 255, (240, 320, 3), dtype=np.uint8) for i in range(3)]
    p = prepare_inputs(live, pb, "Which is closer?", imgs)
    text = pb.tok.decode(p.input_ids[0])
    assert text.startswith("<|im_start|>user\n"), "시스템 프롬프트가 붙었다 — 학습에는 없다"
    assert text.endswith("<think>\n\n</think>\n\n")
    assert text.count("<|vision_start|>") == 3 and "<|video_pad|>" not in text
    n_img = int((p.mm_token_type_ids == 1).sum())
    assert p.geometry.embeds[0].shape[0] == n_img, "기하 임베딩 수 != 이미지 토큰 수"
    assert p.pixel_kwargs["image_grid_thw"].shape == (3, 3)


@needs_tok
def test_placeholders_interleave_when_count_matches(live):
    from live3r.data.prompt import PromptBuilder
    from live3r.eval.consistent import prepare_inputs

    pb = PromptBuilder.from_model(live)
    imgs = [np.zeros((256, 256, 3), dtype=np.uint8)] * 2
    t1 = pb.tok.decode(prepare_inputs(live, pb, "A: <image> B: <image> which?", imgs).input_ids[0])
    assert "A: <|vision_start|>" in t1  # 제자리에 끼웠다
    t2 = pb.tok.decode(prepare_inputs(live, pb, "<image> only one", imgs).input_ids[0])
    assert t2.count("<|vision_start|>") == 2 and "<image>" not in t2  # 어긋나면 지우고 앞에


@needs_tok
def test_adapter_loop_video_goes_through_image_keyframes(live, tmp_path):
    """lmms 어댑터 생성 루프 — 영상 문항이 비디오 모드가 아니라 키프레임 이미지로 들어가는지."""
    from live3r.eval.lmms_live3r import _live3r_generate

    vid = tmp_path / "v.mp4"
    _write_mp4(vid, n=30)
    seen = {}
    orig = live.base.generate

    def spy(**kw):
        seen.update(kw)
        return orig(**kw)

    live.base.generate = spy
    try:
        fake = SimpleNamespace(
            live3r=live, tokenizer=live.tokenizer, rank=0, streaming=False, use_geometry=True,
            keyframe_budget=6, task_dict={"t": {"test": {0: {"v": str(vid)}}}},
            _build_generate_kwargs=lambda g: {"max_new_tokens": 3, "do_sample": False},
            _strip_thinking=lambda a: a,
        )
        req = SimpleNamespace(args=("How many chairs?", {"until": ["\n\n"]},
                                    lambda doc: [doc["v"]], 0, "t", "test"))
        out = _live3r_generate(fake, [req])
    finally:
        live.base.generate = orig
    assert len(out) == 1 and isinstance(out[0], str)
    assert "pixel_values_videos" not in seen, "영상이 비디오 모드로 들어갔다"
    assert seen["image_grid_thw"].shape[0] == 6, "키프레임 수가 예산과 다르다"
    assert int((seen["mm_token_type_ids"] == 2).sum()) == 0


@needs_tok
def test_adapter_loop_streaming_on_real_video_file(live, tmp_path):
    """스트리밍 경로를 실제 mp4 로 — CausalVideoFeed 디코딩 → 세션 → 생성 → 감사 통과."""
    from live3r.eval.lmms_live3r import _live3r_generate

    vid = tmp_path / "v.mp4"
    _write_mp4(vid, n=60)
    fake = SimpleNamespace(
        live3r=live, tokenizer=live.tokenizer, rank=0, streaming=True, use_geometry=True,
        keyframe_budget=4, selector_name="halving", geom_stride=3, stream_mode="deferred",
        visual_mode="image", stream_reports=[],
        task_dict={"t": {"test": {0: {"v": str(vid)}}}},
        _build_generate_kwargs=lambda g: {"max_new_tokens": 3, "do_sample": False},
        _strip_thinking=lambda a: a,
    )
    req = SimpleNamespace(args=("How many chairs?", {}, lambda doc: [doc["v"]], 0, "t", "test"))
    out = _live3r_generate(fake, [req])
    assert len(out) == 1
    r = fake.stream_reports[0]
    assert r["frames_seen"] == 60 and r["causal"] and not r["violations"]
    assert r["keyframes_kept"] >= 4 and r["visual_mode"] == "image"


@needs_tok
def test_adapter_loop_oracle_selector_runs_and_is_flagged(live, tmp_path):
    """오라클은 총 길이를 봐야 돈다 — 예외 없이 돌고, 보고서에 '비인과'로 남아야 한다."""
    from live3r.eval.lmms_live3r import _live3r_generate

    vid = tmp_path / "v.mp4"
    _write_mp4(vid, n=50)
    fake = SimpleNamespace(
        live3r=live, tokenizer=live.tokenizer, rank=0, streaming=True, use_geometry=True,
        keyframe_budget=5, selector_name="uniform_oracle", geom_stride=3, stream_mode="deferred",
        visual_mode="image", stream_reports=[],
        task_dict={"t": {"test": {0: {"v": str(vid)}}}},
        _build_generate_kwargs=lambda g: {"max_new_tokens": 2, "do_sample": False},
        _strip_thinking=lambda a: a,
    )
    req = SimpleNamespace(args=("Q?", {}, lambda doc: [doc["v"]], 0, "t", "test"))
    assert len(_live3r_generate(fake, [req])) == 1
    r = fake.stream_reports[0]
    assert r["causal"] is False and any("인과적이지 않다" in v for v in r["violations"])


def test_video_fps(tmp_path):
    from live3r.eval.consistent import video_fps

    vid = tmp_path / "v.mp4"
    _write_mp4(vid, n=20)  # rate=10
    assert abs(video_fps(vid) - 10.0) < 0.5
    assert video_fps(tmp_path / "없음.mp4") is None


@needs_tok
def test_video_mode_keyframes_match_streaming_layout(live):
    """visual_mode=video — 같은 키프레임을 비디오 블록으로 (게이트 1 의 베이스 최선 형식 기준선용).

    스트리밍 세션의 video 경로와 같은 구성이어야 한다: 2프레임=1 temporal patch, 패치 평균 타임스탬프,
    기하는 2장 평균, 이미지 문항(frame_indices 없음)은 그대로 이미지 모드.
    """
    from live3r.data.prompt import PromptBuilder
    from live3r.eval.consistent import prepare_inputs

    pb = PromptBuilder.from_model(live)
    frames = [np.full((240, 320, 3), 20 * i, dtype=np.uint8) for i in range(6)]
    idx = [0, 10, 20, 30, 40, 50]
    p = prepare_inputs(live, pb, "How many chairs?", frames, idx, visual_mode="video", fps=10.0)
    text = pb.tok.decode(p.input_ids[0])
    assert "pixel_values_videos" in p.pixel_kwargs and "pixel_values" not in p.pixel_kwargs
    assert int(p.pixel_kwargs["video_grid_thw"][0, 0]) == 3            # 6장 → temporal patch 3개
    assert "<0.5 seconds>" in text and "<2.5 seconds>" in text and "<4.5 seconds>" in text
    n_vid = int((p.mm_token_type_ids == 2).sum())
    assert n_vid > 0 and int((p.mm_token_type_ids == 1).sum()) == 0
    assert p.geometry.embeds[0].shape[0] == n_vid, "기하 임베딩 수 != 비디오 토큰 수"
    # 이미지 문항은 visual_mode 와 무관하게 이미지 모드
    q = prepare_inputs(live, pb, "Which?", frames[:2], None, visual_mode="video")
    assert "pixel_values" in q.pixel_kwargs and int((q.mm_token_type_ids == 2).sum()) == 0


@needs_tok
def test_adapter_loop_video_mode_offline(live, tmp_path):
    from live3r.eval.lmms_live3r import _live3r_generate

    vid = tmp_path / "v.mp4"
    _write_mp4(vid, n=30)
    seen = {}
    orig = live.base.generate

    def spy(**kw):
        seen.update(kw)
        return orig(**kw)

    live.base.generate = spy
    try:
        fake = SimpleNamespace(
            live3r=live, tokenizer=live.tokenizer, rank=0, streaming=False, use_geometry=False,
            keyframe_budget=6, visual_mode="video", task_dict={"t": {"test": {0: {"v": str(vid)}}}},
            _build_generate_kwargs=lambda g: {"max_new_tokens": 3, "do_sample": False},
            _strip_thinking=lambda a: a,
        )
        req = SimpleNamespace(args=("How many chairs?", {}, lambda doc: [doc["v"]], 0, "t", "test"))
        out = _live3r_generate(fake, [req])
    finally:
        live.base.generate = orig
    assert len(out) == 1
    assert "pixel_values_videos" in seen and "pixel_values" not in seen
    assert int(seen["video_grid_thw"][0, 0]) == 3
