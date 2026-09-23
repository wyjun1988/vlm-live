"""프롬프트 빌더 — 실제 Qwen3.5 토크나이저·공식 프로세서와 **토큰 단위로** 대조한다.

LIVE3R_TOKENIZER=<Qwen3.5 모델 디렉터리> 를 주면 돈다 (토크나이저·템플릿·전처리 설정 파일만 있으면 된다).
    맥:   scratchpad 에 받은 Qwen3.5-0.8B 토크나이저 파일
    서버: HF 캐시의 Qwen3.5-4B 스냅샷 디렉터리
"""

import os

import numpy as np
import pytest
import torch

TOK = os.environ.get("LIVE3R_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOK, reason="LIVE3R_TOKENIZER 미설정 — 실제 토크나이저 대조 생략")

IDS = dict(image_token_id=248056, video_token_id=248057, vision_start_id=248053, vision_end_id=248054)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOK)


@pytest.fixture(scope="module")
def pb(tok):
    from live3r.data.prompt import PromptBuilder

    return PromptBuilder(tok, **IDS)


@pytest.fixture(scope="module")
def proc():
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(TOK)
    except Exception as exc:  # 전처리 설정 파일이 없으면
        pytest.skip(f"프로세서 로드 불가: {exc}")


def test_special_token_ids_match_config(tok):
    for s, i in (("<|image_pad|>", 248056), ("<|video_pad|>", 248057),
                 ("<|vision_start|>", 248053), ("<|vision_end|>", 248054), ("<|im_end|>", 248046)):
        assert tok.convert_tokens_to_ids(s) == i


def test_single_turn_matches_official_template(tok, pb):
    msgs = [{"role": "user", "content": "How far is the chair?"}, {"role": "assistant", "content": "1.43"}]
    official = tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False).input_ids
    built = pb.build([("How far is the chair?", "1.43")], [], "text")
    assert built.input_ids[0].tolist() == official


def test_generation_prompt_is_thinking_off(tok, pb):
    q = pb.build_query("Q?", [])
    official = tok.apply_chat_template([{"role": "user", "content": "Q?"}],
                                       tokenize=False, add_generation_prompt=True)
    assert tok.decode(q[0]) == official
    assert official.endswith("<think>\n\n</think>\n\n"), "thinking 이 켜진 템플릿이다"


def test_labels_cover_only_answers_and_im_end(tok, pb):
    built = pb.build([("Q1", "A. left"), ("Q2", "2.1")], [], "text")
    sup = tok.decode(built.labels[0][built.labels[0] != -100].tolist())
    assert sup == "A. left<|im_end|>2.1<|im_end|>"


def test_images_match_official_processor(pb, proc):
    from live3r.data.vision import VisionSpec, prepare_image, tokens_per_step

    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (256, 320, 3), dtype=np.uint8),
            rng.integers(0, 255, (288, 256, 3), dtype=np.uint8)]
    spec = VisionSpec(min_pixels=65536, max_pixels=16777216)  # 공식 기본값 → 리사이즈 없음
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "A"},
                                         {"type": "image"}, {"type": "text", "text": "Q?"}]},
            {"role": "assistant", "content": "1.43"}]
    off = proc(text=[proc.apply_chat_template(msgs, tokenize=False)], images=imgs, return_tensors="pt")
    vlm = [prepare_image(im, spec) for im in imgs]
    segs = [pb.image_segment(tokens_per_step(g[0], spec)) for _, g in vlm]
    built = pb.build([("<image>A<image>Q?", "1.43")], segs, "images")
    assert built.input_ids[0].tolist() == off["input_ids"][0].tolist()
    assert built.mm_token_type_ids[0].tolist() == off["mm_token_type_ids"][0].tolist()
    ours = torch.cat([pv for pv, _ in vlm], 0)
    assert (ours - off["pixel_values"].float()).abs().max() < 1e-5


def test_video_matches_official_processor(pb, proc):
    from live3r.data.vision import VisionSpec, frame_timestamps, prepare_video, tokens_per_step

    vid = np.random.default_rng(1).integers(0, 255, (4, 256, 320, 3), dtype=np.uint8)
    spec = VisionSpec(min_pixels=65536, max_pixels=16777216)
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Q?"}]}]
    off = proc(text=[proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)],
               videos=[vid], return_tensors="pt", do_sample_frames=False,
               video_metadata=[{"fps": 2.0, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3]}])
    pv, grid = prepare_video(vid, spec)
    seg = pb.video_segment(tokens_per_step(grid[0], spec), frame_timestamps([0, 1, 2, 3], 2.0, 2))
    assert pb.build_query("<video>Q?", [seg])[0].tolist() == off["input_ids"][0].tolist()
    assert (pv - off["pixel_values_videos"].float()).abs().max() < 1e-5


def test_placeholder_rules(pb):
    from live3r.data.prompt import PromptError

    seg = pb.image_segment(2)
    with pytest.raises(PromptError, match="placeholder"):
        pb.build([("<image> only one", "x")], [seg, seg], "images")
    built = pb.build([("no placeholder", "x")], [seg, seg], "images")  # 없으면 앞에 붙인다
    assert int((built.mm_token_type_ids == 1).sum()) == 4


def test_long_sequence_drops_later_turns_not_images(pb):
    seg = pb.image_segment(50)
    turns = [("<image>Q1", "A1")] + [(f"Q{i} " + "word " * 60, f"A{i}") for i in range(2, 8)]
    built = pb.build(turns, [seg], "images", max_length=300)
    assert built.turns_used < len(turns)
    assert int((built.mm_token_type_ids == 1).sum()) == 50, "비전 토큰이 잘렸다"
