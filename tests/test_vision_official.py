"""비전 전처리를 **transformers 공식 구현과 직접 대조**한다.

랜덤 가중치 초소형 모델로는 "패치가 뒤섞여 들어가는" 버그를 절대 못 잡는다 — 에러 없이
그냥 돈다. 2026-09-23 에 실제로 그랬다 (래스터 순서 평탄화, 실가중치에서 모든 이미지가 뒤섞임).
그래서 공식 patchify / smart_resize 와 비트 단위로 비교한다.
"""

import random

import pytest
import torch

from live3r.data.vision import (
    VisionSpec,
    frame_timestamps,
    normalize,
    patchify,
    prepare_image,
    smart_resize,
    to_float01,
    unpatchify,
)

SPEC = VisionSpec()


def _official_image_patchify():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    fn = getattr(Qwen2VLImageProcessor, "patchify", None)
    if fn is None:
        pytest.skip("이 transformers 판에는 Qwen2VLImageProcessor.patchify 가 없다")
    return fn


def _official_video_patchify():
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    fn = getattr(Qwen3VLVideoProcessor, "patchify", None)
    if fn is None:
        pytest.skip("이 transformers 판에는 Qwen3VLVideoProcessor.patchify 가 없다")
    return fn


@pytest.mark.parametrize("hw", [(64, 96), (96, 64), (128, 128)])
def test_image_layout_matches_official(hw):
    x = torch.randn(1, 3, *hw)
    ours, grid = patchify(x, SPEC)
    theirs, gh, gw = _official_image_patchify()(None, x, 16, 2, 2)
    assert grid.tolist() == [[1, gh, gw]]
    assert torch.equal(ours, theirs[0]), "이미지 패치 평탄화 순서가 공식과 다르다"


@pytest.mark.parametrize("t", [2, 4, 5])  # 5 = 홀수 → 마지막 프레임 복제
def test_video_layout_matches_official(t):
    x = torch.randn(t, 3, 64, 96)
    ours, grid = patchify(x, SPEC)
    theirs, gt, gh, gw = _official_video_patchify()(None, x.unsqueeze(0), 16, 2, 2)
    assert grid.tolist() == [[gt, gh, gw]]
    assert torch.equal(ours, theirs[0]), "비디오 패치 평탄화 순서가 공식과 다르다"


def test_old_raster_layout_would_have_been_wrong():
    """회귀 기록 — 예전 구현(래스터 순서)은 공식과 달랐다. 이 차이가 '이미지 뒤섞임' 이다."""
    x = torch.randn(2, 3, 64, 64)
    p, tp = 16, 2
    gh = gw = 4
    raster = x.reshape(1, tp, 3, gh, p, gw, p).permute(0, 3, 5, 2, 1, 4, 6).reshape(gh * gw, -1)
    ours, _ = patchify(x, SPEC)
    assert not torch.equal(raster, ours)


def test_unpatchify_roundtrip():
    x = torch.randn(6, 3, 96, 64)
    flat, grid = patchify(x, SPEC)
    assert torch.equal(unpatchify(flat, grid, SPEC), x)


def test_smart_resize_matches_official():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as official

    rng = random.Random(0)
    for _ in range(300):
        h, w = rng.randint(20, 3000), rng.randint(20, 3000)
        if max(h, w) / min(h, w) > 150:
            continue
        a = smart_resize(h, w, 32, SPEC.min_pixels, SPEC.max_pixels)
        b = official(h, w, factor=32, min_pixels=SPEC.min_pixels, max_pixels=SPEC.max_pixels)
        assert a == tuple(b), (h, w, a, b)


def test_image_duplicated_over_time_like_official():
    """이미지는 프레임 1장을 temporal_patch 번 복제한 비디오와 같아야 한다."""
    img = torch.randint(0, 255, (256, 320, 3), dtype=torch.uint8)
    pv_img, g_img = prepare_image(img, SPEC)
    x = normalize(to_float01(img))
    pv_vid, g_vid = patchify(torch.cat([x, x], 0), SPEC)
    assert g_img.tolist() == g_vid.tolist()
    assert torch.allclose(pv_img, pv_vid)


def test_timestamps_match_official_formula():
    """공식 Qwen3VLProcessor._calculate_timestamps: 패치 안 첫·끝 프레임 시각의 평균."""
    assert frame_timestamps([0, 1, 2, 3], 2.0, 2) == [0.25, 1.25]
    assert frame_timestamps([0, 10, 20], 10.0, 2) == [0.5, 2.0]  # 홀수 → 마지막 복제
