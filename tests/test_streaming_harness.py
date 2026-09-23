"""스트리밍 평가 하니스 — 제약이 실제로 강제되는지.

여기 테스트가 느슨하면 "라이브로 70점" 주장 전체가 무너진다.
"""

import pytest
import torch
from tiny_model import tiny_model

from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
from live3r.eval.streaming import (
    SELECTORS,
    HalvingSelector,
    ReservoirSelector,
    StreamingAudit,
    StreamingSession,
    StreamingViolation,
    StrideSelector,
    TensorFeed,
    UniformOracleSelector,
)
from live3r.model.live3r import Live3RModel


def make_model():
    cfg = Live3RConfig(
        base_model="tiny", dtype="fp32",
        geometry=GeometryConfig(name="dummy", tap_layers=(0, 1), hidden_size=64, image_size=64),
        fusion=FusionConfig(inject_layers=(0, 1), merge_size=2, zero_init=False),
        lora=LoRAConfig(enabled=False),
    )
    return Live3RModel(cfg, tiny_model())


def feed(T=600, H=72, W=96):
    a = StreamingAudit()
    return TensorFeed(torch.randint(0, 255, (T, H, W, 3), dtype=torch.uint8), a), a


# ------------------------------------------------------------------ 제약 강제
def test_video_length_is_hidden():
    """총 길이를 아는 순간 균등 샘플링이 가능해진다 = 오프라인이다."""
    f, _ = feed(100)
    with pytest.raises(StreamingViolation, match="미래 정보"):
        len(f)
    with pytest.raises(StreamingViolation):
        _ = f.total_frames


def test_feed_is_single_pass():
    f, _ = feed(20)
    list(f.frames())
    with pytest.raises(StreamingViolation, match="1패스"):
        list(f.frames())


def test_non_causal_selector_is_flagged():
    m = make_model()
    f, a = feed(200)
    s = StreamingSession(m, UniformOracleSelector(8, 200), vlm_short_side=64, audit=a)
    s.consume(f)
    s.prefill()
    with pytest.raises(StreamingViolation, match="인과적이지 않다"):
        s.report(strict=True)


def test_incremental_mode_rejects_selectors_needing_eviction():
    """Qwen3.5 는 24/32 층이 선형 어텐션이라 넣은 프레임을 되물릴 수 없다."""
    m = make_model()
    with pytest.raises(ValueError, match="needs_deferred"):
        StreamingSession(m, HalvingSelector(8), mode="incremental")


def test_token_budget_enforced():
    m = make_model()
    f, a = feed(200)
    s = StreamingSession(m, HalvingSelector(16), vlm_short_side=64, audit=a)
    s.consume(f)
    s.prefill()
    with pytest.raises(StreamingViolation, match="예산"):
        s.report(token_budget=4)


# ---------------------------------------------------------------- 선택기 성질
@pytest.mark.parametrize("T", [200, 900, 3600])
def test_halving_uses_full_budget_and_reaches_tail(T):
    sel = HalvingSelector(16)
    for i in range(T):
        sel.offer(i, None)
    k = sel.final_selection()
    assert len(k) == 16, f"예산을 다 못 썼다: {len(k)}"
    assert k[-1] / (T - 1) > 0.9, f"꼬리를 못 봤다: {k[-1]}/{T}"


def test_stride_selector_burns_budget_at_the_front():
    """회귀 방지 — stride 는 긴 스트림에서 앞부분만 본다. 기본값으로 쓰면 안 된다."""
    sel = StrideSelector(16, stride=8)
    for i in range(3600):
        sel.offer(i, None)
    assert sel.final_selection()[-1] / 3599 < 0.1


def test_halving_beats_reservoir_on_uniformity():
    def spread(sel, T):
        for i in range(T):
            sel.offer(i, None)
        k = torch.tensor([float(x) for x in sel.final_selection()])
        return k.diff().std().item()

    assert spread(HalvingSelector(16), 3600) < spread(ReservoirSelector(16, seed=0), 3600)


def test_registry_default_is_causal():
    from live3r.eval.streaming import DEFAULT_SELECTOR

    assert SELECTORS[DEFAULT_SELECTOR].is_causal


# ------------------------------------------------------------------ 세션 동작
def test_session_geometry_runs_on_whole_stream_but_llm_sees_only_keyframes():
    """이중 레이트의 핵심 — 기하는 전 구간, LLM 은 예산만."""
    m = make_model()
    f, a = feed(600)
    s = StreamingSession(m, HalvingSelector(16), geom_stride=3, vlm_short_side=64, audit=a)
    s.consume(f)
    pre = s.prefill()
    r = s.report(token_budget=4096)

    assert r["frames_seen"] == 600
    assert r["geometry_calls"] > 150, "기하가 전 구간을 안 먹었다"
    assert pre["n_keyframes"] == 16
    assert len(set(a.geometry_state_bytes)) == 1, "기하 상태가 상수가 아니다"
    assert r["vision_encoder_calls"] == 1, "비전 타워는 질문 시점 1회만 돌아야 한다"


def test_generation_through_primed_injector():
    """프리필에서만 주입되고 디코딩 스텝은 건너뛰는지."""
    m = make_model()
    f, a = feed(200)
    s = StreamingSession(m, HalvingSelector(8), geom_stride=3, vlm_short_side=64, audit=a)
    s.consume(f)
    pre = s.prefill()

    from tiny_model import VIDEO_TOKEN_ID, VISION_END_ID, VISION_START_ID

    n_patches = int(pre["video_grid_thw"][0, 0])
    per = pre["n_visual_tokens"] // n_patches
    segs = [torch.randint(0, 900, (1, 3))]
    for _ in range(n_patches):
        segs += [torch.tensor([[VISION_START_ID]]),
                 torch.full((1, per), VIDEO_TOKEN_ID),
                 torch.tensor([[VISION_END_ID]])]
    ids = torch.cat(segs, 1)
    mm = (ids == VIDEO_TOKEN_ID).long() * 2

    m.injector.hit_count = m.injector.skip_count = 0
    with torch.no_grad(), m.injector.primed(pre["geometry"].embeds, m.visual_pos_mask(ids)):
        out = m.base.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
            pixel_values_videos=pre["pixel_values_videos"],
            video_grid_thw=pre["video_grid_thw"],
            max_new_tokens=5, do_sample=False,
        )
    assert out.shape[1] > ids.shape[1]
    assert m.injector.hit_count == 2, f"프리필 주입이 안 됐다: {m.injector.hit_count}"
    assert m.injector.skip_count > 0, "디코딩 스텝 건너뛰기가 계측되지 않았다"
