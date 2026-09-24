"""스트리밍 평가 하니스 — 제약이 실제로 강제되는지.

여기 테스트가 느슨하면 "라이브로 70점" 주장 전체가 무너진다.
"""

import pytest
import torch
from live3r.testing import SMALL_IDS, tiny_model

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
        geometry=GeometryConfig(name="dummy", tap_layers=(0, 1), hidden_size=64, image_size=112),
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
    s = StreamingSession(m, UniformOracleSelector(8, 200), audit=a)
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
    s = StreamingSession(m, HalvingSelector(16), audit=a)
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
@pytest.mark.parametrize("visual_mode", ["image", "video"])
def test_session_geometry_runs_on_whole_stream_but_llm_sees_only_keyframes(visual_mode):
    """이중 레이트의 핵심 — 기하는 전 구간, LLM 은 예산만."""
    m = make_model()
    f, a = feed(600)
    s = StreamingSession(m, HalvingSelector(16), geom_stride=3, visual_mode=visual_mode, audit=a)
    s.consume(f)
    pre = s.prefill()
    r = s.report(token_budget=100_000)

    assert r["frames_seen"] == 600
    assert r["geometry_calls"] > 150, "기하가 전 구간을 안 먹었다"
    assert pre["n_keyframes"] == 16
    assert len(set(a.geometry_state_bytes)) == 1, "기하 상태가 상수가 아니다"
    assert r["vision_encoder_calls"] == 1, "비전 타워는 질문 시점 1회만 돌아야 한다"
    steps = 16 if visual_mode == "image" else 8   # 이미지: 키프레임당 1블록 / 비디오: 2장당 1블록
    assert len(pre["step_tokens"]) == steps
    assert pre["geometry"].embeds[0].shape[0] == pre["n_visual_tokens"], "기하 임베딩 수 != 비전 토큰 수"


@pytest.mark.parametrize("visual_mode", ["image", "video"])
def test_generation_through_primed_injector(visual_mode):
    """프리필에서만 주입되고 디코딩 스텝은 건너뛰는지 (두 모드 모두)."""
    m = make_model()
    f, a = feed(200)
    s = StreamingSession(m, HalvingSelector(8), geom_stride=3, visual_mode=visual_mode, audit=a)
    s.consume(f)
    pre = s.prefill()

    pad = SMALL_IDS["image"] if visual_mode == "image" else SMALL_IDS["video"]
    segs = [torch.randint(0, 900, (1, 3))]
    for n in pre["step_tokens"]:
        segs += [torch.tensor([[SMALL_IDS["vision_start"]]]), torch.full((1, n), pad),
                 torch.tensor([[SMALL_IDS["vision_end"]]])]
    ids = torch.cat(segs, 1)
    mm = (ids == SMALL_IDS["image"]).long() + 2 * (ids == SMALL_IDS["video"]).long()

    m.injector.hit_count = m.injector.skip_count = 0
    with torch.no_grad(), m.injector.primed(pre["geometry"].embeds, m.visual_pos_mask(ids)):
        out = m.base.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
            max_new_tokens=5, do_sample=False, **pre["pixel_kwargs"],
        )
    assert out.shape[1] > ids.shape[1]
    assert m.injector.hit_count == 2, f"프리필 주입이 안 됐다: {m.injector.hit_count}"
    assert m.injector.skip_count > 0, "디코딩 스텝 건너뛰기가 계측되지 않았다"


def test_geometry_input_uses_long_side():
    """CUT3R 512 판본은 긴 변 512 로 학습됐다. 짧은 변으로 맞추면 1.8배 큰 입력이 들어간다."""
    m = make_model()
    s = StreamingSession(m, HalvingSelector(4))
    g = s._prep_geom(torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8))
    assert max(g.shape[-2:]) == m.cfg.geometry.image_size


def _with_fake_points(m):
    """dummy 기하에 CUT3R 처럼 점맵·포즈를 붙인다 (가짜 방: 가로 5m × 세로 4m, 카메라가 앞으로 걷는다)."""
    orig = m.geometry.ingest
    m.geometry.decode_points = False
    state = {"n": 0}

    def ingest(frames):
        out = orig(frames)
        if m.geometry.decode_points:
            g = torch.Generator().manual_seed(state["n"])
            h, w = 24, 32
            pts = torch.rand(3, h, w, generator=g)
            pts[0] = pts[0] * 5 - 2.5                      # x
            pts[1] = torch.where(pts[1] > 0.5, torch.tensor(1.5), -1.1 + 2.6 * pts[1])  # 바닥·벽 높이
            pts[2] = pts[2] * 4                            # z
            out.pointmap = pts.unsqueeze(0)
            out.conf = torch.ones(1, 1, h, w)
            c2w = torch.eye(4)
            c2w[2, 3] = 0.01 * state["n"]
            out.extra["c2w"] = c2w.unsqueeze(0)
        state["n"] += 1
        return out

    m.geometry.ingest = ingest
    return m


@pytest.mark.parametrize("visual_mode", ["image", "video"])
def test_scene_map_prompt_goes_after_keyframes_without_injection(visual_mode):
    """장면 지도: 기하가 도는 프레임마다 점이 쌓이고, prefill 이 지도 이미지·텍스트를 키프레임 뒤에 붙인다.
    지도 토큰에는 기하를 주입하지 않는다 (0) — 주입 수 == 시각 토큰 수가 유지돼야 생성이 된다."""
    from live3r.serve.scene_map import SceneMap

    m = _with_fake_points(make_model())
    f, a = feed(120)
    s = StreamingSession(m, HalvingSelector(8), geom_stride=3, visual_mode=visual_mode, audit=a,
                         scene_map=SceneMap(stride=1), map_size=128, map_max_pixels=128 * 128)
    s.consume(f)
    assert s.scene_map.n_frames == a.geometry_calls > 8      # 키프레임만이 아니라 기하가 돈 프레임 전부
    pre = s.prefill()
    n_map = pre["map_tokens"]
    assert n_map > 0 and "square meters" in pre["map_text"]
    assert pre["n_visual_tokens"] == sum(pre["step_tokens"]) + n_map
    assert all(e.shape[0] == pre["n_visual_tokens"] for e in pre["geometry"].embeds)
    assert all(float(e[-n_map:].abs().sum()) == 0.0 for e in pre["geometry"].embeds), "지도 토큰에 주입이 있다"
    assert pre["pixel_kwargs"]["image_grid_thw"].shape[0] == (9 if visual_mode == "image" else 1)

    pad = SMALL_IDS["image"] if visual_mode == "image" else SMALL_IDS["video"]
    segs = [torch.randint(0, 900, (1, 3))]
    for n in pre["step_tokens"]:
        segs += [torch.tensor([[SMALL_IDS["vision_start"]]]), torch.full((1, n), pad),
                 torch.tensor([[SMALL_IDS["vision_end"]]])]
    segs += [torch.tensor([[SMALL_IDS["vision_start"]]]), torch.full((1, n_map), SMALL_IDS["image"]),
             torch.tensor([[SMALL_IDS["vision_end"]]]), torch.randint(0, 900, (1, 4))]
    ids = torch.cat(segs, 1)
    mm = (ids == SMALL_IDS["image"]).long() + 2 * (ids == SMALL_IDS["video"]).long()
    with torch.no_grad(), m.injector.primed(pre["geometry"].embeds, m.visual_pos_mask(ids)):
        out = m.base.generate(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
                              max_new_tokens=3, do_sample=False, **pre["pixel_kwargs"])
    assert out.shape[1] > ids.shape[1]


def test_scene_map_needs_point_producing_geometry():
    from live3r.serve.scene_map import SceneMap

    with pytest.raises(ValueError, match="CUT3R"):
        StreamingSession(make_model(), HalvingSelector(8), scene_map=SceneMap())
