"""시각 프리픽스 캐시 — 캐시를 쓴 답이 매번 전체를 프리필한 답과 같아야 한다 (greedy).

같은 영상에 질문이 여러 개일 때 시각 부분(수천 토큰)을 한 번만 프리필하는 최적화다.
라이브 설계의 핵심과 같은 구조라, 틀리면 속도가 아니라 답이 틀린다.
"""

import os

import pytest
import torch

TOK = os.environ.get("LIVE3R_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOK, reason="LIVE3R_TOKENIZER 미설정")


@pytest.fixture(scope="module")
def setup():
    from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
    from live3r.data.prompt import PromptBuilder
    from live3r.eval.streaming import HalvingSelector, StreamingAudit, StreamingSession, TensorFeed
    from live3r.model.live3r import Live3RModel

    torch.manual_seed(0)
    cfg = Live3RConfig(base_model="tiny", dtype="fp32",
                       geometry=GeometryConfig(name="dummy", tap_layers=(1, 2, 3), hidden_size=64,
                                               image_size=128),
                       fusion=FusionConfig(inject_layers=(0, 1, 2), zero_init=False),
                       lora=LoRAConfig(enabled=False))
    live = Live3RModel.from_pretrained(cfg, tokenizer_path=TOK).eval()
    pb = PromptBuilder.from_model(live)
    frames = torch.randint(0, 255, (120, 240, 320, 3), dtype=torch.uint8)
    out = {}
    for mode in ("image", "video"):
        a = StreamingAudit()
        s = StreamingSession(live, HalvingSelector(6), geom_stride=3, visual_mode=mode, audit=a)
        s.consume(TensorFeed(frames, a))
        out[mode] = s.prefill()
    return live, pb, out


def _full(live, pb, pre, q, n=6):
    from live3r.eval.streaming import StreamingSession

    ids = pb.build_query(q, StreamingSession.segments(pb, pre))
    mm = (ids == live.image_token_id).long() + 2 * (ids == live.video_token_id).long()
    with torch.no_grad(), live.injector.primed(pre["geometry"].embeds, live.visual_pos_mask(ids)):
        out = live.base.generate(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
                                 max_new_tokens=n, do_sample=False, **pre["pixel_kwargs"])
    return pb.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@pytest.mark.parametrize("mode", ["image", "video"])
def test_cached_answers_equal_full_prefill(setup, mode):
    from live3r.eval.prefix_cache import PrefixCache
    from live3r.eval.streaming import StreamingSession

    live, pb, pres = setup
    pre = pres[mode]
    pc = PrefixCache(live, pb, StreamingSession.segments(pb, pre), pre["pixel_kwargs"], pre["geometry"])
    for q in ("How many chairs are there?", "Which is closer, the sofa or the table?", "Room size?"):
        cached, _ = pc.answer(q, max_new_tokens=6, do_sample=False)
        assert cached == _full(live, pb, pre, q), f"[{mode}] 캐시 답이 다르다: {q}"


def test_rope_deltas_restored_between_inputs(setup):
    """다른 입력의 프리필이 끼어도 (rope_deltas 덮어쓰기) 답이 같아야 한다."""
    from live3r.eval.prefix_cache import PrefixCache
    from live3r.eval.streaming import StreamingSession

    live, pb, pres = setup
    a = PrefixCache(live, pb, StreamingSession.segments(pb, pres["image"]), pres["image"]["pixel_kwargs"],
                    pres["image"]["geometry"])
    b = PrefixCache(live, pb, StreamingSession.segments(pb, pres["video"]), pres["video"]["pixel_kwargs"],
                    pres["video"]["geometry"])  # rope_deltas 를 덮어쓴다
    q = "How many chairs are there?"
    assert a.answer(q, max_new_tokens=6, do_sample=False)[0] == _full(live, pb, pres["image"], q)
    assert b.answer(q, max_new_tokens=6, do_sample=False)[0] == _full(live, pb, pres["video"], q)


@pytest.mark.parametrize("mode", ["image", "video"])
def test_cached_first_step_scores_equal_full_prefill(setup, mode):
    """문자열 비교는 랜덤 가중치에서 우연히 같을 수 있다 — generate 첫 스텝 점수(로짓)를 직접 비교한다.

    실제로 쓰는 경로(generate + past_key_values)를 그대로 태운다.
    """
    import copy

    from live3r.eval.prefix_cache import PrefixCache, _inner
    from live3r.eval.streaming import StreamingSession

    live, pb, pres = setup
    pre = pres[mode]
    segs = StreamingSession.segments(pb, pre)
    pc = PrefixCache(live, pb, segs, pre["pixel_kwargs"], pre["geometry"])
    full = pb.build_query("Which object is the largest?", segs)
    mm = (full == live.image_token_id).long() + 2 * (full == live.video_token_id).long()
    kw = dict(max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True)
    with torch.no_grad():
        with live.injector.primed(pre["geometry"].embeds, live.visual_pos_mask(full)):
            ref = live.base.generate(input_ids=full, attention_mask=torch.ones_like(full),
                                     mm_token_type_ids=mm, **pre["pixel_kwargs"], **kw).scores[0][0]
        _inner(live.base).rope_deltas = pc.rope_deltas
        got = live.base.generate(input_ids=full, attention_mask=torch.ones_like(full),
                                 past_key_values=copy.deepcopy(pc.cache), **kw).scores[0][0]
    assert torch.allclose(ref, got, atol=1e-4), f"[{mode}] max|Δ| {(ref - got).abs().max():.2e}"
