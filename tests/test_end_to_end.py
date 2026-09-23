"""초소형 Qwen3.5 로 엔드투엔드 — 가중치 다운로드 없이 CI 에서 돈다."""

import torch
from tiny_model import VIDEO_TOKEN_ID, tiny_model

from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
from live3r.model.live3r import Live3RModel
from live3r.serve.session import LiveSession


def build_model(zero_init=True):
    cfg = Live3RConfig(
        base_model="tiny",
        dtype="fp32",
        geometry=GeometryConfig(name="dummy", tap_layers=(0, 1, 2), hidden_size=96, image_size=224),
        fusion=FusionConfig(inject_layers=(0, 1, 2), merge_size=2, zero_init=zero_init),
        lora=LoRAConfig(enabled=False),
    )
    return Live3RModel(cfg, tiny_model())


def test_geometry_embeds_match_visual_token_count():
    m = build_model()
    T, H, W = 8, 64, 64
    p, sm, tp = m.vision_patch, m.spatial_merge, m.temporal_patch
    gh, gw = H // p, W // p
    n_vis = (T // tp) * (gh // sm) * (gw // sm)
    m.geometry.reset()
    outs = [m.geometry.ingest(torch.randn(1, 3, 224, 224)) for _ in range(T)]
    bundle = m.build_geometry_embeds(outs, (gh // sm, gw // sm))
    assert bundle.embeds[0].shape == (n_vis, m.d_llm)


def test_zero_init_preserves_base_behaviour():
    m = build_model(zero_init=True)
    m.geometry.reset()
    outs = [m.geometry.ingest(torch.randn(1, 3, 224, 224)) for _ in range(4)]
    bundle = m.build_geometry_embeds(outs, (2, 2))
    ids = torch.cat(
        [torch.randint(0, 900, (1, 3)), torch.full((1, 8), VIDEO_TOKEN_ID)], 1
    )
    attn = torch.ones_like(ids)
    with torch.no_grad():
        a = m(input_ids=ids, attention_mask=attn, geometry=bundle).logits
        b = m(input_ids=ids, attention_mask=attn, geometry=None).logits
    assert torch.allclose(a, b, atol=1e-6)


def test_live_session_streams_without_drift():
    m = build_model()
    sess = LiveSession(m, device="cpu")
    for _ in range(16):
        sess.ingest(torch.randn(3, 64, 64), torch.randn(3, 224, 224))
    s = sess.summary()
    assert s["blocks"] == 8
    assert s["drift"] < 2.0
    r = sess.ask(torch.randint(0, 900, (1, 4)), max_new_tokens=4)
    assert r["token_ids"].shape == (1, 4)
