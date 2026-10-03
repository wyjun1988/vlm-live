"""Scaling the visual tokens at the LLM input (live3r.model.visual_scale, BRAINSTORM I-44)."""

import pytest
import torch

from live3r.model.visual_scale import find_visual, parse_scale, scale_visual_tokens
from test_train_step import build, image_batch


def layer0_visual(m, b):
    seen = {}
    h = m._decoder_layers()[0].register_forward_pre_hook(lambda mod, args: seen.setdefault("x", args[0].detach().clone()))
    try:
        with torch.no_grad():
            kw = {k: v for k, v in b.items() if k not in ("geom_outs", "llm_grids", "pool_temporal")}
            m(**kw)
    finally:
        h.remove()
    return seen["x"][m.visual_pos_mask(b["input_ids"])]


def test_gain_multiplies_and_ln_normalises_the_visual_tokens_only():
    torch.manual_seed(0)
    m = build()
    b = image_batch(m)
    base = layer0_visual(m, b)
    h = scale_visual_tokens(m, "3")
    assert torch.allclose(layer0_visual(m, b), 3 * base, atol=1e-5)
    h.remove()
    h = scale_visual_tokens(m, "ln")
    v = layer0_visual(m, b)
    assert torch.allclose(v.mean(-1), torch.zeros(v.shape[0]), atol=1e-4)
    assert torch.allclose(v.std(-1, unbiased=False), torch.ones(v.shape[0]), atol=1e-2)   # LayerNorm eps
    h.remove()
    assert torch.allclose(layer0_visual(m, b), base)            # removed: back to the plain model
    assert scale_visual_tokens(m, "1") is None and find_visual(m) is not None


def test_parse_scale():
    assert parse_scale("2.5") == 2.5 and parse_scale("LN") == "ln"
    with pytest.raises(ValueError):
        parse_scale("0")
