"""형상 계약 — 조용히 틀리면 성능만 깎이고 에러는 안 나는 지점들."""

import pytest
import torch

from live3r.fusion.deepstack import DeepStackInjector
from live3r.fusion.projector import GeometryProjector


def test_projector_resamples_to_llm_grid():
    """기하 격자(37x37) != LLM 격자(8x14) 여도 맞춰 나와야 한다."""
    proj = GeometryProjector(c_geo=96, d_llm=128, merge_size=2, zero_init=False)
    tok = torch.randn(1, 37 * 37, 96)
    out = proj(tok, src_grid=(37, 37), dst_grid=(8, 14))
    assert out.shape == (1, 8 * 14, 128)


def test_projector_handles_odd_grid_with_padding():
    proj = GeometryProjector(c_geo=32, d_llm=64, merge_size=2, zero_init=False)
    out = proj(torch.randn(1, 5 * 7, 32), (5, 7), (3, 3))
    assert out.shape == (1, 9, 64)


def test_projector_rejects_wrong_token_count():
    proj = GeometryProjector(c_geo=32, d_llm=64)
    with pytest.raises(ValueError, match="토큰 수"):
        proj(torch.randn(1, 10, 32), (5, 7), (3, 3))


def test_zero_init_outputs_exactly_zero():
    proj = GeometryProjector(c_geo=32, d_llm=64, zero_init=True)
    out = proj(torch.randn(1, 16, 32), (4, 4), (2, 2))
    assert torch.equal(out, torch.zeros_like(out))


def test_injector_rejects_token_count_mismatch():
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    inj = DeepStackInjector(layers, (0, 1)).attach()
    hidden = torch.randn(1, 10, 8)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :4] = True
    with inj.primed([torch.randn(3, 8), torch.randn(3, 8)], mask):
        with pytest.raises(RuntimeError, match="기하 임베딩"):
            layers[0](hidden)


def test_injector_adds_only_at_visual_positions():
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
    inj = DeepStackInjector(layers, (0,)).attach()
    hidden = torch.zeros(1, 6, 4)
    mask = torch.tensor([[False, True, True, False, False, False]])
    emb = torch.ones(2, 4)
    with inj.primed([emb], mask):
        out = layers[0](hidden)
    assert out[0, 0].abs().sum() == 0
    assert torch.allclose(out[0, 1], torch.ones(4))
    assert torch.allclose(out[0, 2], torch.ones(4))
    assert out[0, 3].abs().sum() == 0
