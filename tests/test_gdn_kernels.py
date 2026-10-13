"""LIVE3R_GDN: the Gated DeltaNet backend switch (live3r.model.kernels) - default torch, fla on request, and the fused
rules called with transformers' arguments. fla itself is CUDA-only, so a stand-in module records the calls."""

import sys
import types

import pytest
import torch

from live3r.model import kernels


@pytest.fixture
def restore_qwen35():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    saved = (q.torch_chunk_gated_delta_rule, q.torch_recurrent_gated_delta_rule)
    yield q
    q.torch_chunk_gated_delta_rule, q.torch_recurrent_gated_delta_rule = saved


def test_default_is_torch_and_bad_values_are_refused(monkeypatch):
    monkeypatch.delenv("LIVE3R_GDN", raising=False)
    assert kernels.gdn_backend() == "torch" and kernels.apply_gdn_backend() == "torch"
    monkeypatch.setenv("LIVE3R_GDN", "triton")
    with pytest.raises(ValueError):
        kernels.gdn_backend()


def test_fla_requested_but_missing_keeps_torch(monkeypatch, restore_qwen35):
    monkeypatch.setenv("LIVE3R_GDN", "fla")
    monkeypatch.setitem(sys.modules, "fla", None)          # import fails
    before = restore_qwen35.torch_chunk_gated_delta_rule
    assert kernels.apply_gdn_backend() == "torch"
    assert restore_qwen35.torch_chunk_gated_delta_rule is before


def test_fla_routes_both_rules_with_transformers_arguments(monkeypatch, restore_qwen35):
    seen = {}

    def fake(name):
        def fn(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False, cu_seqlens=None,
               use_qk_l2norm_in_kernel=False):
            seen[name] = dict(shape=tuple(q.shape), g=g is not None, beta=beta is not None, init=initial_state,
                              final=output_final_state, cu=cu_seqlens, l2=use_qk_l2norm_in_kernel)
            return torch.zeros_like(v), None
        return fn

    mod = types.ModuleType("fla.ops.gated_delta_rule")
    mod.chunk_gated_delta_rule = fake("chunk")
    mod.fused_recurrent_gated_delta_rule = fake("recurrent")
    for name in ("fla", "fla.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", mod)
    monkeypatch.setenv("LIVE3R_GDN", "fla")
    assert kernels.apply_gdn_backend() == "fla"
    q = restore_qwen35
    x = torch.randn(1, 5, 2, 4)
    n_chunk, n_rec = kernels.CALLS["chunk"], kernels.CALLS["recurrent"]
    # the call sites of Qwen3_5GatedDeltaNet.forward: keywords as transformers passes them, chunk_size included
    q.torch_chunk_gated_delta_rule(x, x, x, g=x[..., 0], beta=x[..., 0], initial_state=None, output_final_state=False,
                                   use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    q.torch_recurrent_gated_delta_rule(x, x, x, g=x[..., 0], beta=x[..., 0], initial_state=None,
                                       output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    assert seen["chunk"] == dict(shape=(1, 5, 2, 4), g=True, beta=True, init=None, final=False, cu=None, l2=True)
    assert seen["recurrent"]["final"] is True and seen["recurrent"]["l2"] is True
    assert (kernels.CALLS["chunk"], kernels.CALLS["recurrent"]) == (n_chunk + 1, n_rec + 1)
