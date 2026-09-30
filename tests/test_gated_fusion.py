"""Gated input fusion (fusion.mode gated_input): the math, where it acts, what trains, checkpointing, saving."""

import pytest
import torch

from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
from live3r.fusion.gated import GatedTokenFusion, InputFusionInjector
from live3r.model.live3r import Live3RModel
from live3r.testing import tiny_model
from test_train_step import image_batch


def build(norm=True, bias=0.0, pose=False, hidden=0):
    cfg = Live3RConfig(
        base_model="tiny", dtype="fp32",
        geometry=GeometryConfig(name="dummy", tap_layers=(1,), hidden_size=64, image_size=128, expose_pose_token=pose),
        fusion=FusionConfig(mode="gated_input", merge_size=2, merger_hidden=hidden, gate_norm=norm, gate_bias_init=bias),
        lora=LoRAConfig(enabled=False),
    )
    return Live3RModel(cfg, tiny_model())


def only_fusion_trainable(m):
    m.freeze_base()
    for mod in (m.projectors, m.input_fusion):
        for p in mod.parameters():
            p.requires_grad_(True)


def layer0_input(m, b):
    """The hidden states entering decoder layer 0 (after our fusion hook, which was registered first)."""
    seen = {}
    h = m._decoder_layers()[0].register_forward_pre_hook(lambda mod, args: seen.setdefault("x", args[0].detach().clone()))
    try:
        with torch.no_grad():
            m(**b)
    finally:
        h.remove()
    return seen["x"]


def test_fusion_math():
    torch.manual_seed(0)
    v, g = torch.randn(5, 16), torch.randn(5, 16)
    f = GatedTokenFusion(16, norm=False, bias_init=0.0)
    out, gate = f(v, g)
    assert torch.allclose(gate, torch.full((5, 1), 0.5)) and torch.allclose(out, 0.5 * v + 0.5 * g, atol=1e-6)
    f = GatedTokenFusion(16, norm=True, bias_init=-4.0)
    out, gate = f(v, g)
    ln = torch.nn.functional.layer_norm
    a = torch.sigmoid(torch.tensor(-4.0))
    assert torch.allclose(out, (1 - a) * ln(v, (16,)) + a * ln(g, (16,)), atol=1e-5)
    with pytest.raises(ValueError):
        f(v, g[:, :8])


def test_config_needs_exactly_one_tap():
    with pytest.raises(ValueError, match="one geometry tap"):
        Live3RConfig(geometry=GeometryConfig(tap_layers=(6, 12)), fusion=FusionConfig(mode="gated_input"))
    Live3RConfig(geometry=GeometryConfig(tap_layers=(12,)), fusion=FusionConfig(mode="gated_input"))


def test_model_builds_one_merger_without_scalar_gate():
    m = build(hidden=96)
    assert isinstance(m.injector, InputFusionInjector) and isinstance(m.input_fusion, GatedTokenFusion)
    assert len(m.projectors) == 1 and m.projectors[0].gate is None and m.projectors[0].fc1.out_features == 96
    assert m.projectors[0].fc2.weight.abs().sum() > 0, "the merger must not be zero-initialised"


def test_gate_fully_open_puts_the_geometry_tokens_on_the_visual_positions_in_order():
    """g = 1, no norm: layer 0 sees exactly the merged geometry tokens at the visual positions, the text untouched."""
    torch.manual_seed(0)
    m = build(norm=False, bias=40.0)
    b = image_batch(m)
    x = layer0_input(m, b)
    kw = {k: v for k, v in b.items() if k not in ("geom_outs", "llm_grids", "pool_temporal")}
    base = layer0_input(m, kw)                                     # no geometry: the plain embeddings
    mask = m.visual_pos_mask(b["input_ids"])
    geo = m.build_geometry_embeds(b["geom_outs"], b["llm_grids"], False).embeds[0]
    assert torch.allclose(x[mask], geo.to(x.dtype), atol=1e-5)
    assert torch.equal(x[~mask], base[~mask])
    assert m.injector.hit_count >= 1


def test_gate_closed_without_norm_is_the_base_model():
    torch.manual_seed(0)
    m = build(norm=False, bias=-40.0)
    b = image_batch(m)
    kw = {k: v for k, v in b.items() if k not in ("geom_outs", "llm_grids", "pool_temporal")}
    with torch.no_grad():
        assert torch.allclose(m(**b).logits, m(**kw).logits, atol=1e-5)


def test_one_training_step_reaches_merger_gate_and_norms_only():
    torch.manual_seed(0)
    m = build(norm=True, bias=0.0, pose=True)
    only_fusion_trainable(m)
    for p in m.pose_projector.parameters():
        p.requires_grad_(True)
    m.train()
    b = image_batch(m)
    out = m(**b, keep_primed=True)
    out.loss.backward()
    m.injector.clear()
    mg = m.projectors[0]
    assert mg.fc1.weight.grad.abs().sum() > 0 and mg.fc2.weight.grad.abs().sum() > 0
    f = m.input_fusion
    assert f.gate.weight.grad.abs().sum() > 0 and f.gate.bias.grad.abs().sum() > 0
    assert f.norm_v.weight.grad.abs().sum() > 0 and f.norm_g.weight.grad.abs().sum() > 0
    assert m.pose_projector.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.base.parameters())


def test_loss_decreases_with_the_fusion_path_only():
    torch.manual_seed(0)
    m = build(norm=True, bias=0.0)
    only_fusion_trainable(m)
    m.train()
    b = image_batch(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(15):
        out = m(**b, keep_primed=True)
        opt.zero_grad()
        out.loss.backward()
        m.injector.clear()
        opt.step()
        losses.append(out.loss.item())
    assert losses[-1] < losses[0] - 0.05, f"{losses[0]:.3f} -> {losses[-1]:.3f}"


def test_measure_records_the_gate():
    m = build(bias=-2.0)
    b = image_batch(m)
    m.injector.measure = True
    with torch.no_grad():
        m(**b)
    mean, std = m.injector.last_gate
    assert abs(mean - torch.sigmoid(torch.tensor(-2.0)).item()) < 1e-5 and std < 1e-5
    assert m.injector.last_ratio == {}


def test_checkpointing_gives_the_same_gradients_with_keep_primed():
    torch.manual_seed(0)
    m = build(norm=True, bias=0.5)
    only_fusion_trainable(m)
    m.train()
    b = image_batch(m)

    def grads():
        m.zero_grad(set_to_none=True)
        m(**b, keep_primed=True).loss.backward()
        m.injector.clear()
        return torch.cat([p.grad.flatten() for p in list(m.projectors.parameters()) + list(m.input_fusion.parameters())])

    plain = grads()
    m.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    ckpt = grads()
    assert torch.allclose(plain, ckpt, rtol=1e-4, atol=1e-6), (plain - ckpt).abs().max()


def test_generate_fuses_the_prefill_and_skips_decode_steps():
    m = build(bias=0.0)
    b = image_batch(m)
    bundle = m.build_geometry_embeds(b["geom_outs"], b["llm_grids"], False)
    ids = b["input_ids"]
    m.injector.hit_count = m.injector.skip_count = 0
    with torch.no_grad(), m.injector.primed(bundle.embeds, m.visual_pos_mask(ids)):
        m.base.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=b["pixel_values"],
                        image_grid_thw=b["image_grid_thw"], mm_token_type_ids=b["mm_token_type_ids"],
                        max_new_tokens=3, do_sample=False)
    assert m.injector.hit_count == 1 and m.injector.skip_count >= 1
    assert not m.injector.is_primed


def test_wrong_token_count_is_an_error_not_a_silent_misplacement():
    m = build()
    b = image_batch(m)
    bundle = m.build_geometry_embeds(b["geom_outs"], b["llm_grids"], False)
    with pytest.raises(RuntimeError, match="visual positions"):
        with torch.no_grad(), m.injector.primed([bundle.embeds[0][:-1]], m.visual_pos_mask(b["input_ids"])):
            kw = {k: v for k, v in b.items() if k not in ("geom_outs", "llm_grids", "pool_temporal")}
            m(**kw)
    with pytest.raises(ValueError):
        m.injector.prime([bundle.embeds[0], bundle.embeds[0]], m.visual_pos_mask(b["input_ids"]))


def test_trained_state_saves_and_reloads_the_fusion():
    torch.manual_seed(0)
    m = build(bias=0.3)
    with torch.no_grad():
        m.input_fusion.gate.weight.normal_()
        m.input_fusion.norm_v.bias.fill_(0.1)
    state = m.trainable_state_dict()
    assert {"input_fusion.gate.weight", "input_fusion.gate.bias", "input_fusion.norm_v.weight"} <= set(state)
    assert any(k.startswith("projectors.0.fc1") for k in state)
    fresh = build(bias=0.3)
    res = fresh.load_state_dict(state, strict=False)
    assert not res.unexpected_keys
    b = image_batch(m)
    with torch.no_grad():
        assert torch.allclose(m(**b).logits, fresh(**b).logits, atol=1e-6)


def test_the_trainer_opens_the_fusion_parameters():
    from types import SimpleNamespace

    import live3r.train.train as tr

    cfg = Live3RConfig(
        base_model="tiny", dtype="fp32",
        geometry=GeometryConfig(name="dummy", tap_layers=(1,), hidden_size=64, image_size=128, expose_pose_token=False),
        fusion=FusionConfig(mode="gated_input"), lora=LoRAConfig(enabled=False))
    m = tr.build_model(SimpleNamespace(stage="sft", init_from=None), cfg, is_main=False)
    s = m.trainable_parameter_summary()
    assert s["fusion"] == sum(p.numel() for p in m.input_fusion.parameters()) > 0 and s["projector"] > 0
