"""live3r.model.patch_embed: the Conv3d patch embedding and its linear form compute the same thing."""

import torch

from live3r.model.patch_embed import is_patch_embed, linearize_patch_embed


class PatchEmbed(torch.nn.Module):
    """The shape of transformers' Qwen3_5VisionPatchEmbed (kernel == stride == the patch block)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.patch_size, self.temporal_patch_size, self.in_channels, self.embed_dim = 16, 2, 3, hidden
        k = [2, 16, 16]
        self.proj = torch.nn.Conv3d(3, hidden, kernel_size=k, stride=k, bias=True)

    def forward(self, x):
        x = x.view(-1, 3, 2, 16, 16)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


def test_linear_form_matches_the_conv_and_keeps_the_state_dict():
    torch.manual_seed(0)
    tower = torch.nn.Sequential(PatchEmbed(), torch.nn.Linear(64, 8))
    x = torch.randn(37, 3 * 2 * 16 * 16)
    ref = tower(x)
    keys = set(tower.state_dict())
    assert linearize_patch_embed(tower) == 1 and linearize_patch_embed(tower) == 0      # idempotent
    out = tower(x)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    assert set(tower.state_dict()) == keys
    out.sum().backward()                                                                # gradients still flow
    assert tower[0].proj.weight.grad is not None


def test_only_non_overlapping_patch_convs_are_touched():
    other = torch.nn.Module()
    other.proj = torch.nn.Conv3d(3, 8, kernel_size=3, stride=1)
    other.temporal_patch_size = 2
    assert not is_patch_embed(other) and is_patch_embed(PatchEmbed())


def test_config_patch_embed_reaches_the_model_and_is_validated():
    """patch_embed: linear in the config patches the base model's vision tower when Live3RModel is built."""
    import pytest

    from live3r.config import Live3RConfig
    from live3r.model.live3r import Live3RModel

    with pytest.raises(ValueError):
        Live3RConfig(patch_embed="conv3d")
    for mode, patched in (("conv", False), ("linear", True)):
        cfg = Live3RConfig.from_dict({"base_model": "tiny", "patch_embed": mode, "lora": {"enabled": False},
                                      "geometry": {"name": "dummy", "hidden_size": 32, "tap_layers": [1, 2, 3]}})
        live = Live3RModel.from_pretrained(cfg)
        pe = [m for m in live.base.modules() if is_patch_embed(m)]
        assert pe and all(getattr(m, "_live3r_linear", False) == patched for m in pe)
