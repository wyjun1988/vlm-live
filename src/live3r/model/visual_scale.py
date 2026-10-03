"""Scale the visual tokens before the LLM reads them (BRAINSTORM I-44, an inference-time probe).

The gated fusion's step-0 check (scripts/check_fusion_init.py) found that a LayerNorm on the visual tokens - which
makes them ~5.6x louder on the 4B (norm 9.0 -> 50.6) - lowered the held-out loss before any training. This tests the
effect alone: the vision tower's merged output (`pooler_output`, the embeddings the LLM receives for images and video
alike) is multiplied by a gain, or LayerNormed without affine parameters ("ln"), through a forward hook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def find_visual(model: nn.Module) -> nn.Module:
    """The Qwen3.5 vision tower inside a Live3RModel / HF model / PEFT wrapper."""
    for name, mod in model.named_modules():
        if name.endswith("visual") and "Vision" in type(mod).__name__:
            return mod
    raise AttributeError("no Qwen3.5 vision tower (a module named '...visual') in this model")


def parse_scale(spec) -> float | str:
    """'2' -> 2.0, 'ln' -> 'ln'."""
    if isinstance(spec, str) and spec.lower() == "ln":
        return "ln"
    value = float(spec)
    if value <= 0:
        raise ValueError(f"the visual gain must be positive, got {spec!r}")
    return value


def scale_visual_tokens(model: nn.Module, spec) -> torch.utils.hooks.RemovableHandle | None:
    """Hook the vision tower so its output tokens are scaled; None (no hook) for a gain of 1."""
    scale = parse_scale(spec)
    if scale == 1.0:
        return None

    def hook(module, args, output):
        x = output.pooler_output
        if scale == "ln":
            y = F.layer_norm(x.float(), (x.shape[-1],)).to(x.dtype)
        else:
            y = x * scale
        output.pooler_output = y
        return output

    return find_visual(model).register_forward_hook(hook)
