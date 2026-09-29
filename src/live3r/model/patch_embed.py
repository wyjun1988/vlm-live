"""The vision tower's patch embedding as the linear map it is.

Qwen3.5's `VisionPatchEmbed` is a Conv3d whose kernel equals its stride equals its input block (2 x 16 x 16 per patch
row), i.e. a plain matrix product over each flattened row. Some torch/cuDNN builds route a bf16 Conv3d to a slow
non-cuDNN kernel; the 2026-09-29 server analysis (docs/SERVER_RESULTS_20260929.md §2.3) found the image path costing
4-5 ms per patch row on H100, ~5x an Apple M2, and named this as the prime suspect for the 16 GPU-s per training
sample. `F.linear(x, W.view(out, -1), b)` computes the same result (fp32 rel. diff ~2e-6; bf16 within an ulp, not
bit-identical), so switching it changes numerics only at the last bit - but train and eval must then use the same
form. `scripts/diagnose_throughput.py` measures both on the server before anything adopts it.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _linear_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    w = self.proj.weight
    x = hidden_states.reshape(-1, w[0].numel()).to(dtype=w.dtype)
    return F.linear(x, w.view(w.shape[0], -1), self.proj.bias)


def is_patch_embed(module) -> bool:
    proj = getattr(module, "proj", None)
    return (isinstance(proj, torch.nn.Conv3d) and hasattr(module, "temporal_patch_size")
            and tuple(proj.kernel_size) == tuple(proj.stride))


def linearize_patch_embed(model) -> int:
    """Replace the forward of every Conv3d patch embedding under `model` by the equivalent linear map, in place
    (same parameters, same state dict). Returns how many modules were patched."""
    n = 0
    for mod in model.modules():
        if is_patch_embed(mod) and not getattr(mod, "_live3r_linear", False):
            mod.forward = types.MethodType(_linear_forward, mod)
            mod._live3r_linear = True
            n += 1
    return n
