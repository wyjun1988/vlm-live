"""Fused kernels for Qwen3.5's Gated DeltaNet layers - training speed (2026-10-13).

transformers >= 5.16 runs the 24 Gated DeltaNet layers through `torch_chunk_gated_delta_rule`, a pure-torch loop over
64-token chunks, unless the model is kernelized through the `kernels` hub package: an installed flash-linear-attention
(fla) is no longer used by itself. Run 3 trained at 2.1 samples/s per 8 H100 against the reference's 6.48
(FlashAttention-3, ZeRO-2), and its step profile was tens of thousands of small ops - that loop.

`LIVE3R_GDN=fla` routes the chunk and recurrent rules to fla's fused Triton kernels (package fla-core since fla 0.5;
flash-linear-attention alone is the layers), which take the same arguments
(q, k, v as [B, T, H, D], log-space g, beta; initial_state, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel)
and are what transformers' own hub mapping (kernels-community/fla) loads. Off by default, so that evaluations keep the
numerics of the runs they are compared with; `scripts/diagnose_throughput.py --gdn-check` measures the two on the
server (loss, gradients, time) before a training run turns it on.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

CALLS = {"chunk": 0, "recurrent": 0}     # how often the fused rules ran (the diagnostic checks the patch took effect)


def gdn_backend() -> str:
    """The Gated DeltaNet implementation asked for: LIVE3R_GDN = torch (default) | fla."""
    value = os.environ.get("LIVE3R_GDN", "torch").strip().lower()
    if value not in ("torch", "fla"):
        raise ValueError(f"LIVE3R_GDN must be torch or fla, got {value!r}")
    return value


def use_fla_gated_delta_rule() -> bool:
    """Route transformers' Qwen3.5 Gated DeltaNet to fla's fused kernels, module-wide. False when fla is missing."""
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except Exception as exc:  # noqa: BLE001
        log.warning("LIVE3R_GDN=fla but fla's kernels cannot be imported (%r): since fla 0.5 they are the package "
                    "fla-core (pip install --no-deps flash-linear-attention fla-core einops) - the torch path stays", exc)
        return False
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    def chunk(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, cu_seqlens=None, **_):
        CALLS["chunk"] += 1
        return chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=initial_state,
                                      output_final_state=output_final_state, cu_seqlens=cu_seqlens,
                                      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    def recurrent(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False,
                  cu_seqlens=None, **_):
        CALLS["recurrent"] += 1
        return fused_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=initial_state,
                                                output_final_state=output_final_state, cu_seqlens=cu_seqlens,
                                                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    q.torch_chunk_gated_delta_rule = chunk
    q.torch_recurrent_gated_delta_rule = recurrent
    return True


def apply_gdn_backend() -> str:
    """Apply LIVE3R_GDN; returns what will run: "fla" or "torch"."""
    if gdn_backend() == "fla" and use_fla_gated_delta_rule():
        return "fla"
    return "torch"
