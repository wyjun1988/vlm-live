"""Gated input fusion - geometry enters once, at the LLM input, mixed into each visual token by a learned gate.

Our implementation of the input fusion design the 70.83 reference trains (docs/REFERENCE_70_ANALYSIS.md §6):

    v   = the visual token's embedding (what the LLM would otherwise see at that position)
    geo = the geometry token at the same place (GeometryProjector: one CUT3R decoder tap, 2x2 merge, MLP -> d_llm)
    g   = sigmoid(w · [N_v(v); N_g(geo)] + b)          one scalar per token
    out = (1 - g) · N_v(v) + g · N_g(geo)              replaces v

N_v, N_g are LayerNorms or identities - `norm`: "both" (the reference's gated_token), "geo" (its gated_token_wonorm:
the gate still reads both normalised streams, the output mixes the raw visual token with the normalised geometry -
out = (1 - g) · v + g · N_g(geo)), or "none". True / False mean "both" / "none".

Against deepstack (fusion/deepstack.py): one place instead of several, a replacement instead of a residual add, and
not the identity at the start. With `norm=True` the visual tokens are renormalised and mixed half-and-half with
geometry at step 0 (w = 0, b = `bias_init` = 0 -> g = 0.5): the design is trained together with LoRA in one stage on
a lot of data. `bias_init` sets the starting mix (-4 -> g = 0.018); with `norm=False` and a negative bias the start
is close to the base model.

Where: a forward pre-hook on decoder layer 0 replaces the hidden states at the visual positions - exactly where the
vision embeddings were scattered in, before any layer has read them. `InputFusionInjector` has DeepStackInjector's
interface (prime / clear / primed / measure / hit_count / skip_count), so every training and evaluation path that
primes the injector works unchanged. As there, training keeps the state primed until backward has run: gradient
checkpointing replays layer 0's forward, and this hook with it.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn


NORM_MODES = ("both", "geo", "none")


def norm_mode(norm) -> str:
    """True / False / "both" / "geo" / "none" -> one of NORM_MODES."""
    if norm is True:
        return "both"
    if norm is False:
        return "none"
    if norm not in NORM_MODES:
        raise ValueError(f"gate norm must be one of {NORM_MODES} (or a bool), got {norm!r}")
    return norm


class GatedTokenFusion(nn.Module):
    """[n, d] visual and [n, d] geometry tokens -> [n, d] fused tokens and the [n, 1] gate. Computed in fp32.

    init_std 0: the gate weights start at 0 - every token at the same mix sigmoid(bias_init). init_std 0.02: weights
    N(0, 0.02) as the reference's gate actually started (its checkpoint: std 0.0193, bias 0) - with normalised inputs
    the starting logit is ~N(0, 1.43^2) per token, g spread over 0.14-0.86 (10-90 %)."""

    def __init__(self, d_llm: int, norm=True, bias_init: float = 0.0, init_std: float = 0.0) -> None:
        super().__init__()
        self.mode = norm_mode(norm)
        self.norm_v = nn.LayerNorm(d_llm) if self.mode != "none" else nn.Identity()
        self.norm_g = nn.LayerNorm(d_llm) if self.mode != "none" else nn.Identity()
        self.gate = nn.Linear(2 * d_llm, 1)
        if init_std > 0:
            nn.init.normal_(self.gate.weight, std=float(init_std))
        else:
            nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(bias_init))

    def forward(self, visual: torch.Tensor, geo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if visual.shape != geo.shape:
            raise ValueError(f"visual {tuple(visual.shape)} and geometry {tuple(geo.shape)} tokens differ in shape")
        v, g = self.norm_v(visual), self.norm_g(geo)
        a = torch.sigmoid(self.gate(torch.cat([v, g], dim=-1)))
        base = visual if self.mode == "geo" else v                   # "geo": the visual token itself is not normalised
        return base + a * (g - base), a


class InputFusionInjector:
    """Replaces the visual positions of decoder layer 0's input by `fusion(visual, geometry)`."""

    def __init__(self, layers: nn.ModuleList, fusion: GatedTokenFusion) -> None:
        if len(layers) == 0:
            raise ValueError("no decoder layers")
        self.layers = layers
        self.fusion = fusion
        self.inject_layers = (0,)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._embeds: list[torch.Tensor] | None = None
        self._mask: torch.Tensor | None = None
        #: forwards that fused (prefill), and forwards skipped because they carry no visual position (decode steps)
        self.hit_count = 0
        self.skip_count = 0
        #: measure=True records the gate over the visual tokens: (mean, std) in `last_gate` (training log)
        self.measure = False
        self.last_gate: tuple[float, float] | None = None
        #: DeepStackInjector's per-layer ‖injection‖/‖visual‖ ratio - not meaningful for a replacement; kept empty so
        #: the trainer's overload warning never fires on it
        self.last_ratio: dict[int, float] = {}

    # --------------------------------------------------------------- lifecycle
    def attach(self) -> "InputFusionInjector":
        if not self._handles:
            self._handles.append(self.layers[0].register_forward_pre_hook(self._hook))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _check(self, embeds: list[torch.Tensor]) -> None:
        if len(embeds) != 1:
            raise ValueError(f"gated input fusion takes one geometry embedding (got {len(embeds)})")

    def prime(self, embeds: list[torch.Tensor], visual_pos_mask: torch.Tensor) -> None:
        """Arm for the next forward and keep it until clear() (training: backward replays the hook)."""
        self._check(embeds)
        self._embeds, self._mask = embeds, visual_pos_mask

    def clear(self) -> None:
        self._embeds, self._mask = None, None

    @property
    def is_primed(self) -> bool:
        return self._embeds is not None

    @contextlib.contextmanager
    def primed(self, embeds: list[torch.Tensor], visual_pos_mask: torch.Tensor):
        """Arm for the forwards inside the block (inference). embeds[0]: [n_visual, d]; mask: [B, S] of this forward."""
        self._check(embeds)
        prev = (self._embeds, self._mask)
        self._embeds, self._mask = embeds, visual_pos_mask
        try:
            yield self
        finally:
            self._embeds, self._mask = prev

    # ------------------------------------------------------------------- hook
    def _hook(self, module, args):
        if self._embeds is None or self._mask is None or not args:
            return None
        emb = self._embeds[0]
        hidden = args[0]
        if emb is None or not isinstance(hidden, torch.Tensor):
            return None
        mask = self._mask.to(hidden.device)
        if mask.shape[:2] != hidden.shape[:2]:
            # a generate() decode step: one new token, no visual position
            self.skip_count += 1
            return None
        n_pos = int(mask.sum().item())
        if n_pos != emb.shape[0]:
            raise RuntimeError(f"{n_pos} visual positions != {emb.shape[0]} geometry tokens - the per-frame token "
                               "counts of the vision and geometry paths disagree")
        visual = hidden[mask]
        fused, gate = self.fusion(visual.float(), emb.to(hidden.device).float())
        if self.measure:
            with torch.no_grad():
                self.last_gate = (float(gate.mean()), float(gate.std()) if gate.numel() > 1 else 0.0)
        hidden = hidden.clone()
        hidden[mask] = fused.to(hidden.dtype)
        self.hit_count += 1
        return (hidden,) + tuple(args[1:])
