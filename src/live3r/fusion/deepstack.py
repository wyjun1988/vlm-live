"""DeepStack 주입 — 디코더 여러 깊이의 비전 토큰 위치에 기하 특징을 residual 로 더한다.

Qwen3-VL 에는 이 경로가 네이티브로 있었지만(`deepstack_visual_embeds`),
**Qwen3.5 는 이 코드를 들고 있지 않다** (`deepstack_visual_indexes: []`, modeling 에 처리 없음).
그래서 직접 건다. 다행히 `Qwen3_5DecoderLayer.forward` 는 히든 텐서를 그대로 반환해서
forward hook 이 반환값을 갈아끼우는 것만으로 깔끔하게 주입된다. transformers 를 패치하지 않는다.

두 가지 사용법:
    primed(...)  컨텍스트 — 추론용. 블록을 나가면 주입 상태가 비워진다.
    prime()/clear() — **학습용**. gradient checkpointing 은 역전파 때 레이어 forward 를
        재실행하고, 그때 이 훅도 다시 불린다. 주입 상태가 이미 비워져 있으면 재실행 그래프가
        원래와 달라져(주입 없음) 기울기가 틀리거나 CheckpointError 가 난다.
        그래서 학습 루프는 prime → forward → backward → clear 순서로 쓴다.
"""

from __future__ import annotations

import contextlib
from typing import Iterable

import torch
from torch import nn


class DeepStackInjector:
    """선택한 디코더 레이어 출력에 기하 임베딩을 더하는 훅 관리자."""

    def __init__(self, layers: nn.ModuleList, inject_layers: Iterable[int]) -> None:
        self.layers = layers
        self.inject_layers = tuple(inject_layers)
        n = len(layers)
        bad = [i for i in self.inject_layers if not (0 <= i < n)]
        if bad:
            raise ValueError(f"주입 레이어 인덱스가 범위 밖: {bad} (레이어 수 {n})")
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._embeds: list[torch.Tensor] | None = None
        self._mask: torch.Tensor | None = None
        #: 훅이 실제로 주입한 횟수 — 조용한 미주입을 잡는 계측
        self.hit_count = 0
        #: 프리필이 아닌 forward(=디코딩 스텝)라 건너뛴 횟수.
        #: generate() 는 프리필 1회 + 디코딩 N회를 부른다. 디코딩 스텝에는 비전 토큰이
        #: 아예 없으므로 주입할 것도 없다. 숨기지 않고 센다.
        self.skip_count = 0
        #: measure=True 면 레이어별 ‖주입‖/‖비전 히든‖ 비율을 기록한다 (학습 로그용).
        #: "기하가 실제로 쓰이고 있나"의 직접 지표다 — gate 값보다 훨씬 해석이 쉽다.
        self.measure = False
        self.last_ratio: dict[int, float] = {}

    # --------------------------------------------------------------- lifecycle
    def attach(self) -> "DeepStackInjector":
        if self._handles:
            return self
        for k, layer_idx in enumerate(self.inject_layers):
            self._handles.append(
                self.layers[layer_idx].register_forward_hook(self._make_hook(k))
            )
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _check(self, embeds: list[torch.Tensor]) -> None:
        if len(embeds) != len(self.inject_layers):
            raise ValueError(
                f"embeds {len(embeds)}개 != inject_layers {len(self.inject_layers)}개"
            )

    def prime(self, embeds: list[torch.Tensor], visual_pos_mask: torch.Tensor) -> None:
        """주입 상태를 건다. clear() 전까지 유지된다 (역전파 재계산까지 살아 있어야 할 때)."""
        self._check(embeds)
        self._embeds, self._mask = embeds, visual_pos_mask

    def clear(self) -> None:
        self._embeds, self._mask = None, None

    @property
    def is_primed(self) -> bool:
        return self._embeds is not None

    @contextlib.contextmanager
    def primed(self, embeds: list[torch.Tensor], visual_pos_mask: torch.Tensor):
        """이번 forward 에 쓸 임베딩과 비전 위치 마스크를 걸어둔다 (추론용).

        Args:
            embeds: 길이 = len(inject_layers). 각 [n_visual_total, d_llm]
                (마스크가 True 인 위치 수와 같아야 한다)
            visual_pos_mask: [B, S] bool — **이번 forward 에 들어가는 토큰 구간 기준**.
        """
        self._check(embeds)
        prev = (self._embeds, self._mask)
        self._embeds, self._mask = embeds, visual_pos_mask
        try:
            yield self
        finally:
            self._embeds, self._mask = prev

    # ------------------------------------------------------------------- hooks
    def _make_hook(self, k: int):
        layer_idx = self.inject_layers[k]

        def hook(module, args, output):
            if self._embeds is None or self._mask is None:
                return output
            emb = self._embeds[k]
            if emb is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            mask = self._mask.to(hidden.device)
            if mask.shape[:2] != hidden.shape[:2]:
                # generate() 의 디코딩 스텝 — 새 토큰 1개뿐이라 비전 위치가 없다.
                self.skip_count += 1
                return output
            n_pos = int(mask.sum().item())
            if n_pos != emb.shape[0]:
                raise RuntimeError(
                    f"비전 위치 {n_pos}개 != 기하 임베딩 {emb.shape[0]}개. "
                    "프레임당 토큰 수 계산이 어긋났다."
                )
            emb = emb.to(hidden.device, hidden.dtype)
            if self.measure:
                with torch.no_grad():
                    h_norm = hidden[mask].float().norm(dim=-1).mean()
                    e_norm = emb.float().norm(dim=-1).mean()
                    self.last_ratio[layer_idx] = float(e_norm / h_norm.clamp_min(1e-6))
            hidden = hidden.clone()
            hidden[mask] = hidden[mask] + emb
            self.hit_count += 1
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        return hook
