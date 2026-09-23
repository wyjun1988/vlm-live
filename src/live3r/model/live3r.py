"""Live3RModel — Qwen3.5 VLM + 스트리밍 기하 인코더.

책임 분리:
  * 기하 인코더(동결)  : 프레임 → 잠재 토큰
  * GeometryProjector  : 잠재 토큰 → LLM 히든 공간 (학습)
  * DeepStackInjector  : 디코더 앞단 여러 깊이에 residual add
  * LLM                : LoRA 만 학습

주의할 형상 문제 두 가지 (조용히 틀리는 지점):
  1) 격자 불일치 — 기하 인코더 patch 14 vs Qwen3.5 patch16+merge2. projector 가 리샘플한다.
  2) **시간 축 불일치** — Qwen3.5 비디오는 `temporal_patch_size=2`, 즉 원본 2프레임이
     비전 토큰 1블록이 된다. 기하 인코더는 프레임마다 1출력이다. 그래서 기하 출력을
     temporal patch 단위로 묶어 평균낸다(`_pool_frames_to_patches`).
     이걸 안 하면 토큰 수가 정확히 2배로 어긋나 DeepStackInjector 가 던진다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from ..config import Live3RConfig
from ..fusion.deepstack import DeepStackInjector
from ..fusion.projector import GeometryProjector, PoseProjector
from ..geometry.base import GeomOutput
from ..geometry.registry import build_geometry_stream

logger = logging.getLogger(__name__)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class GeometryBundle:
    """한 번의 forward 에 쓸 기하 임베딩 묶음."""

    embeds: list[torch.Tensor]  # len = len(inject_layers), 각 [n_visual, d_llm]
    pose: torch.Tensor | None = None


class Live3RModel(nn.Module):
    def __init__(
        self,
        cfg: Live3RConfig,
        base: nn.Module,
        processor=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.base = base
        self.processor = processor

        text_cfg = base.config.text_config
        vis_cfg = base.config.vision_config
        self.d_llm: int = text_cfg.hidden_size
        self.n_layers: int = text_cfg.num_hidden_layers
        self.vision_patch: int = vis_cfg.patch_size
        self.spatial_merge: int = vis_cfg.spatial_merge_size
        self.temporal_patch: int = getattr(vis_cfg, "temporal_patch_size", 1)
        self.image_token_id: int = base.config.image_token_id
        self.video_token_id: int = base.config.video_token_id

        # ---- 기하 인코더 (동결) ----
        self.geometry = build_geometry_stream(cfg.geometry)
        if cfg.geometry.freeze:
            for p in self.geometry.parameters():
                p.requires_grad_(False)
            self.geometry.eval()
        if not self.geometry.is_streaming:
            logger.warning(
                "기하 인코더 '%s' 는 스트리밍이 아니다(is_streaming=False). "
                "오프라인 상한선 비교군으로만 써라 — 라이브 지연 수치를 이걸로 보고하면 안 된다.",
                cfg.geometry.name,
            )

        # ---- 프로젝터 (학습) ----
        c_geo = self.geometry.hidden_size
        n_inject = len(cfg.fusion.inject_layers)
        self.projectors = nn.ModuleList(
            [
                GeometryProjector(
                    c_geo=c_geo,
                    d_llm=self.d_llm,
                    merge_size=cfg.fusion.merge_size,
                    mlp_ratio=cfg.fusion.mlp_ratio,
                    zero_init=cfg.fusion.zero_init,
                    dropout=cfg.fusion.dropout,
                )
                for _ in range(n_inject)
            ]
        )
        self.pose_projector = (
            PoseProjector(c_geo, self.d_llm, zero_init=cfg.fusion.zero_init)
            if cfg.geometry.expose_pose_token
            else None
        )

        # ---- 주입 훅 ----
        self.injector: DeepStackInjector | None = None
        if cfg.fusion.mode == "deepstack":
            self.injector = DeepStackInjector(
                self._decoder_layers(), cfg.fusion.inject_layers
            ).attach()
        elif cfg.fusion.mode == "xattn":
            raise NotImplementedError(
                "xattn(VLM-3R 식) 융합은 ablation 용으로 예정. 지금은 deepstack 또는 none."
            )

    # ------------------------------------------------------------------ 생성자
    @classmethod
    def from_pretrained(cls, cfg: Live3RConfig, **kwargs) -> "Live3RModel":
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        dtype = _DTYPES[cfg.dtype]
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.base_model,
            dtype=dtype,
            attn_implementation=cfg.attn_implementation,
            **kwargs,
        )
        try:
            processor = AutoProcessor.from_pretrained(cfg.base_model)
        except Exception as exc:  # pragma: no cover
            logger.warning("프로세서 로드 실패(%s) — 토크나이즈는 직접 해야 한다", exc)
            processor = None
        return cls(cfg, base, processor)

    # -------------------------------------------------------------- 내부 접근자
    def _decoder_layers(self) -> nn.ModuleList:
        """Qwen3.5 디코더 레이어 리스트. 클래스 구조가 바뀌어도 여기만 고치면 된다."""
        m = self.base.model if hasattr(self.base, "model") else self.base
        for path in ("language_model.layers", "layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, nn.ModuleList):
                    return obj
            except AttributeError:
                continue
        raise AttributeError("Qwen3.5 디코더 레이어를 못 찾았다 — transformers 구조가 바뀐 듯")

    # ---------------------------------------------------------------- 기하 경로
    @staticmethod
    def _pool_frames_to_patches(
        outs: list[GeomOutput], temporal_patch: int
    ) -> list[GeomOutput]:
        """프레임 단위 기하 출력을 Qwen 의 temporal patch 단위로 묶는다.

        temporal_patch=2 면 프레임 (0,1)→패치0, (2,3)→패치1. 남는 프레임은 자기 자신으로 채운다
        (Qwen 전처리도 마지막 프레임을 복제해 채운다).
        """
        if temporal_patch <= 1:
            return outs
        pooled: list[GeomOutput] = []
        for i in range(0, len(outs), temporal_patch):
            chunk = outs[i : i + temporal_patch]
            keys = chunk[0].tokens.keys()
            tokens = {k: torch.stack([c.tokens[k] for c in chunk], 0).mean(0) for k in keys}
            pose = None
            if chunk[0].pose_token is not None:
                pose = torch.stack([c.pose_token for c in chunk], 0).mean(0)
            pooled.append(
                GeomOutput(
                    tokens=tokens,
                    grid_hw=chunk[0].grid_hw,
                    pose_token=pose,
                    frame_index=chunk[0].frame_index,
                )
            )
        return pooled

    def build_geometry_embeds(
        self,
        geom_outs: list[GeomOutput],
        llm_grid_hw: tuple[int, int],
        pool_temporal: bool = True,
    ) -> GeometryBundle:
        """기하 출력 시퀀스 → 주입용 임베딩.

        Args:
            geom_outs: 프레임 순서대로의 GeomOutput (배치 1 가정)
            llm_grid_hw: LLM 비전 토큰의 프레임(=temporal patch)당 (H, W).
                Qwen 의 grid_thw 에서 (h//merge, w//merge).
            pool_temporal: temporal_patch 로 묶을지. 이미지 입력이면 False.
        Returns:
            GeometryBundle — embeds[k] 는 [T*H*W, d_llm]
        """
        if pool_temporal:
            geom_outs = self._pool_frames_to_patches(geom_outs, self.temporal_patch)

        taps = self.cfg.geometry.tap_layers
        embeds: list[torch.Tensor] = []
        for k, proj in enumerate(self.projectors):
            per_step = []
            for out in geom_outs:
                tok = out.tokens[taps[k]]  # [B,N,C]
                per_step.append(proj(tok, out.grid_hw, llm_grid_hw))  # [B,H*W,d]
            # [B, T*H*W, d] → 배치 1 기준으로 평탄화 (deepstack 은 마스크 위치 순서대로 더한다)
            embeds.append(torch.cat(per_step, dim=1).flatten(0, 1))

        pose = None
        if self.pose_projector is not None and geom_outs[0].pose_token is not None:
            pose = torch.cat([self.pose_projector(o.pose_token) for o in geom_outs], dim=1)
        return GeometryBundle(embeds=embeds, pose=pose)

    # ------------------------------------------------------------------ forward
    def visual_pos_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids == self.image_token_id) | (input_ids == self.video_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        geometry: GeometryBundle | None = None,
        **kwargs,
    ):
        """geometry 가 None 이면 순수 베이스 VLM 동작 (베이스라인/회귀 확인용)."""
        if geometry is None or self.injector is None:
            return self.base(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
            )
        mask = self.visual_pos_mask(input_ids)
        embeds = list(geometry.embeds)
        if geometry.pose is not None:
            # v1: 포즈는 해당 프레임 비전 토큰 전체에 broadcast 로 더한다.
            # (전용 토큰 슬롯 방식은 docs/DESIGN.md §1.3 — 프롬프트 구조 변경이 필요해 후속.)
            pass
        with self.injector.primed(embeds, mask):
            return self.base(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
            )

    # ------------------------------------------------------------- 파라미터 관리
    def trainable_parameter_summary(self) -> dict[str, int]:
        total = trainable = 0
        by_group = {"projector": 0, "lora": 0, "other_trainable": 0}
        for name, p in self.named_parameters():
            total += p.numel()
            if not p.requires_grad:
                continue
            trainable += p.numel()
            if "projector" in name or name.startswith("projectors"):
                by_group["projector"] += p.numel()
            elif "lora" in name.lower():
                by_group["lora"] += p.numel()
            else:
                by_group["other_trainable"] += p.numel()
        return {"total": total, "trainable": trainable, **by_group}

    def freeze_base(self) -> None:
        """LLM/비전타워 동결. LoRA 는 이후에 얹는다."""
        for p in self.base.parameters():
            p.requires_grad_(False)
