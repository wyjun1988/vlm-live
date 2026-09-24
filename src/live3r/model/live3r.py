"""Live3RModel — Qwen3.5 VLM + 스트리밍 기하 인코더.

책임 분리:
  * 기하 인코더(동결·바닐라) : 프레임 → 잠재 토큰
  * GeometryProjector      : 잠재 토큰 → LLM 히든 공간 (학습)
  * DeepStackInjector      : 디코더 앞단 여러 깊이에 residual add
  * LLM                    : LoRA 만 학습

"스텝" 이라는 단위: LLM 비전 토큰 블록 하나.
    이미지 모드 — 이미지 1장 = 1스텝 (Qwen 이 이미지를 temporal 로 복제해 한 블록을 만든다)
    비디오 모드 — 프레임 temporal_patch(=2)장 = 1스텝
기하 인코더는 프레임마다 1출력이므로, 비디오 모드에서만 2프레임씩 묶어 평균낸다.

조용히 틀리기 쉬운 형상 문제 (전부 테스트로 막아둠):
  1) 격자 불일치 — 기하 인코더 patch 16/14 vs Qwen3.5 patch16+merge2. projector 가 리샘플한다.
  2) 시간축 불일치 — 비디오는 2프레임=1블록. `_pool_frames_to_patches`.
  3) 스텝마다 격자가 다를 수 있다 — 이미지 모드에서 이미지 크기가 제각각이면. 스텝별 격자를 받는다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from ..config import Live3RConfig
from ..data.vision import VisionSpec, geometry_frame, llm_grid, unpatchify
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
    pose: torch.Tensor | None = None  # 스텝별 포즈 임베딩 [steps, d_llm] (embeds[0] 에 이미 더해져 있음)
    steps: int = 0


class Live3RModel(nn.Module):
    def __init__(self, cfg: Live3RConfig, base: nn.Module, processor=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.base = base
        self.processor = processor

        text_cfg = base.config.text_config
        vis_cfg = base.config.vision_config
        self.d_llm: int = text_cfg.hidden_size
        self.n_layers: int = text_cfg.num_hidden_layers
        self.spec = VisionSpec.from_config(
            vis_cfg, min_pixels=cfg.data.min_pixels, max_pixels=cfg.data.max_pixels
        )
        self.vision_patch: int = self.spec.patch
        self.spatial_merge: int = self.spec.merge
        self.temporal_patch: int = self.spec.temporal_patch
        self.image_token_id: int = base.config.image_token_id
        self.video_token_id: int = base.config.video_token_id
        self.vision_start_id: int = base.config.vision_start_token_id
        self.vision_end_id: int = base.config.vision_end_token_id
        self._stop_at_im_end()

        # ---- 기하 인코더 (동결·바닐라) ----
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
                for _ in cfg.fusion.inject_layers
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
        self._auto_geom_orig = None

    # ------------------------------------------------------------------ 생성자
    @classmethod
    def from_pretrained(cls, cfg: Live3RConfig, tokenizer_path: str | None = None, **kwargs) -> "Live3RModel":
        """HF 체크포인트에서 만든다.

        `base_model: tiny` 는 랜덤 초소형 모델 + 실제 Qwen3.5 토크나이저(tokenizer_path 또는
        환경변수 LIVE3R_TOKENIZER) — 학습 루프·DDP 를 가중치 없이 끝까지 돌려보는 용도다.
        """
        import os

        from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

        if cfg.base_model.startswith("tiny"):
            from ..testing import QWEN35_IDS, tiny_model

            base = tiny_model(num_layers=8, hidden=64, ids=QWEN35_IDS)
            tok_path = tokenizer_path or os.environ.get("LIVE3R_TOKENIZER")
            tok = AutoTokenizer.from_pretrained(tok_path) if tok_path else None
            return cls(cfg, base, tok)

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

    @property
    def tokenizer(self):
        if self.processor is None:
            return None
        return getattr(self.processor, "tokenizer", self.processor)

    def _stop_at_im_end(self) -> None:
        """생성이 `<|im_end|>` 에서 멈추게 한다.

        Qwen3.5 로컬 체크포인트에는 generation_config.json 이 없고 config 의 eos 는 `<|endoftext|>`(248044) 뿐이다.
        그러면 generate 가 답 끝(`<|im_end|>`, 248046)에서 안 멈춘다. 베이스는 `<|im_end|>` 뒤에 대개
        `<|endoftext|>` 를 내서 티가 안 났지만, LoRA 학습 후에는 다음 턴(`\n<|im_start|>assistant…`)을 이어
        쓴다 (M2 S2 파일럿: 18%). 채점은 첫 단어라 점수엔 안 보이지만 라이브 답에 쓰레기가 붙고 디코딩 시간을 버린다.
        (lmms-eval 경로는 generate 인자로 eos=<|im_end|> 를 따로 넘겨서 원래 괜찮았다.)
        """
        tok = self.tokenizer
        gc = getattr(self.base, "generation_config", None)
        if tok is None or gc is None:
            return
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if im_end is None or im_end == getattr(tok, "unk_token_id", None):
            return
        eos = gc.eos_token_id
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        if im_end not in eos:
            gc.eos_token_id = [im_end] + eos
        if gc.pad_token_id is None:
            gc.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else im_end

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

    def train(self, mode: bool = True):
        """`model.train()` 이 동결된 기하 인코더까지 train 모드로 바꾸지 않게 막는다.

        안 막으면 CUT3R 의 dropout/drop-path 가 켜져 **학습 때만 기하 토큰이 흔들린다**
        (추론 때와 분포가 달라진다). 동결 인코더는 항상 eval 이다.
        """
        super().train(mode)
        if self.cfg.geometry.freeze:
            self.geometry.eval()
        return self

    # ---------------------------------------------------------------- 기하 경로
    @staticmethod
    def _pool_frames_to_patches(
        outs: list[GeomOutput], temporal_patch: int
    ) -> list[GeomOutput]:
        """프레임 단위 기하 출력을 Qwen 의 temporal patch 단위로 묶는다 (비디오 모드 전용).

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
        llm_grids,
        pool_temporal: bool = True,
    ) -> GeometryBundle:
        """기하 출력 시퀀스 → 주입용 임베딩.

        Args:
            geom_outs: 프레임(또는 이미지) 순서대로의 GeomOutput (배치 1 가정)
            llm_grids: LLM 비전 토큰 격자. (H, W) 하나면 모든 스텝에 공통,
                리스트면 스텝별 — 이미지 크기가 제각각인 이미지 모드에서 필요하다.
            pool_temporal: 비디오 모드면 True (2프레임→1스텝). 이미지 모드면 False.
        Returns:
            GeometryBundle — embeds[k] 는 [Σ스텝토큰, d_llm], 시퀀스 등장 순서대로
        """
        if pool_temporal:
            geom_outs = self._pool_frames_to_patches(geom_outs, self.temporal_patch)
        steps = len(geom_outs)
        if isinstance(llm_grids, tuple) and len(llm_grids) == 2 and isinstance(llm_grids[0], int):
            grids = [llm_grids] * steps
        else:
            grids = [tuple(g) for g in llm_grids]
        if len(grids) != steps:
            raise ValueError(
                f"스텝별 격자 {len(grids)}개 != 기하 스텝 {steps}개. "
                "이미지 모드에서 pool_temporal=True 로 불렀거나, 이미지 수와 기하 출력 수가 다르다."
            )

        taps = self.cfg.geometry.tap_layers
        pose = None
        if self.pose_projector is not None and geom_outs[0].pose_token is not None:
            # 스텝별 포즈 [steps, d] — 그 스텝의 모든 비전 토큰에 더한다 (카메라 임베딩처럼)
            pose = torch.cat([self.pose_projector(o.pose_token)[:, 0] for o in geom_outs], dim=0)

        embeds: list[torch.Tensor] = []
        for k, proj in enumerate(self.projectors):
            per_step = []
            for s, out in enumerate(geom_outs):
                e = proj(out.tokens[taps[k]], out.grid_hw, grids[s])  # [B, H*W, d]
                if k == 0 and pose is not None:
                    e = e + pose[s].view(1, 1, -1)
                per_step.append(e)
            embeds.append(torch.cat(per_step, dim=1).flatten(0, 1))
        return GeometryBundle(embeds=embeds, pose=pose, steps=steps)

    def geometry_frames(self, frames01: torch.Tensor) -> torch.Tensor:
        """[T,3,H,W] ∈ [0,1] → 기하 인코더 입력 [T,3,h,w] (긴 변 기준, 정규화됨)."""
        unit = getattr(self.geometry, "patch_size", 16)
        size = self.cfg.geometry.image_size
        return torch.stack([geometry_frame(f, size, unit) for f in frames01], 0)

    @torch.no_grad()
    def run_geometry(self, geom_frames) -> list[GeomOutput]:
        """동결 인코더를 순서대로 돌린다. 샘플마다 상태를 리셋한다.

        geom_frames: [T,3,h,w] 텐서 또는 스텝마다 크기가 다른 [3,h,w] 들의 리스트.
        """
        self.geometry.reset()
        dev = next(self.projectors.parameters()).device
        return [self.geometry.ingest(f.unsqueeze(0).to(dev)) for f in geom_frames]

    # ------------------------------------------------------------------ forward
    def visual_pos_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids == self.image_token_id) | (input_ids == self.video_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        geometry: GeometryBundle | None = None,
        geom_outs: list[GeomOutput] | None = None,
        llm_grids=None,
        pool_temporal: bool = False,
        keep_primed: bool = False,
        **kwargs,
    ):
        """
        geometry: 미리 만든 임베딩 (추론)
        geom_outs + llm_grids: 기하 출력을 넘기면 **forward 안에서** 임베딩을 만든다 (학습).
            DDP 는 forward 안에서 쓰인 파라미터만 추적하므로, 프로젝터는 반드시 여기서 돌아야 한다.
        keep_primed: True 면 주입 상태를 forward 뒤에도 유지한다 (gradient checkpointing 재계산용).
            호출자가 backward 뒤에 `self.injector.clear()` 를 불러야 한다.
        둘 다 없으면 순수 베이스 VLM 동작 (베이스라인/회귀 확인용).
        """
        if geom_outs is not None:
            geometry = self.build_geometry_embeds(geom_outs, llm_grids, pool_temporal)
        if geometry is None or self.injector is None:
            return self._run_base(input_ids, attention_mask, labels, **kwargs)
        mask = self.visual_pos_mask(input_ids)
        if keep_primed:
            self.injector.prime(list(geometry.embeds), mask)
            return self._run_base(input_ids, attention_mask, labels, **kwargs)
        with self.injector.primed(list(geometry.embeds), mask):
            return self._run_base(input_ids, attention_mask, labels, **kwargs)

    def _run_base(self, input_ids, attention_mask, labels, **kwargs):
        """labels 가 있으면 **감독 위치의 로짓만** 만든다. 손실은 HF 의 전 위치 계산과 같다.

        Qwen3.5 어휘가 248k 라서 전 위치 로짓(CE 때 fp32 로 올린다)은 2k 토큰이면 2GB,
        max_length 8k 면 8GB 에 같은 크기의 기울기가 더 붙는다. 감독되는 건 답변 토큰 수십 개뿐이다
        (M2 실측: 0.8B 파일럿이 LM 헤드 로짓 1.9GB 한 번 할당에서 OOM). 반환하는 `.logits` 도
        그 위치들뿐이다 — 전 위치 로짓이 필요하면 labels 없이 불러라.
        """
        if labels is None:
            return self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        shift = labels[:, 1:]                                  # 위치 p 의 로짓이 labels[p+1] 을 맞힌다
        keep = (shift != -100).any(0).nonzero().squeeze(1)     # 배치 합집합 (배치마다 다른 건 -100 이 가린다)
        if keep.numel() == 0:
            return self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep, **kwargs)
        out.loss = nn.functional.cross_entropy(
            out.logits.float().flatten(0, 1), shift[:, keep].flatten(), ignore_index=-100)
        return out

    # ------------------------------------------------------- 자동 기하 (auto mode)
    def enable_auto_geometry(self, enabled: bool = True) -> "Live3RModel":
        """`base.forward` 를 감싸서 기하 인코딩·주입을 **자동으로** 하게 만든다.

        이게 있으면 `base.generate(...)` 를 그냥 부르는 외부 코드(lmms-eval 등)가
        수정 없이 Live3R 을 쓸 수 있다. 프리필(픽셀이 들어오는 스텝)에서만 기하를 돌리고,
        디코딩 스텝에는 비전 토큰이 없으므로 아무것도 하지 않는다.

        **주의**: VLM 전처리를 마친 픽셀을 기하 인코더에 재사용한다 (unpatchify 로 복원).
        CUT3R 과 Qwen3.5 의 정규화가 mean/std=0.5 로 같아서 성립한다 — 다른 인코더를
        꽂을 때는 정규화를 확인해라. 스트리밍 평가는 이 경로를 쓰지 않는다.
        """
        if not enabled:
            if self._auto_geom_orig is not None:
                self.base.forward = self._auto_geom_orig
                self._auto_geom_orig = None
            return self
        if self._auto_geom_orig is not None:
            return self
        if self.injector is None:
            raise RuntimeError("fusion.mode 가 deepstack 이 아니면 자동 기하를 쓸 수 없다")

        self._auto_geom_orig = self.base.forward
        orig = self._auto_geom_orig

        def wrapped(*args, **kwargs):
            bundle = self._geometry_from_inputs(kwargs)
            if bundle is None:
                return orig(*args, **kwargs)
            mask = self.visual_pos_mask(kwargs["input_ids"])
            with self.injector.primed(bundle.embeds, mask):
                return orig(*args, **kwargs)

        self.base.forward = wrapped
        return self

    @torch.no_grad()
    def _geometry_from_inputs(self, kwargs: dict) -> GeometryBundle | None:
        """프리필 입력의 픽셀에서 프레임을 복원해 기하 토큰을 만든다. 해당 없으면 None."""
        ids = kwargs.get("input_ids")
        if ids is None or not bool(self.visual_pos_mask(ids).any()):
            return None
        has_img = kwargs.get("pixel_values") is not None
        has_vid = kwargs.get("pixel_values_videos") is not None
        if has_img and has_vid:
            raise NotImplementedError("이미지와 비디오가 한 요청에 섞인 경우는 아직 지원하지 않는다")
        if not (has_img or has_vid):
            return None

        if has_vid:
            pv, grid = kwargs["pixel_values_videos"], kwargs["video_grid_thw"]
        else:
            pv, grid = kwargs["pixel_values"], kwargs["image_grid_thw"]

        frames, grids = [], []
        offset = 0
        for row in grid.reshape(-1, 3):
            n = int(row.prod())
            steps = unpatchify(pv[offset : offset + n].float().cpu(), row.view(1, 3), self.spec)
            offset += n
            if has_vid:
                frames.extend(list(steps))  # 비디오: 모든 프레임 (나중에 2장씩 묶는다)
                grids.extend([llm_grid(row, self.spec)] * int(row[0]))
            else:
                frames.append(steps[0])  # 이미지: temporal 로 복제된 것 중 하나
                grids.append(llm_grid(row, self.spec))

        frames01 = [(f * 0.5 + 0.5).clamp(0, 1) for f in frames]  # 정규화 해제
        geom = [self.geometry_frames(f.unsqueeze(0))[0] for f in frames01]
        outs = self.run_geometry(geom)
        return self.build_geometry_embeds(outs, grids, pool_temporal=has_vid)

    # ------------------------------------------------------------- 파라미터 관리
    def trainable_parameter_summary(self) -> dict[str, int]:
        total = trainable = 0
        by_group = {"projector": 0, "lora": 0, "other_trainable": 0}
        for name, p in self.named_parameters():
            total += p.numel()
            if not p.requires_grad:
                continue
            trainable += p.numel()
            if "projector" in name:
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

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """학습된 것만 (프로젝터 + LoRA). 베이스 가중치는 복사하지 않는다."""
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if ("projector" in k or "lora" in k.lower())
        }
