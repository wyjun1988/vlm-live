"""Live3R 설정.

YAML 한 장이 모델 스케일(4B/2B/0.8B) × 기하 인코더 × 융합 방식을 결정한다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class GeometryConfig:
    """기하 인코더 설정."""

    # registry 에 등록된 이름: dummy | cut3r | anchor3r | lingbot | vggt_window
    name: str = "dummy"
    # 체크포인트 경로 (dummy 는 불필요)
    checkpoint: str | None = None
    # 인코더에서 뽑아올 레이어 인덱스. 길이 = 주입 지점 수.
    # SpatialStack 기본값 [11, 17, 23] 을 따른다.
    tap_layers: tuple[int, ...] = (11, 17, 23)
    # 인코더 잠재 차원
    hidden_size: int = 1024
    # 인코더 입력 해상도 (짧은 변)
    image_size: int = 518
    # 인코더를 돌릴 프레임 간격. 1 = 매 프레임. 0.8B 에서 비용 축소용.
    stride: int = 1
    # 인코더 파라미터 동결 여부. 기본 True (docs/DESIGN.md §1.4)
    freeze: bool = True
    # 카메라/포즈 토큰을 별도 슬롯으로 LLM 에 노출할지 (OVO-S-Bench L4 대응)
    expose_pose_token: bool = True
    # 어댑터별 추가 옵션 (예: cut3r 의 repo_path). 어댑터가 해석한다.
    options: dict = field(default_factory=dict)


@dataclass
class FusionConfig:
    """기하 토큰 → LLM 주입 설정."""

    # deepstack: 디코더 여러 깊이에 residual add (SpatialStack 식, 기본값)
    # xattn:     LLM 입력 직전 cross-attention 1회 (VLM-3R 식, ablation)
    # none:      주입 없음 (베이스라인)
    mode: Literal["deepstack", "xattn", "none"] = "deepstack"
    # 주입할 LLM 디코더 레이어 인덱스. geometry.tap_layers 와 길이가 같아야 한다.
    inject_layers: tuple[int, ...] = (0, 1, 2)
    # 기하 토큰 공간 머지 비율 (2 = 2x2 → 1). 토큰 예산 축소 축.
    merge_size: int = 2
    # 프로젝터 은닉 배율
    mlp_ratio: float = 2.0
    # 마지막 Linear 를 0 으로 초기화 → 학습 시작 시 베이스 VLM 동작을 그대로 보존
    zero_init: bool = True
    dropout: float = 0.0


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    # Qwen3.5 는 하이브리드(GatedDeltaNet + Attention)라 타깃 이름이 둘로 갈린다.
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkvz",
        "in_proj_ba",
        "out_proj",
    )


@dataclass
class StreamConfig:
    """라이브 런타임 설정."""

    # 비전 토큰 KV 예산. 초과 시 policy 적용.
    max_visual_tokens: int = 16384
    # keyframe | merge | evict | none
    budget_policy: Literal["keyframe", "merge", "evict", "none"] = "evict"
    # 목표 입력 fps (계측·샘플링 기준)
    target_fps: float = 30.0


@dataclass
class Live3RConfig:
    base_model: str = "Qwen/Qwen3.5-4B"
    # bf16 | fp16 | fp32
    dtype: str = "bf16"
    attn_implementation: str = "sdpa"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)

    def __post_init__(self) -> None:
        if self.fusion.mode == "deepstack":
            n_tap = len(self.geometry.tap_layers)
            n_inj = len(self.fusion.inject_layers)
            if n_tap != n_inj:
                raise ValueError(
                    f"geometry.tap_layers({n_tap}) 와 fusion.inject_layers({n_inj}) 길이가 달라야 할 이유가 없다. "
                    "주입 지점 하나당 탭 하나."
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Live3RConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw or {})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Live3RConfig":
        raw = dict(raw)
        sub = {
            "geometry": GeometryConfig,
            "fusion": FusionConfig,
            "lora": LoRAConfig,
            "stream": StreamConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sub.items():
            kwargs[key] = _build(klass, raw.pop(key, None))
        kwargs.update(raw)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(klass: type, raw: dict[str, Any] | None):
    if raw is None:
        return klass()
    fields = {f.name: f for f in dataclasses.fields(klass)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"{klass.__name__} 에 없는 키: {sorted(unknown)}")
    kwargs = {}
    for k, v in raw.items():
        # tuple 필드는 YAML 에서 list 로 온다 (dict 필드는 그대로 둔다)
        if isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    return klass(**kwargs)
