"""LoRA 적용 — LLM 만, 비전타워·기하인코더는 건드리지 않는다.

왜 LoRA 만 쓰는가:
    OVO-S-Bench 저자들이 "특화 역설"을 보고했다 — 스트리밍/공간 파인튜닝 변형 15개 중 13개가
    자기 베이스 백본보다 낮았다(docs/RESEARCH_NOTES.md §5-1). 전체 파인튜닝은 공간 점수를
    올리면서 일반 능력과 전역 매핑(L4)을 갉아먹는다. LoRA + 동결 + zero-init 프로젝터 +
    일반 데이터 리플레이가 우리의 방어선이다.
"""

from __future__ import annotations

import logging
import re

from torch import nn

from ..config import LoRAConfig

logger = logging.getLogger(__name__)


def apply_lora(model: nn.Module, cfg: LoRAConfig, verbose: bool = True) -> nn.Module:
    """`model.base` 의 언어모델에만 LoRA 를 얹는다.

    비전 타워와 기하 인코더는 타깃에서 제외한다 — 이름이 겹칠 수 있어 명시적으로 막는다.
    """
    if not cfg.enabled:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LoRA 를 쓰려면 peft 가 필요하다: pip install peft") from exc

    targets = _resolve_targets(model, cfg)
    if not targets:
        raise RuntimeError(
            f"LoRA 타깃을 하나도 못 찾았다. target_modules={cfg.target_modules} 를 실제 모듈명과 맞춰라. "
            "Qwen3.5 는 하이브리드라 GatedDeltaNet 쪽 이름(in_proj_qkvz/in_proj_ba/out_proj)이 따로 있다."
        )
    if verbose:
        logger.info("LoRA 타깃 %d개 (예: %s)", len(targets), targets[:4])

    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model.base = get_peft_model(model.base, lora_cfg)
    return model


def _resolve_targets(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """언어모델 안에 있는, 이름이 일치하는 Linear 의 **전체 경로**를 모은다."""
    base = model.base
    wanted = set(cfg.target_modules)
    exclude = re.compile(r"(^|\.)(visual|vision_tower|geometry|projectors|pose_projector)(\.|$)")
    out = []
    for name, mod in base.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if exclude.search(name):
            continue
        if name.rsplit(".", 1)[-1] in wanted:
            out.append(name)
    return out
