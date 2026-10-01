"""LoRA 타깃 이름 — Qwen3.5 실제 모듈명과 맞는지.

Qwen3-Next 의 in_proj_qkvz / in_proj_ba 가 Qwen3.5 에서는 in_proj_qkv/z/b/a 로 쪼개져 있다.
예전 설정은 앞의 이름을 써서 DeltaNet 24층 입력 프로젝션에 LoRA 가 안 붙은 채 조용히 돌 뻔했다.
"""

import pytest
from torch import nn

from live3r.config import LoRAConfig
from live3r.testing import tiny_model


def _lm_linear_names():
    m = tiny_model()
    return {n.rsplit(".", 1)[-1] for n, mod in m.named_modules()
            if isinstance(mod, nn.Linear) and ".language_model." in n}


def test_default_targets_all_exist_in_qwen35():
    missing = set(LoRAConfig().target_modules) - _lm_linear_names()
    assert not missing, f"Qwen3.5 에 없는 LoRA 타깃: {missing}"


def test_deltanet_input_projection_is_targeted():
    assert "in_proj_qkv" in LoRAConfig().target_modules


def test_apply_lora_rejects_unknown_names():
    pytest.importorskip("peft")
    from test_train_step import build

    from live3r.model.lora import apply_lora

    m = build()
    with pytest.raises(RuntimeError, match="in_proj_qkvz"):
        apply_lora(m, LoRAConfig(target_modules=("q_proj", "in_proj_qkvz")), verbose=False)


def test_run3_targets_every_llm_linear_like_the_reference_adapter():
    """configs/server_4b_v3*.yaml: the reference's 12 target names - 8 per DeltaNet layer, 7 per attention layer,
    nothing in the vision tower. On the 4B (24 + 8 layers) that is 248 modules, the count in its adapter."""
    from test_train_step import build

    from live3r.config import Live3RConfig
    from live3r.model.lora import _resolve_targets

    cfg = Live3RConfig.from_yaml("configs/server_4b_v3.yaml").lora
    m = build()
    names = _resolve_targets(m, cfg)
    assert all(".language_model." in n for n in names)
    per_layer: dict[int, int] = {}
    for n in names:
        layer = int(n.split(".layers.")[1].split(".")[0])
        per_layer[layer] = per_layer.get(layer, 0) + 1
    types = m.base.config.text_config.layer_types
    for i, t in enumerate(types):
        assert per_layer[i] == (7 if t == "full_attention" else 8), (i, t, per_layer[i])
    assert 24 * 8 + 8 * 7 == 248
