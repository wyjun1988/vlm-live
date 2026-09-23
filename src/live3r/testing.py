"""테스트·스모크용 초소형 Qwen3.5 — 가중치 다운로드 없이 구조를 검증한다.

실제 Qwen3.5 의 구조적 특징(하이브리드 layer_types, mrope, temporal_patch=2, patch16·merge2)을
유지한 채 차원만 줄인다. 두 가지 ID 체계:
    SMALL_IDS   어휘 1024 — 빠른 단위 테스트용
    QWEN35_IDS  어휘 248,320 + 실제 특수 토큰 ID — **실제 Qwen3.5 토크나이저와 같이 쓸 때**
                (프롬프트 빌더·데이터셋·학습 루프를 진짜 토큰으로 끝까지 돌려볼 수 있다)

한계: 가중치가 랜덤이라 **내용 수준의 버그**(예: 패치 순서가 뒤섞여도 에러 없음)는 못 잡는다.
그런 건 공식 구현과 직접 대조하는 테스트(tests/test_vision_official.py)가 맡는다.
"""

from __future__ import annotations

import torch

SMALL_IDS = dict(vocab=1024, image=1000, video=1001, vision_start=1002, vision_end=1003)
QWEN35_IDS = dict(vocab=248320, image=248056, video=248057, vision_start=248053, vision_end=248054)


def tiny_config(num_layers: int = 8, hidden: int = 128, ids: dict = SMALL_IDS):
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    layer_types = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)
    ]
    text = Qwen3_5TextConfig(
        vocab_size=ids["vocab"],
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=layer_types,
        full_attention_interval=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=hidden,
        num_position_embeddings=64,
    )
    return Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=ids["image"],
        video_token_id=ids["video"],
        vision_start_token_id=ids["vision_start"],
        vision_end_token_id=ids["vision_end"],
        tie_word_embeddings=True,
    )


def tiny_model(num_layers: int = 8, hidden: int = 128, seed: int = 0, ids: dict = SMALL_IDS):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(tiny_config(num_layers, hidden, ids))
    model.eval()
    return model
