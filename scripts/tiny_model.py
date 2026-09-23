"""가중치 없이 도는 초소형 Qwen3.5 — 맥/CI 형상 검증 전용.

실제 Qwen3.5-4B 의 구조적 특징(하이브리드 layer_types, mrope, temporal_patch=2)을
유지한 채 차원만 줄인다. 이게 없으면 구조 버그를 8GB 다운로드 없이 잡을 수 없다.
"""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

# 실모델 토큰 ID 를 그대로 쓴다 (vocab 만 줄이면 id 가 범위를 넘으므로 재배치)
VOCAB = 1024
IMAGE_TOKEN_ID = 1000
VIDEO_TOKEN_ID = 1001
VISION_START_ID = 1002
VISION_END_ID = 1003


def tiny_config(num_layers: int = 8, hidden: int = 128) -> Qwen3_5Config:
    # 실모델과 같은 패턴: linear ×3 + full ×1
    layer_types = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)
    ]
    text = Qwen3_5TextConfig(
        vocab_size=VOCAB,
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
        max_position_embeddings=4096,
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
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        tie_word_embeddings=True,
    )


def tiny_model(num_layers: int = 8, hidden: int = 128, seed: int = 0):
    torch.manual_seed(seed)
    cfg = tiny_config(num_layers, hidden)
    model = Qwen3_5ForConditionalGeneration(cfg)
    model.eval()
    return model
