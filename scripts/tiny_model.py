"""하위호환 — 초소형 Qwen3.5 는 live3r.testing 으로 옮겼다."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from live3r.testing import QWEN35_IDS, SMALL_IDS, tiny_config, tiny_model  # noqa: E402,F401

VOCAB = SMALL_IDS["vocab"]
IMAGE_TOKEN_ID = SMALL_IDS["image"]
VIDEO_TOKEN_ID = SMALL_IDS["video"]
VISION_START_ID = SMALL_IDS["vision_start"]
VISION_END_ID = SMALL_IDS["vision_end"]
