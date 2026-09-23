"""SenseNova-SI 형식의 가짜 데이터 — 실데이터 없이 학습 파이프라인 전체를 검증한다.

    PYTHONPATH=src python scripts/make_mock_sensenova.py /tmp/mock
    → /tmp/mock/sensenova_mock.json, /tmp/mock/media/...

실데이터에서 나올 수 있는 경우를 일부러 섞는다:
  이미지 크기 제각각 · conversations 가 JSON 문자열 · 멀티턴 · placeholder 불일치 1건 · 이미지 누락 1건
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "mock_sensenova")
    media = root / "media"
    for d in ("scene_a", "scene_b"):
        (media / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sizes = [(480, 640), (968, 1296), (720, 1280), (512, 512)]
    paths = []
    for i in range(24):
        h, w = sizes[i % len(sizes)]
        p = f"scene_{'ab'[i % 2]}/img_{i:03d}.jpg"
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(media / p, quality=85)
        paths.append(p)
    recs = []
    for k in range(40):
        n = [5, 6, 2, 1, 8][k % 5]
        imgs = random.Random(k).sample(paths, n)
        q = "".join("<image>\n" for _ in imgs) + "From the viewpoint of Image 2, where is the chair? A. left B. right"
        conv = [{"from": "human", "value": q},
                {"from": "gpt", "value": ["A. left", "1.43", "yes", "B. right"][k % 4]}]
        if k % 7 == 3:
            conv += [{"from": "human", "value": "How far is the table from the sofa in meters?"},
                     {"from": "gpt", "value": "2.1"}]
        rec = {"id": k, "image": imgs, "conversations": conv}
        if k % 6 == 1:
            rec["conversations"] = json.dumps(conv)
        recs.append(rec)
    recs.append({"id": 900, "image": paths[:3],
                 "conversations": [{"from": "human", "value": "<image> only one placeholder"},
                                   {"from": "gpt", "value": "x"}]})
    recs.append({"id": 901, "image": ["scene_a/does_not_exist.jpg"],
                 "conversations": [{"from": "human", "value": "<image>Q"}, {"from": "gpt", "value": "y"}]})
    (root / "sensenova_mock.json").write_text(json.dumps(recs))
    print(f"{len(recs)} 레코드, 이미지 {len(paths)}장 → {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
