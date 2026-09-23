"""배치 구성.

샘플은 `SpatialVQADataset` 이 이미 모델 입력 완성본으로 만든다 (토큰·라벨·픽셀·기하 프레임).
여기서는 묶기만 한다. 비전 토큰 수가 샘플마다 달라서 **배치 1 + gradient accumulation** 을 쓴다.

픽셀 레이아웃·프롬프트 형식은 각각 `vision.py` · `prompt.py` 에 있다.
(예전의 pack_video_patches 는 패치 순서가 공식과 달라 이미지를 뒤섞었다 — 삭제했다.)
"""

from __future__ import annotations


class Live3RCollator:
    def __call__(self, batch: list[dict]) -> dict:
        if len(batch) != 1:
            raise NotImplementedError(
                "배치 1만 지원한다 (비전 토큰 수가 샘플마다 달라서). "
                "gradient_accumulation_steps 와 GPU 수로 유효 배치를 키워라."
            )
        return batch[0]
