"""장면 지도 — 알려진 크기의 가짜 방으로 측정값·렌더링을 확인한다.

CUT3R 규약: 월드 = 첫 프레임 카메라 좌표계 (OpenCV — x 오른쪽, y 아래, z 앞), 미터.
가짜 방: 가로 5m (x) × 세로 4m (z) × 높이 2.6m, 바닥 y=+1.5, 천장 y=−1.1. 카메라는 y=0 에서 돌아다닌다.
"""

import math

import numpy as np
import torch

from live3r.geometry.base import GeomOutput
from live3r.serve.scene_map import SceneMap, keyframe_labels


def _room_points(n, rng):
    """바닥·네 벽·천장 위의 점."""
    pts = []
    for _ in range(n):
        s = rng.integers(0, 6)
        a, b = rng.random(), rng.random()
        if s == 0:
            pts.append((-2.5 + 5 * a, 1.5, 4 * b))          # 바닥
        elif s == 1:
            pts.append((-2.5 + 5 * a, -1.1, 4 * b))         # 천장
        elif s == 2:
            pts.append((-2.5, -1.1 + 2.6 * a, 4 * b))       # 왼벽
        elif s == 3:
            pts.append((2.5, -1.1 + 2.6 * a, 4 * b))        # 오른벽
        elif s == 4:
            pts.append((-2.5 + 5 * a, -1.1 + 2.6 * b, 0.0))  # 뒷벽
        else:
            pts.append((-2.5 + 5 * a, -1.1 + 2.6 * b, 4.0))  # 앞벽
    return np.array(pts, np.float32)


def _frame(i, center, yaw, rng, h=48, w=64):
    pts = _room_points(h * w, rng)
    pm = torch.from_numpy(pts.T.reshape(1, 3, h, w).copy())
    conf = torch.rand(1, 1, h, w, generator=torch.Generator().manual_seed(i)) + 1.0
    c, s = math.cos(yaw), math.sin(yaw)
    rot = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)  # y(아래) 축 회전 = 제자리 회전
    c2w = torch.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = torch.tensor(center, dtype=torch.float32)
    return GeomOutput(tokens={}, grid_hw=(3, 4), pointmap=pm, conf=conf, frame_index=i,
                      extra={"c2w": c2w.unsqueeze(0)})


def _map(n_frames=8, **kw):
    rng = np.random.default_rng(0)
    m = SceneMap(stride=1, **kw)
    for i in range(n_frames):
        m.add(_frame(i, (-1.0 + 2.0 * i / (n_frames - 1), 0.0, 1.0 + 2.0 * i / (n_frames - 1)), 0.3 * i, rng))
    return m


def test_facts_recover_room_size_height_and_path():
    f = _map().facts()
    dims = sorted([f["width_m"], f["length_m"]])
    assert abs(dims[0] - 4.0) < 0.35 and abs(dims[1] - 5.0) < 0.35, dims   # 벽 정렬 bbox (2~98 백분위)
    assert abs(f["height_m"] - 2.6) < 0.3
    assert abs(f["path_m"] - math.hypot(2.0, 2.0)) < 1e-3                  # 직선 궤적 길이
    assert "square meters" in _map().text()


def test_render_draws_points_trajectory_and_labels():
    m = _map()
    img = m.render(size=256, labels=keyframe_labels([0, 3, 7]))
    arr = np.asarray(img)
    assert img.size == (256, 256)
    assert (arr.sum(-1) > 0).mean() > 0.08, "점이 거의 안 그려졌다"
    red = (arr[..., 0] > 200) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
    assert red.sum() > 50, "카메라 궤적·번호 표시가 없다"


def test_voxel_downsample_bounds_memory():
    m = _map(n_frames=6, max_points=5000)
    assert m.n_points <= 5000 and m.n_frames == 6


def test_empty_map_is_harmless():
    m = SceneMap()
    assert m.facts() == {} and m.text() == ""
    assert m.render(size=64).size == (64, 64)


def test_keyframe_labels_are_time_ordered():
    assert keyframe_labels([30, 5, 12]) == {5: 1, 12: 2, 30: 3}


def test_room_facts_sentence_for_question_time_routing():
    """I-27: one sentence of room measurements, no format instruction (the prefix already has it)."""
    t = _map().room_facts()
    assert "square meters" in t and "m high" in t and "Answer" not in t and t.endswith("\n")
    assert SceneMap().room_facts() == ""
