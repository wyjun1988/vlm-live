"""라이브 불변식 — 이게 깨지면 '라이브'라고 부를 수 없다."""

import torch

from live3r.config import GeometryConfig
from live3r.geometry.registry import available, build_geometry_stream


def test_dummy_registered():
    assert "dummy" in available()


def test_state_is_constant_over_long_stream():
    geo = build_geometry_stream(GeometryConfig(name="dummy", hidden_size=64))
    geo.reset()
    sizes = []
    for t in range(200):
        geo.ingest(torch.randn(1, 3, 112, 112))
        if t in (9, 99, 199):
            sizes.append(geo.state_bytes())
    assert len(set(sizes)) == 1, f"상태 크기가 변했다: {sizes}"


def test_reset_clears_counter():
    geo = build_geometry_stream(GeometryConfig(name="dummy", hidden_size=64))
    for _ in range(5):
        geo.ingest(torch.randn(1, 3, 112, 112))
    assert geo.frame_index == 5
    geo.reset()
    assert geo.frame_index == 0


def test_ingest_cost_does_not_grow(tmp_path):
    """프레임당 비용이 스트림 길이에 비례해 커지면 안 된다 (느슨한 상한)."""
    import time

    geo = build_geometry_stream(GeometryConfig(name="dummy", hidden_size=64))
    geo.reset()
    f = torch.randn(1, 3, 112, 112)
    for _ in range(20):  # warmup
        geo.ingest(f)

    def med(n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            geo.ingest(f)
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[n // 2]

    early = med(30)
    for _ in range(300):
        geo.ingest(f)
    late = med(30)
    assert late < early * 3.0, f"프레임당 비용이 {late / early:.2f}배로 늘었다"
