"""대조 프로젝터 (--geom-control shuffled) — 기증자 규칙.

대조군이 진짜 기하를 한 번이라도 보면 "내용 없는 대조군"이 아니다. 규칙:
  * 자기 기하는 절대 안 준다 (첫 샘플은 주입 없이)
  * 같은 격자 모양을 우선한다 (모양 차이가 신호로 섞이지 않게 — 절제 평가와 같은 규칙)
  * 이미지 수가 다르면 기증자 기하를 순환해서 맞춘다
"""

from types import SimpleNamespace

from live3r.train.train import GeomDonors


def geo(tag, *shapes):
    return [SimpleNamespace(grid_hw=sh, tag=f"{tag}{i}") for i, sh in enumerate(shapes)]


def tags(g):
    return None if g is None else [x.tag for x in g]


def test_first_sample_gets_no_geometry_not_its_own():
    d = GeomDonors()
    assert d.swap("a", geo("a", (24, 32))) is None
    assert d.no_donor == 1


def test_never_returns_own_geometry_even_when_seen_before():
    d = GeomDonors()
    d.swap("a", geo("a", (24, 32)))
    # 같은 키가 다시 와도 (에폭 반복·캐시) 자기 것이 아니라 다른 샘플의 것
    d.swap("b", geo("b", (24, 32)))
    assert tags(d.swap("a", geo("a", (24, 32)))) == ["b0"]


def test_prefers_same_grid_shape_over_most_recent():
    d = GeomDonors()
    d.swap("a", geo("a", (24, 32)))
    d.swap("b", geo("b", (32, 24)))           # 가장 최근이지만 모양이 다르다
    out = d.swap("c", geo("c", (24, 32)))
    assert tags(out) == ["a0"]
    assert d.same_shape == 1


def test_falls_back_to_most_recent_and_cycles_image_count():
    d = GeomDonors()
    d.swap("a", geo("a", (24, 32), (24, 32)))
    out = d.swap("b", geo("b", (16, 16), (16, 16), (16, 16)))   # 같은 모양 없음, 이미지 3장
    assert tags(out) == ["a0", "a1", "a0"]
    assert d.same_shape == 0


def test_capacity_bounds_memory():
    d = GeomDonors(capacity=3)
    for k in "abcdef":
        d.swap(k, geo(k, (24, 32)))
    assert len(d.recent) == 3
    assert "대조(shuffled)" in d.summary()


def test_primed_buffer_gives_first_sample_a_donor():
    """학습 전에 채워두면 첫 샘플도 다른 샘플의 기하를 받는다 (S1 은 주입 없는 샘플을 backward 못 한다)."""
    d = GeomDonors()
    d.prime("z", geo("z", (24, 32)))
    assert tags(d.swap("a", geo("a", (24, 32)))) == ["z0"]
    assert d.no_donor == 0
