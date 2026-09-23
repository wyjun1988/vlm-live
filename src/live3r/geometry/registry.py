"""기하 인코더 레지스트리 — 이름 하나로 어댑터를 갈아끼운다."""

from __future__ import annotations

from typing import Callable

from ..config import GeometryConfig
from .base import GeometryStream

_REGISTRY: dict[str, Callable[[GeometryConfig], GeometryStream]] = {}


def register(name: str):
    def deco(fn: Callable[[GeometryConfig], GeometryStream]):
        if name in _REGISTRY:
            raise KeyError(f"기하 인코더 '{name}' 가 이미 등록돼 있다")
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def build_geometry_stream(cfg: GeometryConfig) -> GeometryStream:
    _ensure_loaded()
    if cfg.name not in _REGISTRY:
        raise KeyError(f"모르는 기하 인코더 '{cfg.name}'. 등록된 것: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.name](cfg)


_loaded = False


def _ensure_loaded() -> None:
    """어댑터 모듈을 지연 임포트한다.

    무거운 의존성(CUT3R 체크포인트 등)이 없어도 dummy 는 항상 뜨도록,
    임포트 실패는 조용히 넘긴다 — 실제 사용 시점에 build 에서 터진다.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import dummy  # noqa: F401  (항상 성공해야 한다)

    for mod in ("cut3r", "anchor3r", "lingbot", "vggt_window"):
        try:
            __import__(f"{__package__}.{mod}")
        except Exception:  # pragma: no cover - 선택적 의존성
            pass
