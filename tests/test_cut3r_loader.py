"""CUT3R 로더 — 원본 코드의 임포트 부작용을 막는지 (실체크포인트 없이 확인한다)."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from live3r.geometry.cut3r import _load_cut3r


def test_dust3r_import_does_not_leave_root_logger_at_debug(monkeypatch, tmp_path):
    # 원본 dust3r/model.py 는 임포트될 때 accelerate get_logger(log_level="DEBUG") 로 루트 로거를
    # DEBUG 로 바꾼다 → 학습 로그가 PIL DEBUG 줄로 불어난다. 같은 부작용을 내는 가짜 모듈로 재현.
    fake = types.ModuleType("dust3r.model")

    class _Stub:  # 생성자 이름만 있으면 된다 — 체크포인트가 없어서 여기까지 안 간다
        pass

    fake.ARCroco3DStereo = fake.ARCroco3DStereoConfig = _Stub
    pkg = types.ModuleType("dust3r")
    pkg.model = fake
    monkeypatch.setitem(sys.modules, "dust3r", pkg)
    monkeypatch.setitem(sys.modules, "dust3r.model", fake)

    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.WARNING)
    real_import = __import__

    def importing(name, *a, **k):
        if name == "dust3r.model":
            root.setLevel(logging.DEBUG)
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", importing)
    with pytest.raises(FileNotFoundError):
        _load_cut3r(str(tmp_path / "missing.pth"), None, "cpu")
    assert root.level == logging.WARNING
