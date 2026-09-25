"""scripts/fetch_eval_data.py - unpack once (atomically, resumable) and give eval_vsi_local.py its layout."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("fetch_eval_data", Path(__file__).parent.parent / "scripts" /
                                              "fetch_eval_data.py")
fed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fed)


def _snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    with zipfile.ZipFile(snap / "videos.zip", "w") as zf:
        for name in ("arkitscenes/41069025.mp4", "scannet/scene0011_00.mp4", "scannetpp/abc.mp4"):
            zf.writestr(name, b"x" * 100)
    docs = [{"id": 0, "dataset": "arkitscenes", "scene_name": "41069025"},
            {"id": 1, "dataset": "scannet", "scene_name": "scene0011_00"},
            {"id": 2, "dataset": "scannetpp", "scene_name": "abc"}]
    (snap / "test.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    return snap


def test_unpack_is_atomic_and_idempotent(tmp_path):
    snap = _snapshot(tmp_path)
    dest = tmp_path / "hf" / "vsibench"
    assert fed.unpack(snap, dest) == 3
    assert (dest / "scannet" / "scene0011_00.mp4").stat().st_size == 100
    assert not dest.with_name("vsibench.partial").exists()
    assert fed.unpack(snap, dest) == 0            # lmms-eval also skips an existing directory


def test_unpack_completes_a_directory_an_earlier_run_left_half_done(tmp_path):
    """lmms-eval unpacking cut off midway leaves the directory in place; newer lmms-eval then never looks inside."""
    snap = _snapshot(tmp_path)
    dest = tmp_path / "hf" / "vsibench"
    fed.unpack(snap, dest)
    (dest / "scannetpp" / "abc.mp4").unlink()
    (dest / "scannet" / "scene0011_00.mp4").write_bytes(b"x" * 10)     # truncated
    assert fed.unpack(snap, dest) == 2
    assert (dest / "scannet" / "scene0011_00.mp4").stat().st_size == 100


def test_unpack_resumes_an_interrupted_run(tmp_path):
    snap = _snapshot(tmp_path)
    dest = tmp_path / "hf" / "vsibench"
    part = dest.with_name("vsibench.partial") / "arkitscenes"
    part.mkdir(parents=True)
    (part / "41069025.mp4").write_bytes(b"x" * 100)   # finished before the interruption
    assert fed.unpack(snap, dest) == 2


def test_vsi_links_give_eval_vsi_local_its_layout(tmp_path):
    snap = _snapshot(tmp_path)
    dest = tmp_path / "hf" / "vsibench"
    fed.unpack(snap, dest)
    local = tmp_path / "data" / "eval" / "vsibench"
    fed.link_vsi(snap, dest, local)
    assert (local / "scannetpp" / "abc.mp4").exists() and (local / "test.jsonl").exists()
    fed.link_vsi(snap, dest, local)               # a second run leaves the links alone


def test_vsi_links_fail_loudly_when_a_video_is_missing(tmp_path):
    snap = _snapshot(tmp_path)
    dest = tmp_path / "hf" / "vsibench"
    fed.unpack(snap, dest)
    (dest / "scannetpp" / "abc.mp4").unlink()
    with pytest.raises(SystemExit, match="missing"):
        fed.link_vsi(snap, dest, tmp_path / "local")
