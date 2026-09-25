"""Download evaluation data and unpack it once, where both evaluation paths look for it.

    python scripts/fetch_eval_data.py vsibench     # 5.7 GB
    python scripts/fetch_eval_data.py mmstar       # 0.1 GB
    python scripts/fetch_eval_data.py videomme     # 101 GB download, ~200 GB once unpacked

Why not leave it to lmms-eval: a video task unpacks its archives into $HF_HOME/<cache_dir> on its first run, and
scripts/server_weekend.sh starts several lmms-eval runs at once. Two runs unpacking into one directory race each
other. Here the archives are unpacked once, into `<cache_dir>.partial`, which is renamed only when complete.
lmms-eval then finds the directory and skips unpacking (older versions skip every file that exists). An
interrupted unpack resumes, and a directory an earlier lmms-eval run left half done is completed in place: files
already at full size are not written again.

vsibench also gets data/eval/vsibench/{test.jsonl, arkitscenes/, scannet/, scannetpp/} as links to the same copy,
which is the layout scripts/eval_vsi_local.py reads - one copy on disk, two readers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

# name: (Hugging Face dataset repo, lmms-eval dataset_kwargs.cache_dir or None when nothing is unpacked)
# The repo must be the one the lmms-eval task names, or lmms-eval downloads it again under another name.
TASKS = {
    "vsibench": ("nyu-visionx/VSI-Bench", "vsibench"),
    "videomme": ("lmms-eval/Video-MME", "videomme"),
    "mmstar": ("Lin-Chen/MMStar", None),
}
EXPECTED_VIDEOS = {"vsibench": 288, "videomme": 900}


def hf_home() -> Path:
    return Path(os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))


def unpack(snapshot: Path, dest: Path) -> int:
    """Every zip under `snapshot` -> `dest`. Returns the number of files written.

    A new `dest` is built as `dest.partial` and renamed when complete. An existing `dest` is completed in place:
    an earlier lmms-eval run may have been cut off while unpacking, and newer lmms-eval versions never look inside
    a directory that exists. Files already at full size are never written again, so a complete directory costs
    only a size check per file.
    """
    if list(snapshot.rglob("*.tar*")):
        print("warning: tar archives found and not unpacked - unpack them by hand (lmms-eval will not either, "
              f"once {dest} exists)")
    work = dest if dest.exists() else dest.with_name(dest.name + ".partial")
    work.mkdir(parents=True, exist_ok=True)
    written = 0
    for z in sorted(snapshot.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                target = work / info.filename
                if info.is_dir() or (target.is_file() and target.stat().st_size == info.file_size):
                    continue
                zf.extract(info, work)
                written += 1
        print(f"  {z.name}: checked", flush=True)
    if work != dest:
        work.rename(dest)
    return written


def link_vsi(snapshot: Path, unpacked: Path, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for name, target in [("test.jsonl", snapshot / "test.jsonl")] + [
            (d, unpacked / d) for d in ("arkitscenes", "scannet", "scannetpp")]:
        link = local / name
        if not link.exists():
            if link.is_symlink():            # a dangling link from an earlier layout
                link.unlink()
            link.symlink_to(target.resolve() if name == "test.jsonl" else target)
    docs = [json.loads(line) for line in open(local / "test.jsonl") if line.strip()]
    vids = {(d["dataset"], d["scene_name"]) for d in docs}
    missing = [f"{a}/{b}" for a, b in sorted(vids) if not (local / a / f"{b}.mp4").exists()]
    print(f"{local}: {len(docs)} questions, {len(vids)} videos, {len(missing)} missing")
    if missing:
        raise SystemExit(f"VSI videos missing, e.g. {missing[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--vsi-local", default="data/eval/vsibench",
                    help="vsibench: where scripts/eval_vsi_local.py reads the benchmark")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    repo, cache_dir = TASKS[args.task]
    print(f"{args.task}: downloading {repo} into {hf_home()}", flush=True)
    snap = Path(snapshot_download(repo_id=repo, repo_type="dataset", max_workers=8))
    print(f"  snapshot {snap}")
    if cache_dir is None:
        return 0
    dest = hf_home() / cache_dir
    n = unpack(snap, dest)
    videos = sum(1 for _ in dest.rglob("*.mp4"))
    print(f"  {dest}: {n} files written, {videos} videos")
    want = EXPECTED_VIDEOS.get(args.task)
    if want and videos < want:
        print(f"warning: expected {want} videos, found {videos}")
    if args.task == "vsibench":
        link_vsi(snap, dest, Path(args.vsi_local))
    return 0


if __name__ == "__main__":
    sys.exit(main())
