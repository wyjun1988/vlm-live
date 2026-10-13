"""VSI-590K's images for the VG-LLM annotation of it - download, unpack, check (2026-10-13).

    python scripts/fetch_vsi590k.py --dest /group-volume/wooyeol/data/vsi590k_media \\
        --staging /group-volume/wooyeol/data/vsi590k_archives \\
        --annotation /group-volume/VG-LLM_data/train/vsi_590k_spar.json

The annotation (302,005 VSI-style records - 70 % multiple choice, 28 % numbers - from six environments: Hypersim, ADT,
ProcTHOR, S3DIS, YouTube room tours, robotics; no ScanNet / ScanNet++ / ARKitScenes, so no VSI-Bench scene) is on the
shared volume; its images are not (the data pass found 0 of 1,502). They are the per-source archives of the public
dataset nyu-visionx/VSI-590K (Cambrian-S; Apache-2.0 card, each source keeps its own terms). The six needed are
~96.5 GB (ytb_roomtour 55.3, adt 23.9, hypersim 6.8, procthor 6.5, s3dis 2.9, robotics 1.1); the ScanNet /
ScanNet++ / ARKitScenes archives (139 GB) are not fetched.

Per source: the archive is downloaded (resumable), its top-level entries listed, unpacked under <dest>/vsi_590k/ (the
records' paths are vsi_590k/<source>/...), and deleted unless --keep-archives; a marker makes a re-run skip it. Then a
sample of records per source is resolved under <dest> and the result written as a table (--report). Nothing outside
--dest and --staging is written.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

REPO = "nyu-visionx/VSI-590K"
SIZES_GB = {"ytb_roomtour": 55.3, "adt": 23.9, "hypersim": 6.78, "procthor": 6.52, "s3dis": 2.87, "robotics": 1.1}


def media_of(rec: dict) -> list[str]:
    """The record's image / video paths (relative to the media root)."""
    out = []
    for key in ("image", "images", "video", "videos"):
        v = rec.get(key)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out += [str(x) for x in v]
    return out


def top_entries(archive: Path, n: int = 400) -> list[str]:
    """The distinct first path components among the archive's first n members (its layout)."""
    tops = []
    with tarfile.open(archive, "r:gz") as tf:
        for k, m in enumerate(tf):
            name = m.name[2:] if m.name.startswith("./") else m.name
            top = name.split("/", 1)[0]
            if top and top not in tops:
                tops.append(top)
            if k >= n:
                break
    return tops


def fetch(source: str, staging: Path, dest: Path, keep: bool) -> dict:
    from huggingface_hub import hf_hub_download

    done = dest / "vsi_590k" / f".{source}.done"
    if done.exists():
        return json.loads(done.read_text())
    t0 = time.time()
    archive = Path(hf_hub_download(REPO, f"{source}.tar.gz", repo_type="dataset", local_dir=str(staging)))
    tops = top_entries(archive)
    target = dest / "vsi_590k"
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(target)], check=True)
    info = {"source": source, "archive_gb": round(archive.stat().st_size / 2**30, 2), "top_entries": tops[:10],
            "dir_exists": (target / source).is_dir(), "minutes": round((time.time() - t0) / 60, 1)}
    if not keep:
        archive.unlink()
    done.write_text(json.dumps(info))
    return info


def verify(annotation: Path, dest: Path, per_source: int, seed: int = 0) -> dict:
    """Resolve a reservoir sample of records per source (path prefix vsi_590k/<source>/) under dest."""
    from live3r.data.datasets import iter_json_records

    rng = random.Random(seed)
    seen: dict[str, int] = defaultdict(int)
    pool: dict[str, list[list[str]]] = defaultdict(list)
    for rec in iter_json_records(annotation):
        paths = media_of(rec)
        if not paths:
            continue
        parts = paths[0].split("/")
        source = parts[1] if len(parts) > 2 and parts[0] == "vsi_590k" else parts[0]
        seen[source] += 1
        if len(pool[source]) < per_source:
            pool[source].append(paths)
        elif (j := rng.randrange(seen[source])) < per_source:
            pool[source][j] = paths
    out = {}
    for source, recs in sorted(pool.items()):
        files = [p for r in recs for p in r]
        missing = [p for p in files if not (dest / p).exists()]
        out[source] = {"records": seen[source], "files": len(files), "found": len(files) - len(missing),
                       "missing_examples": missing[:3]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", required=True, help="media root: the images land in <dest>/vsi_590k/<source>/")
    ap.add_argument("--staging", required=True, help="where the archives are downloaded (deleted after unpacking)")
    ap.add_argument("--annotation", default="/group-volume/VG-LLM_data/train/vsi_590k_spar.json")
    ap.add_argument("--sources", default=",".join(SIZES_GB), help="comma list (default: the six the annotation uses)")
    ap.add_argument("--sample", type=int, default=300, help="records per source resolved in the check")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="no download: only the check")
    ap.add_argument("--report", default=None, help="markdown summary (default <dest>/VSI590K.md)")
    a = ap.parse_args()

    dest, staging = Path(a.dest), Path(a.staging)
    sources = [s for s in a.sources.split(",") if s]
    unknown = [s for s in sources if s not in SIZES_GB]
    if unknown:
        print(f"unknown sources {unknown}: the annotation uses {list(SIZES_GB)}")
        return 2
    fetched = []
    if not a.verify_only:
        dest.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=True, exist_ok=True)
        need = sum(SIZES_GB[s] for s in sources) * 2.1      # archive + unpacked, before the archive is deleted
        free = min(shutil.disk_usage(dest).free, shutil.disk_usage(staging).free) / 2**30
        print(f"sources {sources}: ~{need:.0f} GB at the peak (largest archive + its contents, sequentially much less);"
              f" free {free:.0f} GB", flush=True)
        for s in sorted(sources, key=lambda x: SIZES_GB[x]):            # small ones first: the layout shows early
            print(f"[{time.strftime('%H:%M:%S')}] {s} ({SIZES_GB[s]} GB) ...", flush=True)
            info = fetch(s, staging, dest, a.keep_archives)
            fetched.append(info)
            print(f"  {json.dumps(info)}", flush=True)
    check = verify(Path(a.annotation), dest, a.sample)
    lines = ["# VSI-590K media for the VG-LLM annotation", "",
             f"annotation `{a.annotation}` · media root `{dest}` · {time.strftime('%Y-%m-%d %H:%M')}", "",
             "| source | records | sampled files | found | missing (examples) |", "|---|---|---|---|---|"]
    for s, r in check.items():
        lines.append(f"| {s} | {r['records']:,} | {r['files']} | {r['found']} ({100 * r['found'] / max(r['files'], 1):.0f} %)"
                     f" | {', '.join(r['missing_examples']) or '-'} |")
    if fetched:
        lines += ["", "| archive | GB | top-level entries | `<source>/` exists | minutes |", "|---|---|---|---|---|"]
        lines += [f"| {f['source']} | {f['archive_gb']} | {', '.join(f['top_entries'])} | {f['dir_exists']} | {f['minutes']} |"
                  for f in fetched]
    report = Path(a.report) if a.report else dest / "VSI590K.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    ok = check and all(r["found"] == r["files"] for r in check.values())
    print(f"\n{'ALL RESOLVED' if ok else 'NOT ALL RESOLVED - read the table'} · report {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
