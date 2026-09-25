"""Build geometry-only training data from unlabelled video (I-14 in docs/BRAINSTORM.md).

Runs the frozen geometry encoder over evenly spaced frames of each video and turns its own camera poses and
point maps into questions whose answers cannot be produced without reading the geometry ("how far apart were the
camera positions when image 2 and image 6 were taken?"). No human labels, no teacher model.

Why this exists: S1 (projector only) and the 905-sample S2 both learned the answer *format* instead of the
geometry — a control projector trained on shuffled geometry scored the same. These questions remove that
shortcut, because a content-free signal cannot produce the number and no object-category prior helps.

    PYTHONPATH=src python scripts/make_geometry_qa.py \\
        --videos data/eval/vsibench/scannet --out data/geomqa --frames 8 --per-kind 2

Writes <out>/geomqa.jsonl (SenseNova record format, ready for prepare_annotations.py) and <out>/media/*.jpg.
Use videos the evaluation does NOT use — see --holdout-scenes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None, help="video file, or a directory of videos")
    ap.add_argument("--records", default=None,
                    help="instead of videos: an existing annotation file of image sequences (SenseNova format). "
                         "This is what the server has - geometry QA is generated from the training corpus itself")
    ap.add_argument("--media-root", default=None, help="--records: root the record image paths are relative to")
    ap.add_argument("--min-images", type=int, default=4,
                    help="--records: skip sequences with fewer images (too few cameras for these questions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8, help="frames per sequence the LLM will see")
    ap.add_argument("--per-kind", type=int, default=2, help="questions per kind per sequence")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos")
    ap.add_argument("--config", default="configs/m2_zeroshot_4b.yaml",
                    help="only the geometry section is used (the LLM is never loaded)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--holdout-scenes", default=None,
                    help="text file of 'dataset/scene' names to SKIP (keep the eval videos out of training)")
    args = ap.parse_args()

    from live3r.config import Live3RConfig
    from live3r.data.geometry_qa import build_questions, to_record
    from live3r.data.vision import geometry_frame
    from live3r.eval.consistent import read_video_frames
    from live3r.geometry.registry import build_geometry_stream
    from live3r.serve.scene_map import SceneMap
    from PIL import Image

    if not args.videos and not args.records:
        raise SystemExit("give --videos or --records")
    seqs: list[tuple[str, object]] = []          # (name, video path | list of image paths)
    if args.records:
        from live3r.data.datasets import load_records, normalize_record

        root = Path(args.media_root or ".")
        for raw in load_records(args.records):
            try:
                rec = normalize_record(raw)
            except Exception:  # noqa: BLE001
                continue
            if rec.images and len(rec.images) >= args.min_images:
                seqs.append((str(rec.id), [root / p for p in rec.images]))
    src = Path(args.videos) if args.videos else None
    vids = (sorted(src.rglob("*.mp4")) if src.is_dir() else [src]) if src else []
    skip = set()
    if args.holdout_scenes and Path(args.holdout_scenes).is_file():
        skip = {ln.strip() for ln in open(args.holdout_scenes) if ln.strip()}
    vids = [v for v in vids if f"{v.parent.name}/{v.stem}" not in skip]
    seqs += [(f"{v.parent.name}_{v.stem}", v) for v in vids]
    if args.limit:
        seqs = seqs[: args.limit]
    out = Path(args.out)
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)

    cfg = Live3RConfig.from_yaml(args.config)
    geom = build_geometry_stream(cfg.geometry).to(args.device).eval()
    geom.decode_points = True
    unit = getattr(geom, "patch_size", 16)
    print(f"{len(seqs)} sequences · up to {args.frames} frames each · geometry {cfg.geometry.name} "
          f"on {args.device}" + (f" · skipping {len(skip)} held-out scenes" if skip else ""))

    records, n_q = [], {}
    t0 = time.time()
    for vi, (scene, source) in enumerate(seqs):
        try:
            if isinstance(source, list):         # image sequence: use the images as they are
                frames = [np.asarray(Image.open(p).convert("RGB")) for p in source[: args.frames]]
            else:
                frames, _ = read_video_frames(source, args.frames)
        except Exception as exc:  # noqa: BLE001 - a broken file should not stop the run
            print(f"  skip {scene}: {str(exc)[:70]}")
            continue
        if len(frames) < 2:
            continue
        geom.reset()
        smap = SceneMap()
        c2ws = []
        for k, f in enumerate(frames):
            g = geom.ingest(geometry_frame(f, cfg.geometry.image_size, unit).unsqueeze(0).to(args.device))
            smap.add(g, frame_index=k)
            c2ws.append(g.extra["c2w"][0].float().cpu().numpy())
        rel = []
        for k, f in enumerate(frames):
            name = f"{scene}_{k}.jpg"
            Image.fromarray(np.asarray(f)).save(media / name, quality=90)
            rel.append(name)

        qs = build_questions(np.stack(c2ws), facts=smap.facts(), per_kind=args.per_kind, seed=vi)
        for qi, q in enumerate(qs):
            records.append(to_record(f"{scene}_q{qi}", rel, q))
            n_q[q["kind"]] = n_q.get(q["kind"], 0) + 1
        if (vi + 1) % max(1, len(seqs) // 20) == 0 or vi + 1 == len(seqs):
            print(f"  [{vi + 1}/{len(seqs)}] {scene}: {len(records)} records so far · "
                  f"{time.time() - t0:.0f}s", flush=True)

    with open(out / "geomqa.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(records)} records -> {out}/geomqa.jsonl")
    for k, v in sorted(n_q.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v}")
    vals = [r for r in records]
    print(f"images {len(list(media.glob('*.jpg')))} · videos used {len({r['id'].rsplit('_q', 1)[0] for r in vals})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
