"""Capture the raw CUT3R stream of the VSI evaluation videos - per ingested frame the camera pose and the point map
subsample the scene map uses, with its confidence - so that output-side corrections (scale anchors) can be replayed
offline instead of re-running a 1.5-hour evaluation per variant.

    PYTHONPATH=src python scripts/capture_geometry.py --config configs/m2_zeroshot_4b.yaml --videos 60 --seed 1 \\
        --geom-stride 30 --budget 32 --device mps --out outputs/geom_capture/stride30

Same frame schedule as eval_vsi_local.py (every --geom-stride-th frame plus the uniform-oracle keyframes), same
preprocessing (geometry_frame), no LLM. One .npz per video, resumable (existing files are skipped):
    idx [F] video frame index · key [F] keyframe flag · c2w [F,4,4] · pts [F,N,3] · conf [F,N] · meta (json)
    pts_self [F,N,3] · conf_self [F,N]  the head's camera-frame points, when the adapter exposes them
The config must describe the raw stream (no scale_anchor, no reset_every): the corrections are what the replay
(scripts/scale_anchor_lab.py) adds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))          # eval_vsi_local helpers (video pick, frame count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    ap.add_argument("--videos", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--first-videos", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma list of <dataset>/<scene> instead of the seeded pick")
    ap.add_argument("--geom-stride", type=int, default=30)
    ap.add_argument("--budget", type=int, default=32, help="keyframes of the uniform-oracle selector (also ingested)")
    ap.add_argument("--pixel-stride", type=int, default=4, help="point-map subsample, as SceneMap(stride=...)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import av
    from eval_vsi_local import count_frames, pick_videos

    from live3r.config import Live3RConfig
    from live3r.data.vision import geometry_frame
    from live3r.eval.streaming import UniformOracleSelector
    from live3r.geometry.registry import build_geometry_stream

    cfg = Live3RConfig.from_yaml(args.config)
    opts = cfg.geometry.options
    if opts.get("scale_anchor") or int(opts.get("reset_every", 0) or 0):
        raise SystemExit("capture the raw stream: use a config without scale_anchor / reset_every")
    if cfg.geometry.name == "dummy":
        raise SystemExit("the capture needs an encoder that decodes points and poses (cut3r)")
    dev = torch.device(args.device)
    geom = build_geometry_stream(cfg.geometry).to(dev).eval()
    geom.decode_points = True
    unit = getattr(geom, "patch_size", 16)

    root = Path(args.vsi_root)
    docs = [json.loads(ln) for ln in open(root / "test.jsonl") if ln.strip()]
    vids = pick_videos(docs, args.videos, args.seed)
    if args.only:
        vids = [tuple(v.replace("__", "/").split("/", 1)) for v in args.only.split(",")]
    if args.first_videos:
        vids = vids[: args.first_videos]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(vids)} videos · stride {args.geom_stride} + {args.budget} keyframes · {args.device} → {out}")

    s = args.pixel_stride
    for vi, (ds, scene) in enumerate(vids):
        dst = out / f"{ds}__{scene}.npz"
        if dst.exists():
            print(f"  [{vi + 1}/{len(vids)}] {ds}/{scene} exists, skipped")
            continue
        path = root / ds / f"{scene}.mp4"
        total = count_frames(path)
        sel = UniformOracleSelector(args.budget, total)
        geom.reset()
        sel.reset()
        idx, key, c2w, pts, conf, pts_self, conf_self = [], [], [], [], [], [], []
        t0 = time.time()
        with torch.no_grad(), av.open(str(path)) as c:
            st = c.streams.video[0]
            fps = float(st.average_rate) if st.average_rate else 30.0
            for i, frame in enumerate(c.decode(video=0)):
                raw = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                keep, _ = sel.offer(i, raw)
                if not (i % args.geom_stride == 0 or keep):
                    continue
                o = geom.ingest(geometry_frame(raw, cfg.geometry.image_size, unit).unsqueeze(0).to(dev))
                idx.append(i)
                key.append(bool(keep))
                c2w.append(o.extra["c2w"][0].float().cpu().numpy().astype(np.float32))
                pts.append(o.pointmap[0, :, ::s, ::s].reshape(3, -1).T.float().cpu().numpy().astype(np.float32))
                conf.append(o.conf[0, 0, ::s, ::s].reshape(-1).float().cpu().numpy().astype(np.float32))
                if "pointmap_self" in o.extra:      # the head's camera-frame points (SelfViewScale reads the ratio)
                    pts_self.append(o.extra["pointmap_self"][0, :, ::s, ::s].reshape(3, -1).T.float().cpu().numpy()
                                    .astype(np.float32))
                    conf_self.append(o.extra["conf_self"][0, 0, ::s, ::s].reshape(-1).float().cpu().numpy()
                                     .astype(np.float32))
        meta = {"dataset": ds, "scene": scene, "fps": fps, "total_frames": total, "geom_stride": args.geom_stride,
                "budget": args.budget, "pixel_stride": s, "image_size": cfg.geometry.image_size,
                "grid_hw": [int(x) for x in getattr(o, "grid_hw", (0, 0))], "config": args.config}
        extra = {"pts_self": np.stack(pts_self), "conf_self": np.stack(conf_self)} if pts_self else {}
        np.savez_compressed(dst, idx=np.array(idx, np.int64), key=np.array(key, bool), c2w=np.stack(c2w),
                            pts=np.stack(pts), conf=np.stack(conf), meta=json.dumps(meta), **extra)
        if dev.type == "mps":
            torch.mps.empty_cache()
        print(f"  [{vi + 1}/{len(vids)}] {ds}/{scene}: {len(idx)} frames of {total} · {time.time() - t0:.0f}s "
              f"· {dst.stat().st_size / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
