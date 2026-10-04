"""Utonia on the CUT3R reconstruction of a VSI video: does it see the objects and the floor? (feasibility, I-47)

    PYTORCH_ENABLE_MPS_FALLBACK=1 ~/miniforge3/envs/floorplan-vlm/bin/python scripts/utonia_scene_probe.py \\
        --videos 3 --out outputs/utonia_probe

Utonia (Pointcept PTv3, "one encoder for all point clouds") with its released ScanNet-20 linear probe labels every
point; the user's floorplan-vlm project wraps it for MPS (encoders/utonia_encoder.py, utonia_mps.py) and turns the
labels into objects (zeroshot/scene_text.py). Here its input is our own streaming reconstruction: the saved CUT3R
stream (scripts/capture_geometry.py: per ingested frame the pose and a stride-4 point map with confidence), colored
from the video, scaled with the frame-count calibration, turned z-up from the cameras' mean up vector, floor at z = 0.
Per video: the label histogram, objects per class two ways (every connected cluster; floorplan-vlm's own rules, which
also ask for a minimum size), the floor area two ways (occupied floor cells; the wall-aligned box of the floor points,
the room fact's rule) next to that box over all points, how they compare with the video's VSI counting and room-size
ground truth, and a top-down label image.

With --arkit-mesh the input is instead the ground-truth ARKitScenes scan of each VSI scene that has one (the colored
<scene>_3dod_mesh.ply vertices, z-up, metres) - the same labelling, counting and floor area on a clean
reconstruction, which separates Utonia's own error from the streaming reconstruction's.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def setup_floorplan(root: Path):
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "third_party" / "Utonia"))
    from floorplan_vlm.encoders.utonia_encoder import SCANNET20, SemanticHead, UtoniaEncoder
    from floorplan_vlm.zeroshot.scene_text import components, oriented_box

    return SCANNET20, SemanticHead, UtoniaEncoder, components, oriented_box


def frame_colors(video: Path, indices: np.ndarray, long_side: int, unit: int, stride: int) -> dict[int, np.ndarray]:
    """index -> [n, 3] uint8 colors at the capture's stride-4 grid of the geometry input."""
    import av
    import torch

    from live3r.data.vision import geometry_frame

    want = set(int(i) for i in indices)
    out = {}
    with av.open(str(video)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in want:
                g = geometry_frame(torch.from_numpy(frame.to_ndarray(format="rgb24")), long_side, unit)
                rgb = ((g * 0.5 + 0.5).clamp(0, 1) * 255).byte()[:, ::stride, ::stride]
                out[i] = rgb.reshape(3, -1).T.numpy()
            if len(out) == len(want):
                break
    return out


def mesh_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The colored vertices of a ground-truth ARKitScenes scan (<scene>_3dod_mesh.ply: z-up, metres)."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    return xyz, rgb


def up_rotation(c2w: np.ndarray) -> np.ndarray:
    """Rotation taking the cameras' mean up vector (-y of an OpenCV camera) to +z."""
    up = -c2w[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def floor_area(xy: np.ndarray, cell: float = 0.1, min_pts: int = 3) -> float:
    """Occupied top-view cells of floor-labelled points, holes up to a cell filled."""
    from scipy import ndimage

    if len(xy) == 0:
        return 0.0
    lo = xy.min(0)
    ij = np.floor((xy - lo) / cell).astype(int)
    grid = np.zeros(ij.max(0) + 1, dtype=np.int32)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    occ = ndimage.binary_closing(grid >= min_pts, iterations=2)
    occ = ndimage.binary_fill_holes(occ)
    return float(occ.sum() * cell * cell)


def aligned_box_area(xy: np.ndarray) -> float:
    """The wall-aligned 2-98 % box of points seen from above: the room fact's rule (serve/scene_map.py)."""
    if len(xy) < 100:
        return 0.0
    sub = xy[:: max(1, len(xy) // 20000)]
    best = np.inf
    for deg in range(0, 90, 2):
        t = np.deg2rad(deg)
        q = sub @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T
        best = min(best, float(np.prod(np.percentile(q, 98, 0) - np.percentile(q, 2, 0))))
    return best


OBJECT_CLASSES = ("chair", "table", "sofa", "bed", "cabinet", "desk", "toilet", "sink", "refrigerator", "bathtub",
                  "bookshelf", "counter", "picture", "curtain")
OPENINGS = ("door", "window")


def count_objects(xyz, labels, names, components, oriented_box) -> tuple[dict, dict]:
    """Objects per class, two ways. 'naive': every 26-connected cluster of 8 cm voxels with >= 50 points.
    'floorplan': floorplan-vlm's scene_text rules as they are - >= 80 points and a box >= 0.3 m across and 0.1 m
    high; doors and windows: 10 cm clusters of >= 60 points, >= 0.5 m wide and 0.4 m high."""
    naive, fp = {}, {}
    for cls in OBJECT_CLASSES + OPENINGS:
        pts = xyz[labels == names.index(cls)]
        if len(pts) < 50:
            continue
        naive[cls] = len(components(pts, voxel=0.08, min_points=50))
        if cls in OPENINGS:
            boxes = [oriented_box(pts[i], cls) for i in components(pts, voxel=0.1, min_points=60)]
            fp[cls] = sum(int(b.size[:2].max() >= 0.5 and b.size[2] >= 0.4) for b in boxes)
        else:
            boxes = [oriented_box(pts[i], cls) for i in components(pts, voxel=0.08, min_points=80)]
            fp[cls] = sum(int(b.size[:2].max() >= 0.3 and b.size[2] >= 0.1) for b in boxes)
    return naive, fp


COUNT_RE = re.compile(r"How many (.+?)\(s\) are in this room")
VSI_TO_SCANNET = {"chair": "chair", "table": "table", "sofa": "sofa", "bed": "bed", "cabinet": "cabinet",
                  "desk": "desk", "toilet": "toilet", "sink": "sink", "refrigerator": "refrigerator",
                  "bathtub": "bathtub", "bookshelf": "bookshelf", "door": "door", "window": "window",
                  "curtain": "curtain", "counter": "counter", "picture": "picture", "stool": "chair"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", default="outputs/geom_capture/stride30")
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    ap.add_argument("--floorplan", default=str(Path.home() / "work" / "floorplan-vlm"))
    ap.add_argument("--videos", type=int, default=3, help="first N captured videos (0 = all)")
    ap.add_argument("--only", default=None, help="comma list of <dataset>__<scene>")
    ap.add_argument("--arkit-mesh", default=None,
                    help="instead of the captures: the ground-truth scans in this folder "
                         "(<scene>/<scene>_3dod_mesh.ply) of the VSI ARKitScenes scenes")
    ap.add_argument("--conf-quantile", type=float, default=0.3, help="drop each frame's least confident points")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="outputs/utonia_probe")
    a = ap.parse_args()

    import torch

    from live3r.geometry.scale_anchor import FrameCountScale

    SCANNET20, SemanticHead, UtoniaEncoder, components, oriented_box = setup_floorplan(Path(a.floorplan))
    models = Path(a.floorplan) / "storage" / "models" / "utonia"
    enc = UtoniaEncoder(str(models / "utonia.pth")).to(a.device)
    head = SemanticHead(str(models / "utonia_linear_prob_head_sc.pth")).to(a.device)
    docs = [json.loads(ln) for ln in open(Path(a.vsi_root) / "test.jsonl")]
    if a.arkit_mesh:
        vsi = {q["scene_name"] for q in docs if q["dataset"] == "arkitscenes"}
        files = [p / f"{p.name}_3dod_mesh.ply" for p in sorted(Path(a.arkit_mesh).iterdir()) if p.name in vsi]
        files = [f for f in files if f.exists()]
        stem = lambda f: f"arkitscenes__{f.parent.name}"  # noqa: E731
    else:
        files = sorted(Path(a.capture).glob("*.npz"))
        stem = lambda f: f.stem  # noqa: E731
    if a.only:
        files = [f for f in files if stem(f) in set(a.only.split(","))]
    elif a.videos:
        files = files[: a.videos]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        t0 = time.time()
        if a.arkit_mesh:
            ds, scene, n_frames = "arkitscenes", f.parent.name, 0
            xyz, rgb = mesh_points(f)
        else:
            d = np.load(f, allow_pickle=True)
            meta = json.loads(str(d["meta"]))
            ds, scene, n_frames = meta["dataset"], meta["scene"], int(len(d["idx"]))
            video = Path(a.vsi_root) / ds / f"{scene}.mp4"
            cols = frame_colors(video, d["idx"], meta["image_size"], 16, meta["pixel_stride"])
            pts, rgb = [], []
            for k, i in enumerate(d["idx"]):
                keep = d["conf"][k] >= np.quantile(d["conf"][k], a.conf_quantile)
                pts.append(d["pts"][k][keep])
                rgb.append(cols[int(i)][keep])
            pts, rgb = np.concatenate(pts), np.concatenate(rgb)
            scale = 1.0 / np.sqrt(FrameCountScale().ratio(n_frames))         # the read-time calibration (I-37)
            xyz = (pts * scale) @ up_rotation(d["c2w"]).T
        xyz[:, 2] -= np.percentile(xyz[:, 2], 2)
        t1 = time.time()
        if a.device == "mps":
            import gc

            gc.collect()
            torch.mps.empty_cache()
        res = enc(xyz, rgb, return_point_feat=True)
        labels = head(res.point_feat)[res.inverse].cpu().numpy()
        share = np.bincount(labels, minlength=len(SCANNET20)).max() / max(1, len(labels))
        if share > 0.97:                 # one class everywhere: the transient MPS failure seen on 10-04 - redo on CPU
            print(f"  {f.stem}: one class on {100 * share:.0f} % of the points - again on CPU", flush=True)
            res = enc.to("cpu")(xyz, rgb, return_point_feat=True)
            labels = head.to("cpu")(res.point_feat)[res.inverse].numpy()
            enc.to(a.device), head.to(a.device)
        t2 = time.time()
        hist = {SCANNET20[c]: int(n) for c, n in zip(*np.unique(labels, return_counts=True))}
        objects, objects_fp = count_objects(xyz, labels, SCANNET20, components, oriented_box)
        floor_xy = xyz[labels == SCANNET20.index("floor"), :2]
        area, floor_box, all_box = floor_area(floor_xy), aligned_box_area(floor_xy), aligned_box_area(xyz[:, :2])
        qs = [q for q in docs if q["dataset"] == ds and q["scene_name"] == scene]
        counts, room = [], None
        for q in qs:
            if q["question_type"] == "object_counting":
                m = COUNT_RE.search(q["question"])
                name = m.group(1) if m else None
                cls = VSI_TO_SCANNET.get(name)
                counts.append({"object": name, "class": cls, "gt": int(q["ground_truth"]),
                               "utonia": objects.get(cls, 0) if cls else None,
                               "utonia_fp": objects_fp.get(cls, 0) if cls else None})
            if q["question_type"] == "room_size_estimation":
                room = float(q["ground_truth"])
        row = {"video": f"{ds}/{scene}", "frames": n_frames, "points": int(len(xyz)),
               "voxels": res.num_points, "seconds_utonia": round(t2 - t1, 1), "seconds": round(time.time() - t0, 1),
               "labels": hist, "objects": objects, "objects_fp": objects_fp, "floor_area": round(area, 1),
               "floor_box": round(floor_box, 1), "all_box": round(all_box, 1), "room_gt": room, "counts": counts}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("video", "frames", "points", "voxels", "seconds_utonia", "floor_area",
                                                "floor_box", "all_box", "room_gt", "objects", "objects_fp", "counts")},
                         ensure_ascii=False), flush=True)
        # top-down label image (the most frequent classes in fixed colors)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sub = np.random.default_rng(0).choice(len(xyz), min(len(xyz), 60000), replace=False)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(xyz[sub, 0], xyz[sub, 1], c=labels[sub], cmap="tab20", s=0.5, vmin=0, vmax=19)
            ax.set_aspect("equal")
            source = "ground-truth scan" if a.arkit_mesh else "CUT3R reconstruction"
            ax.set_title(f"{ds}/{scene} - Utonia labels on the {source}")
            fig.savefig(out / f"{ds}__{scene}_labels.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print("plot failed:", exc)
    (out / "probe.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
