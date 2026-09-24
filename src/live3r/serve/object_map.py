"""Object-level cognitive map — 2D detections lifted to 3D with CUT3R points, written as text for the prompt.

Why (2026-09-24): the zero-shot scene *map image* had no net effect on VSI (room size +13..+19 from the measured
area, but appearance order / relative direction -7..-10). "Thinking in Space" found that a correct cognitive map
of *objects* is what helps (+20..32% relative distance with the ground-truth map). The VLM already reads object
lists, numbers and coordinates, so the map is given as text — no new modality, no training.

Live design: detections run in the background on keyframes as they arrive; the text sits in the cached prompt
prefix together with the keyframes, so question time only pays for the question tokens.

Detections come from the VLM's own grounding (Qwen3.5 answers `[{"bbox_2d": [x1, y1, x2, y2], "label": ...}]`
with coordinates on a 0..1000 scale). Each box is lifted with the keyframe's CUT3R point map (median of confident
points in the central part of the box — the edges are mostly background), then observations of the same label
closer than `merge_dist` are merged into one instance.
"""

from __future__ import annotations

import re

import numpy as np

_OBJ = re.compile(r'\{[^{}]*"bbox_2d"\s*:\s*\[([^\]]*)\][^{}]*"label"\s*:\s*"([^"]*)"[^{}]*\}')
_OBJ_REV = re.compile(r'\{[^{}]*"label"\s*:\s*"([^"]*)"[^{}]*"bbox_2d"\s*:\s*\[([^\]]*)\][^{}]*\}')


def parse_detections(text: str, max_area: float = 0.8) -> list[dict]:
    """Qwen grounding output -> [{"bbox": [x1, y1, x2, y2] in 0..1, "label": str}].

    Tolerates truncated JSON (the generation can stop mid-list) by matching complete objects only.
    Drops degenerate boxes and boxes covering more than `max_area` of the image (the small model emits
    whole-image boxes for things it cannot localise).
    """
    out = []
    pairs = [(b, lab) for b, lab in _OBJ.findall(text)] + [(b, lab) for lab, b in _OBJ_REV.findall(text)]
    for box, label in pairs:
        try:
            x1, y1, x2, y2 = (float(v) / 1000.0 for v in box.split(","))
        except ValueError:
            continue
        x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
        y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
        area = (x2 - x1) * (y2 - y1)
        if area <= 1e-4 or area > max_area:
            continue
        out.append({"bbox": [x1, y1, x2, y2], "label": label.strip().lower()})
    return out


def detection_prompt(vocab: list[str]) -> str:
    return (
        f"Locate the following objects in the image if they are visible: {', '.join(vocab)}. "
        'Output a JSON list like [{"bbox_2d": [x1, y1, x2, y2], "label": "name"}] using only these names. '
        "Output [] if none of them are visible."
    )


SYNONYMS = {
    "trash can": "trash bin", "garbage can": "trash bin", "recycling bin": "trash bin", "couch": "sofa",
    "television": "tv", "monitor": "tv", "bookcase": "bookshelf", "fridge": "refrigerator", "potted plant": "plant",
    "office chair": "chair", "armchair": "chair", "dining table": "table", "coffee table": "table", "carpet": "rug",
}


def canon(name: str) -> str:
    n = name.strip().lower()
    n = SYNONYMS.get(n, n)
    return n[:-1] if n.endswith("s") and not n.endswith("ss") and len(n) > 3 else n


class ObjectMap:
    """Accumulates lifted detections into object instances and renders them as prompt text.

    v2 (I-28): each observation keeps a small sample of its 3D points, so closest-point distances can be
    measured (VSI measures from the closest points); `merge_dist` grows with the object's own extent so a bed
    seen from different sides stays one instance.
    """

    def __init__(self, merge_dist: float = 0.8, inner: float = 0.5, min_points: int = 15,
                 keep_points: int = 200) -> None:
        self.merge_dist = merge_dist
        self.inner = inner
        self.min_points = min_points
        self.keep_points = keep_points
        self._inst: list[dict] = []   # {"label", "obs": [xyz...], "first": frame_no, "pts": [N,3], "frames": set}

    def add(self, frame_no: int, dets: list[dict], geom) -> int:
        """Lift one keyframe's detections with its CUT3R output (pointmap [1,3,H,W], conf [1,1,H,W]).

        frame_no: the keyframe's number in time order (1-based) — used for first-appearance order.
        Returns the number of detections that could be lifted.
        """
        if geom is None or geom.pointmap is None:
            return 0
        pm = geom.pointmap[0].float().cpu().numpy()          # [3,H,W]
        cf = geom.conf[0, 0].float().cpu().numpy()            # [H,W]
        _, h, w = pm.shape
        thr = np.quantile(cf, 0.3)
        lifted = 0
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * self.inner / 2, (y2 - y1) * self.inner / 2
            c0, c1 = int((cx - hw) * w), int(np.ceil((cx + hw) * w))
            r0, r1 = int((cy - hh) * h), int(np.ceil((cy + hh) * h))
            if c1 <= c0 or r1 <= r0:
                continue
            pts = pm[:, r0:r1, c0:c1].reshape(3, -1).T
            ok = (cf[r0:r1, c0:c1].reshape(-1) >= thr) & np.isfinite(pts).all(1)
            if ok.sum() < self.min_points:
                continue
            good = pts[ok]
            if len(good) > self.keep_points:
                good = good[np.linspace(0, len(good) - 1, self.keep_points).astype(int)]
            self._observe(canon(d["label"]), np.median(good, axis=0), frame_no, good)
            lifted += 1
        return lifted

    def _observe(self, label: str, xyz: np.ndarray, frame_no: int, pts: np.ndarray | None = None) -> None:
        best, best_d = None, None
        for inst in self._inst:
            if inst["label"] != label:
                continue
            dist = float(np.linalg.norm(np.median(inst["obs"], axis=0) - xyz))
            # size-aware: ~half the instance's own extent (with slack), at least merge_dist (a bed spans > 1 m;
            # a partial view's centroid can sit near the object's edge)
            ext = float(np.max(np.ptp(inst["pts"], axis=0))) if len(inst["pts"]) else 0.0
            radius = max(self.merge_dist, 0.6 * ext)
            if dist < radius and (best_d is None or dist < best_d):
                best, best_d = inst, dist
        pts = np.zeros((0, 3), np.float32) if pts is None else pts.astype(np.float32)
        if best is None:
            self._inst.append({"label": label, "obs": [xyz], "first": frame_no, "pts": pts, "frames": {frame_no}})
        else:
            best["obs"].append(xyz)
            best["first"] = min(best["first"], frame_no)
            best["frames"].add(frame_no)
            allp = np.concatenate([best["pts"], pts])
            if len(allp) > 4 * self.keep_points:
                allp = allp[np.linspace(0, len(allp) - 1, 4 * self.keep_points).astype(int)]
            best["pts"] = allp

    def lookup(self, name: str, min_frames: int = 2) -> list[dict]:
        """Instances whose label matches `name` (synonyms, plural) and that were seen in >= min_frames keyframes."""
        want = canon(name)
        return [i for i in self._inst if i["label"] == want and len(i["frames"]) >= min_frames]

    @staticmethod
    def closest_distance(a: dict, b: dict) -> float | None:
        """Closest-point distance between two instances: each point's nearest neighbour in the other set, then
        the 5th percentile — close to the true minimum, robust to a few stray background points in the boxes."""
        if not len(a["pts"]) or not len(b["pts"]):
            return None
        d = np.linalg.norm(a["pts"][:, None, :] - b["pts"][None, :, :], axis=-1)
        return float(np.percentile(np.concatenate([d.min(1), d.min(0)]), 5))

    def pair_distance(self, name_a: str, name_b: str, min_frames: int = 2) -> float | None:
        """Closest-point distance between the nearest instances of two named objects, or None if either is
        missing / seen in fewer than min_frames keyframes (then nothing should be attached — I-28)."""
        A, B = self.lookup(name_a, min_frames), self.lookup(name_b, min_frames)
        ds = [self.closest_distance(a, b) for a in A for b in B]
        ds = [d for d in ds if d is not None]
        return min(ds) if ds else None

    def instances(self) -> list[dict]:
        return [{"label": i["label"], "xyz": np.median(i["obs"], axis=0), "n_obs": len(i["obs"]),
                 "first": i["first"]} for i in self._inst]

    def text(self, to_map=None, n_frames: int | None = None, max_pair_labels: int = 12) -> str:
        """Prompt text. to_map: callable world xyz [N,3] -> map xyz [N,3] (SceneMap.to_map — up-aligned,
        wall-aligned floor plan), so x/y are floor-plan metres and z is height above the lowest points.
        Distances are listed only among the `max_pair_labels` most-observed labels (a 100-word detector
        vocabulary can find dozens of labels; all pairs would cost thousands of tokens)."""
        inst = self.instances()
        if not inst:
            return ""
        xyz = np.stack([i["xyz"] for i in inst])
        if to_map is not None:
            xyz = to_map(xyz)
        labels: dict[str, list[int]] = {}
        for k, i in enumerate(inst):
            labels.setdefault(i["label"], []).append(k)
        order = sorted(labels, key=lambda lab: min(inst[k]["first"] for k in labels[lab]))
        lines = []
        for lab in order:
            ks = labels[lab]
            where = " and ".join(f"({xyz[k][0]:.1f}, {xyz[k][1]:.1f})" for k in ks)
            first = min(inst[k]["first"] for k in ks)
            lines.append(f"- {lab}: {len(ks)} found, at {where}, first seen in frame {first}")
        seen = {lab: sum(inst[k]["n_obs"] for k in labels[lab]) for lab in order}
        top = [lab for lab in order if lab in sorted(seen, key=seen.get, reverse=True)[:max_pair_labels]]
        pair = []
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                d = min(float(np.linalg.norm(xyz[i] - xyz[j]))       # 3D — independent of the frame
                        for i in labels[top[a]] for j in labels[top[b]])
                pair.append(f"{top[a]}-{top[b]} {d:.1f} m")
        span = f"1..{n_frames}" if n_frames else "in time order"
        return (
            "Objects located in 3D from the video (positions (x, y) in meters on a top-down floor plan; "
            f"listed in order of first appearance; frames are numbered {span} in time order):\n"
            + "\n".join(lines)
            + ("\nDistances between object centers: " + "; ".join(pair) + "." if pair else "")
            + "\n"
        )
