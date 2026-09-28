"""Does the scale drift follow the number of frames ingested closely enough to correct it with a calibration curve?

    PYTHONPATH=src python scripts/scale_calibration.py outputs/geom_capture/stride30

The scale anchors (scripts/scale_anchor_lab.py) read the drift from the frame's own floor and carry no per-video
signal (factor-ideal correlation ~0, 2026-09-28). The drift does grow with the frames ingested (room area x1.03 at 32
keyframes, x1.18-1.23 at ~1 fps, x1.39 at ~3 fps). This fits log(measured / true area) against the number of frames
the stream ingested and scores the correction two ways, split-half (fit on odd videos, score even, and the other way):
  global     one factor per video, 1 / sqrt(ratio(n)), on the metric facts at read time (the geometry untouched);
  per-frame  every frame rescaled by 1 / sqrt(ratio(k)) at its own rank k - the smooth inverse of a drift that grows
             with k (if the drift really is about the origin this restores the layout instead of breaking it).
Prints the fit, the split-half room-area MRA of both against `none`, and the ceiling (the scale-free MRA).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_anchor_lab import load_capture, replay  # noqa: E402
from vsi_room_facts_check import mra  # noqa: E402


class RankScaler:
    """A stand-in for the anchor in `replay`: factor 1 / sqrt(ratio(k)) at frame rank k from a fitted curve."""

    def __init__(self, coef: np.ndarray, kind: str) -> None:
        self.coef, self.kind, self.k = coef, kind, 0
        self.history: list[float] = []
        self.skipped = self.frames = 0

    def update(self, c2w, pts, conf) -> float:
        self.k += 1
        f = float(1.0 / np.sqrt(np.exp(predict(self.coef, np.array([self.k]), self.kind)[0])))
        self.history.append(f)
        return f

    @staticmethod
    def apply(factor, c2w, pts):
        out = np.array(c2w, dtype=np.float64, copy=True)
        out[:3, 3] *= factor
        return out, pts * factor


def fit(n: np.ndarray, logratio: np.ndarray, kind: str) -> np.ndarray:
    x = np.log(n) if kind == "log" else n.astype(float)
    return np.polyfit(x, logratio, 1)


def predict(coef: np.ndarray, n: np.ndarray, kind: str) -> np.ndarray:
    x = np.log(n) if kind == "log" else n.astype(float)
    return np.polyval(coef, x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    ap.add_argument("--kind", default="log", choices=("log", "linear"), help="log ratio vs log(n) or vs n")
    args = ap.parse_args()
    truth = {}
    for ln in open(Path(args.vsi_root) / "test.jsonl"):
        if ln.strip():
            d = json.loads(ln)
            if d["question_type"] == "room_size_estimation":
                truth[f"{d['dataset']}/{d['scene_name']}"] = float(d["ground_truth"])
    caps = {}
    for f in sorted(Path(args.capture_dir).glob("*.npz")):
        c = load_capture(f)
        caps[f"{c['meta']['dataset']}/{c['meta']['scene']}"] = c
    vids = [v for v in caps if v in truth]
    none = {v: replay(caps[v], None)[0]["area_m2"] for v in vids}
    n = np.array([len(caps[v]["idx"]) for v in vids])
    lr = np.log(np.array([none[v] / truth[v] for v in vids]))
    print(f"{len(vids)} videos · frames ingested {n.min()}-{n.max()} (median {int(np.median(n))}) · "
          f"median area ratio {np.exp(np.median(lr)):.2f} · corr(log ratio, log n) {np.corrcoef(np.log(n), lr)[0, 1]:+.2f}")
    coef = fit(n, lr, args.kind)
    print(f"fit (all videos, {args.kind}): log ratio = {coef[0]:+.4f} * {'log ' if args.kind == 'log' else ''}n {coef[1]:+.4f}"
          f"  -> ratio at n = 32 / 80 / 240: " + " / ".join(f"{np.exp(predict(coef, np.array([k]), args.kind)[0]):.2f}"
                                                         for k in (32, 80, 240)))
    # split-half: fit on one half, score the other
    halves = [vids[0::2], vids[1::2]]
    rows = {"none": [], "global": [], "per-frame": [], "ceiling (median ratio of the fit half)": []}
    for fit_half, test_half in ((halves[0], halves[1]), (halves[1], halves[0])):
        nf = np.array([len(caps[v]["idx"]) for v in fit_half])
        lf = np.log(np.array([none[v] / truth[v] for v in fit_half]))
        c = fit(nf, lf, args.kind)
        med = float(np.exp(np.median(lf)))
        for v in test_half:
            nv = len(caps[v]["idx"])
            g = float(np.exp(predict(c, np.array([nv]), args.kind)[0]))
            rows["none"].append(mra(none[v], truth[v]))
            rows["global"].append(mra(none[v] / g, truth[v]))
            fixed, _ = replay(caps[v], RankScaler(c, args.kind))
            rows["per-frame"].append(mra(fixed["area_m2"], truth[v]))
            rows["ceiling (median ratio of the fit half)"].append(mra(none[v] / med, truth[v]))
    print("\nsplit-half room-area MRA (fit on one half of the videos, scored on the other):")
    for k, v in rows.items():
        print(f"  {k:<44} {100 * statistics.mean(v):6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
