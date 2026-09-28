"""Replay scale-anchor variants on captured raw CUT3R streams (scripts/capture_geometry.py) and score the room area
each variant's scene map measures against VSI's ground truth - the metric probe of the state's scale drift.

    PYTHONPATH=src python scripts/scale_anchor_lab.py outputs/geom_capture/stride30 \\
        --ref-run outputs/vsi_local/i35_stride30_4b.json --grid quick

Two ways of applying a variant's per-frame factors, scored side by side:
  per-frame  every frame's points and camera are rescaled by its own factor (the v1 anchor, chain 13): columns
             ratio / log sd / MRA / free / <50%. In the 60-video run this destroyed the layout (scale-free MRA
             65 -> 22, directions 75 -> 66): frames scaled differently shift relative to each other.
  global     the geometry is left alone; one factor per video - the geometric mean of the per-frame factors -
             rescales the metric facts at read time: columns g.ratio / g.MRA. Angles are untouched by design.
Also per variant: the median factor applied, the correlation across videos between it and the ideal factor
(sqrt(truth / uncorrected area) - the signal check: a variant can be biased yet track the drift), and the share of
frames the support / gate test skipped. `none` must reproduce the
reference run's areas (--ref-run prints the largest relative difference) - that check validates the capture.
A single variant: --variant "mode=floor_level,window=9,support=0.05".
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vsi_room_facts_check import mra  # noqa: E402

from live3r.geometry.scale_anchor import CameraHeightAnchor, ScaleAnchor  # noqa: E402
from live3r.serve.scene_map import SceneMap  # noqa: E402


def load_capture(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in ("idx", "key", "c2w", "pts", "conf")}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def replay(cap: dict, anchor: ScaleAnchor | None, scene_map_kw: dict | None = None) -> tuple[dict, ScaleAnchor | None]:
    """Feed the captured frames through the anchor (if any) and a SceneMap exactly as the live pipeline does."""
    sm = SceneMap(stride=1, **(scene_map_kw or {}))
    for f in range(len(cap["idx"])):
        c2w = cap["c2w"][f].astype(np.float64)
        pts = cap["pts"][f].astype(np.float64)
        conf = cap["conf"][f]
        if anchor is not None:
            fac = anchor.update(c2w, pts, conf)
            if fac != 1.0:
                c2w, pts = anchor.apply(fac, c2w, pts)
        out = SimpleNamespace(
            pointmap=torch.from_numpy(np.ascontiguousarray(pts.T.reshape(1, 3, 1, -1)).astype(np.float32)),
            conf=torch.from_numpy(conf.reshape(1, 1, 1, -1).astype(np.float32)),
            extra={"c2w": torch.from_numpy(c2w.astype(np.float32))[None]},
            frame_index=int(cap["idx"][f]),
        )
        sm.add(out, frame_index=int(cap["idx"][f]))
    return sm.facts(), anchor


def parse_variant(text: str) -> dict:
    kw: dict = {}
    for part in filter(None, text.split(",")):
        k, v = part.split("=")
        k = k.strip()
        v = v.strip()
        kw[k] = v if k == "mode" else float(v) if "." in v else int(v)
    return kw


def variants(grid: str) -> dict[str, dict | None]:
    out: dict[str, dict | None] = {"none": None, "v1 camera_height (chain 13)": {"_v1": True}}
    if grid == "none":
        return out
    full = grid == "full"
    windows = (5,) if grid == "smoke" else (5, 9, 15)
    supports = (0.0,) if grid == "smoke" else ((0.0, 0.05, 0.10) if full else (0.0, 0.05))
    gates = (0.0,) if grid == "smoke" else ((0.0, 0.25, 0.4) if full else (0.0, 0.25))
    refs = (10, 20) if full else (10,)
    pcts = (1.0, 2.0, 5.0) if full else (2.0,)
    rqs = (0.5, 0.8) if full else (0.5,)
    for mode, w, sup, gate, ref, pct, rq in itertools.product(("camera_height", "floor_level"), windows, supports,
                                                              gates, refs, pcts, rqs):
        name = f"{mode} w{w} sup{sup:.2f} gate{gate:.2f} ref{ref} pct{pct:g} rq{rq:.1f}"
        out[name] = {"mode": mode, "window": w, "support": sup, "gate": gate, "ref_frames": ref, "floor_pct": pct,
                     "ref_quantile": rq}
    return out


def make_anchor(spec: dict | None) -> ScaleAnchor | None:
    if spec is None:
        return None
    if spec.get("_v1"):
        return CameraHeightAnchor()
    return ScaleAnchor(**spec)


def score(pairs: list[tuple[float, float]]) -> dict:
    ratios = [p / t for p, t in pairs]
    med = statistics.median(ratios)
    return {
        "n": len(pairs),
        "median_ratio": med,
        "log_sd": float(np.std(np.log(ratios))),
        "mra": statistics.mean(mra(p, t) for p, t in pairs),
        "mra_scale_free": statistics.mean(mra(p / med, t) for p, t in pairs),
        "within_50": statistics.mean(abs(r - 1) < 0.5 for r in ratios),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    ap.add_argument("--ref-run", help="eval_vsi_local results JSON of the same run - `none` must reproduce its maps")
    ap.add_argument("--grid", default="quick", choices=("none", "smoke", "quick", "full"))
    ap.add_argument("--variant", action="append", default=[], help='e.g. "mode=floor_level,window=9,support=0.05"')
    ap.add_argument("--json", help="write the table here")
    args = ap.parse_args()

    files = sorted(Path(args.capture_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"no captures in {args.capture_dir}")
    caps = {}
    for f in files:
        c = load_capture(f)
        caps[f"{c['meta']['dataset']}/{c['meta']['scene']}"] = c
    truth = {}
    for ln in open(Path(args.vsi_root) / "test.jsonl"):
        if ln.strip():
            d = json.loads(ln)
            if d["question_type"] == "room_size_estimation":
                truth[f"{d['dataset']}/{d['scene_name']}"] = float(d["ground_truth"])
    ref_maps = json.loads(Path(args.ref_run).read_text()).get("maps", {}) if args.ref_run else {}
    print(f"{len(caps)} captured videos · {sum(v in truth for v in caps)} with a room-area truth")

    todo = variants(args.grid)
    for text in args.variant:
        todo[text] = parse_variant(text)
    rows = []
    none_area: dict[str, float] = {}      # the uncorrected area per video: the ideal factor is sqrt(truth / this)
    for name, spec in todo.items():
        pairs, gpairs, applied, ideal, skipped, frames, ref_diff = [], [], [], [], 0, 0, 0.0
        for vid, cap in caps.items():
            facts, anchor = replay(cap, make_anchor(spec))
            if spec is None and facts.get("area_m2"):
                none_area[vid] = facts["area_m2"]
            if anchor is not None:
                skipped += anchor.skipped
                frames += anchor.frames
                if anchor.history and vid in truth and none_area.get(vid):
                    g = float(np.exp(np.mean(np.log(anchor.history))))
                    applied.append(g)
                    ideal.append(float(np.sqrt(truth[vid] / none_area[vid])))
                    gpairs.append((none_area[vid] * g * g, truth[vid]))     # global: the facts rescaled at read time
            if spec is None and vid in ref_maps and ref_maps[vid].get("area_m2"):
                ref_diff = max(ref_diff, abs(facts["area_m2"] - ref_maps[vid]["area_m2"]) / ref_maps[vid]["area_m2"])
            if vid in truth and facts.get("area_m2"):
                pairs.append((facts["area_m2"], truth[vid]))
        if not pairs:
            continue
        corr = float(np.corrcoef(np.log(applied), np.log(ideal))[0, 1]) if len(applied) >= 3 else float("nan")
        gs = score(gpairs) if gpairs else {}
        row = {"variant": name, **score(pairs),
               "global_ratio": gs.get("median_ratio"), "global_mra": gs.get("mra"),
               "global_within_50": gs.get("within_50"),
               "applied_factor": statistics.median(applied) if applied else 1.0,
               "factor_corr": corr,                 # does the correction track the drift video by video?
               "skipped": skipped / frames if frames else 0.0}
        if spec is None and ref_maps:
            row["ref_max_rel_diff"] = ref_diff
        rows.append(row)
        gtxt = f"  global ratio {gs['median_ratio']:.2f} MRA {100 * gs['mra']:.1f}" if gs else ""
        print(f"  {name:<56} per-frame ratio {row['median_ratio']:.2f} sd {row['log_sd']:.2f} MRA {100 * row['mra']:.1f}"
              f" free {100 * row['mra_scale_free']:.1f}{gtxt}  factor {row['applied_factor']:.2f} corr {corr:+.2f}"
              f" skipped {100 * row['skipped']:.0f}%"
              + (f"  ref diff {100 * ref_diff:.1f}%" if "ref_max_rel_diff" in row else ""), flush=True)

    rows.sort(key=lambda r: -max(r["mra"], r["global_mra"] or 0.0))
    print(f"\n{'variant':<56}{'ratio':>7}{'log sd':>8}{'MRA':>7}{'free':>7}{'g.ratio':>9}{'g.MRA':>7}{'g.<50%':>7}"
          f"{'factor':>8}{'corr':>7}{'skip':>6}")
    for r in rows:
        g, gr, gw = r["global_mra"], r["global_ratio"], r["global_within_50"]
        gcols = (f"{gr:>9.2f}{100 * g:>7.1f}{100 * gw:>6.0f}%" if g is not None else f"{'-':>9}{'-':>7}{'-':>7}")
        print(f"{r['variant']:<56}{r['median_ratio']:>7.2f}{r['log_sd']:>8.2f}{100 * r['mra']:>7.1f}"
              f"{100 * r['mra_scale_free']:>7.1f}{gcols}"
              f"{r['applied_factor']:>8.2f}{r['factor_corr']:>+7.2f}{100 * r['skipped']:>5.0f}%")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
