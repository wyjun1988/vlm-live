"""Instance rules for the measured object facts, replayed from dumped observations (I-49).

    python scripts/instance_rules_lab.py outputs/objdump/i49_live128 --run outputs/vsi_local/i49_dump_live_cap128_4b.json

`eval_vsi_local.py --dump-objects` writes, per video, every lifted detection of the 32 measurement frames (set a) and
of 32 frames one second later (set b): label, frame, detector score, floor-plan position and point samples. Here the
online merging (ObjectMap) is replayed on each set and the two object facts this stack attaches are recomputed under
each rule - relative direction (>= 1 view) and absolute distance (>= 3 views) - which instance stands for a named
object, after an optional clean-up of the instance list. Printed per rule: coverage and accuracy against VSI's ground
truth on both sets, and how often the two sets agree - a rule that finds the true object should not care which frames
saw it. The rule 'most_views' with no clean-up is today's; on set a it must reproduce the run's logged facts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from live3r.serve.object_map import ObjectMap, canon  # noqa: E402


def replay(d, tag: str) -> ObjectMap:
    """The online merging, re-run on one set's observations (floor-plan coordinates: distances are unchanged)."""
    om = ObjectMap()
    for lab, fr, sc, xyz, pts, bpts in zip(d[f"{tag}_label"], d[f"{tag}_frame"], d[f"{tag}_score"], d[f"{tag}_xyz"],
                                           d[f"{tag}_pts"], d[f"{tag}_box_pts"]):
        om._observe(str(lab), xyz.astype(np.float64), int(fr), pts[~np.isnan(pts).any(1)],
                    bpts[~np.isnan(bpts).any(1)], score=float(sc))
    return om


def _score_sum(i) -> float:
    return float(np.nansum(i["scores"]))


# which instance stands for a named object
RULES = {
    "most_views": lambda inst: max(inst, key=lambda i: len(i["frames"])),             # today's (first max wins)
    "score_sum": lambda inst: max(inst, key=_score_sum),
    "score_max": lambda inst: max(inst, key=lambda i: float(np.nanmax(i["scores"]))),
    "views_then_score": lambda inst: max(inst, key=lambda i: (len(i["frames"]), _score_sum(i))),
}


def suppress_overlapping_labels(om: ObjectMap, frac: float = 0.5, eps: float = 0.1) -> list[dict]:
    """One physical thing, one label: when two instances of different labels cover the same surface - each one's 3D
    points mostly (>= frac) within eps of the other's - the one with less detector evidence (summed scores) is dropped.
    OWLv2's suppression is per label, so a cabinet front can also come out as a door; an object standing on another
    (a printer on a desk) overlaps it one way only and stays."""
    from scipy.spatial import cKDTree

    inst = [i for i in om._inst if len(i["pts"])]
    cen = [np.median(i["obs"], axis=0) for i in inst]
    trees = [cKDTree(i["pts"]) for i in inst]
    keep = []
    for k, i in enumerate(inst):
        dropped = False
        for j, other in enumerate(inst):
            if (j == k or other["label"] == i["label"] or _score_sum(other) <= _score_sum(i)
                    or np.linalg.norm(cen[j] - cen[k]) > 2.0):
                continue
            d_ij, _ = trees[j].query(i["pts"], k=1)
            d_ji, _ = trees[k].query(other["pts"], k=1)
            if np.mean(d_ij < eps) >= frac and np.mean(d_ji < eps) >= frac:
                dropped = True
                break
        if not dropped:
            keep.append(i)
    return keep


def merge_unseen_together(om: ObjectMap, dist: float) -> list[dict]:
    """Same label, never detected in the same frame, centroids within `dist`: one object seen at different times
    (drift moved it) - merged."""
    groups = [dict(i, obs=list(i["obs"]), scores=list(i["scores"]), frames=set(i["frames"])) for i in om._inst]
    merged = True
    while merged:
        merged = False
        for a, b in itertools.combinations(range(len(groups)), 2):
            A, B = groups[a], groups[b]
            if A["label"] != B["label"] or A["frames"] & B["frames"]:
                continue
            if np.linalg.norm(np.median(A["obs"], axis=0)[:2] - np.median(B["obs"], axis=0)[:2]) < dist:
                A["obs"] += B["obs"]
                A["scores"] += B["scores"]
                A["frames"] |= B["frames"]
                A["pts"] = np.concatenate([A["pts"], B["pts"]])
                del groups[b]
                merged = True
                break
    return groups


CLEANUPS = {
    "none": lambda om: om._inst,
    "overlap": lambda om: suppress_overlapping_labels(om, 0.5, 0.1),
    "merge1.5": lambda om: merge_unseen_together(om, 1.5),
}


def candidates(inst_list, om: ObjectMap, name: str, min_views: int) -> list[dict]:
    want = canon(name)
    return [i for i in inst_list if i["label"] == want and len(i["frames"]) >= min_views]


def direction(inst_list, om, e, rule) -> str | None:
    from eval_vsi_local import direction_label

    pos = []
    for n in (e["a"], e["b"], e["c"]):
        c = candidates(inst_list, om, n, 1)
        if not c:
            return None
        pos.append(np.median(RULES[rule](c)["obs"], axis=0)[:2])
    return direction_label(pos[0], pos[1], pos[2], e["difficulty"])


def distance(inst_list, om, e, rule, today: bool) -> float | None:
    A, B = candidates(inst_list, om, e["a"], 3), candidates(inst_list, om, e["b"], 3)
    if not A or not B:
        return None
    pairs = itertools.product(A, B) if today else [(RULES[rule](A), RULES[rule](B))]
    ds = [ObjectMap.closest_distance(a, b) for a, b in pairs]
    ds = [x for x in ds if x is not None]
    return min(ds) if ds else None


def mra(pred, gt) -> float:
    return float(np.mean([abs(pred - gt) / gt < 1 - t for t in np.arange(0.5, 1.0, 0.05)]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dumps", help="folder of <dataset>__<scene>.npz from --dump-objects")
    ap.add_argument("--run", required=True, help="the dump run's result file (its measurement log and scales)")
    ap.add_argument("--vsi", default="data/eval/vsibench/test.jsonl")
    a = ap.parse_args()

    docs = {json.loads(ln)["id"]: json.loads(ln) for ln in open(a.vsi)}
    run = json.load(open(a.run))
    scales = run.get("metric_scales", {})
    meas = {}
    for e in run["measurements"]:
        if e["kind"] in ("direction", "abs") and "id" in e:
            d = docs[e["id"]]
            meas.setdefault(f"{d['dataset']}/{d['scene_name']}", []).append(e)
    maps = {}
    for f in sorted(Path(a.dumps).glob("*.npz")):
        vid = f.stem.replace("__", "/", 1)
        d = np.load(f)
        maps[vid] = {t: replay(d, t) for t in "ab"}

    def gt_dir(i):
        d = docs[i]
        return next(o.split(".", 1)[1].strip().lower() for o in d["options"]
                    if o.strip().startswith(d["ground_truth"].strip() + "."))

    # the replay must reproduce the run's own facts on set a (today's rule, no clean-up)
    same_dir = same_abs = n_dir = n_abs = 0
    for vid, es in meas.items():
        if vid not in maps:
            continue
        om = maps[vid]["a"]
        for e in es:
            if e["kind"] == "direction":
                n_dir += 1
                same_dir += direction(om._inst, om, e, "most_views") == e["r1"]
            else:
                n_abs += 1
                r = distance(om._inst, om, e, "most_views", today=True)
                same_abs += (r is None and e["d3"] is None) or (r is not None and e["d3"] is not None
                                                               and abs(r - e["d3"]) < 0.05 * max(e["d3"], 0.1))
    print(f"{len(maps)} videos · replay reproduces the logged facts: direction {same_dir}/{n_dir}, "
          f"abs distance {same_abs}/{n_abs} (within 5 %)\n")

    print(f"{'clean-up':10s} {'rule':17s} | direction: cov  right(a) right(b) agree(a,b) | "
          f"abs distance: cov  MRA(a) MRA(b) agree(a,b)")
    for cu, rule in [("none", r) for r in RULES] + [(c, "most_views") for c in CLEANUPS if c != "none"] \
            + [(c, "score_sum") for c in CLEANUPS if c != "none"]:
        res = {"a": {"dir": {}, "abs": {}}, "b": {"dir": {}, "abs": {}}}
        for vid, es in meas.items():
            if vid not in maps:
                continue
            for t in "ab":
                om = maps[vid][t]
                inst = CLEANUPS[cu](om)
                for e in es:
                    if e["kind"] == "direction":
                        res[t]["dir"][e["id"]] = direction(inst, om, e, rule)
                    else:
                        r = distance(inst, om, e, rule, today=(rule == "most_views" and cu == "none"))
                        res[t]["abs"][e["id"]] = None if r is None else r * scales.get(vid, 1.0)
        n = len(res["a"]["dir"])
        cells = []
        cov = [i for i in res["a"]["dir"] if res["a"]["dir"][i] is not None]
        acc = {t: np.mean([res[t]["dir"][i] == gt_dir(i) for i in res[t]["dir"] if res[t]["dir"][i] is not None])
               for t in "ab"}
        both = [i for i in res["a"]["dir"] if res["a"]["dir"][i] is not None and res["b"]["dir"].get(i) is not None]
        agree = np.mean([res["a"]["dir"][i] == res["b"]["dir"][i] for i in both]) if both else float("nan")
        cells.append(f"{100 * len(cov) / max(n, 1):4.0f}%  {100 * acc['a']:6.1f}  {100 * acc['b']:6.1f}  "
                     f"{100 * agree:8.0f}")
        na = len(res["a"]["abs"])
        cova = [i for i in res["a"]["abs"] if res["a"]["abs"][i] is not None]
        m = {t: np.mean([mra(res[t]["abs"][i], float(docs[i]["ground_truth"])) for i in res[t]["abs"]
                         if res[t]["abs"][i] is not None]) for t in "ab"}
        bo = [i for i in cova if res["b"]["abs"].get(i) is not None]
        ag = np.mean([abs(res["a"]["abs"][i] - res["b"]["abs"][i]) < 0.1 * max(res["a"]["abs"][i], 0.1)
                      for i in bo]) if bo else float("nan")
        cells.append(f"{100 * len(cova) / max(na, 1):4.0f}%  {100 * m['a']:5.1f}  {100 * m['b']:5.1f}  "
                     f"{100 * ag:7.0f}")
        print(f"{cu:10s} {rule:17s} | {cells[0]:>34s} | {cells[1]:>34s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
