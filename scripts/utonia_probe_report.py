"""Would Utonia's numbers help? Its counts and floor area (scripts/utonia_scene_probe.py) against the VSI ground truth,
on the videos every given probe covers; with --run, next to what the model answered there (a VSI result file) and the
room facts it was given.

    python scripts/utonia_probe_report.py outputs/utonia_probe/probe.json \\
        --run outputs/vsi_local/i45d_fps1_stack_facts32_4b.json
    python scripts/utonia_probe_report.py outputs/utonia_probe_arkit_scan/probe.json \\
        outputs/utonia_probe_arkit_cut3r/probe.json

VSI's metric for numbers: mean relative accuracy over thresholds 0.50..0.95 (MRA). A measurement is worth attaching
only where it beats the model on the same questions (BRAINSTORM L4/L5).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

COUNT_RE = re.compile(r"How many (.+?)\(s\) are in this room")
FACT_RE = re.compile(r"about ([\d.]+) square")
COUNT_RULES = (("utonia", "every cluster", "all"), ("utonia_fp", "floorplan rules", "fp"))
ROOM_RULES = (("floor_area", "floor cells"), ("floor_box", "floor box"), ("all_box", "box of all points"))


def mra(pred, gt) -> float:
    if pred is None:
        return float("nan")
    gt = float(gt)
    if not gt:
        return float("nan")
    return float(np.mean([abs(float(pred) - gt) / gt < 1 - t for t in np.arange(0.5, 1.0, 0.05)]))


def number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group(0)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probes", nargs="+", help="probe.json files (scripts/utonia_scene_probe.py)")
    ap.add_argument("--run", default=None, help="a VSI result file: the model's answers and the room facts")
    ap.add_argument("--vsi", default="data/eval/vsibench/test.jsonl")
    a = ap.parse_args()

    probes = [{r["video"]: r for r in json.load(open(p))} for p in a.probes]
    labels = [Path(p).parent.name for p in a.probes]
    preds = {p["id"]: p for p in json.load(open(a.run))["predictions"]["oracle-image"]} if a.run else None
    by_video = {}
    for ln in open(a.vsi):
        q = json.loads(ln)
        if preds is None or q["id"] in preds:
            by_video.setdefault(f"{q['dataset']}/{q['scene_name']}", []).append(q)
    videos = sorted(set.intersection(*(set(p) for p in probes)) & set(by_video))

    count_rows, room_rows, n_count = [], [], 0          # per question: gt, per probe its entry, the model's answer
    for v in videos:
        for q in by_video[v]:
            p = preds.get(q["id"]) if preds else None
            model = number(p["pred"]) if p else None
            if q["question_type"] == "object_counting":
                n_count += 1
                m = COUNT_RE.search(q["question"])
                cs = [next((c for c in pr[v]["counts"] if m and c["object"] == m.group(1)), None) for pr in probes]
                if all(c and c["class"] for c in cs):
                    count_rows.append((cs[0]["object"], float(q["ground_truth"]), cs, model))
            if q["question_type"] == "room_size_estimation":
                fact = FACT_RE.search(p.get("attached", "")) if p else None
                room_rows.append((float(q["ground_truth"]), [pr[v] for pr in probes],
                                  float(fact.group(1)) if fact else None, model))

    count_series = [(f"{lab} · {name}", f"{lab}/{short}", k, key) for k, lab in enumerate(labels)
                    for key, name, short in COUNT_RULES if count_rows and key in count_rows[0][2][k]]
    room_series = [(f"{lab} · {name}", k, key) for k, lab in enumerate(labels)
                   for key, name in ROOM_RULES if room_rows and key in room_rows[0][1][k]]
    answer = "the model's answer"
    width = max(len(s[0]) for s in count_series + room_series + [(answer,)])
    print(f"{len(videos)} videos in every probe" + (" and the run" if preds else ""))
    if count_rows:
        gt = np.array([r[1] for r in count_rows])
        print(f"\ncounting questions on a ScanNet-20 class: {len(count_rows)} of {n_count}")
        for name, _, k, key in count_series:
            u = np.array([r[2][k][key] for r in count_rows], dtype=float)
            print(f"  {name:{width}s}  MRA {np.nanmean([mra(x, g) for x, g in zip(u, gt)]):.3f} · exact "
                  f"{100 * np.mean(u == gt):3.0f} % · over {int((u > gt).sum())} / under {int((u < gt).sum())}")
        if preds:
            md = [r[3] for r in count_rows]
            print(f"  {'the model':{width}s}  MRA {np.nanmean([mra(x, g) for x, g in zip(md, gt)]):.3f} · exact "
                  f"{100 * np.mean([x == g for x, g in zip(md, gt)]):3.0f} %")
        per = {}
        for obj, g, cs, md in count_rows:
            per.setdefault(obj, []).append([mra(cs[k][key], g) for _, _, k, key in count_series]
                                           + ([mra(md, g)] if preds else []))
        heads = [s[1] for s in count_series] + (["model"] if preds else [])
        for obj, rows in sorted(per.items(), key=lambda kv: -len(kv[1])):
            m = np.nanmean(np.array(rows), axis=0)
            print(f"    {obj:14s} n={len(rows):3d}  " + " · ".join(f"{h} {x:.2f}" for h, x in zip(heads, m)))
    if room_rows:
        print(f"\nroom size: {len(room_rows)} questions")
        for name, k, key in room_series:
            rows = [(r[0], r[1][k][key]) for r in room_rows if r[1][k][key]]
            print(f"  {name:{width}s}  MRA {np.nanmean([mra(x, g) for g, x in rows]):.3f} (n={len(rows)})"
                  f" · ratio to truth (median) {np.median([x / g for g, x in rows]):.2f}")
        if preds:
            facts = [(r[0], r[2]) for r in room_rows if r[2]]
            print(f"  {'our room fact':{width}s}  MRA {np.nanmean([mra(x, g) for g, x in facts]):.3f} (n={len(facts)})")
            print(f"  {answer:{width}s}  MRA {np.nanmean([mra(r[3], r[0]) for r in room_rows]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
