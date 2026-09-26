"""How good are the room measurements a run's scene maps produced, against VSI's room-size ground truth.

    python scripts/vsi_room_facts_check.py outputs/vsi_local/i33_directions_4b.json [more.json ...]

eval_vsi_local.py stores `maps[video] = SceneMap.facts()` (area_m2, height_m, ...) for every video of a run; VSI has
a ground-truth room area for the videos with a room-size question. Prints, per run: coverage, MRA of the measured
area (the metric VSI uses), the median measured/true ratio (a scale error shows up here - 1.0 is metric), and the
MRA after dividing every area by that median ratio (the *shape* quality with the scale error taken out). Meant for
the encoder bake-off (I-35): a normalised-scale encoder is compared on the last two columns.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def mra(pred: float, gt: float) -> float:
    if gt == 0:
        return float(pred == 0)
    rel = abs(pred - gt) / abs(gt)
    return sum(rel < 1 - (0.5 + 0.05 * i) for i in range(10)) / 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    args = ap.parse_args()
    docs = [json.loads(ln) for ln in open(Path(args.vsi_root) / "test.jsonl") if ln.strip()]
    truth = {}
    for d in docs:
        if d["question_type"] == "room_size_estimation":
            truth[f"{d['dataset']}/{d['scene_name']}"] = float(d["ground_truth"])
    print(f"{'run':<34}{'videos':>7}{'with GT':>8}{'MRA':>7}{'median ratio':>14}{'MRA scale-free':>16}{'within 50%':>12}")
    for path in args.results:
        d = json.loads(Path(path).read_text())
        maps = d.get("maps", {})
        pairs = [(m["area_m2"], truth[v]) for v, m in maps.items() if v in truth and m.get("area_m2")]
        if not pairs:
            print(f"{Path(path).name:<34}{len(maps):>7}{0:>8}   (no room facts)")
            continue
        ratios = [p / t for p, t in pairs]
        med = statistics.median(ratios)
        raw = statistics.mean(mra(p, t) for p, t in pairs)
        norm = statistics.mean(mra(p / med, t) for p, t in pairs)
        within = statistics.mean(abs(r - 1) < 0.5 for r in ratios)
        print(f"{Path(path).name:<34}{len(maps):>7}{len(pairs):>8}{100 * raw:>7.1f}{med:>14.2f}{100 * norm:>16.1f}{100 * within:>11.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
