"""What the measurements attached (or would have attached) at each view-count threshold are worth.

    python scripts/vsi_measurement_table.py outputs/vsi_local/i32_order_4b.json

Reads the `measurements` log that eval_vsi_local.py writes with --route-objects (one entry per routed question,
every threshold 1..6 evaluated from the same run) and, per kind, prints coverage, the measurement's own accuracy
on the questions it covers, and the model's accuracy on the same questions - so the value of attaching is
(measurement - model) x coverage, before spending a run on it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / ".refs" / "lmms-eval"))

from live3r.serve.object_map import canon  # noqa: E402

KS = (1, 2, 3, 4, 5, 6)


def mra(pred: float, gt: float) -> float:
    """lmms-eval's mean relative accuracy (thresholds .5 .. .95 step .05)."""
    if gt == 0:
        return float(pred == 0)
    rel = abs(pred - gt) / abs(gt)
    ths = [0.5 + 0.05 * i for i in range(10)]
    return sum(rel < 1 - t for t in ths) / len(ths)


def option_order(doc: dict) -> list[str]:
    letter = doc["ground_truth"].strip()
    for opt in doc["options"] or []:
        if opt.strip().startswith(letter + "."):
            return [canon(n.strip()) for n in opt.split(".", 1)[1].split(",")]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    args = ap.parse_args()
    d = json.loads(Path(args.result).read_text())
    docs = {x["id"]: x for x in (json.loads(ln) for ln in open(Path(args.vsi_root) / "test.jsonl") if ln.strip())}
    mode = next(iter(d["predictions"]))
    preds = {p["id"]: p for p in d["predictions"][mode]}

    from lmms_eval.tasks.vsibench.utils import extract_number, fuzzy_matching

    by_kind = defaultdict(list)
    for m in d.get("measurements", []):
        if "id" in m:
            by_kind[m["kind"]].append(m)
    print(f"{Path(args.result).name}: {sum(len(v) for v in by_kind.values())} routed questions, kinds {sorted(by_kind)}")

    for kind, rows in sorted(by_kind.items()):
        print(f"\n== {kind} ({len(rows)} questions)")
        print(f"{'min_frames':>10}{'coverage':>10}{'measured':>10}{'model':>10}{'model@all':>10}   value/question")
        for k in KS:
            cov, meas_ok, model_ok, model_all = [], [], [], []
            for m in rows:
                doc, p = docs[m["id"]], preds[m["id"]]
                if kind == "order":
                    model = float(fuzzy_matching(p["pred"]) == doc["ground_truth"].strip())
                    seen = m[f"f{k}"]
                    covered = all(v is not None for v in seen.values()) and len(set(seen.values())) == len(seen)
                    if covered:
                        order = [canon(n) for n in sorted(seen, key=lambda n: seen[n])]
                        meas_ok.append(float(order == option_order(doc)))
                elif kind == "count":
                    gt = float(doc["ground_truth"])
                    try:
                        model = mra(extract_number(p["pred"]), gt)
                    except Exception:  # noqa: BLE001
                        model = 0.0
                    covered = m[f"c{k}"] > 0
                    if covered:
                        meas_ok.append(mra(m[f"c{k}"], gt))
                elif kind == "abs":
                    gt = float(doc["ground_truth"])
                    try:
                        model = mra(extract_number(p["pred"]), gt)
                    except Exception:  # noqa: BLE001
                        model = 0.0
                    covered = m[f"d{k}"] is not None
                    if covered:
                        meas_ok.append(mra(m[f"d{k}"], gt))
                elif kind == "direction":
                    gt = doc["ground_truth"].strip()
                    gt_label = next((o.split(".", 1)[1].strip().lower() for o in doc["options"] if o.strip().startswith(gt + ".")), None)
                    model = float(fuzzy_matching(p["pred"]) == gt)
                    covered = m[f"r{k}"] is not None
                    if covered:
                        meas_ok.append(float(m[f"r{k}"] == gt_label))
                elif kind == "size":
                    gt = float(doc["ground_truth"])
                    try:
                        model = mra(extract_number(p["pred"]), gt)
                    except Exception:  # noqa: BLE001
                        model = 0.0
                    covered = m[f"s{k}"] is not None
                    if covered:
                        meas_ok.append(mra(100 * m[f"s{k}"], gt))
                else:
                    continue
                model_all.append(model)
                cov.append(covered)
                if covered:
                    model_ok.append(model)
            n = len(cov)
            c = sum(cov) / n if n else 0.0
            me = sum(meas_ok) / len(meas_ok) if meas_ok else float("nan")
            mo = sum(model_ok) / len(model_ok) if model_ok else float("nan")
            ma = sum(model_all) / n if n else float("nan")
            gain = c * (me - mo) if meas_ok else 0.0
            print(f"{k:>10}{100 * c:>9.0f}%{100 * me:>10.1f}{100 * mo:>10.1f}{100 * ma:>10.1f}   {100 * gain:+.1f} pts on this type")
    return 0


if __name__ == "__main__":
    sys.exit(main())
