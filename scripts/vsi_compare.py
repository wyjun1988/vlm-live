"""Paired comparison of two eval_vsi_local.py results: Δ overall with a video-level bootstrap interval, per type.

    python scripts/vsi_compare.py outputs/vsi_local/new.json outputs/vsi_local/old.json [--mode oracle-image]

Both runs must have answered the same questions (same --videos / --seed selection). Older result files carry no
per-question score; they are re-scored here with lmms-eval's own VSI metric from the saved prediction and ground
truth, and the video comes from data/eval/vsibench/test.jsonl. The interval resamples videos, so the many
questions about one video move together (they are not independent).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / ".refs" / "lmms-eval"))


def _report_module():
    spec = importlib.util.spec_from_file_location("weekend_report", HERE / "weekend_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rescore(preds: list[dict], video_of: dict) -> list[dict]:
    """Add `score` and `video` to each saved prediction (lmms-eval's metric: accuracy or MRA)."""
    from lmms_eval.tasks.vsibench.utils import vsibench_process_results

    out = []
    for p in preds:
        if "score" in p and "video" in p:
            out.append(p)
            continue
        doc = vsibench_process_results({"question_type": p["type"], "ground_truth": p["gt"]}, [p["pred"]])["vsibench_overall"]
        out.append({**p, "score": float(doc.get("accuracy", doc.get("MRA:.5:.95:.05", 0.0))),
                    "video": video_of[int(p["id"])]})
    return out


def load(path: str, mode: str, video_of: dict) -> list[dict]:
    d = json.loads(Path(path).read_text())
    if mode not in d["predictions"]:
        raise SystemExit(f"{path} has modes {list(d['predictions'])}, not {mode}")
    return rescore(d["predictions"][mode], video_of)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="the new run")
    ap.add_argument("b", help="the reference run")
    ap.add_argument("--mode", default="oracle-image", help="mode of the new run")
    ap.add_argument("--ref-mode", default=None, help="mode of the reference run (default: same as --mode; e.g. "
                                                    "oracle-video to compare against the base's best format)")
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    args = ap.parse_args()
    docs = [json.loads(ln) for ln in open(Path(args.vsi_root) / "test.jsonl") if ln.strip()]
    video_of = {d["id"]: f"{d['dataset']}/{d['scene_name']}" for d in docs}
    wr = _report_module()
    a, b = load(args.a, args.mode, video_of), load(args.b, args.ref_mode or args.mode, video_of)
    ids = set(p["id"] for p in a) & set(p["id"] for p in b)
    a, b = [p for p in a if p["id"] in ids], [p for p in b if p["id"] in ids]
    oa, ta = wr.overall(a)
    ob, tb = wr.overall(b)
    d, lo, hi, nv, nq = wr.paired_bootstrap(a, b)
    print(f"{Path(args.a).name}  {100 * oa:.1f}   vs   {Path(args.b).name}  {100 * ob:.1f}")
    print(f"Δ overall {d:+.1f}  95% [{lo:+.1f}, {hi:+.1f}]   ({nv} videos, {nq} common questions)")
    print(f"\n{'type':<28}{'new':>8}{'ref':>8}{'Δ':>8}")
    for t in sorted(set(ta) | set(tb)):
        print(f"{t:<28}{100 * ta.get(t, float('nan')):8.1f}{100 * tb.get(t, float('nan')):8.1f}"
              f"{100 * (ta.get(t, 0) - tb.get(t, 0)):+8.1f}")
    # where did the answers change?
    pa, pb = {p["id"]: p for p in a}, {p["id"]: p for p in b}
    changed = defaultdict(lambda: [0, 0.0])
    for i in ids:
        if pa[i]["pred"] != pb[i]["pred"]:
            changed[pa[i]["type"]][0] += 1
            changed[pa[i]["type"]][1] += pa[i]["score"] - pb[i]["score"]
    if changed:
        print("\nanswers that changed (count, net score change):")
        for t, (n, s) in sorted(changed.items()):
            print(f"  {t:<28}{n:5d}  {s:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
