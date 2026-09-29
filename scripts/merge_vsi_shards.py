"""Combine the shards of one sharded VSI run (eval_vsi_local.py --shard k --num-shards n) into one result file.

    python scripts/merge_vsi_shards.py outputs/run2/vsi/s2_real.json outputs/run2/vsi/shards/s2_real.*.json

Checks that the shards are one run (same settings, shard indices 0..n-1 each exactly once), concatenates the
per-question predictions and logs, restores the video order of the unsharded selection, and re-aggregates the
scores from the shards' full scored records with lmms-eval's own function - the result equals what a single process
would have written (the per-question answers are deterministic; only the aggregation is recomputed). The merged
file drops the bulky records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
if (HERE.parent / ".refs" / "lmms-eval").exists():
    sys.path.insert(0, str(HERE.parent / ".refs" / "lmms-eval"))

# keys that must agree across shards (everything that defines the run); the rest is per-shard data
SETTINGS = ("model", "budget", "config", "weights", "scene_map", "format_hint", "map_image", "object_map", "detector",
            "oracle_vocab", "self_map", "route_facts", "route_objects", "min_frames", "interpret", "type_hints",
            "describe_first", "facts_hint", "hint_after", "order_facts", "count_facts", "direction_facts",
            "direction_min_frames", "route_plan_facts", "detect_stride", "view_gap", "frame_labels", "thinking",
            "think_tokens", "patch_embed")
LISTS = ("measurements", "interpretations", "descriptions", "thinking_log")
DICTS = ("maps", "objects", "self_maps", "metric_scales")


def aggregate(records: list[dict]) -> dict:
    """lmms-eval's aggregation, with eval_vsi_local.py's fallback when a question type is absent."""
    from lmms_eval.tasks.vsibench.utils import _compute_all_subscores

    try:
        return {k: float(v) for k, v in _compute_all_subscores(records).items()}
    except KeyError:
        by: dict[str, list] = {}
        for r in records:
            t = r["question_type"]
            by.setdefault("object_rel_direction" if t.startswith("object_rel_direction") else t, []).append(
                r.get("accuracy", r.get("MRA:.5:.95:.05", 0.0)))
        out = {f"{k}_partial": sum(v) / len(v) for k, v in by.items()}
        out["overall"] = sum(out.values()) / len(out)
        return out


def format_fail(records: list[dict]) -> float:
    from lmms_eval.tasks.vsibench.utils import MCA_QUESTION_TYPES, extract_number, fuzzy_matching

    def bad(r):
        if r["question_type"] in MCA_QUESTION_TYPES:
            return fuzzy_matching(r["prediction"]) not in ("A", "B", "C", "D")
        try:
            return extract_number(r["prediction"]) is None
        except Exception:  # noqa: BLE001
            return True

    return sum(bad(r) for r in records) / max(1, len(records))


def merge(shards: list[dict]) -> dict:
    if not shards:
        raise ValueError("no shards")
    n = shards[0].get("shard", [0, 1])[1] if shards[0].get("shard") else 1
    idx = sorted(s["shard"][0] for s in shards if s.get("shard"))
    if n < 2 or idx != list(range(n)) or any((s.get("shard") or [0, 0])[1] != n for s in shards):
        raise ValueError(f"expected shards 0..{n - 1} of one run exactly once, got {idx} (of {n})")
    for k in SETTINGS:
        vals = {json.dumps(s.get(k), sort_keys=True) for s in shards}
        if len(vals) > 1:
            raise ValueError(f"shards disagree on {k!r}: {sorted(vals)}")
    if any(s.get("records") is None for s in shards):
        raise ValueError("a shard has no scored records - was it written by eval_vsi_local.py --num-shards?")
    by_idx = {s["shard"][0]: s for s in shards}
    modes = list(shards[0]["records"])
    out = {k: shards[0].get(k) for k in SETTINGS}
    # the unsharded order: video i of the selection went to shard i % n, position i // n
    longest = max(len(s["videos"]) for s in shards)
    out["videos"] = [by_idx[j]["videos"][i] for i in range(longest) for j in range(n) if i < len(by_idx[j]["videos"])]
    records = {m: [r for s in shards for r in s["records"][m]] for m in modes}
    order = {v: i for i, v in enumerate(out["videos"])}
    for m in modes:
        records[m].sort(key=lambda r: (order.get(f"{r['dataset']}/{r['scene_name']}", 1 << 30), r["id"]))
    preds = {m: [p for s in shards for p in s["predictions"][m]] for m in modes}
    for m in modes:
        preds[m].sort(key=lambda p: (order.get(p["video"], 1 << 30), p["id"]))
    out["n_questions"] = len(records[modes[0]])
    out["scores"] = {m: aggregate(records[m]) for m in modes}
    out["format_fail"] = {m: format_fail(records[m]) for m in modes}
    out["predictions"] = preds
    for k in LISTS:
        out[k] = [x for s in shards for x in (s.get(k) or [])]
    for k in DICTS:
        merged: dict = {}
        for s in shards:
            merged.update(s.get(k) or {})
        out[k] = merged
    out["shards"] = n
    out["records"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args()
    shards = [json.loads(Path(p).read_text()) for p in args.shards]
    merged = merge(shards)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    sc = merged["scores"]
    print(f"{args.out}: {len(shards)} shards · {merged['n_questions']} questions · "
          + " · ".join(f"{m} {100 * v['overall']:.1f}" for m, v in sc.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
