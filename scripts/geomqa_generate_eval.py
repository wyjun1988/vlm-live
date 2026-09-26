"""Generate answers to the held-out geometry questions (I-14) and score them per kind.

    PYTHONPATH=src python scripts/geomqa_generate_eval.py --config configs/m2_pilot_08b.yaml \\
        --weights real=outputs/m2_geomqa_s1_real/final.pt control=outputs/m2_geomqa_s1_control/final.pt \\
        --ann data/geomqa/holdout_v.jsonl --media-root data/geomqa/media --device mps --out outputs/m2_geomqa_gen.json

The holdout-loss ablation (eval_geometry_ablation.py) is the pre-registered decisive number; this is the readable
one: turn direction is a 3-way choice (33% by guessing), "closest image" a 4-way choice (25%), the rest are
numbers scored with VSI's mean relative accuracy. For each projector three conditions are generated:
  real      this record's geometry
  shuffled  another record's geometry (same grid) - content wrong, signal present
  none      no injection at all (the frozen LLM alone)
If the projector reads the geometry's content, `real` beats `shuffled` on turn direction and displacement; a
projector that only learned the answer format scores the same in both.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

CHOICE_KINDS = {"camera_turn", "closest_image"}


def mra(pred: float, gt: float) -> float:
    """VSI's mean relative accuracy (thresholds .5 .. .95 step .05)."""
    if gt == 0:
        return float(pred == 0)
    rel = abs(pred - gt) / abs(gt)
    return sum(rel < 1 - (0.5 + 0.05 * i) for i in range(10)) / 10


def first_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def score(kind: str, pred: str, truth: str) -> float:
    pred = pred.strip().split("\n")[0].strip().lower().rstrip(".")
    truth = truth.strip().lower()
    if kind == "camera_turn":          # exactly one direction word, and the right one
        want = "same" if "same" in truth else ("left" if "left" in truth else "right")
        return float({o for o in ("same", "left", "right") if o in pred} == {want})
    if kind == "closest_image":
        m = re.search(r"image\s*(\d+)", pred)
        return float(m is not None and f"image {m.group(1)}" == truth)
    p, t = first_number(pred), first_number(truth)
    return 0.0 if p is None or t is None else mra(p, t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", nargs="+", required=True, help="label=path.pt ... (projector weights, S1)")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--n", type=int, default=0, help="records to use (0 = all)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from live3r.config import Live3RConfig
    from live3r.data.datasets import SpatialVQADataset, normalize_record
    from live3r.data.prompt import PromptBuilder
    from live3r.train.train import build_model

    dev = torch.device(args.device)
    cfg = Live3RConfig.from_yaml(args.config)
    model = build_model(argparse.Namespace(stage="align", init_from=None, tokenizer=None), cfg, True).to(dev).eval()
    weights = {}
    for spec in args.weights:
        label, _, path = spec.partition("=")
        weights[label] = {k: v.to(dev) for k, v in torch.load(path, map_location="cpu").items()}
    prompt = PromptBuilder.from_model(model)
    ds = SpatialVQADataset(args.ann, args.media_root, prompt, model.spec, geom_long_side=cfg.geometry.image_size,
                           geom_unit=getattr(model.geometry, "patch_size", 16), data_cfg=cfg.data, max_fail_rate=1.0)
    n = min(args.n, len(ds)) if args.n else len(ds)
    print(f"{n} held-out records · projectors {list(weights)} · {args.device}")

    def generate(item, prompt_ids, bundle):
        mm = item["mm_token_type_ids"][:, : prompt_ids.shape[1]].to(dev)
        kw = dict(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), mm_token_type_ids=mm,
                  pixel_values=item["pixel_values"].to(dev), image_grid_thw=item["image_grid_thw"].to(dev),
                  max_new_tokens=args.max_new_tokens, do_sample=False)
        with torch.no_grad():
            if bundle is None:
                out = model.base.generate(**kw)
            else:
                with model.injector.primed(list(bundle.embeds), model.visual_pos_mask(prompt_ids)):
                    out = model.base.generate(**kw)
        return model.tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)

    rows, prev = [], None      # prev: (llm_grids, geom_outs) of the previous record -> the shuffled donor
    t0 = time.time()
    for i in range(n):
        try:
            item = ds.build(i)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {i}: {str(exc)[:80]}")
            continue
        rec = normalize_record(ds.records[i])
        kind = ds.records[i].get("kind", "?")
        truth = rec.turns[0][1]
        ids, labels = item["input_ids"], item["labels"]
        first = int((labels[0] != -100).nonzero()[0])
        prompt_ids = ids[:, :first].to(dev)
        geom_outs = model.run_geometry(item["geom_frames"])
        donor = prev[1] if prev is not None and prev[0] == item["llm_grids"] and len(prev[1]) == len(geom_outs) else None
        row = {"id": rec.id, "kind": kind, "truth": truth, "pred": {}, "score": {}}
        for label, state in weights.items():
            model.load_state_dict(state, strict=False)
            conds = {"real": geom_outs, "shuffled": donor, "none": None}
            for cond, g in conds.items():
                if cond == "shuffled" and g is None:
                    continue
                bundle = None if g is None else model.build_geometry_embeds(g, item["llm_grids"], pool_temporal=False)
                pred = generate(item, prompt_ids, bundle)
                key = f"{label}/{cond}"
                row["pred"][key] = pred
                row["score"][key] = score(kind, pred, truth)
        rows.append(row)
        prev = (item["llm_grids"], geom_outs)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  [{i + 1}/{n}] {time.time() - t0:.0f}s", flush=True)

    keys = sorted({k for r in rows for k in r["score"]})
    table: dict[str, dict[str, float]] = {}
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        table[kind] = {k: sum(r["score"][k] for r in sub if k in r["score"]) / max(1, sum(k in r["score"] for r in sub))
                       for k in keys}
        table[kind]["n"] = len(sub)
    chance = {"camera_turn": 1 / 3, "closest_image": 1 / 4}
    print(f"\n{'kind':<22}{'n':>5}" + "".join(f"{k:>18}" for k in keys) + "   chance")
    for kind, v in table.items():
        print(f"{kind:<22}{v['n']:>5}" + "".join(f"{100 * v[k]:>18.1f}" for k in keys)
              + (f"   {100 * chance[kind]:.0f}" if kind in chance else ""))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"table": table, "rows": rows, "weights": args.weights,
                                              "config": args.config, "ann": args.ann}, indent=1, ensure_ascii=False))
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
