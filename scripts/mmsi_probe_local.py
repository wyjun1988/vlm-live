"""MMSI-Bench probes without lmms-eval's harness - on the M2 (zero-shot on the base) or on a trained checkpoint.

    PYTHONPATH=src python scripts/mmsi_probe_local.py --config configs/m2_zeroshot_4b_anchor_v2.yaml \\
        --types "Cam.-Cam.,Motion (Cam.)" --variants plain,camera,labels --out outputs/mmsi_local/cam_4b.json

The same inputs as the server's path (`live3r.eval.consistent.prepare_inputs`: images first, then the question with
the task's post-prompt; greedy), scored with the task's own extractor (backticks, braces, a bare letter), per
question type. Variants, each a change of the prompt only (docs/MMSI_BENCH_ANALYSIS_20261015.md §3B-C):
    plain    the question as the task gives it
    labels   "Image k: " before every image block
    camera   the camera motion between the two images from the geometry encoder's poses, attached to the camera
             types (Cam-Cam, Motion (Cam.)); other questions keep the plain prompt
    hint     the format line in front
    ls768 / ls1024   the VLM input at that long side (shrink-only)
--format-line puts the format line in front of EVERY variant: the base model otherwise reasons in prose and never
reaches a letter (the smoke of 10-15: 2 of 3 answers unparsable), as on VSI before the instruction; the trained
model answers a letter either way.
A question whose prompt is identical under two variants is answered once (greedy). Paired bootstrap over questions
between variants. The dataset comes from the HF hub (RunsenXu/MMSI-Bench, cached by `datasets`).

The camera facts can be computed apart from the LLM pass (another encoder, or the M2's memory - VGGT-1B next to the
4B swaps): `--facts-only --geom-config <yaml>` writes {question id: fact} from that config's geometry stream alone;
`--facts-from <json>` makes the `camera` variant read them instead of running the model's own encoder.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

VARIANTS = ("plain", "labels", "camera", "hint", "ls768", "ls1024")


def norm_type(t: str) -> str:
    return str(t).replace("–", "-").replace("—", "-")


def short_type(t: str) -> str:
    t = norm_type(t)
    m = re.search(r"\((.+?)\)", t)
    return m.group(1) if m else t


def load_docs(types: list[str], limit: int):
    from datasets import load_dataset

    ds = load_dataset("RunsenXu/MMSI-Bench", split="test")
    docs = []
    for row in ds:
        if types and not any(t in norm_type(row["question_type"]) for t in types):
            continue
        docs.append(row)
        if limit and len(docs) >= limit:
            break
    return docs


def build_model(config: str, weights: str | None, device: str, dtype: str):
    from live3r.config import Live3RConfig
    from live3r.model.live3r import Live3RModel

    cfg = Live3RConfig.from_yaml(config)
    if dtype:
        cfg = replace(cfg, dtype=dtype)
    live = Live3RModel.from_pretrained(cfg)
    if weights:
        from live3r.eval.lmms_live3r import _load_trained
        from live3r.model.lora import apply_lora

        if cfg.lora.enabled:
            live = apply_lora(live, cfg.lora)
        _load_trained(live, weights)
    return live.to(device).eval()


def facts_only(a, docs, is_camera_question) -> int:
    """The camera facts from --geom-config's geometry stream alone, {question id: sentence} to --out."""
    from live3r.config import Live3RConfig
    from live3r.eval.lmms_live3r import wants_axes_form
    from live3r.geometry.registry import build_geometry_stream
    from live3r.serve.camera_facts import camera_motion_from_images, camera_motion_text

    cfg = Live3RConfig.from_yaml(a.geom_config or a.config)
    geom = build_geometry_stream(cfg.geometry).to(a.device).eval()
    unit = getattr(geom, "patch_size", 16)
    facts, failed, t0 = {}, 0, time.time()
    for n, doc in enumerate(docs, 1):
        imgs = list(doc["images"])
        if not (is_camera_question(doc) and len(imgs) >= 2):
            continue
        try:
            rel = camera_motion_from_images(geom, imgs, cfg.geometry.image_size, unit, a.device)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{doc['id']}] failed: {exc!r}", flush=True)
            rel = None
        if rel is None:
            failed += 1
            continue
        facts[str(doc["id"])] = camera_motion_text(rel, axes_form=wants_axes_form(doc["question"]))
        if a.device == "mps" and n % 10 == 0:
            torch.mps.empty_cache()
        if n % 20 == 0 or n == len(docs):
            print(f"  [{n}/{len(docs)}] {time.time() - t0:.0f}s · {len(facts)} facts · {failed} failed", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"geom_config": a.geom_config or a.config, "geometry": cfg.geometry.name,
                               "n": len(facts), "failed": failed, "facts": facts}, indent=1, ensure_ascii=False))
    print(f"saved {out}: {len(facts)} facts, {failed} failed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--types", default="", help="comma list of substrings of question_type (dash-insensitive)")
    ap.add_argument("--variants", default="plain,camera,labels")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--format-line", action="store_true", help="the format line in front of every variant (the base)")
    ap.add_argument("--facts-only", action="store_true",
                    help="compute the camera facts only (no LLM) with --geom-config's geometry stream, to --out")
    ap.add_argument("--geom-config", default=None, help="the yaml whose geometry section builds the facts' encoder")
    ap.add_argument("--facts-from", default=None, help="a --facts-only output: the camera variant reads these facts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from lmms_eval.tasks.mmsi_bench.utils import MMSI_DIRECT_POST_PROMPT, extract_single_choice_with_word_boundary

    from live3r.data.prompt import PromptBuilder
    from live3r.eval.consistent import generate, prepare_inputs
    from live3r.eval.lmms_live3r import FORMAT_LINE, camera_fact_text, is_camera_question

    variants = [v for v in a.variants.split(",") if v]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; known: {VARIANTS}")
    types = [norm_type(t) for t in a.types.split(",") if t]
    docs = load_docs(types, a.limit)
    print(f"{len(docs)} questions" + (f" of types {types}" if types else "") + f" · variants {variants}", flush=True)
    if a.facts_only:
        return facts_only(a, docs, is_camera_question)
    facts_from = json.load(open(a.facts_from))["facts"] if a.facts_from else None
    live = build_model(a.config, a.weights, a.device, a.dtype)
    pb = PromptBuilder.from_model(live)
    tok = live.tokenizer
    base_spec = live.spec

    records = []
    cache: dict[tuple, str] = {}
    t0 = time.time()
    for n, doc in enumerate(docs, 1):
        imgs = list(doc["images"])
        text0 = doc["question"].strip() + MMSI_DIRECT_POST_PROMPT
        rec = {"id": doc["id"], "type": norm_type(doc["question_type"]), "gt": doc["answer"], "n_images": len(imgs),
               "variants": {}}
        for v in variants:
            text, labels, long_side, fact = (FORMAT_LINE + text0 if a.format_line else text0), False, 0, ""
            if v == "labels":
                labels = True
            elif v == "camera":
                if is_camera_question(doc) and len(imgs) >= 2:
                    fact = (facts_from.get(str(doc["id"]), "") if facts_from is not None
                            else camera_fact_text(live, imgs, doc["question"]))
                    text = fact + text
            elif v == "hint" and not a.format_line:
                text = FORMAT_LINE + text
            elif v.startswith("ls"):
                long_side = int(v[2:])
            key = (doc["id"], text, labels, long_side)
            if key not in cache:
                live.spec = replace(base_spec, long_side=long_side) if long_side else base_spec
                try:
                    with torch.no_grad():
                        prepared = prepare_inputs(live, pb, text, imgs, None, use_geometry=False, image_labels=labels)
                        ans = generate(live, prepared, tok, max_new_tokens=a.max_new_tokens, do_sample=False)
                    if ans.strip().startswith("!!!") and a.device == "mps":   # the transient MPS failure: once more
                        torch.mps.empty_cache()
                        with torch.no_grad():
                            prepared = prepare_inputs(live, pb, text, imgs, None, use_geometry=False, image_labels=labels)
                            ans = generate(live, prepared, tok, max_new_tokens=a.max_new_tokens, do_sample=False)
                finally:
                    live.spec = base_spec
                cache[key] = ans.strip()
            ans = cache[key]
            score = extract_single_choice_with_word_boundary(ans, doc["answer"])
            rec["variants"][v] = {"pred": ans[:120], "score": 0.0 if score is None else float(score),
                                  "parsed": score is not None, "fact": fact.strip() or None}
        records.append(rec)
        if a.device == "mps" and n % 10 == 0:
            torch.mps.empty_cache()
        if n % 20 == 0 or n == len(docs):
            done = {v: np.mean([r["variants"][v]["score"] for r in records]) for v in variants}
            print(f"  [{n}/{len(docs)}] {time.time() - t0:.0f}s · " + " · ".join(f"{v} {100 * x:.1f}" for v, x in done.items()),
                  flush=True)

    # ---- summary
    types_seen = sorted({r["type"] for r in records})
    lines = [f"MMSI-Bench probe · {len(records)} questions · {a.config}" + (f" + {a.weights}" if a.weights else ""), ""]
    head = f"{'type':38s}" + "".join(f"{v:>10s}" for v in variants) + "     n"
    lines += [head, "-" * len(head)]
    for t in types_seen + ["ALL"]:
        rows = [r for r in records if t == "ALL" or r["type"] == t]
        cells = "".join(f"{100 * np.mean([r['variants'][v]['score'] for r in rows]):10.1f}" for v in variants)
        lines.append(f"{(short_type(t) if t != 'ALL' else 'ALL'):38s}{cells}{len(rows):6d}")
    unparsed = {v: sum(not r["variants"][v]["parsed"] for r in records) for v in variants}
    lines += ["", "unparsed answers: " + ", ".join(f"{v} {k}" for v, k in unparsed.items())]
    if len(variants) > 1:
        rng = np.random.default_rng(0)
        lines += ["", "paired vs plain (bootstrap over questions, 95 %):"]
        base_v = variants[0]
        for v in variants[1:]:
            d = np.array([r["variants"][v]["score"] - r["variants"][base_v]["score"] for r in records])
            if v == "camera":
                sub = np.array([r["variants"][v]["fact"] is not None for r in records])
                d_sub = d[sub] if sub.any() else d
                bs = [d_sub[rng.integers(0, len(d_sub), len(d_sub))].mean() for _ in range(2000)]
                lines.append(f"  {v:8s} on the {int(sub.sum())} questions with a fact: {100 * d_sub.mean():+.1f} "
                             f"[{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lines.append(f"  {v:8s} all: {100 * d.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, "
                         f"{100 * np.percentile(bs, 97.5):+.1f}] · answers changed {int((d != 0).sum())}")
    print("\n" + "\n".join(lines))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": a.config, "weights": a.weights, "types": types, "variants": variants,
                               "format_line": a.format_line, "max_new_tokens": a.max_new_tokens,
                               "facts_from": a.facts_from,
                               "device": a.device, "summary": lines, "records": records}, indent=1, ensure_ascii=False))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
