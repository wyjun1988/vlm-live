"""How far does the gated input fusion start from the base model? Held-out loss at step 0, before any training.

    PYTHONPATH=src python scripts/check_fusion_init.py --config configs/m2_zeroshot_08b.yaml \
        --ann data/sensenova_1k/sn1k.holdout.jsonl --media-root data/sensenova_1k/media -n 16

The fusion replaces every visual token (fusion/gated.py), so unlike the zero-init deepstack it is not the base model
at step 0; how far it starts decides how much of a run goes into undoing the start (and what a short run leaves of
the base model's general ability). Variants, all on the same randomly initialised merger (real CUT3R geometry):
    base          no geometry (the reference point)
    norm, b=0     LayerNorm on both streams, g = 0.5 everywhere (the design as the reference trains it)
    norm, b=-4    LayerNorm on both streams, g = 0.018
    raw,  b=-4    no norm, g = 0.018 (closest to the base model)
    raw,  b=0     no norm, g = 0.5
Per variant: mean loss on the supervised answer tokens, mean |loss - base|, and how often the greedy first answer token
differs from the base model's. Also the typical norm of a visual token at the LLM input against a LayerNorm output
(sqrt(d)) - the size of the renormalisation the norm variants apply.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from live3r.config import Live3RConfig  # noqa: E402
from live3r.data.datasets import SpatialVQADataset  # noqa: E402
from live3r.data.prompt import PromptBuilder  # noqa: E402
from live3r.fusion.gated import GatedTokenFusion  # noqa: E402
from live3r.model.live3r import Live3RModel  # noqa: E402

VARIANTS = (("norm, b=0", True, 0.0), ("norm, b=-4", True, -4.0), ("raw, b=-4", False, -4.0), ("raw, b=0", False, 0.0))
PIX = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--media-root", required=True)
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = Live3RConfig.from_yaml(a.config)
    cfg.geometry.tap_layers = (cfg.geometry.tap_layers[-1],)
    cfg.fusion.mode = "gated_input"
    cfg.geometry.expose_pose_token = False
    torch.manual_seed(a.seed)
    m = Live3RModel.from_pretrained(cfg).to(a.device).eval()
    prompt = PromptBuilder.from_model(m)
    ds = SpatialVQADataset(a.ann, a.media_root, prompt, m.spec, cfg.geometry.image_size,
                           getattr(m.geometry, "patch_size", 16), cfg.data, max_samples=a.n, seed=a.seed,
                           require_media=True)
    items = [ds[i] for i in range(len(ds))]
    print(f"{len(items)} samples · {cfg.base_model} · merger {sum(p.numel() for p in m.projectors.parameters()) / 1e6:.1f} M "
          f"params (random init, seed {a.seed}) · device {a.device}", flush=True)

    def run(it, geo):
        kw = {k: it[k].to(a.device) for k in PIX if k in it}
        out = m(input_ids=it["input_ids"].to(a.device), attention_mask=it["attention_mask"].to(a.device),
                labels=it["labels"].to(a.device), mm_token_type_ids=it["mm_token_type_ids"].to(a.device),
                geom_outs=geo, llm_grids=it.get("llm_grids") if geo is not None else None,
                pool_temporal=it.get("pool_temporal", False), **kw)
        return float(out.loss), int(out.logits[0, 0].argmax())       # logits start at the first supervised token

    # the visual tokens' own scale at the LLM input
    seen: list[float] = []
    hook = m._decoder_layers()[0].register_forward_pre_hook(
        lambda mod, args: seen.append(float(args[0][m.visual_pos_mask(cur["ids"])].float().norm(dim=-1).mean())))
    cur: dict = {}
    base, geos = [], []
    with torch.no_grad():
        for it in items:
            cur["ids"] = it["input_ids"].to(a.device)
            base.append(run(it, None))
            geos.append(m.run_geometry(it["geom_frames"]))
    hook.remove()
    d = m.d_llm
    print(f"visual token norm at the LLM input: mean {sum(seen) / len(seen):.2f} (a LayerNorm output: sqrt(d) = "
          f"{d ** 0.5:.1f})")
    rows = [("base", sum(x for x, _ in base) / len(base), 0.0, 0.0)]
    for name, norm, bias in VARIANTS:
        fusion = GatedTokenFusion(d, norm=norm, bias_init=bias).to(a.device)
        m.input_fusion = fusion
        m.injector.fusion = fusion
        got = []
        with torch.no_grad():
            for it, g in zip(items, geos):
                got.append(run(it, g))
        loss = sum(x for x, _ in got) / len(got)
        dl = sum(abs(x - y) for (x, _), (y, _) in zip(got, base)) / len(got)
        flip = sum(t != u for (_, t), (_, u) in zip(got, base)) / len(got)
        rows.append((name, loss, dl, flip))
        print(f"  {name:<11} loss {loss:.3f}  |Δ| {dl:.3f}  first token differs {100 * flip:.0f} %", flush=True)
    print("\n| start | loss | mean abs Δ vs base | first answer token differs |\n|---|---|---|---|")
    for name, loss, dl, flip in rows:
        print(f"| {name} | {loss:.3f} | {dl:.3f} | {100 * flip:.0f} % |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
