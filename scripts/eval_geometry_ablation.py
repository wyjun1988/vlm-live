"""기하 절제 평가 — "기하 토큰이 실제로 쓰이고 있나"를 판정한다.

손실이 내려가도 그게 기하 덕인지 LoRA·데이터 덕인지는 모른다. 같은 홀드아웃 샘플에
대해 손실을 세 번 잰다:

    real      이 샘플의 진짜 기하
    shuffled  **다른 이미지 세트**의 기하 — 분포는 같고 내용만 틀린, 가장 엄한 대조군
    none      기하 주입 없음

판정:
    real < shuffled          모델이 기하의 **내용**을 쓴다  ← 원하는 것
    real ≈ shuffled < none   기하를 "뭔가 들어왔다" 신호로만 쓴다 (내용 무시)
    real ≈ none              기하가 무시된다

    PYTHONPATH=src python scripts/eval_geometry_ablation.py \\
        --config configs/live3r_4b.yaml --base-model /path/Qwen3.5-4B \\
        --geometry-checkpoint checkpoints/cut3r_512_dpt_4_64.pth --cut3r-repo third_party/CUT3R \\
        --weights outputs/4b_s1/final.pt --stage align \\
        --ann data/sensenova.holdout.jsonl --media-root data/sensenova_media --n 300
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from live3r.config import Live3RConfig  # noqa: E402
from live3r.data.datasets import SpatialVQADataset  # noqa: E402
from live3r.data.prompt import PromptBuilder  # noqa: E402
from live3r.train.train import build_model  # noqa: E402


def mean_ci(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


@torch.no_grad()
def loss_of(model, item, device, geom_outs):
    pix = {k: item[k].to(device) for k in
           ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw") if k in item}
    out = model(
        input_ids=item["input_ids"].to(device),
        attention_mask=item["attention_mask"].to(device),
        labels=item["labels"].to(device),
        mm_token_type_ids=item["mm_token_type_ids"].to(device),
        geom_outs=geom_outs,
        llm_grids=item.get("llm_grids") if geom_outs is not None else None,
        pool_temporal=item.get("pool_temporal", False),
        **pix,
    )
    return float(out.loss)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None, help="학습 결과 (없으면 zero-init = 베이스 동작)")
    ap.add_argument("--stage", choices=["align", "sft"], default="align",
                    help="sft 면 LoRA 를 얹은 뒤 가중치를 로드한다 (학습 때와 같은 구조)")
    ap.add_argument("--ann", required=True, help="홀드아웃 어노테이션")
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--geometry-checkpoint", default=None)
    ap.add_argument("--cut3r-repo", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Live3RConfig.from_yaml(args.config)
    if args.base_model:
        cfg.base_model = args.base_model
    if args.geometry:
        cfg.geometry.name = args.geometry
    if args.geometry_checkpoint:
        cfg.geometry.checkpoint = args.geometry_checkpoint
    if args.cut3r_repo:
        cfg.geometry.options = dict(cfg.geometry.options, repo_path=args.cut3r_repo)

    ns = argparse.Namespace(stage=args.stage, init_from=args.weights, tokenizer=args.tokenizer)
    if args.stage == "align" and args.weights:
        model = build_model(argparse.Namespace(stage="align", init_from=None, tokenizer=args.tokenizer),
                            cfg, True)
        state = torch.load(args.weights, map_location="cpu")
        res = model.load_state_dict(state, strict=False)
        print(f"가중치 로드: {len(state) - len(res.unexpected_keys)}/{len(state)} 텐서")
    else:
        model = build_model(ns, cfg, True)
    model.to(device).eval()

    prompt = PromptBuilder.from_model(model)
    ds = SpatialVQADataset(args.ann, args.media_root, prompt, model.spec,
                           geom_long_side=cfg.geometry.image_size,
                           geom_unit=getattr(model.geometry, "patch_size", 16),
                           data_cfg=cfg.data, max_samples=args.n, max_fail_rate=1.0)

    # (cache_key, 격자모양, geom_outs). 대조군은 **같은 격자 모양**의 다른 이미지 세트를 우선한다.
    # 종횡비가 다른 샘플의 기하를 쓰면 "내용"이 아니라 "격자 모양" 차이까지 섞여 판정이 오염된다
    # (가짜 기하로 돌려보니 내용이 없는데도 '내용을 쓴다'가 나왔다 — 그 원인이 이것).
    recent: deque = deque(maxlen=32)
    rows = []
    n_same = 0
    for i in range(min(args.n, len(ds))):
        try:
            item = ds.build(i)  # 재시도 없이 — 같은 샘플을 결정적으로 본다
        except Exception as exc:  # 망가진 레코드는 건너뛴다
            print(f"  건너뜀 {i}: {str(exc)[:80]}")
            continue
        if not item.get("geom_frames"):
            continue
        geo = model.run_geometry(item["geom_frames"])
        shape = tuple(g.grid_hw for g in geo)
        cands = [(k, sh, g) for k, sh, g in reversed(recent) if k != item["cache_key"]]
        same = [g for _, sh, g in cands if sh == shape]
        other = same[0] if same else (cands[0][2] if cands else None)
        recent.append((item["cache_key"], shape, geo))
        if other is None:
            continue
        n_same += bool(same)
        shuffled = [other[s % len(other)] for s in range(len(geo))]
        rows.append({
            "id": item["id"],
            "real": loss_of(model, item, device, geo),
            "shuffled": loss_of(model, item, device, shuffled),
            "none": loss_of(model, item, device, None),
        })
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} 샘플", flush=True)

    if not rows:
        print("평가할 샘플이 없다")
        return 1
    res = {}
    for k in ("real", "shuffled", "none"):
        res[k] = mean_ci([r[k] for r in rows])
    d_shuf = mean_ci([r["shuffled"] - r["real"] for r in rows])
    d_none = mean_ci([r["none"] - r["real"] for r in rows])
    win_shuf = sum(r["real"] < r["shuffled"] for r in rows) / len(rows)

    print(f"\n홀드아웃 {len(rows)} 샘플 · 가중치 {args.weights or '(없음 = zero-init)'}")
    for k, (m, ci) in res.items():
        print(f"  loss[{k:8s}] = {m:.4f} ± {ci:.4f}")
    print(f"  shuffled − real = {d_shuf[0]:+.4f} ± {d_shuf[1]:.4f}   (양수여야 '내용'을 쓴다)")
    print(f"  none − real     = {d_none[0]:+.4f} ± {d_none[1]:.4f}")
    print(f"  real 이 shuffled 보다 나은 샘플 비율 = {win_shuf:.1%}")
    print(f"  대조군이 같은 격자 모양이었던 비율 = {n_same / len(rows):.1%} "
          "(낮으면 판정에 격자 모양 차이가 섞인다 — n 을 늘려라)")

    lo = d_shuf[0] - d_shuf[1]
    if lo > 0:
        verdict = "기하의 내용을 쓴다 (shuffled − real 의 95% 하한 > 0)"
    elif d_none[0] - d_none[1] > 0:
        verdict = "기하를 신호로만 쓴다 — 내용은 무시 (none 보다는 낫지만 shuffled 와 구분 안 됨)"
    else:
        verdict = "기하가 무시된다"
    if n_same / len(rows) < 0.5:
        verdict += " — ⚠️ 신뢰 낮음: 대조군 절반 이상이 격자 모양이 달랐다 (n 을 늘리거나 같은 소스끼리 평가)"
    print(f"\n판정: {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n": len(rows), "loss": res, "shuffled_minus_real": d_shuf, "none_minus_real": d_none,
             "win_rate_vs_shuffled": win_shuf, "same_shape_rate": n_same / len(rows),
             "verdict": verdict, "rows": rows}, indent=2,
            ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
