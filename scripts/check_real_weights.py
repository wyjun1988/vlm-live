"""실가중치 검증 — 우리 입력 파이프라인이 공식 경로와 같은 결과를 내는지 **실제 Qwen3.5 로** 확인.

지금까지의 검증은 (a) 공식 전처리 함수와 텐서 비교, (b) 랜덤 초소형 모델로 형상 검증이었다.
둘 다 "실가중치 모델이 우리 입력을 제대로 보는가"는 직접 보여주지 못한다.
여기서는 같은 이미지·질문을

    [공식]  AutoProcessor → Qwen3_5ForConditionalGeneration
    [우리]  PromptBuilder + vision.prepare_image → 같은 모델
    [옛것]  예전 래스터 순서 평탄화 (2026-09-23 에 고친 버그) → 같은 모델
    [주입]  Live3RModel(zero-init 기하 주입 활성) → 같은 모델

에 넣고 마지막 토큰 로짓·생성 결과를 비교한다. 이미지에는 **글자**를 넣는다 — 패치가 뒤섞이면
가장 먼저 글자를 못 읽는다.

    PYTHONPATH=src python scripts/check_real_weights.py --model checkpoints/qwen3.5-0.8b --device mps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _font(size: int):
    from PIL import ImageFont

    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 오래된 Pillow
        return ImageFont.load_default()


def make_images():
    """해상도는 32 배수 + 공식 min/max 픽셀 범위 안 → 공식 프로세서가 리사이즈하지 않는다."""
    from PIL import Image, ImageDraw

    text = Image.new("RGB", (640, 256), "white")
    d = ImageDraw.Draw(text)
    d.text((40, 80), "SPATIAL 42", fill="black", font=_font(88))  # 640 폭 안에 다 들어가게

    shapes = Image.new("RGB", (512, 384), "white")
    d = ImageDraw.Draw(shapes)
    d.ellipse((40, 110, 200, 270), fill=(220, 30, 30))       # 왼쪽 빨간 원
    d.rectangle((320, 110, 480, 270), fill=(30, 60, 220))    # 오른쪽 파란 사각형
    return [
        (text, "What text is written in the image? Answer with the exact text."),
        (shapes, "What color is the circle, and is it on the left or the right? Answer briefly."),
    ]


def old_raster_pixel_values(img, spec):
    """2026-09-23 이전 구현 — 패치를 래스터 순서로 평탄화했다 (공식은 2×2 merge 블록 순서)."""
    from live3r.data.vision import normalize, to_float01

    x = normalize(to_float01(img))                      # [1,3,H,W]
    x = torch.cat([x, x], 0)                            # temporal_patch=2 복제
    tp, c, h, w = x.shape
    p = spec.patch
    gh, gw = h // p, w // p
    x = x.reshape(1, tp, c, gh, p, gw, p).permute(0, 3, 5, 2, 1, 4, 6)
    return x.reshape(gh * gw, c * tp * p * p)


@torch.no_grad()
def last_logits(model, **kw):
    return model(**kw).logits[0, -1].float().cpu()


@torch.no_grad()
def gen(model, tok, max_new_tokens=24, **kw):
    out = model.generate(**kw, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, kw["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/qwen3.5-0.8b")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
    from live3r.data.prompt import PromptBuilder
    from live3r.data.vision import VisionSpec, prepare_image, tokens_per_step
    from live3r.eval.consistent import prepare_inputs
    from live3r.model.live3r import Live3RModel

    dev = torch.device(args.device)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=DTYPES[args.dtype]).to(dev).eval()
    tok = proc.tokenizer
    print(f"모델 로드 {time.time() - t0:.1f}s · {sum(p.numel() for p in base.parameters()) / 1e9:.2f}B "
          f"· {args.device} {args.dtype}")

    # 공식 이미지 프로세서의 기본 범위와 같게 → 두 경로 모두 리사이즈 없음 → 픽셀이 비트 단위로 같다
    ip = proc.image_processor
    spec = VisionSpec(min_pixels=ip.size["shortest_edge"], max_pixels=ip.size["longest_edge"])
    cfg = base.config
    pb = PromptBuilder(tok, cfg.image_token_id, cfg.video_token_id,
                       cfg.vision_start_token_id, cfg.vision_end_token_id)

    fails = 0
    for img, q in make_images():
        print("\n" + "=" * 78)
        print(f"Q: {q}   (이미지 {img.size[0]}x{img.size[1]})")

        # [공식]
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        # enable_thinking=False 를 명시해야 한다 — 4B·9B 템플릿은 기본이 thinking on 이다
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        off = {k: v.to(dev) for k, v in proc(text=[text], images=[img], return_tensors="pt").items()}
        off["pixel_values"] = off["pixel_values"].to(base.dtype)
        lg_off = last_logits(base, **off)

        # [우리]
        pv, grid = prepare_image(img, spec)
        ids = pb.build_query("<image>" + q, [pb.image_segment(tokens_per_step(grid[0], spec))])
        ours = dict(input_ids=ids.to(dev), attention_mask=torch.ones_like(ids).to(dev),
                    mm_token_type_ids=((ids == cfg.image_token_id).long()).to(dev),
                    pixel_values=pv.to(dev, base.dtype), image_grid_thw=grid.to(dev))
        same_ids = ours["input_ids"][0].tolist() == off["input_ids"][0].tolist()
        px_diff = float((pv.to(dev, base.dtype) - off["pixel_values"]).abs().max())
        lg_ours = last_logits(base, **ours)

        # [옛것] 래스터 순서
        old = dict(ours, pixel_values=old_raster_pixel_values(img, spec).to(dev, base.dtype))
        lg_old = last_logits(base, **old)

        def cmp(a, b):
            top_a, top_b = a.topk(5).indices.tolist(), b.topk(5).indices.tolist()
            return float((a - b).abs().max()), top_a[0] == top_b[0], len(set(top_a) & set(top_b))

        d_ours, top1_ours, ov_ours = cmp(lg_off, lg_ours)
        d_old, top1_old, ov_old = cmp(lg_off, lg_old)
        print(f"  토큰열 동일: {same_ids} · 픽셀 max|Δ| {px_diff:.2e}")
        print(f"  [우리 vs 공식] 로짓 max|Δ| {d_ours:.3e} · top1 일치 {top1_ours} · top5 겹침 {ov_ours}/5")
        print(f"  [옛것 vs 공식] 로짓 max|Δ| {d_old:.3e} · top1 일치 {top1_old} · top5 겹침 {ov_old}/5")

        a_off = gen(base, tok, **off)
        a_ours = gen(base, tok, **ours)
        a_old = gen(base, tok, **old)
        print(f"  공식 답: {a_off!r}")
        print(f"  우리 답: {a_ours!r}")
        print(f"  옛것 답: {a_old!r}   ← 래스터 순서 버그가 실가중치에서 하던 일")
        ok = same_ids and px_diff < 1e-5 and a_ours == a_off and top1_ours
        fails += not ok
        print(f"  → {'일치' if ok else '불일치'}")

    # [주입] zero-init 기하 주입이 켜져 있어도 출력이 베이스와 같아야 한다
    print("\n" + "=" * 78)
    lcfg = Live3RConfig(base_model=args.model, dtype=args.dtype,
                        geometry=GeometryConfig(name="dummy", tap_layers=(6, 9, 12), hidden_size=768,
                                                image_size=512),
                        fusion=FusionConfig(inject_layers=(0, 1, 2), zero_init=True),
                        lora=LoRAConfig(enabled=False))
    lcfg.data.min_pixels, lcfg.data.max_pixels = spec.min_pixels, spec.max_pixels
    live = Live3RModel(lcfg, base, tok).to(dev).eval()
    img, q = make_images()[0]
    prep = prepare_inputs(live, pb, q, [img])
    kw = dict(input_ids=prep.input_ids.to(dev), attention_mask=torch.ones_like(prep.input_ids).to(dev),
              mm_token_type_ids=prep.mm_token_type_ids.to(dev),
              **{k: v.to(dev) for k, v in prep.pixel_kwargs.items()})
    kw["pixel_values"] = kw["pixel_values"].to(base.dtype)
    live.injector.hit_count = 0
    with torch.no_grad():
        with_geo = live(**kw, geometry=prep.geometry).logits[0, -1].float().cpu()
        no_geo = base(**kw).logits[0, -1].float().cpu()
    d = float((with_geo - no_geo).abs().max())
    print(f"[주입] zero-init 기하 주입 {live.injector.hit_count}회 · 베이스 대비 로짓 max|Δ| {d:.2e} "
          f"→ {'베이스 동작 보존' if d < 1e-4 and live.injector.hit_count == 3 else '보존 실패'}")
    fails += not (d < 1e-4 and live.injector.hit_count == 3)

    print(f"\n{'전부 일치' if not fails else f'{fails}건 불일치'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
