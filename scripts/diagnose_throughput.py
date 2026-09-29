"""Why does a training sample cost ~16 GPU-seconds on the server? A ten-minute check on ONE idle GPU.

    PYTHONPATH=src python scripts/diagnose_throughput.py --model /group-volume/wooyeol/models/Qwen3.5-4B \\
        | tee outputs/diagnose_throughput.txt

docs/SERVER_RESULTS_20260929.md §2.3 / §8.1: the weekend arms ran at 15.5-16.4 GPU-s per sample (plan ~0.42) and saw
~2 % of an epoch. The cost does not scale with the LLM (2B == 4B), is not CUT3R or the DataLoader, and tracks the
image path at 4-5 ms per vision patch row (~5x an Apple M2). Two suspects, both settled here:
  1. the bf16 Conv3d patch embed on a slow kernel -> fix F1 (the exact linear form, live3r.model.patch_embed);
  2. the Gated DeltaNet on transformers' pure-torch fallback (fla / causal-conv1d missing) -> fix F2 (pip install).
Sections: [env] versions and which kernel implementations transformers resolved; [patch] Conv3d vs its linear form,
synthetic, with a profiler table; [vision] (--model) the real vision tower on a 32-image keyframe batch, stock vs
linear, ms per patch row; [llm] (--model) a 4,096-token text prefill of the language model. Ends with a verdict.
Nothing here trains or writes outside stdout; it changes no default anywhere.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def timed(fn, dev: torch.device, reps: int = 3) -> float:
    """Median wall ms over `reps` calls after one warm-up."""
    fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[len(ts) // 2]


def resolved(func) -> str:
    """Which implementation a transformers `use_kernel_func_from_hub_with_fallback` wrapper resolved to."""
    for f in (func, getattr(func, "__wrapped__", None)):
        if f is None:
            continue
        cv = inspect.getclosurevars(f).nonlocals if f.__closure__ else {}
        impl = cv.get("implementation")
        if impl is not None:
            return f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__name__', '?')}"
    return f"{getattr(func, '__module__', '?')}.{getattr(func, '__name__', '?')} (unwrapped)"


def section_env(dev: torch.device) -> dict:
    print("[env]")
    print(f"  python {platform.python_version()} · torch {torch.__version__} · device {dev}")
    if dev.type == "cuda":
        print(f"  cuda {torch.version.cuda} · cudnn {torch.backends.cudnn.version()} enabled={torch.backends.cudnn.enabled}"
              f" · {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")
    import transformers

    peft = importlib.util.find_spec("peft")
    print(f"  transformers {transformers.__version__} · peft {importlib.import_module('peft').__version__ if peft else '-'}")
    pk = {m: bool(importlib.util.find_spec(m)) for m in ("fla", "causal_conv1d", "flash_attn", "kernels")}
    print("  packages: " + " ".join(f"{k}={'yes' if v else 'NO'}" for k, v in pk.items()))
    impls = {}
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                     "causal_conv1d_update"):
            f = getattr(q, name, None)
            if f is not None:
                impls[name] = resolved(f)
                print(f"  qwen3_5.{name} -> {impls[name]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not inspect qwen3_5 kernels: {exc!r})")
    fallback = any(v.startswith("transformers.") for k, v in impls.items() if "gated_delta" in k)
    print(f"  Gated DeltaNet path: {'TORCH FALLBACK' if fallback else 'fused (fla)'}")
    return {"packages": pk, "gdn_fallback": fallback}


def section_patch(dev: torch.device, rows: int, hidden: int) -> dict:
    print(f"\n[patch] Conv3d(3 -> {hidden}, 2x16x16) vs its linear form, bf16, {rows} patch rows")
    dt = torch.bfloat16 if dev.type != "cpu" else torch.float32
    conv = torch.nn.Conv3d(3, hidden, (2, 16, 16), (2, 16, 16)).to(dev, dt)
    x = torch.randn(rows, 3 * 2 * 16 * 16, device=dev, dtype=dt)
    with torch.no_grad():
        a = conv(x.view(-1, 3, 2, 16, 16)).view(-1, hidden)
        b = F.linear(x, conv.weight.view(hidden, -1), conv.bias)
        rel = ((a.float() - b.float()).norm() / a.float().norm()).item()
        t_conv = timed(lambda: conv(x.view(-1, 3, 2, 16, 16)), dev)
        t_lin = timed(lambda: F.linear(x, conv.weight.view(hidden, -1), conv.bias), dev)
    print(f"  conv3d {t_conv:9.2f} ms  ({1e3 * t_conv / rows:.3f} µs/row)")
    print(f"  linear {t_lin:9.2f} ms  ({1e3 * t_lin / rows:.3f} µs/row)   ratio {t_conv / max(t_lin, 1e-6):.1f}x · rel. diff {rel:.1e}")
    if dev.type == "cuda":
        with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                 torch.profiler.ProfilerActivity.CUDA]) as prof:
            conv(x.view(-1, 3, 2, 16, 16))
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=6))
    return {"conv_ms": t_conv, "linear_ms": t_lin}


def section_model(dev: torch.device, model_path: str, images: int, llm_tokens: int) -> dict:
    from transformers import Qwen3_5ForConditionalGeneration

    from live3r.model.patch_embed import linearize_patch_embed

    dt = torch.bfloat16 if dev.type != "cpu" else torch.float32
    t0 = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=dt).to(dev).eval()
    print(f"\n[model] {model_path} loaded in {time.perf_counter() - t0:.0f}s")
    vis = model.model.visual if hasattr(model.model, "visual") else model.visual
    vc = model.config.vision_config
    # a keyframe as the live path sends it: max_pixels 131072 -> 16 x 32 patches = 512 rows -> 128 LLM tokens
    gh, gw = 16, 32
    rows = images * gh * gw
    pix = torch.randn(rows, vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2, device=dev, dtype=dt)
    grid = torch.tensor([[1, gh, gw]] * images, device=dev)
    out = {}
    with torch.no_grad():
        ref = vis(pix, grid_thw=grid)
        ref = ref[0] if isinstance(ref, (tuple, list)) else getattr(ref, "last_hidden_state", ref)
        t_stock = timed(lambda: vis(pix, grid_thw=grid), dev)
        n = linearize_patch_embed(vis)
        new = vis(pix, grid_thw=grid)
        new = new[0] if isinstance(new, (tuple, list)) else getattr(new, "last_hidden_state", new)
        rel = ((ref.float() - new.float()).norm() / ref.float().norm()).item()
        t_lin = timed(lambda: vis(pix, grid_thw=grid), dev)
    print(f"[vision] {images} images x {gh * gw} patch rows = {rows} rows")
    print(f"  stock  {t_stock:9.1f} ms  = {t_stock / rows:.3f} ms/row   (vision tower only; M2 0.8B: 0.055 ms/row. The weekend"
          " estimate of 4-5 ms/row is the whole image path, tower + LLM prefill of the visual tokens)")
    print(f"  linear {t_lin:9.1f} ms  = {t_lin / rows:.3f} ms/row   ({n} patch embed patched; output rel. diff {rel:.1e} -"
          " ~3e-2 is ordinary bf16 error: on the M2 both forms sit 3.0e-2 from an fp32 reference, and are exact in fp32)")
    out.update(vision_stock_ms_row=t_stock / rows, vision_linear_ms_row=t_lin / rows)
    ids = torch.randint(1000, 20000, (1, llm_tokens), device=dev)
    with torch.no_grad():
        lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
        t_llm = timed(lambda: lm(input_ids=ids, use_cache=False), dev, reps=2)
    print(f"[llm] text prefill {llm_tokens} tokens: {t_llm:.0f} ms  ({1e3 * t_llm / llm_tokens:.1f} µs/token)")
    out["llm_prefill_ms"] = t_llm
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Qwen3.5 checkpoint; without it only [env] and [patch] run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--rows", type=int, default=16384, help="[patch] rows (32 keyframes x 512 = 16,384)")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--images", type=int, default=32)
    ap.add_argument("--llm-tokens", type=int, default=4096)
    args = ap.parse_args()
    dev = torch.device(args.device)
    env = section_env(dev)
    patch = section_patch(dev, args.rows, args.hidden)
    model = section_model(dev, args.model, args.images, args.llm_tokens) if args.model else {}

    print("\n[verdict]")
    ratio = patch["conv_ms"] / max(patch["linear_ms"], 1e-6)
    if ratio > 5:
        print(f"  F1 NEEDED: the Conv3d patch embed is {ratio:.0f}x slower than its linear form on this stack.")
    else:
        print(f"  Conv3d is fine here ({ratio:.1f}x of the linear form): F1 is not the fix; look at [vision] / a profile.")
    if model:
        gain = model["vision_stock_ms_row"] / max(model["vision_linear_ms_row"], 1e-9)
        print(f"  vision tower: {model['vision_stock_ms_row']:.3f} -> {model['vision_linear_ms_row']:.3f} ms/row with F1 ({gain:.1f}x)")
    print("  F2 NEEDED: Gated DeltaNet runs the torch fallback - install flash-linear-attention and causal-conv1d."
          if env["gdn_fallback"] else "  Gated DeltaNet uses the fused kernels: F2 is not needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
