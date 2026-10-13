"""Why does a training sample cost ~16 GPU-seconds on the server? A ten-minute check on ONE idle GPU.

    PYTHONPATH=src python scripts/diagnose_throughput.py --model /group-volume/wooyeol/models/Qwen3.5-4B \\
        | tee outputs/diagnose_throughput.txt

docs/SERVER_RESULTS_20260929.md §2.3 / §8.1: the weekend arms ran at 15.5-16.4 GPU-s per sample (plan ~0.42) and saw
~2 % of an epoch. The cost does not scale with the LLM (2B == 4B), is not CUT3R or the DataLoader, and tracks the
image path at 4-5 ms per vision patch row (~5x an Apple M2). Two suspects, both settled here:
  1. the bf16 Conv3d patch embed on a slow kernel -> fix F1 (the exact linear form, live3r.model.patch_embed);
  2. the Gated DeltaNet on transformers' pure-torch path -> fix F2 (LIVE3R_GDN=fla; --gdn-check compares).
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


GDN_PROBE = r"""
import json, os, sys, time, torch
sys.path.insert(0, os.environ["LIVE3R_SRC"])
from live3r.model import kernels
backend = kernels.apply_gdn_backend()          # LIVE3R_GDN of this process, before the model is built
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForConditionalGeneration
path, lengths, out, dev = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3], sys.argv[4]
m = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
           "in_proj_a", "in_proj_b", "out_proj"]             # run 3's LoRA (configs/server_4b_v3*.yaml)
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets))
m.gradient_checkpointing_enable()
m.train()
def sync():
    torch.cuda.synchronize() if dev.startswith("cuda") else None
res = {"backend": backend}
for n in lengths:
    g = torch.Generator().manual_seed(n)
    ids = torch.randint(1000, 20000, (1, n), generator=g).to(dev)
    def step():
        m.zero_grad(set_to_none=True)
        o = m(input_ids=ids, labels=ids)
        o.loss.backward()
        return o.loss
    step(); sync()                                     # warm-up (Triton autotuning on the fused path)
    torch.cuda.reset_peak_memory_stats() if dev.startswith("cuda") else None
    t = time.perf_counter(); loss = step(); sync(); ms = (time.perf_counter() - t) * 1e3
    gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)).item()
    mem = torch.cuda.max_memory_allocated() / 2**30 if dev.startswith("cuda") else 0.0
    res[str(n)] = {"loss": float(loss), "grad_norm": gn, "ms": ms, "mem_gb": mem}
res["calls"] = dict(kernels.CALLS)
json.dump(res, open(out, "w"))
"""


def section_gdn(dev: torch.device, model_path: str, lengths: str) -> dict:
    """One LoRA training step (forward + backward, gradient checkpointing - as train.py) of the 4B with transformers'
    torch Gated DeltaNet and with fla's fused one (LIVE3R_GDN, live3r.model.kernels), in two processes: loss, LoRA
    gradient norm, ms and peak memory per length, and whether the fused rule really ran. transformers >= 5.16 no longer
    picks fla up by itself, so an installed fla says nothing about the path a run takes - this does."""
    import json
    import os
    import subprocess
    import tempfile

    print(f"\n[gdn] torch vs fla Gated DeltaNet, LoRA forward + backward at {lengths} tokens of {model_path}")
    try:
        have_ops = importlib.util.find_spec("fla.ops.gated_delta_rule") is not None
    except ModuleNotFoundError:
        have_ops = False
    if not have_ops:
        print("  fla's kernels are not installed (since fla 0.5 they are fla-core, which --no-deps leaves out):"
              " pip install --no-deps flash-linear-attention fla-core einops - nothing to compare")
        return {"gdn": None}
    res = {}
    src = str(Path(__file__).resolve().parents[1] / "src")
    with tempfile.TemporaryDirectory() as td:
        for name in ("torch", "fla"):
            out = f"{td}/{name}.json"
            r = subprocess.run([sys.executable, "-c", GDN_PROBE, model_path, lengths, out, str(dev)],
                               env={**os.environ, "LIVE3R_GDN": name, "LIVE3R_SRC": src}, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {name} path failed:\n{r.stderr[-2500:]}")
                return {"gdn": False}
            res[name] = json.load(open(out))
    a, b = res["fla"], res["torch"]
    print(f"  backends: {b['backend']} / {a['backend']} · fused rule calls in the fla process: {a['calls']}")
    ok = a["backend"] == "fla" and a["calls"]["chunk"] > 0
    speed = []
    for n in lengths.split(","):
        x, y = a[n], b[n]
        dloss, gratio = abs(x["loss"] - y["loss"]), x["grad_norm"] / max(y["grad_norm"], 1e-12)
        ok = ok and dloss < 0.05 and 0.9 < gratio < 1.1
        speed.append(y["ms"] / max(x["ms"], 1e-6))
        print(f"  {n:>6} tokens: loss {y['loss']:.4f} / {x['loss']:.4f} (|d| {dloss:.4f}) · grad norm ratio {gratio:.3f}"
              f" · step {y['ms']:.0f} -> {x['ms']:.0f} ms ({speed[-1]:.1f}x) · peak {y['mem_gb']:.1f} / {x['mem_gb']:.1f} GB")
    print(f"  GDN {'OK - train with LIVE3R_GDN=fla' if ok else 'MISMATCH or not taken - keep the torch path'}")
    return {"gdn": ok, "gdn_speedup": min(speed) if speed else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Qwen3.5 checkpoint; without it only [env] and [patch] run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--rows", type=int, default=16384, help="[patch] rows (32 keyframes x 512 = 16,384)")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--images", type=int, default=32)
    ap.add_argument("--llm-tokens", type=int, default=4096)
    ap.add_argument("--gdn-check", action="store_true",
                    help="with --model: a LoRA training step with the torch and the fla Gated DeltaNet (two processes)")
    ap.add_argument("--gdn-lengths", default="4096,12288", help="token lengths of the --gdn-check steps")
    ap.add_argument("--fla-check", dest="gdn_check", action="store_true", help=argparse.SUPPRESS)   # run 2's name
    args = ap.parse_args()
    dev = torch.device(args.device)
    env = section_env(dev)
    patch = section_patch(dev, args.rows, args.hidden)
    model = section_model(dev, args.model, args.images, args.llm_tokens) if args.model else {}
    gdn = section_gdn(dev, args.model, args.gdn_lengths) if (args.model and args.gdn_check) else {}

    print("\n[verdict]")
    ratio = patch["conv_ms"] / max(patch["linear_ms"], 1e-6)
    if ratio > 5:
        print(f"  F1 NEEDED: the Conv3d patch embed is {ratio:.0f}x slower than its linear form on this stack.")
    else:
        print(f"  Conv3d is fine here ({ratio:.1f}x of the linear form): F1 is not the fix; look at [vision] / a profile.")
    if model:
        gain = model["vision_stock_ms_row"] / max(model["vision_linear_ms_row"], 1e-9)
        print(f"  vision tower: {model['vision_stock_ms_row']:.3f} -> {model['vision_linear_ms_row']:.3f} ms/row with F1 ({gain:.1f}x)")
    print("  F2 NEEDED: Gated DeltaNet runs transformers' torch path - with transformers >= 5.16 an installed fla is not"
          " used by itself: train with LIVE3R_GDN=fla once --gdn-check says OK."
          if env["gdn_fallback"] else "  Gated DeltaNet uses fused kernels: F2 is not needed.")
    if gdn.get("gdn") is not None:
        print(f"  gdn check: {'OK - LIVE3R_GDN=fla for training' if gdn['gdn'] else 'MISMATCH - keep the torch path'}"
              + (f" ({gdn['gdn_speedup']:.1f}x faster step, the slowest length)" if gdn.get("gdn_speedup") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
