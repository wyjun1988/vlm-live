# Latency (gate 3) - what the numbers measure, what they say, what to measure next

Materials: `eval/outputs/weekend/REPORT.md` lines 155-168 (the only latency output uploaded; `latency_stream.json`,
`latency_stream.log`, `latency_live.log`, `STATUS` and `p0_*.log` are not in the upload or in the snapshot tarball).
Code read: `scripts/bench_latency.py`, `src/live3r/eval/latency.py`, `src/live3r/serve/session.py`,
`src/live3r/eval/streaming.py`, `src/live3r/eval/prefix_cache.py`, `src/live3r/fusion/deepstack.py`,
`scripts/server_weekend.sh` (lines 387-392), `scripts/weekend_report.py` (`section_latency`), transformers 5.16.1
`models/qwen3_5/modeling_qwen3_5.py`. `latency.py` / `session.py` / `prefix_cache.py` have not changed in a way that
touches timing since 9a46695 (09-24), before the server's script commit, so the local code is what produced the
numbers.

Computations (CPU only, no LLM weights loaded):

```
source ~/miniforge3/etc/profile.d/conda.sh && conda activate live3r
PYTHONPATH=src python outputs/server_0929/analysis/latency_analysis.py  > outputs/server_0929/analysis/latency_analysis.out
PYTHONPATH=src python outputs/server_0929/analysis/latency_opcount.py   > outputs/server_0929/analysis/latency_opcount.out
```

`latency_analysis.py` reads the Qwen3.5-4B safetensors headers (shapes only), recomputes the token/patch counts
with the repo's own preprocessing, does the arithmetic on the reported numbers, counts the aten ops of the Gated
DeltaNet torch fallback, and simulates the HalvingSelector. `latency_opcount.py` counts the ops of one forward of a
randomly initialised Qwen3.5 text model with the real 32-layer layout (24 linear + 8 full attention) at tiny widths.

---

## TL;DR

1. **Gate 3 formally passes** (question-only TTFT 164 ms < 1 s; drift 1.0 < 1.2), but it passes on the easy half
   of the design: the 164 ms assumes a visual prefix that took **61.3 s** to build, and the drift was measured on
   the incremental path, which is not the live design.
2. **The 61.3 s visual prefill is 390-780x slower than an H100 should be** for 3,840 tokens (first-principles
   78-157 ms of compute). It is not a CPU run (15.3 GB of CUDA-allocator peak memory, CUT3R at 25.6 ms/frame) and GPU
   sharing cannot produce it. The cost is **linear in vision patches at 4.0-5.1 ms per patch** in both benches,
   matching a known PyTorch 2.9.x / cuDNN<9.15 regression in which Qwen-VL's bf16 `Conv3d` patch embed falls back to
   `slow_conv_dilated3d` (external report: 3.5 ms/patch). Prime suspect, not proven; one command settles it.
3. **The text path is launch-bound**: fla / causal-conv1d are not in `requirements.txt`, so the Gated DeltaNet runs
   the torch fallback - 26,160 aten ops for a 24-token question forward and 4,030 per decode step. 164 ms / 26,160 =
   6.3 us per op, i.e. the question-only TTFT is almost all kernel-launch overhead.
4. **At 61 s per rebuild the "maintain the prefix in the background" design is infeasible**: at 1 fps the 32-keyframe
   HalvingSelector set changes every 1 s (first minute), 4 s (1-5 min), 8 s (5-15 min), 32 s (15-30 min), and only a
   median 7-9 of 32 leading keyframes survive a change. With the suspected fix (estimate: ~0.1-1 s per rebuild) it
   becomes feasible with rebuild coalescing.
5. The incremental path (1,160.9 ms per frame, 0.86 fps max) cannot run at 1 fps, and it also ignores the configured
   16,384 visual-token cap (it reached 28,672). It is a diagnostic, not the design.
6. `offline_ttft_ms` 570,745 ms is not an offline baseline: it is the incremental path replayed (1,114.7 ms per frame
   vs ingest p50 1,160.9).

---

## 1. What each number measures

Common to both benches: `Live3RModel.from_pretrained(configs/server_4b.yaml)` = base Qwen3.5-4B bf16, sdpa, CUT3R
(512 long side, `cut3r_512_dpt_4_64`), zero-init projectors. **No trained weights and no LoRA** (`bench_latency.py`
has no `--weights`; the orchestrator passes only `--config`). Frames are synthetic uniform-random uint8 at the
script defaults **256x448** (not VSI's native 480x640). Batch 1. Device auto-selected (`cuda` if visible), pinned by
`on_gpu 4` (CUDA_VISIBLE_DEVICES=4) in `role_sft`; no other job in the script uses GPU 4.

Token accounting (recomputed, section B of the .out): 256x448 -> 16x28 patch grid = **448 Conv3d rows -> 112 LLM
tokens** per image or per 2-frame video block; scene map 512x512 -> 1,024 rows -> 256 tokens; CUT3R input 288x512.

### "Streaming ingest" line = `bench_latency.py --frames 512 --offline` -> `benchmark_stream` (the **incremental** path)

| number | what it is (code) |
|---|---|
| frames 512, visual_tokens 28672 | 512 frames -> 256 video blocks (temporal_patch 2) x 112 tokens = 28,672 (matches). **Exceeds `stream.max_visual_tokens` 16,384 by 1.75x** - `LiveSession` stores the cap and never enforces it. |
| frame_ingest_ms p50 1160.9 / p95 1197.6 | For each of the 256 calls to `LiveSession.ingest` that complete a 2-frame block: wall time of that call / 2. Includes the 2nd frame's CUT3R ingest (stride 1: CUT3R on every frame), CPU preprocessing (bicubic resizes, patchify), `build_geometry_embeds`, and the `model.base` forward (vision tower on 448 rows + LLM prefill of 114 tokens onto the growing cache). Synchronised by the `.item()` after the forward. The 1st frame's call (CUT3R only) is excluded from the percentiles. Block p50 = 2,321.8 ms. |
| ttft_ms 164.0 | `LiveSession.ask`: one forward of 24 random token ids on the ~29k-token cache, argmax. **No CUDA sync before the timer is read** - an enqueue time if the forward has no internal sync. |
| tpot_ms 63.3 | 31 manual decode steps (max_new_tokens 32), also unsynchronised. |
| drift 1.0 | mean per-frame cost of the last 10 % of blocks / first 10 % (`summary()`); the report prints it with one decimal. |
| peak_mem_mb 15336 | `torch.cuda.max_memory_allocated()` - only returned when the device is `cuda` (0 on CPU). |
| geometry_state_mb 7.9 | CUT3R state bytes at the last block. |
| offline_ttft_ms 570745.5 | `_measure_offline_ttft`: a fresh `LiveSession` re-ingests all 512 frames **through the same incremental path** and asks once. 570,745.5 / 512 = 1,114.7 ms per frame (0.96x the ingest p50). Not a batched offline encode; the "3,480x faster than offline" it implies is meaningless. |
| meets_live_budget False | The harness's own rule (`latency.py`): ttft < 500 AND ingest p95 < 33 ms (30 fps) AND drift < 1.2. Fails only on ingest p95. **Not the pre-registered gate.** |

### "Live design" block = `bench_latency.py --mode deferred --scene-map --frames 512` -> `benchmark_deferred`

512 random frames at nominal 30 fps (17.1 s of video), `HalvingSelector(32)`, geom_stride 3, image mode, scene map
image in the prefix. Simulated: geometry ran on 276/512 frames, final set 32 frames (indices 0..504).

| number | what it is (code) |
|---|---|
| geometry frame 25.6 ms (39 fps) | median synced wall time of `push()` on frames where CUT3R ran (every 3rd + keyframes, first two dropped): geometry preprocessing + CUT3R with point decoding + `scene_map.add`. 39 fps = 1000/25.6. |
| other frames 0.01 ms | frames where only the selector ran. |
| TTFT 61,558 ms | 110 + 61,284 + 164 (adds up exactly). |
| 110 ms (select + geometry embed) | `sess.prefill()`: final selection, 32x `prepare_image` (CPU), `build_geometry_embeds`, map render + prepare, pixel tensors to GPU. |
| **61,284 ms (vision tower + LLM visual prefill)** | the `PrefixCache` constructor: tokenise the prefix, one `model.base` forward with `pixel_values` for 33 images (32 keyframes + map = **15,360 Conv3d rows**) and ~3,840 visual + ~100-200 text tokens, deepstack hooks on layers 0-2; synced. |
| question-only TTFT 164 ms | `pc.answer(question, max_new_tokens=1)`: re-tokenise the full prompt, `deepcopy` the cache, `generate()` over the question tokens -> 1 token; synced. Assumes the prefix already exists. |
| TPOT 51.1 ms | (16-token generate from the cache - t_q) / 15. |
| vision tokens 3840 | 32 x 112 + 256 (matches). |
| scene map 256 tokens / 30 ms | one extra CPU render of the map. |

**Dropped from REPORT.md**: `weekend_report.py` keeps only lines of `latency_live.log` containing "ms" or "TTFT".
That removed the header (label, **device**, frames, budget, stride) and the "real-time follow" line
(`live_fps_max` from the mean per-frame cost). Estimated from the p50s instead (not the dropped value): mean
~13.8 ms/frame -> ~72 fps.

## 2. Gate 3 against the pre-registered rule

Rule 5 of `docs/SERVER_WEEKEND.md`: *question-only TTFT < 1 s (live design, map in the prefix); ingest drift < 1.2*.

| criterion | value | verdict | caveat |
|---|---|---|---|
| question-only TTFT | 164 ms | PASS (6x margin) | best case: prefix freshly built; synthetic frames; no LoRA; measured with the map **image** in the prefix, not the routed-facts text configuration the VSI runs use (`--no-map-image`, OWLv2 object map, per-question routed facts - none of that is timed). |
| ingest drift | 1.0 | PASS | measured on the **incremental** path, not the deferred design; rounded to 0.1; over 17 s of video only. The deferred path's per-frame cost (CUT3R + scene-map accumulation up to 400k points) was never measured for drift. |

Which numbers are diagnostics: everything else - ingest p50/p95, `meets_live_budget`, `offline_ttft_ms`, the
61.6 s full TTFT, TPOT, peak memory.

Reading: the gate's literal criteria pass. The gate was written assuming a background-maintained prefix is cheap;
the same run shows it costs 61 s on this stack, so the pass does not yet show that the live design meets < 1 s.

## 3. Why is the visual prefill 61 s? (data, then interpretation)

**Data** (section C/D of `latency_analysis.out`):

- Expected on an H100 SXM (989 TFLOP/s dense bf16, 3.35 TB/s): prefix = 2 x 3.570 B (LLM body) x ~3,990 tokens
  = 28.5 TFLOP + 2 x 0.334 B (vision) x 15,360 rows = 10.2 TFLOP -> **78-157 ms** at 50-25 % MFU. Measured 61,284 ms =
  **391-782x slower**; implied 0.63 TFLOP/s (0.064 % of peak).
- Per-unit cost is the same in both benches: incremental 2,321.8 ms per block - 2 x 25.6 ms CUT3R = 2,270.6 ms for
  448 rows = **5.07 ms/row**; deferred 61,284 ms / 15,360 rows = **3.99 ms/row**.
- The patch embed itself is a 1536->1024 GEMM over 15,360 rows = 48 GFLOP (0.2 ms as a GEMM).
- External reference (axolotl PR #3897): Qwen-VL `Conv3d` patch embed without cuDNN, 11,716 patches -> 41,000 ms =
  **3.50 ms/patch** (cuDNN 9.20: 16 ms; as a linear: 1.4 ms). PyTorch issue #174051 / #166122 and a PyTorch-forums
  thread report the same `Conv3d` bf16 cliff on torch 2.9.x, fixed with cuDNN >= 9.15 or torch 2.10+.

**Not a CPU run**: `peak_mem_mb` 15,336 comes from `torch.cuda.max_memory_allocated()` (0 on CPU) and is consistent
with 9.08 GB of bf16 Qwen weights + CUT3R + a 28.7k-token cache; CUT3R at 25.6 ms per 288x512 frame is a GPU speed
(it would be seconds on CPU); the text-only path (164 ms, 51 ms) is GPU-plausible for eager mode.

**Not GPU sharing**: in `role_sft` the latency block is the only user of GPU 4 (the concurrent VSI / gate / ablation
jobs use GPUs 0-2); time-slicing with another process gives ~2x, not ~400x; and sharing cannot make a cost that is
linear in patches at a fixed 4-5 ms/row. Host-CPU contention from the concurrent jobs can inflate the launch-bound
numbers (TTFT, TPOT) somewhat, but not by this much.

**Interpretation**: the prime suspect is the torch 2.9.x + cuDNN<9.15 `Conv3d` regression
(`aten::slow_conv_dilated3d`). `requirements.txt` pins only `torch>=2.6`, so the server's torch/cuDNN versions are
unknown here (they are printed in `outputs/weekend/p0_<role>.log`). What would also fit: cuDNN disabled or
unavailable for another reason (no `cudnn` setting exists in `src/`, `scripts/`, `configs/`; `third_party/CUT3R` is
not available locally to check). What settles it in one line on the server:

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled)"
```

and a 30-second profile of `model.base.model.visual.patch_embed` on 15,360 rows (look for `slow_conv_dilated3d`).

**Consequence beyond latency (hypothesis)**: every VSI eval builds one such prefix per video (`eval_vsi_local.py`
uses `PrefixCache` with 32 keyframes), and training runs the same frozen vision tower. If the suspect is confirmed,
it predicts ~1 min of prefill per VSI video (~5 h per 288-video run) and a large share of the 0.3-0.5 samples/s
training speed (smoke A with dummy geometry on 1 GPU: 0.1 samples/s). Not testable from these materials; the VSI
logs' per-video timings and one profiled training step would settle it.

## 4. The text path: question-only TTFT and TPOT are launch-bound

Op counts (`latency_opcount.out`, tiny widths, real 32-layer layout, torch fallback as on a machine without fla):

| forward | aten ops | measured | implied cost |
|---|---|---|---|
| 24-token question on a cached prefix | 26,160 (23,016 of them in the chunked delta-rule fallback: 959 per layer) | 164 ms | 6.3 us/op |
| 1 decode step | 4,030 | 51.1 ms (generate) | 12.7 us/op incl. generate() overhead |
| ~3,990-token prefix (LLM only) | 116,902 | - | ~0.74 s at 6.3 us/op |

The chunked fallback has a fixed ~960-op cost per layer per call regardless of length (a 63-step Python loop inside
each chunk), so any multi-token forward pays ~23k launches. Decode is ~20x its weight-read floor (8.4 GB / 3.35 TB/s =
2.5 ms). TPOT 63.3 (incremental, 29k context) vs 51.1 (deferred, ~4k context): +12.2 ms that cannot be attributed
from these materials (different loop - manual vs `generate()` - and no sync in the incremental loop).

Interpretation: installing `flash-linear-attention` (and `causal-conv1d`) should remove most of these ops; the
question-only TTFT would then be well under 100 ms (estimate, not measured). The gate passes even without it.

## 5. Is the background prefix feasible at 1 fps with 32 keyframes?

The live design needs the prefix (32 keyframes + map + facts) ready when a question arrives. Two facts from the
code: Qwen3.5's 24 linear-attention layers carry a recurrent state, so a changed keyframe invalidates everything
after it; and `PrefixCache` rebuilds from scratch (no per-keyframe state checkpoints).

HalvingSelector(32) simulation (section F):

| stream time | set changes (1 fps input) | median gap | median unchanged leading keyframes |
|---|---|---|---|
| 0-1 min | 60 | 1.0 s | 9/32 |
| 1-5 min | 74 | 4.0 s | 7/32 |
| 5-15 min | 51 | 8.0 s | 7/32 |
| 15-30 min | 32 | 32.0 s | 7/32 |

(At 30 fps input: gaps 0.1 / 2.1 / 8.5 / 34.1 s, same 7-8/32 survival.) So each change forces recomputing ~75 % of
the prefix even with state checkpoints, and the whole prefix without them.

- **At the measured 61.3 s per rebuild**: never catches up (every gap < 61 s through 30 min); the prefix is always
  61-120 s stale (the newest keyframes and the map lag by 1-2 min), a question arriving mid-rebuild waits up to
  61 s, and the rebuild monopolises the GPU that also has to run CUT3R (25.6 % duty at 30 fps / stride 3, 2.6 % at
  1 fps). Infeasible.
- **After the suspected fix (estimate)**: vision tower at GEMM speed + the LLM prefix at ~0.74 s of launches with the
  current fallback -> ~1 s per rebuild; with fla, ~0.1-0.3 s. At 1 fps, coalescing rebuilds to at most one per 4 s costs ~25 % GPU
  duty at ~1 s per rebuild (~2.5-7.5 % with fla), and the prefix is at most ~5 s stale. Feasible.
- The incremental path is not an alternative: 1,160.9 ms per frame = 0.86 fps max now; even healthy it cannot evict
  frames, so it cannot hold a uniform 32-keyframe budget, and it already overran the 16,384-token cap in this bench.

## 6. What to measure next time (idle GPU; confirm with `nvidia-smi` that GPU 4 has no other process)

1. Environment line in the latency log: torch / CUDA / cuDNN versions, `cudnn.enabled`, whether `fla` and
   `causal_conv1d` import, GPU name, host load.
2. Component microbenchmarks with `torch.profiler` (top 10 CUDA ops): `patch_embed` alone on 15,360 rows (Conv3d vs
   the equivalent `F.linear`), the vision tower alone on 33 images, the LLM prefix alone (3,840 visual tokens), the
   question forward, one decode step. Each synced.
3. Rerun the deferred bench after the fix (cuDNN >= 9.15 / torch >= 2.10, or patch embed as a linear; plus fla) with:
   trained S2 weights and LoRA (merged and unmerged), real VSI video at native resolution instead of random 256x448,
   the configuration VSI actually uses (routed facts text, OWLv2 object map, `--no-map-image`) as well as the map
   image, and a long stream (>= 10 min) so drift is measured on the **deferred** path (CUT3R + scene map).
4. The live-design end-to-end number the gate really needs: a background rebuild policy running while frames
   stream, questions arriving at random times -> TTFT distribution (p50/p95), prefix staleness, geometry fps while a
   rebuild runs (same GPU).
5. Harness fixes to request (not made here - `scripts/` and `src/` are read-only for this analysis):
   `weekend_report.py` should keep the device header and `live_fps_max` lines; `latency_stream.json` should include
   the per-block geometry / vision / prefill split that `IngestStats` already records; `LiveSession.ask` and
   `ingest` sub-timings need a CUDA sync; `offline_ttft_ms` should be a single batched encode of all frames (or be
   dropped); `LiveSession` should enforce `max_visual_tokens`.

## 7. What these materials cannot settle

- The server's torch / cuDNN versions and whether `Conv3d` actually hit `slow_conv_dilated3d` -> `p0_<role>.log`,
  or the one-liner above.
- The device the deferred bench ran on (its header line was filtered out) -> `latency_live.log` line 1.
- When the latency jobs ran and what else was on the node -> `STATUS` timestamps and `master.sft.log` (the jobs are
  skipped when their outputs exist, so they may date from an earlier attempt).
- The per-block split of the 2.3 s incremental block -> not saved anywhere; needs a rerun.
- Whether the same slowdown affects VSI evaluation and training -> per-video timings in `vsi/*.log`, a profiled
  training step.
- The TPOT difference between the two paths (63.3 vs 51.1 ms).

## Sources (external)

- [PyTorch #174051: PyTorch 2.9.x Conv3d regression may significantly slow Qwen3-VL vision forward](https://github.com/pytorch/pytorch/issues/174051)
- [PyTorch #166122: 4x performance regression for 3D convs with AMP on torch 2.9.0](https://github.com/pytorch/pytorch/issues/166122)
- [PyTorch forums: Conv3D runs very slow in fp16 and bf16 (fixed with torch 2.9.1 + cuDNN 9.15)](https://discuss.pytorch.org/t/conv3d-runs-very-slow-in-fp16-and-bf16/223940)
- [axolotl PR #3897: run Qwen/GLM vision patch_embed as its GEMM equivalent (11,716 patches: 41,000 ms -> 1.4 ms)](https://github.com/axolotl-ai-cloud/axolotl/pull/3897)
