# Training throughput and the data budget: why every arm saw ~2% of an epoch

Materials: `eval/outputs/weekend/REPORT.md` (training table, budget line, latency section) and the code
(`scripts/server_weekend.sh`, `scripts/weekend_report.py`, `src/live3r/train/train.py`, `src/live3r/data/*.py`,
`src/live3r/geometry/cut3r.py`, `src/live3r/model/live3r.py`, `src/live3r/fusion/deepstack.py`, `configs/server_4b.yaml`,
transformers 5.16.1 `models/qwen3_5/modeling_qwen3_5.py`). The arm logs (`outputs/weekend/<arm>.log`, which carry the
per-phase `data / geom / fb ms/샘플` split), `STATUS`, `budget*.env` and `p0_*.log` were **not** uploaded. Local
references: `data/sensenova_1k` (905 train records), `logs/s2_4b_real.log` and `logs/latency_map_4b.log` (Mac M2 runs).

All computations are CPU-only, with no LLM weights loaded. Every script is in this folder:

```
source ~/miniforge3/etc/profile.d/conda.sh && conda activate live3r && cd outputs/server_0929/analysis
python tp_budget_check.py      > tp_budget_check.txt     # plan() arithmetic, observed rates, targets
python tp_data_cost.py         > tp_data_cost.txt        # real SpatialVQADataset.build() on 905 local records, 1 thread
python tp_opcount.py sft|align                          # ATen op count per training sample -> tp_opcount_{sft,align}.jsonl
python tp_opcount_ckpt.py                               # with/without grad checkpointing -> tp_opcount_ckpt.json
python tp_opcount_infer.py                              # streaming step + 4k prefill -> tp_opcount_infer.json
python tp_opcount_ttft.py                               # question-only TTFT + decode step -> tp_opcount_ttft.json
python tp_rank_gating.py       > tp_rank_gating.txt      # DDP slowest-rank simulation
python tp_per_image.py         > tp_per_image.txt        # server timings per vision patch / per image
python tp_summary_numbers.py   > tp_summary_numbers.txt  # derived numbers quoted below
python tp_conv3d_linear_check.py                        # Conv3d patch embed == Linear (numerical check)
```

The op counts come from a Qwen3.5 model built with the 4B's exact layer structure (32 decoder layers with 24 Gated
DeltaNet and 8 attention, 32/16 linear heads, vision depth 24) at tiny widths. It is wrapped exactly as train.py wraps
the real model: deepstack on layers 0-2, LoRA r32 for S2, non-reentrant gradient checkpointing, and
`keep_primed -> backward -> clear`. It runs on real Sensenova samples. A `TorchDispatchMode` counts every ATen call in
the forward, recompute and backward passes. "Launching" ops exclude pure view and metadata ops. "Fused" swaps the
torch gated-delta-rule fallback for a 4-op differentiable stand-in, which approximates what fla's fused kernels cost
in launches.

---

## TL;DR

1. **The budget arithmetic reproduces exactly.** `plan()` computes `steps = min(epoch, hours·3600·smoke_sps·frac) // 128`.
   With a smoke speed of 0.30 samples/s (8 GPUs, grad-accum 2) this gives S1 = 106 steps (13,568 samples, 1.63% of
   830,135) and S2 = 136 steps (17,408 samples, 2.10%). The 2B gets 57/71 steps (0.88% / 1.09%). The smoke speed
   implies **26.7 GPU-seconds per sample** (29.6 after the 0.9 margin). At that speed one epoch would take ~850 h,
   against caps of 14-18 h. The plan had assumed ~19 samples/s, i.e. 0.42 GPU-s per sample.
2. **The arms ran 1.66× faster than the smoke** (0.49-0.51 vs 0.306 samples/s, i.e. 15.8-16.4 GPU-s per sample). They
   therefore finished in about **53-54% of their hour caps** (S1: 7.4 h of 14 h; S2: 9.7 h of 18 h). The smoke ran at
   grad-accum 2 and the arms at 16. The ratio (0.60) is exactly what slowest-rank gating predicts when per-sample cost
   is proportional to sample length.
3. **The dominant cost scales with the number of images; LLM FLOPs barely matter.** On this server each image costs
   about **2.3 s (4-5 ms per vision patch row)**. Three independent server measurements agree:
   - training: 5.4 ms/patch
   - streaming bench step: 5.2 ms/patch
   - deferred 32-keyframe prefill: 4.0 ms/patch

   That is about 1000× the vision tower's FLOP cost. Text-only paths on the same server are normal: question-only
   TTFT is 164 ms and TPOT is 51 ms, the same as a Mac. Four observations rule out the other candidates:
   - The 2B and 4B have **the same own cost per sample (10.5 vs 10.7 s)**. They share the vision tower, CUT3R and the
     data path; their LLMs differ.
   - `s2_sft`, which never runs CUT3R, is no faster than `s2_real`.
   - S1 (no LoRA) runs at the same speed as S2.
   - The latency benches have no DataLoader.

   The mechanism is not proven. The prime suspect is the bf16 `Conv3d` patch embed dropping to a non-cuDNN kernel.
   One version print and one profile settle it, and an exact fix exists (the Conv3d as a Linear, checked numerically
   here).
4. **The second term: the 24 Gated DeltaNet layers almost certainly run transformers' pure-PyTorch fallback.** Neither
   fla nor causal-conv1d is in `requirements.txt`, and transformers falls back silently. The fallback costs 78k
   launching ops per mean sample instead of 11.8k (6.6×). The server's TTFT and TPOT fit the fallback at 13-23 µs per
   op; they do not fit fused kernels. The fallback costs about 1.0 s per sample today, and it becomes the dominant term
   once item 3 is fixed.
5. **Rank gating** gives 0.67 efficiency at 8 ranks × 16 micro-steps, so about 33% of rank time is spent waiting. At 32
   ranks × 4 (one 4-node job) it would be 0.37. Length-bucketed sampling raises this to 0.92 and 0.78.
6. **CUT3R and data loading are not bottlenecks today.** CUT3R costs at most ~0.15 s per sample: it runs in fp32 (803M
   parameters), one frame at a time. Data loading is 148 ms of CPU per sample on 1 thread, with 6 workers per rank.
7. **Target: one epoch in 24 h needs 9.61 samples/s.** That means ≤0.83 GPU-s per sample on one node (**19× faster**
   than now), or ≤3.33 GPU-s per sample on 4 nodes as one job (4.8×). Estimates:
   - fixing item 3 alone gives ~1.4 GPU-s per sample (~40 h per epoch per node);
   - adding fla gives ~0.45-0.6 (~13-18 h per epoch per node), which meets the target;
   - adding bucketing, bf16 CUT3R and no checkpointing gives ~0.25-0.3 (~7-9 h).

---

## 1. How the smoke speed became the step budget

`server_weekend.sh`, lines 82-83 and 202-227:

- `sps_of` averages the last 6 `N samp/s` values of `p1_<which>_<role>_smoke_c.log` (S1) and `smoke_d.log` (S2).
- train.py's `samp/s` is global: `all_mean(samples_since/dt) × world`.
- `fit(hours, sps)`:
  - `s = min(N, hours·3600·sps·frac)`
  - `steps = max(50, s // 128)`
  - `epochs = ceil(steps·128/N)`
- `frac` is 0.9 for a 4B arm alone on a node and 0.425 for each 2B arm on half a node.
- The real and control arms read one shared `budget.env`.

Reproduced in `tp_budget_check.txt`:

| plan | hours × 3600 × 0.30 × frac | steps × 128 | share of 830,135 | unrounded smoke speed consistent with the step count |
|---|---|---|---|---|
| 4B S1 (14 h, 0.9) | 13,608 | 106 → 13,568 | 1.63% | [0.2991, 0.3019) |
| 4B S2 / sft / joint (18 h, 0.9) | 17,496 | 136 → 17,408 | 2.10% | [0.2985, 0.3007) |
| 2B S1 (16 h, 0.425) | 7,344 | 57 → 7,296 | 0.88% | [0.2980, 0.3033) |
| 2B S2 (20 h, 0.425) | 9,180 | 71 → 9,088 | 1.09% | [0.2970, 0.3012) |

The smoke implies 8 GPUs / 0.30 = **26.7 GPU-s per sample** (29.6 including `frac`). The plan's own expectation
(`docs/SERVER_WEEKEND.md`: "~19 samples/s → an epoch in ~12 h / ~15 h") corresponds to 0.42-0.52 GPU-s per sample.

**What the arms actually did.** REPORT's `elapsed` spans the first to the last logged step, and `samples/s` is the mean
of the log lines after the first.

| run | samples between first and last log | elapsed | samples/s | GPU-s per sample |
|---|---|---|---|---|
| 4B smoke_c / smoke_d (8 GPUs, accum 2) | 880 | 0.8 h | 0.306 | 26.2 |
| s1_real / s1_control (accum 16) | 10,240 | 5.6 / 5.7 h | 0.508 / 0.499 | 15.8 / 16.0 |
| s2_real / s2_control / s2_sft | 12,800 | 7.1 / 7.2 / 7.3 h | 0.501 / 0.494 / 0.487 | 16.0 / 16.2 / 16.4 |
| s1_real_2b / s2_real_2b (4 GPUs, accum 32) | 2,560 / 5,120 | 2.3 / 4.7 h | 0.309 / 0.303 | 12.9 / 13.2 |
| 4B smoke_a (1 GPU, dummy geometry, no DDP) | 25 | 0.1 h | ~0.1 (report) | ~7-20 |

Because production ran 1.66× faster than the smoke, the steps that were sized to fill 14 h and 18 h took 7.4 h and
9.7 h (53% and 54%). The smoke was taken at grad-accum 2 (`p1`: `--grad-accum 2`), but the arms run at 16 (4B) and
32 (2B). Section 2.3 shows this difference is rank gating.

## 2. Where the ~16 GPU-seconds per sample go

### 2.1 Not GPU FLOPs

- **FLOP floor per mean sample** (655 tokens and 4.2 images, from `tp_data_cost.txt`):
  - LLM with checkpointing and LoRA: 6 × 4e9 × 655 ≈ 15.7 TFLOP;
  - vision tower: ~0.61 GFLOP per patch row × 1,982 rows ≈ 1.2 TFLOP;
  - CUT3R: ≤25.6 ms per frame measured (REPORT), so ~0.11-0.15 s.

  Together that is roughly **0.06-0.15 s per sample per GPU**. The observed figure is 16.
- **The Mac comparison.** The same 4B S2 on the same data distribution ran at ~8 s per sample on a Mac MPS GPU
  (`logs/s2_4b_real.log`: data 115-153, geom 1346-1892, fb 6378-10375 ms per sample). The H100 node is twice as slow
  per sample as a laptop GPU, so compute cannot be the limit.

### 2.2 The dominant cost is the same for the 2B and the 4B, and it is not CUT3R, LoRA or DDP

- **2B vs 4B.** Own cost per sample = GPU-s per sample × gating efficiency (section 2.3). With length-proportional
  cost: 4B 16.0 × 0.671 = **10.7 s**, 2B 13.05 × 0.808 = **10.5 s**. With 30% fixed cost: 11.9 vs 11.2 s.
  - The two LLMs differ a lot: 32 vs 24 layers and hidden size 2560 vs 2048 (`checkpoints/qwen3.5-*/config.json`).
  - The **vision tower is identical** (depth 24, width 1024), and so are CUT3R and the data path.
- **Not CUT3R.** `s2_sft` (`--no-geometry`, so the encoder never runs) is at 0.487 samples/s, against 0.501 for
  `s2_real` and 0.494 for `s2_control`.
- **Not LoRA.** S1 (projector only) runs at 0.508 / 0.499 and S2 (with LoRA) at 0.501 / 0.494.
- **Not DDP.** The 1-GPU smoke_a, with dummy geometry and no DDP, still runs at ~0.1 samples/s.

### 2.3 The cost scales with sample length: rank gating (`tp_rank_gating.txt`)

DDP syncs once per optimizer step (`no_sync` on the other micro-steps). Step time is therefore the maximum over ranks
of each rank's sum of `accum` sample costs. The simulation draws per-sample sizes from the 905 local records.

The local length distribution is bimodal:
- 80% of records have ≤2 images and 316 tokens on average;
- 15.5% have ≥16 images, 2,357 tokens on average, and carry 56% of all tokens (`tp_data_cost.txt`).

| per-sample cost model | eff 8×2 (smoke) | eff 8×16 (4B arm) | eff 4×32 (2B arm) | predicted smoke/arm speed ratio |
|---|---|---|---|---|
| ∝ tokens (f = 0) | 0.401 | 0.671 | 0.807 | **0.60** |
| 30% fixed + 70% ∝ tokens | — | 0.744 | 0.856 | 0.66 |
| ∝ launching ops, torch DeltaNet (≈70% fixed) | 0.692 | 0.872 | 0.934 | 0.79 |
| **observed** | | | | **0.60** (rounding range 0.56-0.65) |

At least ~70% of the per-sample cost must scale with length, whether counted in tokens or images. An op-count-bound
LLM would have predicted 0.79.

What length-bucketed sampling buys (every rank draws from the same length decile): 8×16 goes 0.671 → 0.920, and 32
ranks × 4 goes 0.366 → 0.779.

### 2.4 The server's own latency runs locate the cost in the image path (`tp_per_image.txt`)

| measurement (server unless noted) | wall time | vision patch rows | ms per patch row | ms per image |
|---|---|---|---|---|
| deferred bench prefill: 32 keyframes at 256×448 + 256-token map (no DataLoader, no backward) | 61,284 ms | 15,360 | **3.99** | 1,857 |
| incremental streaming step: 2 frames, 112 tokens (no DataLoader) | 2 × 1,160.9 ms | 448 | **5.18** | 2,322 |
| training, own cost of a mean sample (10.7 s) | 10,700 ms | 1,981 | **5.40** | 2,522 |
| Mac MPS, same deferred prefill (480×640) | 12,390 ms | 16,000 | 0.77 | 375 |

Paths with no image input are normal on the server:
- question-only TTFT: 164 ms on the server vs 226 ms on the Mac;
- TPOT: 51.1 ms on the server vs 51.3-52.6 ms on the Mac.

CUT3R is also healthy on the server: 25.6 ms per frame, against 412.6 ms on the Mac.

**Interpretation.** The vision tower's FLOPs predict about 3 µs per patch row (0.61 GFLOP at 200 TFLOP/s). The server
spends about 4-5 ms per patch row in the image path, whether training or running inference.

The prime suspect is the patch embed. It is `nn.Conv3d(3, 1024, kernel=stride=(2,16,16))` on N = #patches rows in bf16.
If cuDNN does not take a Conv3d, PyTorch falls back to `slow_conv_dilated3d`. The parallel latency analysis
(`latency.md` in this folder) cites external reports of exactly this regression: bf16 Conv3d on torch 2.9.x with cuDNN
< 9.15, measured at 3.5 ms/patch on a Qwen-VL patch embed. I did not verify those references.

The server's torch/cuDNN versions are not in the upload. A code comment in `src/live3r/geometry/cut3r.py` (09-24)
says "server torch 2.7.1". If that is still true, the regression would have to come from cuDNN being unavailable or
disabled rather than from the 2.9.x bug. The Mac numbers show a healthy image path at 0.77 ms/row on MPS; that Mac
runs torch 2.14.0.

Another image-path cost is certain from the config (`attn_implementation: sdpa`): the vision attention loops over
images with one `lengths.tolist()` sync per block. At normal kernel speeds that costs milliseconds, not seconds.

If the image-path cost is ~4.5 ms per row, then 1,982 rows × 4.5 ms ≈ 8.9 s. Add ~1.0 s for the LLM graph (2.5) and
~0.15 s for CUT3R, and the total is ≈10 s, matching the 10.7 s own cost.

### 2.5 The Gated DeltaNet runs its torch fallback (`tp_summary_numbers.txt`)

transformers 5.16.1 resolves each kernel in this order: hub `kernels` (only with `use_kernels=True`, which we do not
pass), then the `fla` and `causal_conv1d` packages, then a pure-torch path. The torch path loops 63 times per layer
per call in Python, plus one iteration per 64-token chunk, in fp32. There is **no warning**, so the arm logs cannot
show which path ran. `requirements.txt` lists neither package, and p0 only checks
torch/transformers/peft/lmms_eval/roma/omegaconf/scipy/yaml.

Launching ATen ops per sample:

| path | fitted per-sample launches | mean sample (655 tokens) |
|---|---|---|
| S2, torch fallback | 54,939 + 35.6/token | **78,237** |
| S2, fused stand-in | 10,911 + 1.37/token | **11,806** (6.6× fewer) |
| S1, torch fallback | 50,397 + 35.6/token | 73,695 |
| S1, fused stand-in | 6,369 + 1.37/token | 7,264 |

| inference path | torch fallback | fused stand-in |
|---|---|---|
| streaming step | 13,353 | 2,601 |
| 4,069-token prefill | 40,133 | 4,061 |
| question-only TTFT | 12,689 | 2,345 |
| decode step | 2,195 | 1,811 |

The TTFT/TPOT argument that the server runs the fallback:

| | cost per launching op if fallback | cost per launching op if fused |
|---|---|---|
| server TTFT (164 ms) | 12.9 µs | 69.9 µs |
| server TPOT (51.1 ms) | 23.3 µs | 28.2 µs |
| Mac, where the fallback is certain (TTFT 226 ms, TPOT 52.6 ms) | 17.8 / 24.0 µs | — |

Under "fused", the server would need 70 µs per op on the text prefill but 28 µs per op on decode, and a host 4× slower
than a Mac on one path while equal on the other. Under "fallback", both numbers are ordinary eager-CUDA costs and match
the Mac. **The fallback is the consistent reading.**

Cost today: 78k launches × 12.9 µs ≈ **1.0 s per mean sample** (≈10% of the own cost). With fused kernels it would
be 0.15 s. On the Mac CPU (1 thread, tiny widths), the fallback adds 1.6 s of pure host time to the smallest
(152-token) sample: 97k extra ATen ops at 16.5 µs each.

### 2.6 CUT3R: frozen, fp32, one frame at a time

- `run_geometry` ingests frames one at a time at batch 1 (`live3r.py:290-297`).
- The checkpoint is 1,248 tensors, all float32, 803M parameters. Nothing in `src/` or `scripts/` casts it or uses
  autocast.
- Measured on the server: ≤25.6 ms per frame, and that includes the DPT head, which training does not run. The bench
  frame was 288×512 (576 patches); training frames average 759 patches. That gives ~0.11-0.15 s for 4.2 frames.
- It is about 1% of the step today (`s2_sft` ≈ `s2_real`). After the image-path and fla fixes it becomes 20-30%.
- **A precomputed geometry cache is impractical:** 830k records × 4.2 frames × 3 taps × 759 × 768 × 2 bytes ≈ 12 TB.

### 2.7 Data loading

- Sensenova is image-only: all 905 local records are image lists. Video decode (decord, or PyAV's full-decode
  fallback) is irrelevant for this run.
- The real `build()` on 1 thread with local SSD takes mean **148 ms per sample** (p50 73, p90 370, p99 1,263). Heavy
  records take 513 ms, light ones 61 ms (`tp_data_cost.txt`).
- Six workers per rank give ~40 samples/s of capacity per rank against a consumption of 0.06. Data is not the limit
  unless `/group-volume` adds >~80 s per sample.
- The latency benches reproduce the per-image cost without any DataLoader.

### 2.8 Minor terms

- **Gradient checkpointing** adds 1.27-1.40× launches (`tp_opcount_ckpt.json`).
- **DDP**:
  - `find_unused_parameters=True` traverses the graph only on the sync micro-step (1 in 16);
  - the all-reduce happens once per 16 samples (~156M trainable fp32 parameters in the local S2 run, a few ms on
    NVLink);
  - `check_rank_sync` and `time_is_up` are tiny collectives.
- **Skips**: 5-8 per arm on rank 0, out of 1,696-2,176 samples, i.e. 0.3-0.4%.
- **HoldoutProbe**: runs only at the start and end, since `probe_every` 500 exceeds the step count, and it is outside
  the `samp/s` windows. Under the image-path hypothesis each probe costs ~15 min of wall time: 32 samples × 3 forwards
  plus 6 generations.

## 3. Ranked hypotheses

| # | hypothesis | share of the 16 GPU-s today | evidence for | evidence against / open |
|---|---|---|---|---|
| 1 | **Image path costs ~2.3 s per image** (~4.5 ms per patch row). Suspect: Conv3d patch embed on a non-cuDNN kernel | ~85-90% of own cost (~9 s) | same ms/patch in training, streaming and prefill (2.4); 2B = 4B own cost; ∝ length (2.3); text paths healthy; CUT3R/LoRA/DDP/data excluded (2.2, 2.7) | mechanism not observed; server torch/cuDNN unknown |
| 2 | **Budget sized from a smoke at grad-accum 2** | explains 1.66× under-use of the caps, not the slowness | arithmetic (1); simulated gating ratio 0.60 = observed | — |
| 3 | **DeltaNet torch fallback** | ~1.0 s (~10%); dominant after #1 | TTFT/TPOT consistency (2.5); requirements lack fla/causal-conv1d | package list on the server not seen |
| 4 | **Slowest-rank gating** | ×1/0.67 at 8×16 (~33% idle) | 2.3 | per-rank timings are not logged (rank 0 only) |
| 5 | CUT3R fp32, batch 1 | ~1% | `s2_sft` ≈ `s2_real` | grows to 20-30% after #1 and #3 |
| 6 | Data / network filesystem | ~0 expected | 148 ms CPU per sample; benches without a DataLoader show the same cost | `/group-volume` latency unknown; the `data` field settles it |
| 7 | Grad checkpointing, `find_unused`, all-reduce, skips | few % | op counts; sizes | — |

## 4. What to grep and run on the server, with predictions

Reading the log line (`train.py:704-709`):
- `data X geom Y fb Z ms/샘플` are **rank 0 only**, averaged per micro-step over the log window (20 steps × 16).
- `fb` includes the wait for the slowest rank at the all-reduce.

```
grep -a "samp/s" outputs/weekend/s1_real.log outputs/weekend/s2_real.log outputs/weekend/s2_sft.log | tail -n 6
#   predict: data <~300, geom ~100-150 (0 for s2_sft), fb ~15,000 ms per sample. data >~1,000 -> hypothesis 6.
grep -a "samp/s" outputs/weekend/p1_4b_real_smoke_a.log outputs/weekend/p1_4b_real_smoke_c.log | tail -n 4
#   smoke_a (1 GPU, dummy geometry, no DDP): fb ~7,000-20,000 -> the cost is in the model path, not in DDP or CUT3R.
#   smoke_c (accum 2) fb ~25,000 vs s1_real (accum 16) fb ~15,000 -> the difference is gating wait.
grep -a "probe step" outputs/weekend/s1_real.log outputs/weekend/s2_real.log
#   "(n=32, XXXs)": forward-only, no DDP. Hypothesis 1 predicts ~600-1,200 s; a small value (<~100 s) refutes it.
grep -a "torch\|GPUs" outputs/weekend/p0_*.log | head            # torch / transformers / peft versions per node
grep -a "RoPE2D\|curope" outputs/weekend/p0_shared.log outputs/weekend/s1_real.log | head -n 3
grep -a "건너뜀\|완료. 건너뛴\|non-finite" outputs/weekend/s2_real.log | head
```

One-minute checks on one GPU, when a GPU is free:

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled)"
python -c "import importlib.util as u; print({m: bool(u.find_spec(m)) for m in ('fla','causal_conv1d','flash_attn','kernels')})"
# patch embed alone, 2,000 rows, bf16: conv vs its exact linear; look for slow_conv_dilated3d in the profile
python - <<'EOF'
import torch, time, torch.nn.functional as F
c = torch.nn.Conv3d(3, 1024, (2, 16, 16), (2, 16, 16)).cuda().bfloat16(); x = torch.randn(2000, 1536, device="cuda", dtype=torch.bfloat16)
for name, f in (("conv3d", lambda: c(x.view(-1, 3, 2, 16, 16))), ("linear", lambda: F.linear(x, c.weight.view(1024, -1), c.bias))):
    f(); torch.cuda.synchronize(); t = time.perf_counter(); f(); torch.cuda.synchronize(); print(name, (time.perf_counter() - t) * 1e3, "ms")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as p:
    c(x.view(-1, 3, 2, 16, 16)); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))
EOF
```

Then run a `torch.profiler` pass over two micro-steps of `train.py` on one GPU (`--max-steps 1 --grad-accum 2`) and
look at the top 20 ops by CPU and CUDA time. While an arm runs, `nvidia-smi dmon -s u -c 20` should show SM% near 0
if the step is host- or slow-kernel-bound.

## 5. Fixes and expected speed-ups

All numbers are estimates. They assume 12.9 µs per launching op (the server's own TTFT) and FLOP-level vision cost
once the image path is fixed.

| fix | removes | expected | cost / risk |
|---|---|---|---|
| **F1** Profile first. Then run the patch embed as a Linear (`F.linear(x, W.view(1024,-1), b)`, exact: fp32 rel. diff 2e-6, bf16 within one ulp; `tp_conv3d_linear_check.txt`), or move to a torch/cuDNN without the Conv3d cliff | ~9 s per sample of image path | own cost 10.7 → ~1.2 s; **16 → ~1.4 GPU-s per sample (~11×)** | a few lines in `Live3RModel.from_pretrained`; must also apply to eval so train and eval match |
| **F2** Install flash-linear-attention and causal-conv1d; p0 asserts `import fla.ops.gated_delta_rule, causal_conv1d` | 66k of 78k launches per sample | LLM host time 1.0 → 0.15 s; after F1: **~0.45-0.6 GPU-s per sample** | pip on 4 nodes; check that one batch's loss matches the torch path within bf16 |
| **F3** Length-bucketed global sampler (all ranks draw from the same length bucket at each step; seeded so real and control see the same data) | slowest-rank idle | 8×16: 0.67 → 0.92 (≈1.37×); 32×4: 0.37 → 0.78 (≈2.1×) | sampler code; epoch order changes (not the data) |
| **F4** CUT3R in bf16 autocast or TF32, batched across records with equal frame counts | fp32 803M model at batch 1 | ~0.12 → ~0.04 s per sample | changes the geometry numerics, so the next training and all evals must use the same setting |
| **F5** Drop `--grad-checkpointing` after F2 | recompute forward | 1.27-1.40× fewer launches | peak memory 27-31 GB now; test on the longest sample (6,897 tokens) |
| **F6** Size the budget from the production configuration: smoke at accum 16/32, or re-plan after the real arm's first 20 steps | 1.66× under-use of the hour caps | same hours, 1.66× the samples | trivial |
| **F7** Sequence packing (varlen: fla `cu_seqlens`, FA2 varlen in the 8 attention layers, per-sample M-RoPE and injection masks) or per-GPU batch > 1 | fixed ~10.9k launches per call | ~11.8k → ~2.3k launches per sample at 8 per pack | large change; only if still host-bound after F1-F5 |
| **F8** Pre-resize images offline (VLM ≤131k px, CUT3R 512 long side) into shards on local NVMe | 148 ms CPU per sample plus filesystem latency | insurance at ≥10 samples/s per node | none needed today |

Not recommended:
- a geometry cache (~12 TB);
- more DataLoader workers (the workers are idle).

Projected throughput for the 4B S2 arm:

| state | own cost (s/sample) | gating eff. | GPU-s/sample | samples/s per node | 1 epoch, 1 node | 1 epoch, 4 nodes as one job (bucketed) |
|---|---|---|---|---|---|---|
| now (measured) | 10.7 | 0.67 | 16.0 | 0.50 | 19 days | — |
| + F1 | ~1.2 | ~0.87 | ~1.4 | ~5.8 | ~40 h | ~10 h |
| + F1 + F2 | ~0.35-0.45 | ~0.75-0.8 | ~0.45-0.6 | ~13-18 | ~13-18 h | ~4 h |
| + F3, F4, F5 | ~0.2-0.25 | ~0.95 | ~0.25-0.3 | ~27-32 | ~7-9 h | ~2-3 h |

## 6. Targets

| goal | total samples/s | per GPU | GPU-s per sample | speed-up vs now (16 GPU-s) |
|---|---|---|---|---|
| 1 epoch (830,135) in 24 h on one 8-H100 node | 9.61 | 1.20 | **≤0.83** | **19.2×** |
| the same on 4 nodes as one 32-GPU job | 9.61 | 0.30 | ≤3.33 | 4.8× |
| the weekend matrix (14 arm-stage epochs incl. the 2B) in 3 days on 32 GPUs | — | — | ≤0.71 | 22× |
| as measured | 0.50 per node | 0.063 | 16.0 | one epoch per arm-stage = 19 node-days |

Notes on the targets:
- The 4-node option needs a real multi-node `torchrun` job. Today nodes only coordinate through marker files.
- It also needs F3: with random sampling, 32 ranks at effective batch 128 (accum 4) waste 63% of rank time on
  gating.
- F1 + F2 alone are expected to meet the one-node target.

## 7. What these materials cannot settle, and what would

- **The mechanism behind the ~4.5 ms-per-patch image path.**
  - Needs: the torch/CUDA/cuDNN versions (`p0_<role>.log` or the one-liner above) and a profile of
    `visual.patch_embed` and of one training step.
  - Everything here is inferred from wall times.
- **Whether fla and causal-conv1d are installed.**
  - Needs: the import check above. Section 2.5 is an indirect inference from TTFT and TPOT.
- **The actual `data / geom / fb` split and the probe durations** in the arm logs (section 4 gives predictions).
- **Per-rank imbalance.** Only rank 0 logs; the gating numbers are simulated from the 905-record local sample.
  - Whether its image-count distribution matches the 830k set needs a histogram over `data/sensenova.jsonl` on the
    server.
- **`/group-volume` read latency.** It matters only once the step is ~20× faster (the `data` field and a cold-cache
  read test would show it).
- **Absolute per-op costs on an H100.** Op counts are exact for the graph; microseconds per op are calibrated only
  from the server's own TTFT and TPOT (12.9-23.3 µs), so every projection in section 5 carries that uncertainty.
