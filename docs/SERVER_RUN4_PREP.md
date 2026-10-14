# Run 4 preparation — training speed and VSI-590K (2026-10-13)

Run 3 reproduced the reference (plain 71.3) and the live design holds (plain at 128 live frames 72.7,
`docs/SERVER_RUN3.md`). Training candidates for run 4: another pass (the learning curve still rose, 68.9 / 69.9 / 70.6
/ 71.0 at the quarters), the VSI-590K data, and either of them several times faster. One script prepares both:
`scripts/server_prep_run4.sh`.

## 1. Training speed: the Gated DeltaNet ran on transformers' torch path

- **Symptom.** Run 3 trained at 2.1 samples/s per 8 H100; the reference at 6.48 (FlashAttention-3, ZeRO-2). The
  profiled step (REPORT.md, d1) was CPU-bound with tens of thousands of small ops per micro-step (copy_ 150,819,
  add_ 62,098, SelectBackward 39,624 in 4 micro-steps).
- **Cause.** transformers >= 5.16 runs Qwen3.5's 24 Gated DeltaNet layers through `torch_chunk_gated_delta_rule`, a
  Python loop over 64-token chunks, unless the model is kernelized through the `kernels` hub package. p0 installs
  flash-linear-attention (fla), but fla is no longer picked up by itself - run 2's check predates this - and since
  fla 0.5 its kernels are the separate package `fla-core`, which a `--no-deps` install leaves out (p0 and the prep
  script now install both, pinned 0.5.2; fla-core wants torch >= 2.7 and triton >= 3.3).
- **Fix (ours, opt-in).** `LIVE3R_GDN=fla` routes the chunk and recurrent rules to fla's fused kernels - the ones
  transformers' own hub mapping loads, same arguments (`src/live3r/model/kernels.py`; the loader logs which one runs).
  Off by default, so evaluations keep the numerics of the runs they are compared with.
- **Check before use.** `scripts/diagnose_throughput.py --gdn-check`: one LoRA training step (forward + backward,
  gradient checkpointing) at 4,096 and 12,288 tokens with each, in two processes. OK = |Δ loss| < 0.05, gradient norm
  within 10 %, and the fused rule actually called. Then run 4 trains with `LIVE3R_GDN=fla`.
- **Result (10-14, one H100, `gdn_check.txt`):** 4,096 tokens: step 2,980 → 607 ms (**4.9×**), loss |Δ| 0.0001,
  gradient-norm ratio 1.001, peak memory 12.3 → 10.7 GB; 12,288 tokens: 12,811 → 1,772 ms (**7.2×**), |Δ| 0.0010,
  ratio 0.996, 19.5 → 14.8 GB. GDN OK: run 4 trains with the fused kernels. The whole step gains less (vision tower,
  data, DDP unchanged); the calibration step of run 4 gives the samples/s.
- **transformers 5.18 (the server's version) uses fla by itself** once `fla` is importable: the modeling module
  resolves `chunk_gated_delta_rule` to fla at import time (`use_kernel_func_from_hub_with_fallback`, priority: hub
  kernels, then the package, then torch) with no device check. So on the server every training and evaluation now
  runs the fused kernels automatically (the evaluations of 10-13 and before ran the torch path; the check puts the
  numerics within 1e-3); `LIVE3R_GDN=fla` is needed only on 5.16 / 5.17 and is harmless on 5.18. The flip side: the
  test suite builds tiny models on the CPU, and fla on CPU tensors raises "Pointer argument cannot be accessed from
  Triton" - the p0 failure on run440212 right after p0 installed fla-core. `tests/conftest.py` now blocks `fla` for
  the suite (`LIVE3R_TEST_FLA=1` lets it through), so p0 passes on every node.
- **Smaller items for run 4:** `--no-find-unused` for the plain arm (DDP's extra graph walk; nothing is unused there);
  the vision tower runs SDPA per image (no flash-attn on the nodes).

**Found on the way - what run 3 actually trained with.** The run trained at `5d963f6`, before `e6b67b0` / `3da9a51`:
LoRA on 10 module types (200 modules; `in_proj_a/b` came with e6b67b0), "\n" after every image block
(`image_newline` true; false since 3da9a51 - a training-only flag), the earlier SPAR mark rules, CUT3R without bf16
autocast. The evaluation prompt has no such newline in any run (10-12 or 10-13); `ROLE=format` in
`docs/SERVER_RUN3.md` also evaluates the model in its trained form (`--image-newline`).

## 2. VSI-590K (the `live_vsi` mix)

The VG-LLM annotation `train/vsi_590k_spar.json` (302,005 records; 70 % multiple choice, 28 % numbers) is on the
shared volume; its images are not (0 of 1,502 found in the data pass). They are the per-source archives of
[nyu-visionx/VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K) (Cambrian-S, arXiv 2511.04670; Apache-2.0
card - each source keeps its own terms). The annotation uses six of them:

| archive | GB | | archive | GB |
|---|---|---|---|---|
| ytb_roomtour | 55.3 | | procthor | 6.5 |
| adt | 23.9 | | s3dis | 2.9 |
| hypersim | 6.8 | | robotics | 1.1 |

~96.5 GB in all; the ScanNet / ScanNet++ / ARKitScenes archives (139 GB) are not fetched - those environments could
share scenes with VSI-Bench. `scripts/fetch_vsi590k.py` downloads (resumable), unpacks under
`/group-volume/wooyeol/data/vsi590k_media/vsi_590k/<source>/` (the records' paths), deletes each archive, and resolves
a sample of records per source. The registry entry `vsi590k` reads the same annotation with that media root; the
mix `live_vsi` = `live` + `vsi590k` (1,745,782 records).

## Run

```
cd /group-volume/wooyeol/vlm-live && git pull && mkdir -p outputs/run4_prep
nohup bash scripts/server_prep_run4.sh > outputs/run4_prep/master.$(hostname -s).log 2>&1 &
```

It can run next to the run-3 evaluations: the check takes one GPU for ~15-30 min (the first visible one), the
download needs network and ~100 GB on `/group-volume` (peak ~150 GB while an archive and its contents coexist).
Send back `outputs/run4_prep/prep_to_send.tar.gz` (the GDN check, the VSI-590K table, the logs).
