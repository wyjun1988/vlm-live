# The reference system (VSI 70.83) and what separates it from ours

Source: two read-only inspection passes over their files on the server on 2026-09-30 (checkpoint configs and
`trainer_state.json`, the lmms-eval results JSON and 50 sample records, the dataset annotation files, their latency
profiles, the training log). The reference team's reply to `docs/QUESTIONS_FOR_EXISTING_SYSTEM.md` was a
prediction-file runbook; the facts below are read from their files.

**Rule (2026-09-30): we do not take their code.** We note how their system is built and used, and implement it
ourselves; their training data may be used (`configs/data/vgllm.yaml`). The local bundles keep only data and
metadata - the copies of their scripts and sources that the passes had collected were deleted, and the inspector
that collected them was replaced by a data-only one (`scripts/inspect_reference_data.py`).

## 1. What it is

| item | value | where |
|---|---|---|
| VSI-Bench overall | **70.83** (lmms-eval `vsibench`, all 5,130 questions, greedy, thinking off) | results JSON 20260728_140541 |
| base / adapter | Qwen/Qwen3.5-4B + LoRA r16 α32 dropout 0.05 on all linear layers of the LLM (attention q/k/v/o, Gated DeltaNet in_proj_qkv/z/a/b + out_proj, MLP) | `adapter_config.json` |
| also trained | `geometry_merger` (MLP, 23 M params, with `ln_q`) and `feature_fusion` (gate projection + two norms): `modules_to_save` | `adapter_config.json`, tensor groups |
| geometry | CUT3R `cut3r_512_dpt_4_64.pth` — **byte-identical to ours** (sha256); `geometry_encoder_type cut3r`, patch 16, `reference_frame first`, target size 512 | inventory, `config.json` |
| fusion | `feature_fusion_method gated_token`, `fusion_num_layers 1`, `geometry_merger_type mlp`: geometry tokens merged to the LLM width and combined with the visual tokens by a learned gate at the LLM input (one place) | `config.json`, training scripts |
| training (the CUT3R "all" recipe) | datasets `spar_234k, llava_hound_64k, vlm3r_scannet, vlm3r_scannetpp, vlm3r_arkit, sensenova_800k`; 1 epoch; LoRA lr 5e-5, merger/fusion lr 1e-5, vision lr 1e-6 (vision frozen); global batch 64; cosine, warmup 3 %; ≤ 576 tokens per image, video 4-8 frames; one stage | `scripts/train/train_qwen35_4B_all_gt_cut3r_lora2.sh` |
| the 70.83 run itself | `ckpt/cot/qwen3.5-4b-allcot-sqa-gt-cut3r-lora2` = the "all" mix + CoT data + SQA3D ("gt" = gated-token fusion, not ground truth); **31,702 steps × 64 ≈ 2.03 M samples**, one epoch, loss 1.45 → 0.33 | `trainer_state.json`, run name |
| evaluation | their lmms-eval model `vgllm_qwen35`: the **video path**, 32 frames sampled uniformly over the whole video, `target_size 512` → **256 tokens per frame**, `max_length 12800`, batch 8, greedy; prompt "These are frames of a video. {question} {post-prompt}" with the standard VSI post-prompts | results `config` / task config / samples |
| answers | one word or letter: median 4 characters, max 54, **no chain of thought** in any answer (the CoT data did not make answers long) | samples (5,130) |

## 2. Per type: the reference against ours (all 5,130 questions)

| run | overall | appear | abs dist | count | rel dist | obj size | room size | route | rel dir |
|---|---|---|---|---|---|---|---|---|---|
| **reference** (2.03 M samples, their data mix) | **70.8** | 80.1 | 57.5 | 73.8 | 72.1 | 78.4 | 74.1 | 46.4 | 84.3 |
| ours: plain SFT, weekend (17.4 k Sensenova samples, no geometry) | 62.3 | 72.7 | 47.9 | 68.9 | 67.3 | 67.4 | 59.0 | 42.3 | 72.9 |
| ours: S1→S2 real geometry, weekend | 60.7 | 67.3 | 45.5 | 67.4 | 61.4 | 66.1 | 61.7 | 46.4 | 69.7 |
| ours: base + format hint (no training) | 52.2 | 62.5 | 37.9 | 54.7 | 58.5 | 66.0 | 50.0 | 33.0 | 54.8 |
| **gap: reference − our plain SFT** | **+8.5** | +7.4 | +9.6 | +4.9 | +4.8 | +11.0 | **+15.1** | +4.1 | **+11.4** |

The gap is largest on the metric and 3D-relational types (room size, relative direction, object size, absolute
distance) and smallest on counting, relative distance and route — i.e. where training data with 3D-derived answers
teaches the most. The evaluation paths differ (their video path at 256 tokens/frame vs our image-mode keyframes at
128), so these rows are not a paired comparison; §5 turns it into one.

## 3. Setup side by side

| | reference | ours, weekend | ours, run 2 (running) |
|---|---|---|---|
| data | the six sets above (+ CoT + SQA3D) | Sensenova only | Sensenova only |
| samples seen | ≈ 2.03 M (1 epoch) | 17.4 k (S2), 13.6 k (S1) | ≤ 830 k (1 epoch; the calibrated speed decides) |
| LoRA | r16 α32, all linear LLM layers | r32 α64, the same layers except in_proj_a/b | same |
| geometry → LLM | trained MLP merger + gated-token fusion at the input; one stage | zero-init projectors into decoder layers 0-2 (deepstack) + pose token; S1 (projector) then S2 | same, S1 = 5 % of an epoch |
| learning rates, batch | LoRA 5e-5, merger 1e-5; 64 | LoRA 1e-4, projector 3e-5; 128 | same |
| tokens per image | train ≤ 576, eval 256 (512 px) | 128 | 128 |
| evaluation | video path, 32 uniform frames of the whole video (offline) | image mode, 32 uniform keyframes (the live path) | same |

## 4. What the gap is made of (ranked; the evidence, and what would settle each)

1. **Data volume** — 2.03 M samples against 17 k at the weekend (×120). Run 2's one Sensenova epoch is still 0.4× theirs.
2. **In-domain spatial QA** — as published, VLM-3R built VSI-style question templates on ScanNet / ScanNet++ /
   ARKitScenes (the collections VSI-Bench is drawn from) and SPAR generates 3D-grounded QA on ScanNet-family scenes.
   The per-type gap (§2) sits exactly on the types those answers are computed from 3D annotations. Sensenova alone
   does not supply this, so **run 2 cannot reach 70 whatever its throughput**. To verify: the annotation files'
   scene ids against VSI-Bench's (a scene overlap would inflate their number; the published papers use training
   splits and VSI-Bench validation splits).
3. **Resolution** — 256 tokens per frame at evaluation (576 per image in training) against our 128: twice the visual
   tokens. Measurable on their own model with `TARGET_SIZE` (§5).
4. **Fusion design** — a trained merger with a gated input fusion in one stage, against our zero-init deepstack with a
   separate projector stage. At equal (tiny) data our geometry path cost 1.6 against plain SFT; theirs is proven at
   2 M samples. Only an A/B at equal data settles which is better.
5. **Evaluation path** — offline video path over the whole video vs our live image-mode keyframes. Also measurable on
   their model (`vsibench_multiimage`, `FPS=1`).

Not a factor: the CUT3R checkpoint (identical) and chain of thought in the answers (none; thinking off).

## 5. What to do with it

- **The live question for their model is theirs to measure.** Their evaluation already has knobs for our live
  constraints (target size, frame cap, 1 fps sampling, frames as images); their 70.83 under those settings would be
  the realistic live target. That is a request to their team - we do not run their code.
- **Run 2 stays what it is:** the throughput fixes and a Sensenova-only one-epoch reference for our architecture, with
  the weekend recovered. It is not the route to 70.
- **Run 3 (proposal, after run 2):** the same data mix at one epoch, trained on our stack with our implementation
  of their fusion design, against a no-geometry arm at equal data (§7).

The second pass (2026-09-30) read the dataset annotation files (paths, sizes, record counts), their 1-fps latency
profiles and the head and tail of the checkpoint's `train.log`.

---

## 6. Second pass (2026-09-30, `outputs/reference_70_v2/`)

### The data (all under `/group-volume/VG-LLM_data`, which we may use)

| dataset | records | | dataset (CoT variant, `*_qwen_reasoning_jcho_cleaning.json`) | records |
|---|---|---|---|---|
| spar_234k | 234,277 | | spar_234k_cot | 87,433 |
| llava_hound_64k | 63,750 | | llava_hound_64k_cot | 61,122 |
| vlm3r_scannet | 53,821 | | vlm3r_scannet_cot | 24,898 |
| vlm3r_scannetpp | 152,462 | | vlm3r_scannetpp_cot | 70,143 |
| vlm3r_arkit | 1,375 | | vlm3r_arkit_cot | 770 |
| sensenova_800k | 832,132 | | sensenova_800k_cot | 340,857 |
| sqa3d | 79,445 | | scanqa | 26,515 |

Also registered: vsi_590k (302,005), vsti_camera (66,513), vsti_camera_obj (48,628), spatial457_20k, robot datapacks.
**The 70.83 run's mix reconciles exactly:** the six "all" sets (1,337,817) + the six CoT variants (585,223) + sqa3d +
scanqa = 2,029,000 records against 31,704 planned steps × 64 = 2,029,056. Training input: images ≤ 576 tokens; a video
becomes 4-8 frames (one per 2 s) given as images; SPAR records carry `spar_info` marks drawn onto the frames.

### Their speed on the same cluster

`train.log`: **6.48 samples/s on one 8-GPU node** (1.23 GPU-s per sample; 87 h for the 2.03 M-sample epoch) with
DeepSpeed ZeRO-2, FlashAttention-3, gradient checkpointing, batch 1 × accumulation 8. Our weekend arms: 0.5 samples/s
(16 GPU-s) — the cluster is fine, our software stack was not (run 2's d0 / calib measure ours; their environment is the
proven fast one).

Their own profiles of the 70.83 model (`profiling/results`, 1 fps sampling, whole video at query time, medians):

| frames | visual tokens | CUT3R | vision tower | LLM prefill | video decode | total per question |
|---|---|---|---|---|---|---|
| 63 | 9,070 | 2.86 s | 0.13 s | 0.18 s | 1.87 s | 6.95 s |
| 159 | 22,900 | 7.75 s | 0.32 s | 0.47 s | 3.79 s | 19.3 s |
| 331 | 47,700 | 14.8 s | 0.67 s | 1.08 s | 8.84 s | 37.9 s |
| 533 | 76,800 | 25.5 s | 1.11 s | 1.95 s | 21.2 s | 71.5 s |
| 1,000 | 144,000 | 48 s | 2.05 s | 4.57 s | 31.7 s | 120 s |

Per frame: CUT3R ~48 ms, vision ~2 ms; LLM prefill ~20-32 µs per token. Offline, everything happens after the
question; live, CUT3R and the vision tower run as frames arrive, and a 32-keyframe prefix (8,192 tokens) prefills in
~0.16 s — **their model on their stack fits the 1 s answer budget even without prefix caching.**

### The fusion design (in our words)

A geometry merger: CUT3R's final decoder tokens (768-d, patch 16) → 2×2 spatial merge → normalisation → an MLP (3,072 →
4,096 → 2,560), i.e. one geometry token per visual token (both cover 32 px). A per-token gate at the LLM input:
`g = sigmoid(w·[LN(v); LN(geo)] + b)`, `fused = (1 − g)·LN(v) + g·LN(geo)`, which replaces the visual token's
embedding. CUT3R runs over the sample's frames in one call, the first frame as the reference. Merger, gate and LoRA
train together in one stage.

## 7. Recommendation: implement the design here, train it on the permitted data, and make it live

Their score comes from data (2.03 M samples of VSI-domain QA), resolution and a simple input fusion trained at that
scale; none of it needs their code. Our harness already has the live half (streaming CUT3R, causal keyframes, prefix
cache, question-time latency). So:

1. **Our implementation of the fusion design** - `fusion.mode: gated_input` in Live3R: a merger from CUT3R's last
   decoder tap to one token per visual token, and a learned per-token gate that mixes it into the visual embedding at
   the LLM input, trained in one stage with LoRA (`configs/server_4b_v3.yaml`).
2. **The data, read by our loader** - the registry `configs/data/vgllm.yaml` (paths and media roots of the 17
   annotation files, the `ref70` and `all6` mixes), `MixtureDataset` (per-source media roots, fixed subsets and
   repeats), videos as a few frames given as images. The record forms our loader must handle (frame directories or
   files, the `spar_info` marks, multi-turn records) are read by `scripts/inspect_reference_data.py` (data only, CPU,
   minutes); the marks are then drawn by our own code.
3. **Run 3** - the `ref70` mix (or the subset the calibrated speed allows) on 4B: gated input fusion against a
   no-geometry arm at equal data, evaluated on our live path (and on the offline protocol for the comparison with
   70.83).

Their checkpoint is not part of this plan. Our deepstack / S1→S2 design stays available as an A/B arm; run 2 (running)
still settles our stack's speed and a Sensenova-only baseline for it.

## 8. Status (2026-09-30, evening)

Implemented here (tests: `tests/test_gated_fusion.py`, `test_mixture.py`, `test_data_forms.py`, the trainer end to end
on a two-source mix): the gated input fusion (`fusion.mode: gated_input`), the registry and mixture loader, videos
as frames given as images (files and frame directories), long-side-512 input (the geometry and visual grids
coincide), the one-time preparation (`scripts/prepare_mix.py`) and run 3 (`scripts/server_run3.sh`,
`docs/SERVER_RUN3.md`).

The data pass (`outputs/reference_data_srv/`, all 17 files, media found) settled the rest: the SPAR marks are
`{color}_point` / `{color}_bbox` / `bbox_list` in 0-1000 coordinates with a frame index per mark, now drawn by our code
(`src/live3r/data/marks.py`); the CoT sets answer `<think>…</think>` after a system message - **the lead dropped them
for the live system** (run 3 trains the `live` mix: ref70 without the six CoT variants, 1,443,777 records, any leftover
reasoning stripped to the final answer); and one VSI-Bench scene (`scene0685_00`) appears in the SPAR training images,
so every record naming a VSI-Bench scene is left out at preparation.

Step 0 of our implementation on the M2 (4B, 12 Sensenova holdout samples, random merger, real CUT3R): the
reference-style start (LayerNorm on both streams, g = 0.5) changes 75 % of the first answer tokens against the base
model but lowers the held-out loss (1.71 → 1.51): the LayerNorm alone makes the visual tokens 5.6× louder. A start
without the norm and g = 0.018 stays at the base model (8 %, 1.72). Run 3 trains both (gated, gated_raw); the loss
drop from louder visual tokens is a zero-shot idea of its own (BRAINSTORM I-44).
