# The reference system (VSI 70.83) and what separates it from ours

Source: `scripts/inspect_reference_system.py` run on the server on 2026-09-30 against `/group-volume/jcho` (bundle in
`outputs/reference_70/reference_70/` on the M2: `INVENTORY.md`, the checkpoint's configs and `trainer_state.json`, the
training and evaluation scripts, the lmms-eval results JSON and 50 sample records). The reference team's reply to
`docs/QUESTIONS_FOR_EXISTING_SYSTEM.md` was a prediction-file runbook; the facts below are read from their files.

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

- **The live question is now cheap to answer on their model.** Their `scripts/evaluation/eval_lora_multi.sh` already
  exposes our live constraints as knobs: `TARGET_SIZE` (358 px ≈ our 128 tokens/frame), `MAX_NUM_FRAMES`, `FPS=1`
  (frames at 1 fps, capped), and the `vsibench_multiimage` task (frames as images). Their 70.83 under those settings
  is the realistic live target — the number every one of our runs should be compared with, on the same evaluation.
  Their own 1-fps latency profiles of this model (`profiling/results`) exist too.
- **Run 2 stays what it is:** the throughput fixes and a Sensenova-only one-epoch reference for our architecture, with
  the weekend recovered. It is not the route to 70.
- **Run 3 (proposal, after run 2 and the live evaluation of their model):** their data mix at one epoch, with their
  fusion and ours as the A/B, at the resolution the live evaluation says we can afford.

Second inspection pass (CPU, ~1 min, `python scripts/inspect_reference_system.py`): the dataset table with every
annotation file's path, size and record count; the gated-token fusion code; their 1-fps latency profiles; the head and
tail of the checkpoint's `train.log`. The first bundle's searches were drowned by their lmms-eval fork (under `src/`);
the second pass searches the training package only.
