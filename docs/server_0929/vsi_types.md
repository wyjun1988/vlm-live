# VSI-Bench deep dive - trained models, per type and per question (server snapshot 2026-09-29)

All numbers computed on CPU from `outputs/server_0929/eval/outputs/weekend/vsi/*.json` and `data/eval/vsibench/test.jsonl` by the scripts in this folder (`make_vsi_types.py` regenerates this file). Aggregation = lmms-eval (relative-direction difficulties averaged first, then the mean of 8 types). Intervals = 95% paired bootstrap over videos (2,000 resamples, seed 0; identical to `weekend_report.paired_bootstrap` for the overall number). `*` = interval excludes 0. Per-type intervals are not corrected for multiple comparisons (8 types x 14 comparisons).

## Run conditions (read from the JSON flags)

| run | mode | model | --format-hint | routed facts | weights | effective prompt | overall |
|---|---|---|---|---|---|---|---|
| base_hint | oracle-image | Qwen3.5-4B | yes | no | - | hint | 52.2 |
| base_routed | oracle-image | Qwen3.5-4B | no | yes | - | hint+facts | 54.4 |
| s1_real | oracle-image | Qwen3.5-4B | no | no | s1_real/final.pt | plain | 52.9 |
| s2_real | oracle-image | Qwen3.5-4B | no | no | s2_real/final.pt | plain | 60.7 |
| s2_real_routed | oracle-image | Qwen3.5-4B | no | yes | s2_real/final.pt | hint+facts | 61.1 |
| s2_sft | oracle-image | Qwen3.5-4B | no | no | s2_sft/final.pt | plain | 62.3 |
| s2_sft_hint | oracle-image | Qwen3.5-4B | yes | no | s2_sft/final.pt | hint | 62.3 |
| base_hint_2b | oracle-image | Qwen3.5-2B | yes | no | - | hint | 39.3 |
| base_plain_2b_img | oracle-image | Qwen3.5-2B | no | no | - | plain | 24.5 |
| base_plain_2b_vid | oracle-video | Qwen3.5-2B | no | no | - | plain (video mode) | 42.7 |

Notes: the routed runs (`base_routed`, `s2_real_routed`) have `format_hint = false` in the JSON but the harness appends the SAME sentence ("Answer directly in the requested format without explanation.") to the prefix whenever routing is on (`eval_vsi_local.py`, the `route_facts or route_objects` branch), so their effective prompt = hint + facts. Facts were attached only to room-size (288/288) and abs-distance (226/834) questions. There is no 4B base run without the hint in these materials (`base_plain` 4B is not uploaded), so `s1_real` / `s2_real` / `s2_sft` (plain) vs `base_hint` always mixes training with the hint.

Sanity (`s0_sanity.py`): every saved per-question score equals the lmms-eval rescore of (pred, gt); all 10 prediction sets cover the same 5,130 ids with gt/type/video identical to test.jsonl; the overalls reproduce REPORT.md. Determinism: `base_routed` and `base_hint` give byte-identical answers on every question that received no attachment (appear 618/618, count 565/565, rel dist 710/710, route 194/194, rel dir 968/968; obj size 948/953), so within-server differences are model differences, not decoding noise. Training-seed noise is NOT measured (the `s2_real_seed1` VSI run failed).

## 1. Paired comparisons, overall and per type

Cells: Δ = A - B in points [95% interval]. Prompt column says whether the two runs saw the same prompt.

| A - B | what it measures | prompt | A / B | Δ overall | appear | abs dist | count | rel dist | obj size | room size | route | rel dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_real - s2_sft | value of the whole geometry path | same prompt (plain, no hint) | 60.7 / 62.3 | **-1.6 [-3.1, -0.1] *** | -5.3 [-9.5, -1.3] * | -2.5 [-5.3, +0.2] | -1.5 [-3.7, +0.9] | -5.9 [-10.5, -1.2] * | -1.3 [-2.5, +0.1] | +2.6 [+0.1, +5.5] * | +4.1 [-3.9, +11.8] | -3.1 [-7.4, +1.4] |
| s2_real - base_hint | training (S2 real) vs base | DIFFERS: A plain, B + format hint | 60.7 / 52.2 | **+8.5 [+6.7, +10.3] *** | +4.9 [+0.2, +9.5] * | +7.5 [+4.6, +10.5] * | +12.8 [+9.2, +16.5] * | +3.0 [-1.4, +7.3] | +0.1 [-1.5, +1.8] | +11.7 [+7.6, +16.0] * | +13.4 [+3.9, +22.3] * | +15.0 [+10.3, +19.6] * |
| s2_sft - base_hint | training (plain SFT) vs base | DIFFERS: A plain, B + format hint | 62.3 / 52.2 | **+10.1 [+8.4, +11.8] *** | +10.2 [+6.1, +13.6] * | +10.0 [+7.0, +13.0] * | +14.3 [+11.0, +17.6] * | +8.9 [+4.4, +13.4] * | +1.4 [-0.2, +3.1] | +9.0 [+5.1, +13.0] * | +9.3 [+0.0, +17.8] | +18.1 [+13.8, +22.1] * |
| s2_sft_hint - base_hint | training (plain SFT) vs base, same prompt | same prompt (both + hint) | 62.3 / 52.2 | **+10.1 [+8.4, +11.7] *** | +9.7 [+5.6, +13.2] * | +9.3 [+6.2, +12.5] * | +15.2 [+11.9, +18.5] * | +8.7 [+4.1, +13.2] * | +1.4 [-0.3, +3.0] | +10.1 [+6.2, +14.1] * | +8.2 [-1.6, +16.8] | +18.3 [+14.1, +22.2] * |
| s1_real - base_hint | S1 vs base | DIFFERS: A plain, B + format hint | 52.9 / 52.2 | **+0.7 [-0.7, +2.1]** | -3.1 [-7.8, +1.6] | +0.2 [-2.2, +2.5] | +10.7 [+7.2, +14.5] * | -2.8 [-6.9, +1.1] | -1.3 [-2.9, +0.4] | +3.9 [+0.4, +7.6] * | +2.6 [-5.2, +10.1] | -4.6 [-8.0, -1.2] * |
| s2_real - s1_real | S2 over S1 (real arm) | same prompt (plain) | 60.7 / 52.9 | **+7.8 [+6.2, +9.5] *** | +7.9 [+3.2, +12.6] * | +7.3 [+4.8, +9.8] * | +2.0 [+0.4, +3.6] * | +5.8 [+1.5, +10.0] * | +1.4 [-0.2, +3.0] | +7.8 [+4.2, +11.6] * | +10.8 [+2.7, +19.6] * | +19.6 [+14.4, +24.6] * |
| s2_sft_hint - s2_sft | format hint after plain SFT | hint is the tested factor | 62.3 / 62.3 | **-0.0 [-0.7, +0.6]** | -0.5 [-1.5, +0.4] | -0.7 [-1.8, +0.4] | +0.9 [-0.7, +2.5] | -0.1 [-1.4, +1.1] | +0.0 [-0.7, +0.7] | +1.1 [-0.2, +2.5] | -1.0 [-4.7, +2.7] | +0.2 [-0.9, +1.2] |
| s2_real_routed - s2_real | routed facts after S2 real | A has hint line + facts, B neither | 61.1 / 60.7 | **+0.4 [-0.4, +1.4]** | +0.8 [-0.7, +2.2] | -1.2 [-3.2, +0.8] | +0.2 [-0.5, +0.8] | -0.8 [-2.1, +0.6] | -0.1 [-0.7, +0.5] | +4.2 [-0.3, +8.5] | +1.5 [-2.7, +5.8] | -1.2 [-2.7, +0.3] |
| base_routed - base_hint | routed facts, base | same prompt except facts | 54.4 / 52.2 | **+2.2 [+1.6, +2.9] *** | +0.0 [+0.0, +0.0] | +2.1 [+0.6, +3.7] * | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [-0.0, +0.0] | +15.6 [+10.8, +20.6] * | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| s2_real_routed - base_routed | S2 real vs base, both routed | same prompt (hint + facts) | 61.1 / 54.4 | **+6.7 [+5.1, +8.2] *** | +5.7 [+0.8, +10.5] * | +4.2 [+1.8, +6.7] * | +12.9 [+9.4, +16.5] * | +2.1 [-2.2, +6.5] | +0.0 [-1.8, +1.7] | +0.2 [-1.0, +1.3] | +14.9 [+5.4, +23.8] * | +13.8 [+9.1, +18.4] * |
| s2_real_routed - s2_sft_hint | best geometry system vs plain SFT + hint | A has facts, both hint | 61.1 / 62.3 | **-1.2 [-2.7, +0.4]** | -4.0 [-8.1, -0.4] * | -2.9 [-5.7, +0.0] | -2.2 [-4.7, +0.4] | -6.6 [-11.6, -1.7] * | -1.4 [-2.7, -0.1] * | +5.7 [+1.5, +10.1] * | +6.7 [-0.6, +14.7] | -4.5 [-8.8, -0.0] * |
| base_hint_2b - base_plain_2b_img | 2B: hint, image mode | hint is the tested factor (image mode) | 39.3 / 24.5 | **+14.8 [+13.3, +16.3] *** | +14.6 [+9.8, +19.6] * | -0.9 [-2.2, +0.5] | -0.8 [-2.2, +0.5] | +38.6 [+33.5, +43.8] * | +4.7 [+3.0, +6.7] * | +10.9 [+7.6, +14.3] * | +22.2 [+15.5, +28.6] * | +29.3 [+26.0, +32.5] * |
| base_hint_2b - base_plain_2b_vid | 2B: hint image vs plain video | DIFFERS: hint and input format | 39.3 / 42.7 | **-3.4 [-5.1, -1.8] *** | -28.5 [-36.5, -20.7] * | -2.3 [-4.5, -0.1] * | +8.7 [+5.1, +12.3] * | -0.6 [-5.2, +3.7] | +0.1 [-1.8, +2.0] | +4.5 [+0.7, +8.7] * | -0.5 [-5.6, +4.6] | -8.7 [-12.6, -4.8] * |
| base_plain_2b_img - base_plain_2b_vid | 2B: image vs video mode, plain | input format is the tested factor | 24.5 / 42.7 | **-18.2 [-19.7, -16.8] *** | -43.0 [-49.3, -36.6] * | -1.4 [-3.5, +0.9] | +9.5 [+5.6, +13.4] * | -39.2 [-44.5, -34.3] * | -4.6 [-6.9, -2.3] * | -6.4 [-10.6, -1.9] * | -22.7 [-28.7, -16.2] * | -38.0 [-41.3, -34.7] * |

Route (194 questions) has the widest intervals (about ±8 points); room size has 288 questions, one per video. Excluding route, s2_real - s2_sft on the 7-type mean is -2.4 [-3.7, -1.0] * (`s7`/inline check).

## (a) Numeric types: answer distributions, collapse, ratio to ground truth

Computed by `s2_numeric.py`. Predictions parsed with lmms-eval `extract_number` (what the metric scores). `top1/top3` = share of all questions answered with the most frequent 1 / 3 values. `ratio` = pred / gt (median [q25, q75]). `±25%` = share of parsed answers within 25% of gt. `MRA=0` / `MRA=1` = share of questions scoring 0 (off by >50%) / full marks (within 5%). `rho(gt)` = Spearman rank correlation of pred with gt (does the model track the scene?); `rho(prior)` = with the leave-one-video-out blind prior (category median; shown for count and size only - for distance and room size the prior is a single global value).

### object_abs_distance (n=834)

Ground truth: 72 distinct values, median 1.80 [q25 1.10, q75 2.90]; most frequent: 1 (5%), 0.9 (4%), 0.7 (4%), 1.5 (4%), 1.4 (4%). Blind baselines fitted on the test gt (optimistic): best single constant 1.87 -> MRA 32.0; leave-one-video-out median -> MRA 30.6.

| run | MRA | parse fail | distinct | top1 | top3 | most frequent answers | ratio pred/gt | ±25% | MRA=0 | MRA=1 | rho(gt) | rho(prior) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | 37.9 | 0 | 29 | 20% | 49% | 2.1 (20%), 1.7 (17%), 2 (12%), 3.1 (11%), 2.6 (7%) | 1.15 [0.79, 1.62] | 38% | 36% | 8% | 0.62 | n/a |
| base_routed | 40.1 | 0 | 51 | 15% | 38% | 2.1 (15%), 1.7 (13%), 2 (10%), 3.1 (8%), 2.6 (5%) | 1.06 [0.74, 1.44] | 40% | 35% | 9% | 0.61 | n/a |
| s1_real | 38.1 | 0 | 30 | 22% | 52% | 2.1 (22%), 1.7 (18%), 2.4 (12%), 1.1 (8%), 2 (7%) | 1.08 [0.74, 1.57] | 37% | 36% | 8% | 0.55 | n/a |
| s2_real | 45.5 | 0 | 30 | 25% | 48% | 2 (25%), 1 (14%), 2.1 (9%), 2.4 (8%), 1.7 (7%) | 0.95 [0.69, 1.25] | 45% | 23% | 10% | 0.67 | n/a |
| s2_real_routed | 44.3 | 0 | 51 | 19% | 38% | 2 (19%), 1 (13%), 2.1 (7%), 2.4 (7%), 1.7 (6%) | 0.93 [0.67, 1.25] | 45% | 27% | 9% | 0.64 | n/a |
| s2_sft | 47.9 | 0 | 28 | 19% | 42% | 1 (19%), 2.1 (12%), 2 (11%), 2.4 (8%), 1.4 (8%) | 0.89 [0.67, 1.17] | 49% | 21% | 10% | 0.72 | n/a |
| s2_sft_hint | 47.2 | 0 | 30 | 17% | 41% | 1 (17%), 2 (13%), 2.1 (11%), 1.4 (9%), 2.4 (8%) | 0.87 [0.67, 1.15] | 48% | 21% | 10% | 0.72 | n/a |
| base_hint_2b | 32.3 | 0 | 36 | 27% | 70% | 1.2 (27%), 2 (23%), 2.1 (21%), 1.1 (5%), 1 (5%) | 0.88 [0.58, 1.42] | 29% | 38% | 5% | 0.42 | n/a |
| base_plain_2b_vid | 34.6 | 0 | 16 | 30% | 78% | 2 (30%), 1.2 (29%), 2.1 (19%), 2.5 (6%), 3 (4%) | 1.00 [0.66, 1.50] | 33% | 38% | 7% | 0.43 | n/a |

### object_size_estimation (n=953)

Ground truth: 252 distinct values, median 96.00 [q25 59.00, q75 169.00]; most frequent: 78 (1%), 91 (1%), 96 (1%), 58 (1%), 55 (1%). Blind baselines fitted on the test gt (optimistic): best single constant 93 -> MRA 29.8; leave-one-video-out category median -> MRA 64.0 (rho(prior, gt) = 0.83).

| run | MRA | parse fail | distinct | top1 | top3 | most frequent answers | ratio pred/gt | ±25% | MRA=0 | MRA=1 | rho(gt) | rho(prior) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | 66.0 | 0 | 89 | 12% | 25% | 200 (12%), 30 (7%), 100 (6%), 120 (6%), 60 (5%) | 0.96 [0.82, 1.08] | 69% | 9% | 23% | 0.89 | 0.88 |
| base_routed | 66.0 | 0 | 90 | 12% | 25% | 200 (12%), 30 (7%), 100 (6%), 120 (6%), 60 (5%) | 0.96 [0.82, 1.08] | 69% | 9% | 23% | 0.89 | 0.88 |
| s1_real | 64.7 | 0 | 88 | 10% | 25% | 210 (10%), 200 (8%), 100 (7%), 110 (5%), 50 (4%) | 0.99 [0.86, 1.14] | 67% | 13% | 26% | 0.88 | 0.90 |
| s2_real | 66.1 | 0 | 98 | 8% | 22% | 120 (8%), 100 (8%), 200 (6%), 210 (5%), 80 (5%) | 1.00 [0.88, 1.16] | 70% | 11% | 26% | 0.89 | 0.90 |
| s2_real_routed | 66.0 | 0 | 100 | 9% | 25% | 120 (9%), 100 (9%), 200 (7%), 210 (5%), 30 (4%) | 1.00 [0.88, 1.16] | 69% | 12% | 26% | 0.88 | 0.91 |
| s2_sft | 67.4 | 0 | 87 | 10% | 24% | 100 (10%), 211 (7%), 210 (7%), 120 (5%), 30 (5%) | 1.00 [0.89, 1.16] | 71% | 10% | 27% | 0.89 | 0.89 |
| s2_sft_hint | 67.4 | 0 | 91 | 10% | 24% | 100 (10%), 210 (7%), 120 (6%), 211 (6%), 30 (5%) | 1.00 [0.89, 1.16] | 71% | 10% | 27% | 0.89 | 0.89 |
| base_hint_2b | 58.4 | 0 | 35 | 34% | 51% | 100 (34%), 200 (10%), 190 (7%), 50 (6%), 40 (6%) | 0.96 [0.77, 1.14] | 59% | 14% | 16% | 0.85 | 0.86 |
| base_plain_2b_vid | 58.3 | 0 | 31 | 28% | 46% | 100 (28%), 200 (10%), 40 (8%), 50 (8%), 180 (7%) | 0.95 [0.78, 1.14] | 60% | 16% | 17% | 0.84 | 0.87 |

### room_size_estimation (n=288)

Ground truth: 204 distinct values, median 17.50 [q25 11.15, q75 26.68]; most frequent: 12.3 (2%), 5.4 (1%), 26.4 (1%), 10.6 (1%), 14.8 (1%). Blind baselines fitted on the test gt (optimistic): best single constant 16.2 -> MRA 33.1; leave-one-video-out median -> MRA 33.0.

| run | MRA | parse fail | distinct | top1 | top3 | most frequent answers | ratio pred/gt | ±25% | MRA=0 | MRA=1 | rho(gt) | rho(prior) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | 50.0 | 0 | 35 | 23% | 55% | 18 (23%), 10 (17%), 20 (16%), 30 (11%), 16 (8%) | 1.03 [0.81, 1.33] | 51% | 22% | 11% | 0.81 | n/a |
| base_routed | 65.6 | 0 | 68 | 6% | 15% | 14 (6%), 5 (5%), 7 (5%), 18 (5%), 17 (4%) | 1.03 [0.87, 1.19] | 68% | 7% | 18% | 0.92 | n/a |
| s1_real | 53.9 | 0 | 49 | 16% | 41% | 10 (16%), 18 (16%), 20 (9%), 30 (9%), 14 (7%) | 0.93 [0.77, 1.22] | 56% | 15% | 10% | 0.86 | n/a |
| s2_real | 61.7 | 0 | 40 | 20% | 45% | 10 (20%), 16 (13%), 20 (12%), 30 (10%), 14 (8%) | 0.91 [0.73, 1.08] | 62% | 7% | 14% | 0.89 | n/a |
| s2_real_routed | 65.8 | 0 | 115 | 4% | 11% | 18 (4%), 14 (4%), 17 (3%), 16 (3%), 7 (3%) | 1.03 [0.87, 1.19] | 68% | 6% | 18% | 0.91 | n/a |
| s2_sft | 59.0 | 0 | 53 | 18% | 38% | 10 (18%), 20 (11%), 16 (9%), 30 (9%), 14 (7%) | 0.92 [0.73, 1.10] | 59% | 9% | 14% | 0.86 | n/a |
| s2_sft_hint | 60.1 | 0 | 52 | 14% | 34% | 10 (14%), 30 (11%), 20 (10%), 16 (9%), 14 (5%) | 0.95 [0.76, 1.14] | 63% | 9% | 14% | 0.89 | n/a |
| base_hint_2b | 46.3 | 0 | 21 | 39% | 61% | 15 (39%), 25 (12%), 12.5 (10%), 30 (9%), 20 (8%) | 1.08 [0.82, 1.38] | 47% | 28% | 11% | 0.62 | n/a |
| base_plain_2b_vid | 41.8 | 0 | 9 | 27% | 66% | 15 (27%), 30 (20%), 20 (19%), 25 (10%), 10 (9%) | 1.26 [1.00, 1.68] | 43% | 34% | 10% | 0.83 | n/a |

### object_counting (n=565)

Ground truth: 25 distinct values, median 2.00 [q25 2.00, q75 4.00]; most frequent: 2 (53%), 3 (18%), 4 (10%), 5 (5%), 6 (3%). Blind baselines fitted on the test gt (optimistic): best single constant 2.04 -> MRA 61.8; leave-one-video-out category median -> MRA 58.9 (rho(prior, gt) = 0.40).

| run | MRA | parse fail | distinct | top1 | top3 | most frequent answers | ratio pred/gt | ±25% | MRA=0 | MRA=1 | rho(gt) | rho(prior) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | 54.7 | 0 | 16 | 48% | 82% | 2 (48%), 1 (18%), 3 (16%), 4 (4%), 6 (4%) | 1.00 [0.50, 1.00] | 49% | 9% | 40% | 0.62 | 0.44 |
| base_routed | 54.7 | 0 | 16 | 48% | 82% | 2 (48%), 1 (18%), 3 (16%), 4 (4%), 6 (4%) | 1.00 [0.50, 1.00] | 49% | 9% | 40% | 0.62 | 0.44 |
| s1_real | 65.4 | 0 | 11 | 81% | 92% | 2 (81%), 3 (6%), 4 (5%), 10 (3%), 1 (2%) | 1.00 [0.67, 1.00] | 58% | 11% | 54% | 0.53 | 0.40 |
| s2_real | 67.4 | 0 | 9 | 80% | 93% | 2 (80%), 4 (7%), 3 (7%), 10 (4%), 11 (1%) | 1.00 [0.67, 1.00] | 61% | 10% | 55% | 0.61 | 0.44 |
| s2_real_routed | 67.6 | 0 | 9 | 79% | 93% | 2 (79%), 3 (8%), 4 (6%), 10 (5%), 6 (1%) | 1.00 [0.67, 1.00] | 61% | 9% | 55% | 0.63 | 0.43 |
| s2_sft | 68.9 | 0 | 9 | 70% | 90% | 2 (70%), 3 (10%), 4 (10%), 10 (6%), 6 (2%) | 1.00 [0.67, 1.00] | 62% | 7% | 56% | 0.65 | 0.52 |
| s2_sft_hint | 69.8 | 0 | 12 | 65% | 89% | 2 (65%), 3 (14%), 4 (11%), 10 (6%), 6 (2%) | 1.00 [0.71, 1.00] | 64% | 6% | 57% | 0.67 | 0.49 |
| base_hint_2b | 54.8 | 0 | 11 | 47% | 84% | 2 (47%), 3 (21%), 1 (17%), 10 (6%), 4 (5%) | 1.00 [0.50, 1.00] | 49% | 11% | 42% | 0.59 | 0.42 |
| base_plain_2b_vid | 46.1 | 0 | 10 | 48% | 86% | 2 (48%), 1 (28%), 3 (11%), 4 (6%), 10 (4%) | 0.67 [0.50, 1.00] | 39% | 15% | 34% | 0.54 | 0.40 |

### Where the numeric gains come from

Counting split by ground truth (53% of counting questions have gt = 2; answering 2 everywhere scores MRA 61.8). Abs distance split by gt tertile (cuts 1.4 m / 2.5 m).

| run | count MRA, gt=2 (n=302) | count MRA, gt!=2 (n=263) | answers 2 when gt!=2 | abs MRA gt<=1.4 m | abs MRA middle | abs MRA gt>2.5 m |
|---|---|---|---|---|---|---|
| base_hint | 63.9 | 44.0 | 34% | 12.7 | 58.7 | 45.8 |
| base_routed | 63.9 | 44.0 | 34% | 19.6 | 58.1 | 45.1 |
| s1_real | 95.8 | 30.5 | 64% | 14.4 | 64.8 | 37.8 |
| s2_real | 97.0 | 33.5 | 60% | 35.2 | 64.1 | 37.7 |
| s2_real_routed | 97.0 | 33.9 | 58% | 33.9 | 61.5 | 38.3 |
| s2_sft | 91.5 | 43.0 | 45% | 41.4 | 59.4 | 43.5 |
| s2_sft_hint | 88.7 | 48.2 | 38% | 42.1 | 58.4 | 41.4 |
| base_hint_2b | 64.0 | 44.2 | 31% | 24.5 | 51.5 | 21.0 |
| base_plain_2b_vid | 58.7 | 31.6 | 40% | 19.3 | 57.8 | 28.0 |

Paired video bootstrap of counting MRA split by gt (same questions in both runs):

| comparison | count, gt = 2 | count, gt != 2 |
|---|---|---|
| s2_real - s2_sft | +5.5 [+2.7, +8.5] * | -9.5 [-13.1, -5.7] * |
| s1_real - base_hint | +31.9 [+26.7, +37.2] * | -13.6 [-17.1, -10.0] * |
| s2_real - base_hint | +33.0 [+28.1, +38.1] * | -10.5 [-13.9, -6.9] * |
| s2_sft_hint - base_hint | +24.8 [+19.3, +30.5] * | +4.1 [+0.8, +7.4] * |

### MRA distribution per numeric type (share of questions)

MRA takes the values 0, 0.1, ..., 1.0 (10 thresholds from 50% to 5% relative error).

| type | run | 0 | 0.1-0.3 | 0.4-0.6 | 0.7-0.9 | 1.0 | mean |
|---|---|---|---|---|---|---|---|
| abs dist | base_hint | 36% | 14% | 23% | 19% | 8% | 37.9 |
| abs dist | s1_real | 36% | 16% | 19% | 21% | 8% | 38.1 |
| abs dist | s2_real | 23% | 19% | 22% | 26% | 10% | 45.5 |
| abs dist | s2_sft | 21% | 19% | 21% | 29% | 10% | 47.9 |
| abs dist | base_hint_2b | 38% | 20% | 20% | 18% | 5% | 32.3 |
| obj size | base_hint | 9% | 11% | 19% | 37% | 23% | 66.0 |
| obj size | s1_real | 13% | 10% | 17% | 34% | 26% | 64.7 |
| obj size | s2_real | 11% | 10% | 18% | 35% | 26% | 66.1 |
| obj size | s2_sft | 10% | 11% | 17% | 35% | 27% | 67.4 |
| obj size | base_hint_2b | 14% | 15% | 21% | 34% | 16% | 58.4 |
| room size | base_hint | 22% | 12% | 28% | 28% | 11% | 50.0 |
| room size | s1_real | 15% | 16% | 25% | 34% | 10% | 53.9 |
| room size | s2_real | 7% | 16% | 24% | 40% | 14% | 61.7 |
| room size | s2_sft | 9% | 16% | 27% | 33% | 14% | 59.0 |
| room size | base_hint_2b | 28% | 14% | 20% | 27% | 11% | 46.3 |
| count | base_hint | 9% | 29% | 19% | 3% | 40% | 54.7 |
| count | s1_real | 11% | 13% | 22% | 1% | 54% | 65.4 |
| count | s2_real | 10% | 12% | 21% | 1% | 55% | 67.4 |
| count | s2_sft | 7% | 13% | 22% | 2% | 56% | 68.9 |
| count | base_hint_2b | 11% | 29% | 16% | 1% | 42% | 54.8 |

## (b) Where does s2_real lose to s2_sft?

Paired video-level bootstrap of s2_real - s2_sft (same prompt: plain, no hint) inside slices. Per-type cells are point deltas; `*` = 95% interval excludes 0 (per-type intervals are in section 1).

#### By source dataset

| slice | videos/q | s2_real | s2_sft | Δ overall [95%] | Δ appear | Δ abs dist | Δ count | Δ rel dist | Δ obj size | Δ room size | Δ route | Δ rel dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 288/5130 | 60.7 | 62.3 | -1.6 [-3.1, -0.1] * | -5.3 * | -2.5 | -1.5 | -5.9 * | -1.3 | +2.6 * | +4.1 | -3.1 |
| arkitscenes | 150/1601 | 60.2 | 63.1 | -2.9 [-5.4, -0.3] * | - | -1.3 | +2.5 | -13.2 * | -2.3 * | +0.4 | -2.9 | -3.3 |
| scannet | 88/2071 | 63.3 | 61.8 | +1.5 [-0.9, +3.8] | -6.7 * | -4.3 | -3.0 | -1.5 | +0.6 | +7.2 * | +23.4 * | -3.5 |
| scannetpp | 50/1458 | 55.4 | 59.5 | -4.0 [-7.7, -0.7] * | -3.1 | -1.1 | -3.2 | -5.6 | -2.5 | +1.4 | -15.4 | -2.6 |

#### By questions per video (tertiles)

Cuts at 6 / 19 questions per video.

| slice | videos/q | s2_real | s2_sft | Δ overall [95%] | Δ appear | Δ abs dist | Δ count | Δ rel dist | Δ obj size | Δ room size | Δ route | Δ rel dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions/video T1 | 103/467 | 58.9 | 59.2 | -0.3 [-3.9, +3.4] | - | -1.9 | -3.0 | - | -0.9 | +1.4 | +3.1 | - |
| questions/video T2 | 89/1092 | 65.2 | 65.3 | -0.0 [-3.0, +3.0] | +0.0 | -0.8 | +2.2 | -5.8 | -2.9 * | +5.7 * | +8.9 | -7.7 |
| questions/video T3 | 96/3571 | 60.2 | 62.3 | -2.1 [-4.2, -0.2] * | -5.4 * | -3.2 | -2.8 | -5.9 * | -0.5 | +1.1 | +1.4 | -1.5 |

#### By video length (frame-count tertiles, from the local mp4 metadata)

Cuts at 1771 / 3025 frames (video_meta.json, read with PyAV; the harness always samples 32 uniform keyframes, so a longer video = sparser keyframes).

| slice | videos/q | s2_real | s2_sft | Δ overall [95%] | Δ appear | Δ abs dist | Δ count | Δ rel dist | Δ obj size | Δ room size | Δ route | Δ rel dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames T1 | 96/1205 | 63.7 | 64.0 | -0.4 [-6.5, +2.6] | -7.0 | -2.9 | +1.1 | -2.4 | -2.0 | +3.1 | +13.2 * | -6.3 |
| frames T2 | 96/1596 | 62.9 | 65.4 | -2.5 [-5.0, +0.1] | -7.5 | -3.8 | -3.0 | -11.9 * | -0.1 | +3.2 | +4.3 | -1.5 |
| frames T3 | 96/2329 | 56.0 | 58.7 | -2.7 [-5.4, -0.1] * | -4.1 | -1.1 | -1.9 | -4.4 | -1.7 | +1.6 | -7.0 | -2.6 |

#### By VSI-Bench-Debiased v1 membership

`pruned_ids.txt` = the 2,768 questions the Debiased v1 pilot removed as answerable through non-visual shortcuts; the other 2,362 are the debiased set.

| slice | videos/q | s2_real | s2_sft | Δ overall [95%] | Δ appear | Δ abs dist | Δ count | Δ rel dist | Δ obj size | Δ room size | Δ route | Δ rel dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| debiased (kept) | 274/2362 | 51.0 | 54.8 | -3.8 [-6.0, -1.6] * | -3.8 | -5.7 * | -7.6 * | -6.8 | -2.7 * | +1.7 | +1.8 | -7.2 |
| pruned (shortcut-prone) | 282/2768 | 67.8 | 68.1 | -0.3 [-2.8, +2.1] | -7.0 * | +1.0 | +3.4 * | -5.2 | -0.4 | +4.8 | +7.5 | -6.1 |

#### Do the slices differ? (bootstrap of the difference of two slice deltas, `s7_interaction.py`)

| comparison | slice X - slice Y | Δ(X) - Δ(Y) |
|---|---|---|
| s2_real - s2_sft | scannet - (arkitscenes + scannetpp) | +4.4 [+1.2, +7.8] * |
| s2_real - s2_sft | same, route questions excluded | +0.9 [-1.8, +3.7] |
| s2_real - s2_sft | longest third - shortest third of videos (frames) | -2.3 [-6.3, +4.4] |
| s2_real - s2_sft | debiased - pruned questions | -3.5 [-6.6, -0.4] * |
| s2_sft_hint - base_hint | debiased - pruned questions | -2.3 [-6.3, +1.4] |
| s2_real - base_hint | debiased - pruned questions | -5.9 [-10.3, -1.4] * |
| s2_real - s2_sft | M2 60-video subset - the other 228 videos | -3.3 [-7.5, +1.1] |

The dataset interaction disappears once the 194 route questions are excluded (route is +23.4 on 64 scannet questions and -15.4 on 26 scannetpp ones). Seven tests, uncorrected.

#### Per-video spread of s2_real - s2_sft (unweighted mean of per-question score differences)

Videos with Δ<0: 152, Δ=0: 22, Δ>0: 114 of 288; median -0.6, mean -1.4 points. Most negative 10 videos: arkitscenes/47115469 (-39, 7q), arkitscenes/47430007 (-38, 14q), scannetpp/13c3e046d7 (-36, 5q), scannet/scene0629_01 (-32, 18q), arkitscenes/47331262 (-32, 6q), arkitscenes/47334380 (-31, 17q), arkitscenes/45261637 (-30, 5q), scannetpp/7b6477cb95 (-29, 48q), arkitscenes/42899720 (-28, 4q), arkitscenes/47331654 (-27, 18q).

Sum of per-question differences over all videos: -126.7 score-units (questions); the worst 10% of videos (28) contribute -149.0.

## Debiased split for every run

Overall on the VSI-Bench-Debiased v1 questions (kept) vs the pruned (shortcut-prone) questions, same lmms-eval aggregation. Then paired deltas on the debiased set for the key comparisons.

| run | full | debiased (kept) | pruned | pruned - kept |
|---|---|---|---|---|
| base_hint | 52.2 | 45.5 | 56.5 | +11.0 |
| base_routed | 54.4 | 48.2 | 57.9 | +9.7 |
| s1_real | 52.9 | 43.9 | 59.6 | +15.6 |
| s2_real | 60.7 | 51.0 | 67.8 | +16.9 |
| s2_real_routed | 61.1 | 51.9 | 67.3 | +15.5 |
| s2_sft | 62.3 | 54.8 | 68.1 | +13.3 |
| s2_sft_hint | 62.3 | 54.7 | 68.0 | +13.3 |
| base_hint_2b | 39.3 | 32.4 | 45.4 | +13.0 |
| base_plain_2b_img | 24.5 | 19.1 | 29.8 | +10.6 |
| base_plain_2b_vid | 42.7 | 34.7 | 49.0 | +14.3 |

| comparison | split | Δ overall [95%] | Δ appear | Δ abs dist | Δ count | Δ rel dist | Δ obj size | Δ room size | Δ route | Δ rel dir |
|---|---|---|---|---|---|---|---|---|---|---|
| s2_real - s2_sft | debiased | -3.8 [-6.0, -1.6] * | -3.8 | -5.7 * | -7.6 * | -6.8 | -2.7 * | +1.7 | +1.8 | -7.2 |
| s2_real - s2_sft | pruned | -0.3 [-2.8, +2.1] | -7.0 * | +1.0 | +3.4 * | -5.2 | -0.4 | +4.8 | +7.5 | -6.1 |
| s2_sft_hint - base_hint | debiased | +9.2 [+6.9, +11.4] * | +9.4 * | +11.5 * | +13.1 * | +9.0 * | -2.1 | +10.9 * | +7.0 | +14.7 * |
| s2_sft_hint - base_hint | pruned | +11.5 [+8.6, +14.6] * | +10.0 * | +6.9 * | +16.8 * | +8.5 * | +3.5 * | +8.4 * | +10.0 | +28.0 * |
| s2_real - base_hint | debiased | +5.5 [+3.1, +7.9] * | +5.0 | +6.5 * | +1.8 | +2.3 | -4.7 * | +11.0 * | +12.3 | +9.7 * |
| s2_real - base_hint | pruned | +11.4 [+7.4, +15.2] * | +4.7 | +8.6 * | +21.6 * | +3.5 | +3.0 * | +13.3 * | +15.0 * | +21.2 |
| s2_real - s1_real | debiased | +7.0 [+4.7, +9.4] * | +6.3 * | +9.3 * | +4.2 * | +3.5 | +0.0 | +8.1 * | +10.5 | +14.3 * |
| s2_real - s1_real | pruned | +8.3 [+4.1, +11.9] * | +9.7 * | +5.1 * | +0.3 | +7.5 * | +2.3 * | +7.0 | +11.2 | +23.1 |
| base_routed - base_hint | debiased | +2.7 [+1.9, +3.4] * | +0.0 | +3.6 * | +0.0 | +0.0 | -0.0 | +17.8 * | +0.0 | +0.0 |
| base_routed - base_hint | pruned | +1.4 [+0.0, +2.7] * | +0.0 | +0.5 | +0.0 | +0.0 | +0.0 | +10.7 * | +0.0 | +0.0 |
| s2_real_routed - base_routed | debiased | +3.7 [+1.6, +5.9] * | +6.3 * | +3.8 * | +2.4 | +0.3 | -4.6 * | -0.3 | +13.2 | +8.5 * |
| s2_real_routed - base_routed | pruned | +9.5 [+5.8, +13.1] * | +5.0 | +4.7 * | +21.4 * | +3.5 | +2.7 * | +1.3 | +17.5 * | +19.6 |

## (c) Per-dataset breakdown of every run

lmms-eval aggregation inside each dataset (types present in that dataset only; rel dir = mean of its difficulties). Question counts per dataset and type first.

| dataset | videos | questions | appear | abs dist | count | rel dist | obj size | room size | route | rel dir |
|---|---|---|---|---|---|---|---|---|---|---|
| arkitscenes | 150 | 1601 | 0 | 277 | 166 | 167 | 363 | 150 | 104 | 374 |
| scannet | 88 | 2071 | 390 | 347 | 223 | 259 | 357 | 88 | 64 | 343 |
| scannetpp | 50 | 1458 | 228 | 210 | 176 | 284 | 233 | 50 | 26 | 251 |

### arkitscenes

| run | prompt | overall | appear | abs dist | count | rel dist | obj size | room size | route | rel dir |
|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | hint | **52.3** | - | 37.1 | 54.3 | 54.5 | 70.8 | 55.2 | 39.4 | 54.8 |
| base_routed | hint+facts | **53.8** | - | 38.9 | 54.3 | 54.5 | 70.8 | 63.5 | 39.4 | 54.8 |
| s1_real | plain | **55.0** | - | 38.7 | 69.3 | 53.3 | 70.1 | 59.8 | 44.2 | 49.3 |
| s2_real | plain | **60.2** | - | 46.3 | 72.1 | 56.9 | 70.6 | 64.0 | 44.2 | 67.4 |
| s2_real_routed | hint+facts | **59.6** | - | 42.3 | 71.6 | 55.1 | 70.9 | 64.0 | 47.1 | 65.8 |
| s2_sft | plain | **63.1** | - | 47.6 | 69.6 | 70.1 | 72.9 | 63.6 | 47.1 | 70.7 |
| s2_sft_hint | hint | **63.3** | - | 45.6 | 71.1 | 70.7 | 73.0 | 63.9 | 48.1 | 71.0 |
| base_hint_2b | hint | **46.9** | - | 33.1 | 53.0 | 47.3 | 66.7 | 53.1 | 38.5 | 36.8 |
| base_plain_2b_img | plain | **31.7** | - | 33.3 | 54.5 | 12.6 | 62.8 | 41.9 | 11.5 | 5.5 |
| base_plain_2b_vid | plain (video mode) | **47.5** | - | 36.1 | 46.1 | 44.9 | 68.5 | 49.1 | 41.3 | 46.6 |

### scannet

| run | prompt | overall | appear | abs dist | count | rel dist | obj size | room size | route | rel dir |
|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | hint | **51.9** | 65.1 | 40.4 | 59.4 | 61.4 | 61.0 | 45.2 | 25.0 | 58.0 |
| base_routed | hint+facts | **55.4** | 65.1 | 42.4 | 59.4 | 61.4 | 61.0 | 70.6 | 25.0 | 58.0 |
| s1_real | plain | **51.5** | 59.7 | 40.7 | 68.0 | 57.1 | 59.0 | 49.8 | 26.6 | 50.9 |
| s2_real | plain | **63.3** | 67.4 | 48.6 | 69.0 | 64.5 | 61.5 | 61.7 | 56.2 | 77.2 |
| s2_real_routed | hint+facts | **64.4** | 67.7 | 48.8 | 69.8 | 63.3 | 61.3 | 70.7 | 56.2 | 77.4 |
| s2_sft | plain | **61.8** | 74.1 | 52.9 | 72.1 | 66.0 | 61.0 | 54.5 | 32.8 | 80.7 |
| s2_sft_hint | hint | **61.5** | 73.6 | 52.5 | 71.6 | 66.8 | 61.2 | 57.0 | 29.7 | 79.7 |
| base_hint_2b | hint | **36.2** | 12.8 | 34.4 | 60.7 | 37.8 | 53.6 | 44.0 | 18.8 | 27.8 |
| base_plain_2b_img | plain | **23.1** | 0.3 | 36.1 | 61.3 | 3.5 | 47.3 | 32.4 | 1.6 | 2.3 |
| base_plain_2b_vid | plain (video mode) | **39.8** | 43.3 | 34.3 | 53.7 | 45.6 | 50.5 | 35.9 | 14.1 | 41.0 |

### scannetpp

| run | prompt | overall | appear | abs dist | count | rel dist | obj size | room size | route | rel dir |
|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | hint | **48.3** | 57.9 | 35.1 | 49.0 | 58.1 | 66.3 | 42.8 | 26.9 | 50.7 |
| base_routed | hint+facts | **51.2** | 57.9 | 37.7 | 49.0 | 58.1 | 66.2 | 63.2 | 26.9 | 50.7 |
| s1_real | plain | **48.5** | 58.8 | 33.1 | 58.4 | 55.6 | 65.0 | 43.4 | 23.1 | 50.8 |
| s2_real | plain | **55.4** | 67.1 | 39.2 | 61.0 | 61.3 | 66.2 | 54.6 | 30.8 | 63.4 |
| s2_real_routed | hint+facts | **56.4** | 68.9 | 39.5 | 61.0 | 61.3 | 65.7 | 62.8 | 30.8 | 61.3 |
| s2_sft | plain | **59.5** | 70.2 | 40.3 | 64.3 | 66.9 | 68.8 | 53.2 | 46.2 | 66.0 |
| s2_sft_hint | hint | **59.3** | 69.7 | 40.8 | 66.4 | 65.5 | 68.2 | 54.0 | 42.3 | 67.6 |
| base_hint_2b | hint | **35.6** | 18.0 | 27.6 | 49.1 | 46.1 | 52.8 | 30.2 | 19.2 | 41.9 |
| base_plain_2b_img | plain | **20.4** | 0.0 | 28.2 | 49.4 | 1.4 | 49.3 | 21.0 | 3.8 | 10.2 |
| base_plain_2b_vid | plain (video mode) | **38.2** | 43.0 | 33.0 | 36.4 | 41.9 | 54.2 | 30.2 | 23.1 | 43.8 |

## (d) Answer agreement

Answers normalized the way lmms-eval scores them (option letter via `fuzzy_matching`, first number via `extract_number`). For multiple-choice types, when the answers differ: `A only` = A right and B wrong, `B only` = the reverse, `both wrong`. For numeric types: `A better` / `B better` = higher MRA, `tie` = same MRA with different numbers. `net` = sum of score differences (A - B) over the changed answers.

### s2_real (A) vs s2_sft (B), per type

| type | n | same answer | differ | A only / A better | B only / B better | both wrong / tie | net (A-B) |
|---|---|---|---|---|---|---|---|
| appear | 618 | 80.9% | 118 | 34 | 67 | 17 | -33.0 |
| abs dist | 834 | 24.7% | 628 | 227 | 278 | 123 | -20.8 |
| count | 565 | 76.6% | 132 | 38 | 73 | 21 | -8.4 |
| rel dist | 710 | 72.5% | 195 | 65 | 107 | 23 | -42.0 |
| obj size | 953 | 40.2% | 570 | 160 | 194 | 216 | -12.1 |
| room size | 288 | 42.0% | 167 | 71 | 69 | 27 | +7.6 |
| route | 194 | 55.7% | 86 | 34 | 26 | 26 | +8.0 |
| rel dir | 968 | 76.4% | 228 | 91 | 117 | 20 | -26.0 |

### s2_sft_hint (A) vs s2_sft (B), per type

| type | n | same answer | differ | A only / A better | B only / B better | both wrong / tie | net (A-B) |
|---|---|---|---|---|---|---|---|
| appear | 618 | 97.6% | 15 | 5 | 8 | 2 | -3.0 |
| abs dist | 834 | 54.8% | 377 | 127 | 158 | 92 | -5.9 |
| count | 565 | 90.1% | 56 | 35 | 18 | 3 | +5.2 |
| rel dist | 710 | 96.9% | 22 | 8 | 9 | 5 | -1.0 |
| obj size | 953 | 75.7% | 232 | 60 | 64 | 108 | +0.1 |
| room size | 288 | 53.5% | 134 | 45 | 32 | 57 | +3.1 |
| route | 194 | 89.2% | 21 | 6 | 8 | 7 | -2.0 |
| rel dir | 968 | 97.1% | 28 | 12 | 11 | 5 | +1.0 |

### Summary of agreement for several pairs

| pair | what differs | n | same (normalized) | same (raw string) | changed | net (A-B) |
|---|---|---|---|---|---|---|
| s2_real vs s2_sft | the geometry path (key) | 5130 | 58.6% | 49.1% | 2124 | -126.7 |
| s2_sft_hint vs s2_sft | one prompt line (calibration) | 5130 | 82.7% | 82.6% | 885 | -2.5 |
| s2_real_routed vs s2_real | hint + routed facts | 5130 | 76.9% | 70.7% | 1185 | -11.1 |
| s2_real vs s1_real | S2 on top of S1 | 5130 | 51.4% | 45.9% | 2492 | +413.6 |
| s2_sft_hint vs base_hint | plain SFT, same prompt | 5130 | 47.3% | 15.4% | 2705 | +525.0 |

## (e) Is the M2 60-video subset representative?

Subset = `eval_vsi_local.pick_videos(docs, 60, 1)`: 60 videos, 1151 questions. Same set as the first 60 of the full run's ordering: True; same as `data/eval/vsibench/holdout_scenes.txt`: True.

Composition (share of questions):

| set | questions | arkit | scannet | scannetpp | appear | abs dist | count | rel dist | obj size | room size | route | rel dir | pruned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 5130 | 31% | 40% | 28% | 12.0% | 16.3% | 11.0% | 13.8% | 18.6% | 5.6% | 3.8% | 18.9% | 54% |
| subset | 1151 | 22% | 39% | 40% | 12.9% | 16.3% | 10.9% | 14.9% | 18.9% | 5.2% | 3.2% | 17.7% | 54% |
| complement | 3979 | 34% | 41% | 25% | 11.8% | 16.2% | 11.1% | 13.5% | 18.5% | 5.7% | 3.9% | 19.2% | 54% |

Server runs scored on the subset and on its complement (228 videos):

| run | full (288 v) | subset (60 v) | complement (228 v) | subset - full |
|---|---|---|---|---|
| base_hint | 52.2 | 53.2 | 51.8 | +1.0 |
| base_routed | 54.4 | 55.3 | 54.1 | +1.0 |
| s1_real | 52.9 | 53.6 | 52.6 | +0.8 |
| s2_real | 60.7 | 61.8 | 60.4 | +1.1 |
| s2_real_routed | 61.1 | 62.5 | 60.8 | +1.4 |
| s2_sft | 62.3 | 65.9 | 61.3 | +3.6 |
| s2_sft_hint | 62.3 | 64.9 | 61.5 | +2.6 |
| base_hint_2b | 39.3 | 41.7 | 38.7 | +2.5 |
| base_plain_2b_img | 24.5 | 23.4 | 24.8 | -1.1 |
| base_plain_2b_vid | 42.7 | 44.6 | 42.2 | +1.9 |

Paired deltas (video bootstrap) on the full set, the subset and the complement:

| comparison | full 288 | subset 60 | complement 228 |
|---|---|---|---|
| s2_real - s2_sft | -1.6 [-3.1, -0.1] * | -4.2 [-8.3, -0.2] * | -0.9 [-2.5, +0.7] |
| s2_real - base_hint | +8.5 [+6.7, +10.3] * | +8.6 [+4.4, +12.5] * | +8.6 [+6.6, +10.5] * |
| s2_sft_hint - base_hint | +10.1 [+8.4, +11.7] * | +11.7 [+8.3, +14.8] * | +9.7 [+7.8, +11.6] * |
| s1_real - base_hint | +0.7 [-0.7, +2.1] | +0.5 [-2.8, +3.8] | +0.8 [-0.8, +2.4] |
| base_routed - base_hint | +2.2 [+1.6, +2.9] * | +2.2 [+0.8, +3.5] * | +2.2 [+1.5, +3.0] * |
| s2_real_routed - s2_real | +0.4 [-0.4, +1.4] | +0.7 [-1.2, +2.8] | +0.4 [-0.6, +1.3] |
| s2_sft_hint - s2_sft | -0.0 [-0.7, +0.6] | -1.1 [-2.5, +0.2] | +0.2 [-0.4, +0.9] |
| base_hint_2b - base_plain_2b_vid | -3.4 [-5.1, -1.8] * | -2.8 [-7.0, +0.9] | -3.5 [-5.4, -1.8] * |

### Cross-check: the local M2 runs (Mac) against the server runs on the same 1,151 questions

Local files re-scored with lmms-eval (`vsi_compare.rescore`). Same configuration flags; different hardware / software stack (Mac MPS vs server CUDA) and, for M2, a local model copy.

| config | local file | local (M2) | server run | server on same q | server - local | 95% (video bootstrap) | same answer | questions |
|---|---|---|---|---|---|---|---|---|
| 4B + format hint | hint_4b_oracle-image.json | 53.7 | base_hint | 53.2 | -0.5 | [-1.6, +0.7] | 88.1% | 1151 |
| 4B routed facts (I-29b = ROUTED config) | i29b_4b_oracle-image.json | 56.0 | base_routed | 55.3 | -0.6 | [-1.8, +0.6] | 86.5% | 1151 |
| 2B + format hint | hint_2b_oracle-image.json | 40.3 | base_hint_2b | 41.7 | +1.4 | [+0.2, +2.8] | 88.2% | 1151 |

## Format compliance: why `s1_real - base_hint` is not the value of S1

Format failure = lmms-eval cannot parse the answer (first word not A-D / no number). Computed by `s4_agree_subset.format_section`.

| run | prompt | format fail | appear | abs dist | count | rel dist | obj size | room size | route | rel dir |
|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | hint | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| base_routed | hint+facts | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| s1_real | plain | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| s2_real | plain | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| s2_real_routed | hint+facts | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| s2_sft | plain | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| s2_sft_hint | hint | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| base_hint_2b | hint | 12.5% | 48% | 0% | 0% | 10% | 0% | 0% | 21% | 24% |
| base_plain_2b_img | plain | 45.8% | 99% | 0% | 0% | 93% | 0% | 15% | 79% | 90% |
| base_plain_2b_vid | plain (video mode) | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| M2 local 4B, plain image mode (q4b_60v_image, 60 v) | plain | 51.0% | 100% | 0% | 2% | 100% | 0% | 38% | 100% | 100% |

The plain 4B base in image mode (M2, same 60 videos, Mac) scores 24.7 with 51% unparseable answers (all multiple-choice types explain instead of answering). `s1_real` (plain prompt, projector only, LLM frozen) scores 53.6 on the same questions with 0% failures: +28.9 [+25.9, +31.8] * - a cross-stack comparison (Mac vs server; the stack effect measured in the cross-check above is within about ±1.5 points), but it shows S1's first effect is format compliance, i.e. the same thing the one-line hint does. Against the hint base, S1 is +0.7 [-0.7, +2.1] overall.

## Routed measurements: copy rate and the value of the number itself

Only the questions that received an attached measurement in the routed runs (room size: all 288; abs distance: the 226 whose two objects were seen in >= 3 keyframes). `facts verbatim` = MRA of answering with the attached number. `copy` = share of answers equal to the attached number (|diff| < 0.051). Computed by `s5_facts.py`.

| type | questions | facts verbatim (MRA) | run | prompt | run MRA on these q | copy |
|---|---|---|---|---|---|---|
| room_size_estimation | 288 | 65.6 | base_hint | hint | 50.0 | 6% |
| room_size_estimation | 288 | 65.6 | base_routed | hint+facts | 65.6 | 97% |
| room_size_estimation | 288 | 65.6 | s2_real | plain | 61.7 | 8% |
| room_size_estimation | 288 | 65.6 | s2_real_routed | hint+facts | 65.8 | 71% |
| room_size_estimation | 288 | 65.6 | s2_sft | plain | 59.0 | 7% |
| room_size_estimation | 288 | 65.6 | s2_sft_hint | hint | 60.1 | 7% |
| object_abs_distance | 226 | 46.1 | base_hint | hint | 38.2 | 4% |
| object_abs_distance | 226 | 46.1 | base_routed | hint+facts | 46.1 | 100% |
| object_abs_distance | 226 | 46.1 | s2_real | plain | 48.6 | 4% |
| object_abs_distance | 226 | 46.1 | s2_real_routed | hint+facts | 46.1 | 100% |
| object_abs_distance | 226 | 46.1 | s2_sft | plain | 54.4 | 3% |
| object_abs_distance | 226 | 46.1 | s2_sft_hint | hint | 53.5 | 4% |

## Logged measurements vs the trained models, on the questions each measurement covers

`base_routed.json` logs, for every question, what the CUT3R + OWLv2 object map would answer at each view-count threshold k (min keyframes an object was seen in). Coverage = questions with a measured answer at that k. The model columns are each run's score on exactly those covered questions (accuracy or MRA, plain mean). Computed by `s6_measured.py`. `order`, `count`, `direction`, `reldist`, `route` were never attached in these runs; the next run adds direction facts at k=1 (I-33).

| kind | k views | coverage | measured | base_hint | s1_real | s2_real | s2_sft | s2_sft_hint |
|---|---|---|---|---|---|---|---|---|
| direction | 1 | 686/968 | 71.6 | 54.8 | 50.9 | 71.9 | 72.4 | 72.4 |
| direction | 2 | 342/968 | 84.8 | 53.8 | 51.8 | 75.1 | 76.3 | 75.4 |
| direction | 3 | 139/968 | 82.7 | 58.3 | 59.7 | 81.3 | 79.1 | 77.7 |
| reldist | 1 | 614/710 | 45.8 | 58.8 | 55.2 | 62.1 | 67.1 | 67.3 |
| reldist | 2 | 379/710 | 55.7 | 65.4 | 59.1 | 70.7 | 71.2 | 72.0 |
| reldist | 3 | 240/710 | 51.7 | 70.4 | 62.5 | 70.4 | 72.5 | 72.9 |
| order | 1 | 101/618 | 84.2 | 81.2 | 75.2 | 83.2 | 90.1 | 88.1 |
| order | 2 | 51/618 | 86.3 | 72.5 | 70.6 | 84.3 | 92.2 | 90.2 |
| order | 3 | 14/618 | 92.9 | 71.4 | 85.7 | 92.9 | 92.9 | 92.9 |
| route | 1 | 29/182 | 65.5 | 44.8 | 48.3 | 44.8 | 51.7 | 55.2 |
| route | 2 | 19/182 | 63.2 | 47.4 | 52.6 | 47.4 | 57.9 | 52.6 |
| route | 3 | 11/182 | 63.6 | 54.5 | 54.5 | 54.5 | 63.6 | 54.5 |
| count | 1 | 509/565 | 32.4 | 56.5 | 66.1 | 67.8 | 68.9 | 70.0 |
| count | 2 | 455/565 | 41.4 | 57.8 | 65.1 | 66.8 | 68.2 | 69.4 |
| count | 3 | 393/565 | 37.6 | 59.3 | 64.8 | 66.6 | 67.8 | 69.6 |
| abs | 1 | 647/834 | 31.3 | 38.9 | 38.8 | 45.8 | 49.5 | 48.6 |
| abs | 2 | 387/834 | 44.8 | 41.4 | 42.6 | 48.2 | 52.3 | 50.5 |
| abs | 3 | 226/834 | 46.1 | 38.2 | 41.3 | 48.6 | 54.4 | 53.5 |

