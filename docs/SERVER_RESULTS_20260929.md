# Server results, weekend run 2026-09-26..29 (Sensenova-only baseline): analysis

Scope: the only server outputs we have, the 2026-09-29 09:45 KST snapshot: `REPORT.md` (generated 00:31 UTC),
9 VSI prediction files (`vsi/*.json`, all 288 videos / 5,130 questions) and the S1 `final.pt` of six arms. The plan
and the pre-registered reading rules are in `docs/SERVER_WEEKEND.md`. The lead's M2 holdout ablation of the S1
checkpoints (`ablation_s1_local.md`) is included.

How to read this document:
- `A/` = `outputs/server_0929/analysis/` on the M2 (not in git: the scratch scripts and outputs). The seven analysis
  write-ups are copied to `docs/server_0929/` (vsi_types, facts_after_training, checkpoints, throughput,
  missing_results, latency, ablation_s1_local). Every number is from a script output in `A/` or from `REPORT.md`; the
  file is cited next to it. The headline paired comparisons were recomputed once more for this document with
  `scripts/weekend_report.py:paired_bootstrap` (`A/doc_check/headline.py` -> `A/doc_check/headline.out`).
- Each analyst finding was checked by two independent recomputations. Only findings that survived are used; where
  a check corrected a finding, the corrected version is used; where the two checks disagree, both numbers are given.
  Findings that no one re-checked are collected in the appendix and marked **[U]** where they are mentioned.
- Intervals are 95 % paired bootstraps over videos (288), which cover question sampling only. **No interval in this
  document includes training-seed variance**: every arm has one evaluated seed.
- Times are UTC (server clock). KST = UTC + 9.
- "Data shows" and "interpretation" are kept apart; "cannot be determined" names the file that would settle it.
- A later critic pass re-derived over 250 numbers independently and corrected a few; see **Verification notes** at the end.

---

## 1. TL;DR

1. **Every arm trained on about 2 % of an epoch.** S1 = 106 steps x 128 = 13,568 samples (1.63 % of 830,135),
   S2 = 136 steps = 17,408 samples (2.10 %); 2B 57 / 71 steps (0.88 % / 1.09 %). The step budget was sized from the
   smokes' `SMOKE_SPS` 0.30 samples/s (26.7 GPU-s/sample; the smoke's own logged window reads 26.2); the arms ran at
   15.5-16.4 GPU-s/sample (0.49-0.52 samples/s per 8-GPU node), against the plan's ~0.42. Every result below is a
   "2 % of an epoch" result.
2. **Sensenova LoRA SFT is worth about +10 VSI points even at 17.4k samples.** Same prompt: `s2_sft_hint - base_hint`
   = **+10.1 [+8.4, +11.7]** (62.3 vs 52.2); +9.2 [+6.9, +11.5] on the VSI-Debiased questions. After SFT the format
   hint changes nothing (-0.0 [-0.7, +0.6]). One seed, local harness only. About 1.9 of the 10.1 points is the
   counting floor (§3.5): against the base with its 0/1 counting answers raised to 2, SFT gains +8.3 [+6.6, +9.8].
3. **The geometry path has not paid for itself yet.** Same prompt: `s2_real - s2_sft` = **-1.6 [-3.1, -0.1]**
   (p about 0.04); -2.4 [-3.8, -1.1] without route; -3.8 [-6.0, -1.7] on the Debiased questions vs -0.3 [-3.0, +2.2]
   on the pruned ones. No type shows a significant gain (room size +2.6 [-0.07, +5.5]). No single-type loss survives
   a Holm correction.
4. **The key number (rule 1, `s2_real - s2_control`) does not exist.** `s2_control`, both seed-1 VSI runs and the S2
   holdout ablation were never produced. So these materials cannot say whether the trained model uses the geometry's
   content.
5. **S1 checkpoints (holdout loss on the M2, and weight forensics):** the injection path is worth ~2.1 nats for real
   *and* control (it carries answer format and priors). The real projector's loss depends on the geometry's content
   (+0.037 ± 0.031) and the control's does not (-0.006 ± 0.012). Against the control, the value of content is
   **-0.026 ± 0.041**, not distinguishable from 0 (seed noise ± 0.021). In function space only 3-7 % of the
   real-minus-control difference reproduces across seeds.
6. **Training shifted numeric answer priors.** The geometry arms answer "2" on 80 % of counting questions (truth 53 %).
   On counts other than 2, the S2 real weights are **-10.2 [-13.5, -6.8]** below the base under the same prompt
   (`s2_real_routed`: hint sentence, nothing attached to counting). The entire counting gain of every trained arm is
   explained by moving the base's 0/1 answers to 2. The extra shift to "2" that only the geometry arms show is a net
   cost in the point estimate (counting vs plain SFT -1.5 [-3.8, +0.9], n.s.); it is significant on gt ≠ 2
   (-9.5 [-13.1, -5.8]) and on Debiased counting.
7. **Routed facts after training:** still copied (room 97 %, distance 100 %), whether or not the fact is better.
   The routing gain fell from +2.2 [+1.6, +2.9] (base) to +0.4 [-0.4, +1.4] (S2 real) because the model caught up:
   room fact 65.6 vs S2 real 61.7; distance fact 46.1 vs S2 real 48.6 vs plain SFT 54.4. In-sample, averaging the
   fact with the model's own answer is worth +1.15 [+0.74, +1.56] overall, almost all from room size.
8. **Throughput root cause (not yet proven):** the image path costs 4.0-5.2 ms per vision patch row on the server,
   about 5x a Mac M2 on the same prefill. Prime suspect: the bf16 `Conv3d` patch embed on a slow non-cuDNN kernel.
   Second suspect: the Gated DeltaNet running transformers' pure-torch fallback (fla / causal-conv1d are not in
   `requirements.txt`; whether they are on the server is unknown). One version print and one profile settle both.
9. **Gate 3 passes on its literal criteria** (question-only TTFT 164 ms, drift 1.0). But the full TTFT is 61.6 s:
   one prefix build takes 61.4 s, longer than the gap between keyframe-set changes for the first ~34 min, so the
   background-prefix design is not demonstrated. **Gates 1-2 have no data**: lmms-eval swallowed an exception and
   exited 0, so the three `gate_*.done` markers are false.
10. **Next:** fix throughput first; the target is ≤0.83 GPU-s/sample per node (~19x) for one epoch per arm per day.
    Recover the missing pre-registered evaluations now (~36-48 GPU-h, by hand, no 6 h timeout). Only then run the
    matrix, at ≥1 epoch, with ≥2 seeds per key arm, step checkpoints and the probe.

---

## 2. What was actually trained: the single most important caveat

### 2.1 Data budget per arm

Source: `REPORT.md` training table; rates recomputed from its `elapsed` column (first to last logged step, printed
to 0.1 h) in `A/skeptic_tp_own_cost/rates.txt`; budget arithmetic in `A/tp_budget_check.txt`.

| arm | GPUs x accum | steps (last logged) | samples | % of 830,135 | logged window | samples/s | GPU-s / sample | train loss first -> last logged |
|---|---|---|---|---|---|---|---|---|
| 4B smoke c/d (sizing run) | 8 x 2 | 60 (60) | 880 in window | - | 0.8 h | 0.306 | 26.2 | - |
| s1_real / s1_control | 8 x 16 | 106 (100) | 13,568 | 1.63 % | 5.6 / 5.7 h | 0.508 / 0.499 | 15.75 / 16.03 | 0.935 -> 0.560 / 0.942 -> 0.579 |
| s1_real_seed1 / s1_control_seed1 | 8 x 16 | 106 (100) | 13,568 | 1.63 % | 5.6 h | 0.508 | 15.75 | 1.005 -> 0.540 / 1.027 -> 0.567 |
| s2_real / s2_control | 8 x 16 | 136 (120) | 17,408 | 2.10 % | 7.1 / 7.2 h | 0.501 / 0.494 | 15.97 / 16.20 | 0.511 -> 0.369 / 0.548 -> 0.379 |
| s2_real_seed1 / s2_control_seed1 | 8 x 16 | 136 (120) | 17,408 | 2.10 % | 6.9 / 7.1 h | 0.515 / 0.501 | 15.53 / 15.97 | 0.551 -> 0.371 / 0.569 -> 0.379 |
| s2_sft (no geometry) / s2_joint (no S1) | 8 x 16 | 136 (120) | 17,408 | 2.10 % | 7.3 / 7.1 h | 0.487 / 0.501 | 16.43 / 15.97 | 0.797 -> 0.365 / 0.805 -> 0.372 |
| s1_real_2b / s1_control_2b | 4 x 32 | 57 (40) | 7,296 | 0.88 % | 2.3 / 2.4 h | 0.309 / 0.296 | 12.94 / 13.50 | 0.927 -> 0.600 / 0.933 -> 0.625 |
| s2_real_2b / s2_control_2b | 4 x 32 | 71 (60) | 9,088 | 1.09 % | 4.7 h | 0.303 | 13.22 | 0.587 -> 0.533 / 0.571 -> 0.534 |

- The final losses are single logged windows with no uncertainty. The real/control pairs are nearly equal
  (0.369 vs 0.379, 0.371 vs 0.379, 2B 0.533 vs 0.534). **This is not rule 1 and must not be read as it.**
- Whether each arm ran every planned step is not readable from the materials; the REPORT shows the last *logged*
  step. Steps after the last logged one carry 0.04 % (S1) and 0.30 % (S2) of the lr mass; for the 2B, 4.5 % (S1,
  last logged step 40 of 57) and 0.7 % (S2), so the 2B S1 "last" loss is the least representative
  (`A/doc_check/critic_ckpt.out`, from `train.py:cosine_with_warmup`, warmup 3 %).
- The REPORT's `skipped` column counts rank-0 dataset retries (samples that failed to build and were replaced), not
  non-finite losses. It is equal within every real/control pair (5/5, 6/6, 8/8), so it does not break pairing.
  `sync OK` in every multi-GPU arm: the DDP replicas stayed identical.
- "~15k samples" in some notes is 120 x 128. The run was planned for, and saved `final.pt` after, 136 x 128 = 17,408.

### 2.2 How 2 % of an epoch happened (verified, with corrections)

| fact | value | source |
|---|---|---|
| `plan()`: steps = min(epoch, hours x 3600 x smoke_sps x frac) // 128 | smoke_sps = 0.300 exactly (mean of six `%.1f` log values) -> 106 / 136 / 57 / 71 steps | `A/skeptic_tp_accum/check.txt` |
| Planned speed in `SERVER_WEEKEND.md` | ~19 samples/s = 0.42 GPU-s/sample | doc |
| Measured, 4B arms | 15.5-16.4 GPU-s/sample (mean 15.94) | `A/skeptic_tp_own_cost/rates.txt` |
| 4B arms vs the smoke, per GPU | 1.62-1.72x faster | `A/skeptic_tp_gradaccum/recompute.txt` |
| 4B arms vs the plan's derated 0.27 samples/s | 1.80-1.91x faster -> the budget used **52-55 %** of the 14 h / 18 h caps | same |
| Wall clock from checkpoint mtimes | s1_real -> s1_real_seed1 = 17.16 h for S2 seed 0 + S1 seed 1 (53.6 % of 32 h) | same |
| 2B arms | ~2.0x the smoke per GPU, 2.3-2.4x the plan; 41-43 % of the 16 h / 20 h caps | same |

Why the arms beat the smoke: the smoke ran at grad-accum 2 and the arms at 16 (4B) or 32 (2B). DDP waits for the
slowest rank once per optimizer step. A rank-gating simulation with cost proportional to sample length predicts
smoke/arm ratios of 0.58-0.60 (4B) and 0.48-0.50 (2B); observed are 0.60 and 0.49. The 2B ratio is an independent
check. This is **plausible but not measured**. A fixed ~25 s per-step overhead fits the two ratios too, but nothing
in `train.py` costs that much (`A/skeptic_tp_rankgate_v3/window_noise.txt`).

### 2.3 Where the ~16 GPU-seconds per sample go

The arm logs with the `data / geom / fb ms/샘플` split were not uploaded, so everything here is inferred from
cross-run comparisons, the latency benches and op counts.

| component | finding (verified / corrected) | share of today's cost | source |
|---|---|---|---|
| **Image path** (vision tower on patch rows, plus the LLM prefill of visual tokens) | Server benches: 3.95-3.99 ms per patch row in image mode (deferred prefill, 33 images per forward), 5.13-5.18 ms per row streaming (video mode). Mac M2 MPS, same prefill: 0.75-0.77 ms per row, **5.2x faster**. Mean training sample: 4.24 images, 1,981 rows -> 7.8-10.3 s/sample. | **66-98 %** of the gated per-sample cost (10.3-11.9 s); the like-for-like image-mode rate gives about 74 %. Training itself was never measured. | `A/skeptic_tp_perimage_v3/summary.txt`, `rows.txt` |
| LLM size | After gating correction the 2B and 4B cost the same per sample: 11.0 s [10.3, 11.7] vs 10.9 s [10.4, 11.4], ratio 0.99 [0.95, 1.04]. The two smokes (same config, same data order) both took 0.8 h for 880 samples. | LLM FLOP-proportional share ≤ ~9 % (≤ ~21 % if launch-bound) | `A/skeptic_tp_own_cost/` |
| CUT3R | `s2_sft` (CUT3R never runs) is the *slowest* 4B arm (16.43 GPU-s); the run-to-run spread is ~4 % | ≤ ~5 % | `A/skeptic_tp_sharedcost/check.txt` |
| LoRA | S1 ≈ S2 (15.82 vs 15.92 GPU-s). Not diagnostic: S1 also backpropagates through the whole LLM. | - | same |
| DDP itself | Not established. The 1-GPU smoke_a gives 7.2-21.6 s/sample (rounded to 0.1 h and 0.1 samples/s); its 2B printed 0.2 vs 0.1 samples/s for the 4B, inconclusive. | unknown | same |
| **Gated DeltaNet torch fallback** (if running) | +~67k launching ATen ops per mean S2 micro-step: 80.0k vs 12.9k fp32 (6.2-6.6x depending on the fused stand-in), ~5.4x in bf16, 2.8-5.1x once CUT3R's launches are counted in the geometry arms | ~0.66-0.87 s/sample host time at 9.9-12.9 µs per launch (plausible range 0.3-1.5 s) | `A/skeptic_tp_gdn_indep/analysis.out`, `A/skeptic_tp_fallback_indep/recompute.txt` |
| Slowest-rank gating | efficiency 0.67 at 8 x 16 (admissible cost models 0.65-0.77); 0.37 at 32 x 4 (0.34-0.49). Length-decile bucketing: 0.92 / 0.78. Coarse image-count buckets: 0.83 / 0.57. | ~23-35 % of rank time idle | `A/skeptic_tp_rankgate_v3/gate_sim.txt`, `check.txt` |
| Data loading | Real `build()` on 1 thread: 147.5 ms CPU per sample (p99 1.24 s). The real DataLoader (6 workers) delivered 23.2-26.1 samples/s on the Mac against 0.0625 consumed per rank: **~370-420x headroom**. | ~0 | `A/skeptic_tp_buildcost/`, `A/skeptic_tp_dataload_v3/` |
| Gradient checkpointing | +3,205 launches per training call (constant), 1.30-1.40x in the fused graph | small | `A/skeptic_tp_ckptpack_v4/recompute.log` |

Interpretation: the dominant, LLM-size-independent cost sits in the pixel-input path. Its mechanism is not observed.
The FLOP floor for a mean sample is ~0.02-0.12 GPU-s, plus 0.11-0.15 s of fp32 CUT3R (`A/skeptic_tp_v3/recheck.txt`).
Today's 16 GPU-s are two orders of magnitude above that. For scale: the Mac M2 trains the same 4B S2 on one device
at 8.9 s/sample wall (800 samples, no gating; `A/skeptic_tp_perimage_v3/summary.txt`), about the ~10.7 s own cost of
one H100 here (different sample mix: the local 1k sample). An Apple-silicon GPU matching an H100 per sample is
itself evidence of a broken kernel path on the server, whatever the mechanism.

### 2.4 What an epoch-scale run needs (verified arithmetic; the projections are estimates)

| goal | requirement | vs today | source |
|---|---|---|---|
| 1 epoch (830,135) in 24 h on one 8-H100 node | 9.61 samples/s, **≤0.83 GPU-s/sample** | 18.6-19.7x faster | `A/skeptic_tp_epoch/recompute.txt` |
| same on 4 nodes as one 32-GPU job, effective batch 128 | ≤3.33 GPU-s/sample | today's code at 32 x 4 would cost ~26-29 GPU-s (gating), so **~8-9x**; ~5.5x at effective batch 512; ~4.8x only with length bucketing. No multi-node job path exists in the code. | `A/skeptic_tp_epoch/`, `A/skeptic_tp_v3/recheck.txt` |
| the whole weekend matrix (14 arm-stage epochs) | ≤0.71 GPU-s/sample pooled in 3 days on 32 GPUs; **≤0.62** with integer packing of single-node jobs (≤0.83 in 4 days) | 21-26x | `A/skeptic_tp_v3/recheck.txt` |

Projections, which are **estimates** and not measured:
- Fixing the image path alone gives ~1.4 GPU-s/sample by the bottom-up count. Top-down (own cost minus the measured
  per-row cost) leaves 0.5-3.2 GPU-s.
- Adding fla gives ~0.45-0.6 GPU-s. That meets the one-node target only if the image path really costs ≥ ~4.6 ms
  per row, and only if fla is absent today.

---

## 3. VSI results on the full 5,130 questions

Local harness, live path: 32 uniform keyframes, image mode, greedy, 16 new tokens, lmms-eval scoring and
aggregation (the three rel-dir difficulties averaged first, then the mean of 8 types). Integrity checks passed: every
saved score equals an independent rescore, and all runs cover the same 5,130 ids (`A/s0_sanity.py`,
`A/skeptic_vsitypes_recompute.log`). Decoding is near-deterministic even across eval-code versions and entry points:
`base_routed` and `base_hint` give the same answer on 4,602 of 4,616 questions with nothing attached, and the 14 that
differ sum to -0.1 score points (`A/doc_check/critic_vsi.out`). The two routed runs saw byte-identical facts
(measurement logs, maps, object-map texts and attached strings all equal; `A/doc_check/critic_unused.out`).

### 3.1 Every run

`hint` = the one-line format instruction. The routed runs carry the same sentence, although their `format_hint` flag
is False (`eval_vsi_local.py` appends it whenever routing is on). There is **no un-hinted 4B base run** (`base_plain`
is missing).

| run | weights | prompt | overall | appear | abs dist | count | rel dist | obj size | room size | route | rel dir | Debiased (2,362 q) | pruned (2,768 q) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_hint | 4B base | hint | **52.2** | 62.5 | 37.9 | 54.7 | 58.5 | 66.0 | 50.0 | 33.0 | 54.8 | 45.5 | 56.5 |
| base_routed | 4B base | hint + facts | **54.4** | 62.5 | 40.1 | 54.7 | 58.5 | 66.0 | 65.6 | 33.0 | 54.8 | 48.2 | 57.9 |
| s1_real | S1 real | plain | **52.9** | 59.4 | 38.1 | 65.4 | 55.6 | 64.7 | 53.9 | 35.6 | 50.2 | 43.9 | 59.6 |
| s2_real | S2 real | plain | **60.7** | 67.3 | 45.5 | 67.4 | 61.4 | 66.1 | 61.7 | 46.4 | 69.7 | 51.0 | 67.8 |
| s2_real_routed | S2 real | hint + facts | **61.1** | 68.1 | 44.3 | 67.6 | 60.6 | 66.0 | 65.8 | 47.9 | 68.5 | 51.9 | 67.3 |
| s2_sft | plain SFT | plain | **62.3** | 72.7 | 47.9 | 68.9 | 67.3 | 67.4 | 59.0 | 42.3 | 72.9 | 54.8 | 68.1 |
| s2_sft_hint | plain SFT | hint | **62.3** | 72.2 | 47.2 | 69.8 | 67.2 | 67.4 | 60.1 | 41.2 | 73.0 | 54.7 | 68.0 |
| base_hint_2b | 2B base | hint | **39.3** | 14.7 | 32.3 | 54.8 | 43.4 | 58.4 | 46.3 | 29.4 | 35.1 | 32.4 | 45.4 |
| base_plain_2b | 2B base | plain, image | **24.5** | 0.2 | 33.2 | 55.6 | 4.8 | 53.7 | 35.4 | 7.2 | 5.8 | 19.1 | 29.8 |
| base_plain_2b | 2B base | plain, video | **42.7** | 43.2 | 34.6 | 46.1 | 43.9 | 58.3 | 41.8 | 29.9 | 43.8 | 34.7 | 49.0 |

Sources: `REPORT.md`; Debiased columns from `A/vsi_types.md`, re-derived in `A/doc_check/critic_vsi.out`. Debiased =
the 2,362 ids *not* in `pruned_ids.txt` (= `test_debiased.parquet`); pruned = the 2,768 ids in it
(= `test_pruned.parquet`). Format failures: ≤ 0.04 % for every 4B run (2 answers each in base_hint and base_routed;
REPORT rounds to 0.0 %); 2B 12.5 % (hint, image), 45.8 % (plain, image), 0.0 % (video).

Missing rows: `s2_control`, `s2_real_seed1`, `s2_control_seed1`, `s2_joint`, the 2B trained arms, `s1_control`,
`s2_real_hint`, 4B `base_plain`. See section 7. Never planned, but needed as a same-prompt partner: `s2_sft_routed`
(today the "best geometry system vs plain SFT" row compares a routed run with an unrouted one; §7.3 item 13).

### 3.2 The comparisons that matter

Recomputed for this document (`A/doc_check/headline.out`). `*` = the interval excludes 0.

| question | A - B | prompts | Δ [95 %] |
|---|---|---|---|
| Value of Sensenova SFT (same prompt) | s2_sft_hint - base_hint | both hint | **+10.1 [+8.4, +11.7] *** |
| Value of the whole geometry path (rule 2) | s2_real - s2_sft | both plain | **-1.6 [-3.1, -0.1] *** |
| Training vs one prompt line (rule 6) | s2_real - base_hint | plain vs hint | **+8.5 [+6.7, +10.3] *** |
| Training on top of routed prompting (rule 6) | s2_real_routed - base_routed | both hint + facts | **+6.7 [+5.1, +8.2] *** |
| S2 on top of S1 | s2_real - s1_real | both plain | +7.8 [+6.2, +9.5] * |
| S1 alone vs the hint | s1_real - base_hint | plain vs hint | +0.7 [-0.7, +2.1] |
| Format hint after SFT | s2_sft_hint - s2_sft | tested factor | -0.0 [-0.7, +0.6] |
| Routed facts, base | base_routed - base_hint | facts only | +2.2 [+1.6, +2.9] * |
| Routed facts after S2 real | s2_real_routed - s2_real | facts + hint line | +0.4 [-0.4, +1.4] |
| Best geometry system vs plain SFT | s2_real_routed - s2_sft_hint | A adds facts | -1.2 [-2.7, +0.4] |
| Size, zero-shot | base_hint_2b - base_hint | both hint | -12.9 [-14.7, -10.9] * |
| 2B, hint (image) vs plain video | base_hint_2b - base_plain_2b:video | differ | -3.4 [-5.1, -1.8] * |
| 2B, image vs video, plain | base_plain_2b image - video | input format | -18.2 [-19.7, -16.8] * |

### 3.3 Value of training (verified; two corrections)

- **Like for like.** base_hint (older eval code, `--model`) and s2_sft_hint (`--config`) differ in code path. That
  path is harmless: base_routed goes through the config path and matches base_hint on 99.7 % of the questions with
  nothing attached. The +10.1 is also not format repair, since base + hint already fails format on 0.04 % of
  answers.
- **Counting floor.** Raising base_hint's 0/1 counting answers to 2 (a post-processing, 54.0 overall) leaves SFT at
  +8.3 [+6.6, +9.8], s2_real at +6.7 [+4.9, +8.4] and s1_real at -1.2 [-2.5, +0.2] (`A/doc_check/critic_extra.out`).
  About 1.9 points of every trained arm's gain is this VSI-specific prior (only 1 of 565 true counts is below 2).
- Per type (same prompt): rel dir +18.3, count +15.2, room +10.1, appear +9.7, abs +9.3, rel dist +8.7 (all
  significant); route +8.2 and object size +1.4 are not significant.
- **Is s2_real's +8.5 a same-prompt number?** s2_real_hint failed. Proxy: on the 4,616 questions where the routed run
  attached nothing, its prompt is s2_real's plus the hint sentence, and there the hint changes s2_real by
  -0.04 [-0.79, +0.72]. So +8.5 is effectively same-prompt, apart from the 288 room-size and 226 attached
  abs-distance questions, estimated at ~0.1-0.3 points (`A/skeptic_sft10/hint_s2real.out`).
- **Debiased vs pruned.**
  - SFT's gain is not shortcut-driven: +9.2 Debiased vs +11.5 pruned. The difference is -2.3 [-6.2, +1.5], or
    -1.65 [-4.85, +1.59] with rel dir pooled; neither is significant.
  - s2_real's +8.5 is uneven: +5.3 / +5.5 Debiased vs +11.3 / +11.4 pruned. The difference is -6.1 [-9.5, -2.7]
    (verifier 2, rel dir pooled) or -5.9 [-10.3, -1.4] (verifier 1).
- **Caveats.**
  - One SFT seed.
  - The lmms-eval gate produced no result, so the official-path value is undetermined.
  - The local 905-record Sensenova sample contains VSI-style question templates: 2 exact VSI template strings, and
    ~17 % match a rough regex. Part of the gain may therefore be in-distribution template exposure; its size is not
    measured.
  - Whether Sensenova scenes overlap VSI scenes cannot be checked without the training manifest.

### 3.4 Value of the geometry path, per type (corrected claim)

`s2_real - s2_sft`, same prompt, identical flags except `weights`, 10,000 video resamples. Source:
`A/skeptic_vsitypes_recompute.log`; the second verifier's intervals (`A/skeptic_vt_s2vsft_claim/recompute.out`)
agree to within 0.2. Bootstrap p-values move with the resampling seed: a third recomputation
(`A/doc_check/critic_vsi.out`, seed 0, per-type resampling) gives appearance order p = 0.008 (Holm 0.062) and the
same conclusions.

| type | n | s2_real | s2_sft | Δ [95 %] | raw p | Holm p (8 types) |
|---|---|---|---|---|---|---|
| appearance order | 618 | 67.3 | 72.7 | -5.3 [-9.7, -1.4] | 0.011 | 0.086 |
| rel distance | 710 | 61.4 | 67.3 | -5.9 [-10.6, -1.2] | 0.012 | 0.086 |
| rel direction | 968 | 69.7 | 72.9 | -3.1 [-7.5, +1.1] | 0.15 | 0.46 |
| abs distance | 834 | 45.5 | 47.9 | -2.5 [-5.2, +0.3] | 0.07 | 0.29 |
| counting | 565 | 67.4 | 68.9 | -1.5 [-3.8, +0.9] | 0.21 | 0.46 |
| object size | 953 | 66.1 | 67.4 | -1.3 [-2.5, -0.02] | 0.046 | 0.27 |
| room size | 288 | 61.7 | 59.0 | +2.6 [-0.07, +5.5] | 0.058 | 0.29 |
| route | 194 | 46.4 | 42.3 | +4.1 [-3.9, +12.0] | 0.34 | 0.46 |
| **overall** | 5,130 | 60.7 | 62.3 | **-1.6 [-3.2, -0.1]** (2,000 resamples: [-3.1, -0.1]) | 0.036 (sign-flip permutation 0.042) | - |
| 7 types, no route | | | | -2.4 [-3.8, -1.1] | < 0.001 | - |

What the data shows:
- The overall loss is small and nominally significant. It is spread over 6 of 8 types; appearance order and
  relative distance carry ~87 % of it.
- The two runs give identical raw answers on only 49.1 % of questions (2,518 / 5,130; normalized 58.6 %); the
  hint line on the SFT weights leaves 82.6-82.7 % unchanged. Among the MCQ answers that differ, s2_sft is right more often: appear 34 vs 67 (McNemar p = 0.0013),
  rel dist 65 vs 107 (p = 0.0017).
- **Room size is not a significant gain.** Its lower bound is below 0 in 20 of 20 bootstrap seeds at 10,000
  resamples (`A/skeptic_vsitypes_seeds.log`); at 2,000 resamples it is above 0 in 1 of 10 (`A/skeptic_vt_s2vsft_claim/extra.out`)
  or 3 of 20 seeds (`A/doc_check/critic_vsi.out`). The earlier "[+0.1, +5.5] *" was a seed-0 edge case. Object size
  is the mirror case: its upper bound is below 0 in 17 of 20 seeds, so its "significant" loss is equally fragile.
- **Debiased vs pruned.** Debiased 50.99 vs 54.78 = -3.8 [-6.0, -1.7] (p = 0.0004); pruned -0.3 [-3.0, +2.2];
  interaction -3.5 [-6.6, -0.2] (p about 0.04, borderline).
- **The interaction comes from counting and absolute distance**, not from the headline types.
  - Per-type interaction: count -11.1 [-15.7, -6.6], abs -6.7 [-11.5, -1.8]. Appearance order goes the other way:
    it loses more on the pruned set (-7.0) than on Debiased (-3.8).
  - Mechanism: s2_real answers the modal value "2" more often than s2_sft. On Debiased counting that is 64 % vs 49 %,
    where truth is 2 on only 20 %. On pruned counting it is 93 % vs 86 %, where truth is 2 on 80 %. For absolute
    distance it is 23 % vs 11 % (Debiased) and 26 % vs 12 % (pruned).
- **Per video the verifiers disagree.** 148 negative / 114 positive / 26 zero (verifier 1, who calls 152 / 22 a float
  artefact) vs 152 / 114 / 22 (verifier 2). Both give median -0.63 and mean -1.39.
- **Dataset slices.** The scannet-vs-others dependence (+4.4 [+1.2, +7.8]) comes from route alone: dropping route
  gives +0.9 [-1.9, +3.8]; dropping any other type leaves it at +4.1 to +5.6.
  - Lmms-eval weighting gives 64 scannet route questions 1/8 of scannet's score, and s2_real is an outlier on
    scannet route (56.2 % vs 25-33 % for every other 4B run; exact sign test p = 0.004, post hoc, not surviving
    ~23-cell correction).
  - Without route the datasets are homogeneous (-2.9 / -1.6 / -2.4, homogeneity p = 0.72-0.75), which only rules out
    differences larger than ~3-4 points.
- **Video length:** no robust interaction (longest minus shortest frame tertile -2.3 [-6.3, +4.5]; duration tertiles
  -0.5 [-5.5, +6.1]). Length is confounded with dataset and fps.

Interpretation:
- "The geometry path costs ~1.6-2.4 points at 2 % of an epoch" is what the data supports. It is **not** evidence
  about geometry *content*: s2_real also differs from s2_sft by the S1 stage and by the injected tokens.
- The gap is a single-seed contrast. Whether -1.6 exceeds training-seed noise cannot be determined:
  `s2_real_seed1` VSI was still running at the snapshot, and there is no s2_sft replica.

### 3.5 Answer distributions: the shift toward "2" (counting; verified, reframed)

Sources: `A/skeptic_counting_decomp*.log`, `A/skeptic_collapse2/{check,discrim,switch}.out`.

| run | share of counting answers = "2" | MRA all | MRA gt = 2 (302 q) | MRA gt ≠ 2 (263 q) |
|---|---|---|---|---|
| ground truth | 53.5 % | - | - | - |
| always answer 2 | 100 % | 61.8 | 100 | - |
| base_hint | 48.1 % | 54.7 | 63.9 | 44.0 |
| base_hint, answers < 2 raised to 2 (post-process) | 66.9 % | **69.5** | 90.0 | 45.9 |
| s1_real | 80.7 % | 65.4 | 95.8 | 30.5 |
| s2_real | 79.8 % | 67.4 | 97.0 | 33.5 |
| s2_real_routed | 78.8 % | 67.6 | 97.0 | 33.9 |
| s2_sft | 69.6 % | 68.9 | 91.5 | 43.0 |
| s2_sft_hint | 64.6 % | 69.8 | 88.7 | 48.2 |

- **The whole counting gain of every trained arm, geometry or not, comes from the floor.** base_hint answers 0 or 1
  on 106 questions, but only 1 of 565 true counts is below 2. Every trained arm answers 2 there, which is worth
  +13.8 (s1_real) to +14.8 points. The post-processed base (69.5) matches or beats every trained arm.
- **The geometry arms' extra shift is a net cost.**
  - They also change 82-89 of base's 3+ answers to 2 (s2_sft 32, s2_sft_hint 14). For s2_real that nets -0.44
    points: +4.0 on true-2 questions, -4.5 on the rest.
  - They also shrink their remaining 3+ answers, a further -1.65 (s2_real) and -1.82 (s1_real).
  - The shift is fully present after S1, which trains only the projector: s2_real minus s1_real share of "2" is
    -0.9 pp [-3.7, +1.9].
- **Same-prompt deficit on gt ≠ 2.** s2_real_routed (hint, nothing attached to counting) - base_hint =
  **-10.2 [-13.5, -6.8]**; s2_real - s2_sft = -9.5 [-13.1, -5.8]. For s1_real (-13.6 [-17.1, -10.0]) no same-prompt
  base exists, but the largest hint effect measured on these questions is +5.2.
- **It is a bias toward the mode, not a scene-blind answer.** The trained models separate gt = 2 from other counts
  better than the base (log-odds 2.90 for s2_real vs 1.07). Rank correlation with the truth is intact (Spearman on
  gt ≠ 2: 0.668 vs 0.677). The rate of answering 2 falls as the true count rises.
- What is geometry-specific is that the shift is larger and less selective: 59.6 % of s2_real's switches to 2 are
  correct vs 71-79 % for SFT.
- Whether "2" is the Sensenova training prior cannot be checked: the local 1k sample has too few counting questions.

**[U] single-analyst, not independently checked** (`A/vsi_types.md` §a):
- Object size looks saturated by a blind per-category prior: leave-one-video-out category median MRA 64.0 vs
  64.7-67.4 for every 4B run.
- On absolute distance, training pulls answers toward the middle of the range: s2_real gains on gt ≤ 1.4 m
  (35.2 vs 12.7) and loses on gt > 2.5 m (37.7 vs 45.8).

### 3.6 S1 alone: equal to the hint on average, not the same answerer (corrected)

Sources: `A/skeptic_s1_eq_hint/*.out`, `A/skeptic_s1eqhint/*.out`.

- **On average S1 matches the hint.** s1_real (plain) - base_hint = +0.7 [-0.7, +2.1]. A synthetic predictor
  ("base_hint, but answer 2 on every counting question") is indistinguishable from s1_real: -0.2 [-1.5, +1.1].
- **Format.** On the 60-video M2 subset, the plain 4B base (Mac) scores 24.7 with 51 % unparseable answers (100 % on
  every MCQ type). s1_real scores 53.6 with 0 % unparseable, +28.9 over the plain base. On the Mac, the hint alone
  also gives +28.9 over the plain base. This is a cross-stack comparison; the same config agrees between stacks to
  -0.5 [-1.6, +0.7], with 87.9 % identical answers.
- **But not the same answerer.** Only 47 % of s1_real's normalized answers (letter / first number) match base_hint's
  (raw strings: 21.6 %). That is as far as plain SFT is from base_hint (46.8 %), whereas the hint changes only 17 %
  of the SFT arm's answers (`A/doc_check/critic_vsi.out` §4).
- **The rel-dir loss (-4.6 [-8.0, -1.0]) is not a letter prior.**
  - It sits in easy questions (-7.4 [-14.1, -0.9]; s1 at 53.9 %, chance level on the two-way questions) and medium
    ones (-5.3). Hard questions are flat (-1.1).
  - Answers of the letter "D" exist only in the hard set: s1 7.2 %, truth 24.9 %, base_hint 45.3 %.
- **Numeric content tracking changes both ways.**
  - Abs-distance rank correlation falls: -0.073 [-0.127, -0.025]. The hint alone leaves it unchanged on the base.
  - Room-size ranking rises: +0.051 [+0.007, +0.104]. The room gain survives an optimal global rescale (+2.7
    [-0.8, +6.2] left, not significant).
- **With a Bonferroni correction over 8 types only counting survives** (+10.7 [+6.0, +15.9]).

### 3.7 The 2B

- **Zero-shot, the 2B is far behind.** 2B + hint scores -12.9 [-14.7, -10.9] below the 4B + hint.
- **Its best zero-shot input is plain video mode (42.7).** In image mode the 2B fails format: 45.8 % unparseable
  plain, 12.5 % with the hint (appear 48 %, rel dir 24 %, route 21 %).
  - Hint image vs plain video: -3.4 [-5.1, -1.8], with appearance order -28.5 and rel dir -8.7, but count +8.7 and
    room +4.5.
  - Plain image vs plain video: -18.2 [-19.7, -16.8].
- **Whether SFT closes the 2B-4B gap cannot be determined:** the trained 2B arms were never evaluated. Any future 2B
  comparison should use the base's video mode as the reference, not the hint row.

### 3.8 The 60-video M2 subset

- It is the same set as `holdout_scenes.txt` (60 videos, 1,151 questions).
- **Zero-shot levels and prompt deltas transfer.** Server runs read 0.7-1.1 points higher on the subset for
  base_hint, base_routed, s1_real and s2_real (s2_sft: +3.6); the routed-minus-hint delta is +2.2 on the full set,
  the subset and the complement alike (+2.22 / +2.19 / +2.22; `A/doc_check/critic_vsi.out` §8,
  `A/doc_check/critic_extra.out`).
- **Trained-model comparisons do not transfer.** s2_real - s2_sft is -4.2 [-8.3, -0.2] on the subset vs
  -0.9 [-2.5, +0.7] on the other 228 videos (recomputed in `A/doc_check/critic_vsi.out`); the difference is
  -3.3 [-7.5, +1.1] [U]. Paired intervals are 2.3-2.8x wider at 60 videos than at 288 (`A/resolution_60_vs_288.out`).
- The local Mac harness reproduces the server's base_hint on the same questions: -0.5 [-1.6, +0.7] (verified,
  `A/skeptic_s1_eq_hint/format_subset.out`).

---

## 4. Routed facts after training

Only room size (288 / 288 questions) and absolute distance (226 / 834 = 27.1 %, both objects seen in ≥ 3 keyframes)
were attached. Direction labels were **logged (968) but never attached** (`direction_facts=False`). This is the
pre-I-33 routed configuration ("M2 56.0"), not the one that reached 57.8. Sources: `A/facts_after_training.md` and the
verifier outputs cited in each row.

The log also holds counting (565), appearance-order (618), relative-distance (710) and route (182) entries that were
never attached. Scored against the truth, the logged count, order and rel-distance values are below *every* model,
zero-shot base included, at every view threshold (count k ≥ 2: fact 41.4 vs base_hint 57.8 / s2_real 66.8 MRA;
rel dist rp2: 55.7 vs 65.4 / 70.7 %; order k ≥ 2: 39.2 vs 58.0 / 66.4 %; `A/doc_check/critic_unused.out`, a
single critic computation). Routing only room size and distance was the right call for this map.

### 4.1 Are the pairs like for like?

| pair | identical answers off the attached questions | what differs |
|---|---|---|
| base_routed vs base_hint | 4,602 / 4,616 (99.7 %) | the facts only; nearly clean (different eval-script versions and model builds; the 14 flips sum to -0.1) |
| s2_real_routed vs s2_real | 3,599 / 4,616 (78.0 %) | facts + the format sentence (`s2_real_hint` failed) |
| s2_sft_hint vs s2_sft (reference for the sentence alone) | 3,956 / 4,616 (85.7 %) on the same ids | the sentence |

- The sentence moves individual types on s2_real but cancels overall: -0.02 [-0.71, +0.63] as the unattached
  questions' contribution to the 8-type overall (hybrid prediction set, `A/facts_after_training.md` §6), or
  -0.04 [-0.79, +0.72] as the 7-type mean over the unattached questions alone (§3.3). Same data, two aggregations. Rel-dir hard -2.4 [-4.6, -0.3] is the only per-type interval of 9 that excludes 0, which is weak
  evidence.
- The sentence's share of the room gain cannot be separated, because every room question carries a fact. The SFT
  proxy for the sentence alone on room size is +1.1 [-0.2, +2.5].

### 4.2 Coverage, fact accuracy, copy rate, and what routing bought

| | room (288) | abs distance, attached (226) |
|---|---|---|
| fact MRA (displayed value) | **65.6** | **46.1** |
| base_hint / base_routed | 50.0 / 65.6 | 38.2 / 46.1 |
| s1_real | 53.9 | 41.3 |
| s2_real / s2_real_routed | 61.7 / 65.8 | 48.6 / 46.1 |
| s2_sft / s2_sft_hint (never routed) | 59.0 / 60.1 | 54.4 / 53.5 |
| copy rate, base -> S2 real (within 5 %; room also counts width x length) | 99.3 % -> 97.2 % (paired McNemar p = 0.07) | 226/226 -> 226/226 |
| exact reproduction, base -> S2 real | 98.3 % -> 84.4 % counting w x l (p = 1e-10); 96.2 % -> 71.2 % exact displayed integer (S2 writes "20.0" for "20") | 100 % |
| copy rate when the fact is better vs worse than the model's own answer (S2 real) | 97.1 % vs 96.2 %, difference [-3.5, +5.6] pp | 100 % in every bin (trivial) |
| fact - s2_real | +3.9 [-0.6, +8.5] (n.s.) | -2.6 [-8.8, +3.8] (n.s.) |
| fact - s2_sft | +6.6 [+2.1, +11.0] | **-8.4 [-15.0, -1.7]** |
| routed - plain, base | +15.6 [+10.8, +20.6] | +7.8 [+2.2, +13.6] |
| routed - plain, S2 real | +4.2 [-0.3, +8.5] (includes the sentence) | -2.6 [-8.8, +3.8] |

- **Training did not make the model ignore facts, and it did not make it selective.**
  - S2 real lands within 5 % of the fact on 96.9 % of the room questions where its own answer was more than 5 % away.
  - Its non-copies do not track whether the fact is right.
  - Routed minus fact is +0.2 [-0.9, +1.4] on room and exactly 0 on distance, so routed - plain = fact - model by
    construction.
- **The overall S2 routing gain decomposes exactly.** +0.42 [-0.44, +1.36] = attached questions +0.43 (room +0.52,
  distance -0.09) + unattached -0.02. No part excludes 0.
- **The routing gain shrank because the model caught up.** On the attached questions, S2 real minus base_hint is
  +11.7 on room and +10.4 on distance, both significant. Plain SFT does the same, so this is a training effect, not
  a geometry one.

### 4.3 Distance facts at every view threshold (corrected)

Fact = logged d_k rounded to 0.1 m (what the model reads). Source: `A/skeptic_facts_recompute.log`,
`A/skeptic_abs_fact_k.log`.

| views ≥ k | n | fact | s2_real | s2_sft | s1_real | fact - s2_real [95 %] | fact - s2_sft |
|---|---|---|---|---|---|---|---|
| 1 | 647 | 31.3 | 45.8 | 49.5 | 38.8 | -14.5 [-18.5, -10.4] | -18.2 * |
| 2 | 387 | 44.8 | 48.2 | 52.3 | 42.6 | -3.5 [-8.6, +1.3] | -7.5 * |
| 3 (attached) | 226 | 46.1 | 48.6 | 54.4 | 41.3 | -2.6 [-8.8, +3.5] | -8.4 * |
| 4 | 133 | 42.9 | 48.0 | 55.3 | 40.9 | -5.2 [-13.9, +2.7] | -12.5 * |
| 5 | 73 | 47.7 | 51.1 | 57.3 | 38.9 | -3.4 [-14.3, +7.3] | -9.6 * |
| 6 | 39 | 42.3 | 50.8 | 61.3 | 34.4 | -8.5 [-23.4, +7.9] | -19.0 * |

- The fact is not better than either S2 model at any k. Against S2 SFT it is worse at every k. Against S2 real it is
  worse only at k = 1; at k ≥ 2 it ties (the thresholds are nested subsets, not six confirmations).
- The fact beats S1 real at k ≥ 2 (point estimates), so "worse than the trained model" holds for the S2 models only.
- **The error is not a scale error.**
  - The median fact/truth ratio is 1.00, and bias is 8 % of the log-MSE.
  - Only ~12-20 % of the error variance is between videos.
  - A per-video oracle scale anchor (room ground truth) moves the fact by +0.3 to +0.9 at k = 1..3 and hurts at
    k = 5-6.
  - The best in-sample global rescale (x0.89-0.90) reaches 48.5, a tie with S2 real.
- **The error is a lower tail of gross underestimates.** 20 % of the attached facts are under half the truth, 4 % over
  double, and 2.7 % read "0.0 m". This fits a wrong-instance or collapsed-closest-point failure; no per-instance
  geometry was uploaded to confirm it.

### 4.4 Direction facts (logged, never attached)

Sources: `A/skeptic_dirfacts*.out`, `A/skeptic_fat_dir2/recompute*.out`, `A/skeptic_vt_facts_k/check_dir*.out`.

| band (views of the objects) | n (videos) | fact | s2_real | s2_sft | s2_real_routed | fact - s2_real, video-level [95 %] |
|---|---|---|---|---|---|---|
| ≥ 1 | 686 (101) | 71.6 | 71.9 | 72.4 | 70.0 | -0.3 [-10.2, +9.1] |
| exactly 1 | 344 (55) | 58.4 | 68.6 | 68.6 | - | -10.1 [-25.2, +5.8]; question-level McNemar p = 0.004 |
| exactly 2 | 203 (32) | 86.2 | 70.9 | 74.4 | - | +15.1 [-1.6, +31.5]; McNemar p = 0.0005 |
| ≥ 2 | 342 (53) | 84.8 | 75.1 | 76.3 | 72.5 | +9.6 [-2.0, +20.7] (vs routed: +12.3 [+0.6, +23.4]) |
| ≥ 3 | 139 (26) | 82.7 | 81.3 | 79.1 | - | +0.5 [-15.5, +12.6]; McNemar p = 0.84 |

- **The simulated benefit of attaching directions is about 0 at ≥ 1 view, the planned setting.** The simulation
  assumes a 93 % take-rate (M2's base 4B), taken independently of correctness.
  - At ≥ 1 view (`--direction-min-frames 1`, the flag queued for the next run): -0.04 overall against s2_real and
    +0.07 [-0.77, +0.90] against s2_real_routed, which is the right baseline for that run. Zero-shot base_hint:
    +1.28 [+0.53, +1.99].
  - At ≥ 2 views: +0.38 against s2_real, +0.47 [-0.01, +0.96] against s2_real_routed. The advantage sits entirely
    in the exactly-2 band (≥ 3 is a tie), and k was picked from 3 tried, in-sample.
- **Wrong facts are mostly left/right mirrors, clustered by video.** 142 of the 195 wrong facts are exact left/right
  mirrors. 10 videos (7 ARKitScenes) are ≥ 75 % mirrored. Excluding the 18 worst videos (selected with ground truth,
  so not achievable) the fact scores 88.2 vs 73.3 for s2_real on 544 questions. Interpretation: a handedness or
  up-axis flip in the floor-plan projection, or instance bookkeeping. Not confirmable here: per-video map transforms
  are not logged.
- The take-rate of a *trained* model for direction facts (letter mapping) is unmeasured. The 2B followed direction
  facts half as often as the 4B on the M2.
- [U] The server's direction labels are worse than the M2's on the same 60 videos: 68.2 % vs 76.4 %, with 22
  left/right flips in 5 videos.

### 4.5 Ceiling vs ignore: what is the fact worth after training? (in-sample what-if, verified)

Answering with the arithmetic mean of the fact and the trained model's own no-fact answer (room uses width x length;
nothing fitted; 2 rules tried). Sources: `A/skeptic_fusion_step*.py`, `A/skeptic_fusion_v2/check*.out`.

| | copy the fact | model alone | 50/50 mean | mean - copy | mean - alone |
|---|---|---|---|---|---|
| room, S2 real | 65.6-65.7 | 61.7 | **73.5** | +7.9 [+4.7, +11.0] | +11.9 [+8.7, +15.0] |
| distance, S2 real | 46.1 | 48.6 | 51.6 | +5.6 [+1.6, +9.5] | +3.0 [-1.6, +7.6] (n.s.) |
| room, S2 SFT | 65.6-65.7 | 59.0 | 72.0-72.2 | +6.4-6.5 | +13.0 [+9.9, +16.2] |
| room, base | 65.6-65.7 | 50.0 | 65.5 | -0.2 [-4.1, +3.7] | - |

- **Overall, the averaging hybrid gains +1.15 [+0.74, +1.56] over s2_real_routed and +1.57 [+0.76, +2.35] over
  s2_real.** Almost all of it is room: room only gives +0.96 [+0.58, +1.35]. On distance, averaging instead of
  dropping the fact adds only +0.10 [-0.06, +0.26].
- **The gain comes from the model's own answer to that question, not from shrinkage.**
  - Averaging with shuffled answers or with a constant scores far below copying (p < 0.002).
  - The rule holds on held-out video halves (room +8.2 over copy).
  - The errors of the fact and of the model are nearly independent: log-error correlation 0.13 for room.
- **It is not geometry-specific.** Plain SFT gets the same room gain. For the base model an even split does nothing;
  a fact weight of 0.7-0.8 gains ~+2.7, but that weight was picked in-sample.
- Deployment needs a second, no-fact decode per attached question. It is untested on new videos.

---

## 5. S1 checkpoints: does real differ from control beyond seed noise?

### 5.1 Materials

| file | state |
|---|---|
| `s1_real`, `s1_control`, `s1_control_seed1` (4B), `s1_real_2b`, `s1_control_2b` | load on CPU, all finite, fp32; 95.1 M parameters (4B), 68.7 M (2B); projectors + pose projector only, no LoRA/LLM keys |
| `s1_real_seed1/final.pt` | **Truncated upload.** Full length (380,308,683 bytes) but zero from byte 346,291,712, no zip central directory. |

Salvage of `s1_real_seed1`:
- Two independent CRC-32 walks of the zip local headers agree bit for bit (`A/salvage_seed1.log`,
  `A/skeptic_ckpt_f5/salvage.py`); the output is `A/s1_real_seed1_salvaged.pt`.
- **Recovered: 24 of 28 tensors.** All three deepstack projectors (21 tensors), all gates, and the pose LayerNorm.
- **Lost:** `pose_projector.net.1.{weight,bias}` (2560x768, 2560) and `net.3.{weight,bias}` (2560x2560, 2560).
- **Use:** fit for weight- and function-level comparisons at L0-L2 *without* the pose path. **Unusable for
  evaluation**, because `expose_pose_token` uses the lost tensors.

### 5.2 Holdout-loss ablation on the M2 (the lead's measurement, `A/ablation_s1_local.md`)

Setup: 4B, stage align, 94 held-out Sensenova samples (`sn1k.holdout.jsonl`; expected overlap with the 13.6k S1
training samples ~1.6 %). The same samples in both runs: the `none` loss is identical on 94 / 94. Donors are
same-grid on 72.3 % of rows.

| quantity (loss, lower is better; mean ± 95 % CI) | value |
|---|---|
| s1_real loss: real / shuffled / none | 0.6286 / 0.6652 / 2.7657 |
| s1_control loss: real / shuffled (its training condition) / none | 0.6088 / 0.6024 / 2.7657 |
| real projector reads content (shuffled - real, within s1_real) | **+0.0366 ± 0.0314** (real wins 66 %) |
| control projector reads content | -0.0063 ± 0.0123 |
| **value of content vs control** = s1_control(shuffled) - s1_real(real) | **-0.0262 ± 0.0408** |
| seed noise at the training condition = control_seed1 - control | +0.0005 ± 0.0214 |
| value of the injection path (none - real), s1_real / s1_control | +2.137 ± 0.545 / +2.157 ± 0.548 |

What the ablation shows:
- The injection path carries almost all of the S1 effect (~2.1 nats) in both arms. That is the answer format and
  distribution; it is not geometry.
- The real projector's loss depends on content, the control's does not. But the control reaches a lower loss
  overall, so against the control the value of content is not distinguishable from 0.
- The same data read another way: s1_real(shuffled) - s1_control(shuffled) = +0.063 [+0.018, +0.115], while
  control - control_seed1 = -0.0005. The two arms differ at the loss level by more than seed noise, but not in the
  direction "real is better" (`A/skeptic_sameinit_v3/`).
- No S1 seed-1 real holdout number exists: the file is truncated.

### 5.3 Weight and function forensics (only what survived verification)

Setup: same-seed real and control start from **bit-identical initial weights** and the same data order. Same-seed fc1
differences are ≤ ~1.1 x sum(lr), cross-seed ~22x. A direct |real - control| vs |control - control_seed1| comparison
is therefore uninformative. The analyst's "same-seed is closer on every metric" is **refuted**: gates, the L0 LN
weight and the holdout loss go the other way. The fair test is the 2 x 2 (condition x seed) on the projector's
output for identical CUT3R tokens.

| question | result | source |
|---|---|---|
| Does real - control reproduce across seeds? | Barely. Energy-weighted cross-seed correlation of (real - control): 0.067 / 0.041 / 0.034 at L0 / L1 / L2; video-bootstrap CIs [0.050, 0.082] / [0.034, 0.051] / [0.021, 0.051], so small but not 0. The condition energy exceeds the interaction by 5.1-5.2 / 2.9 / 2.6 % of output energy. Replicated on 6 other videos x 32 keyframes (0.077 / 0.047 / 0.034). Positive control: the *seed* effect replicates across conditions at 0.30-0.34. | `A/skeptic_seedspec/fn2x2*.log` |
| Where the reproducible part sits | In the scene-dependent part: between-frame cross-seed correlation 0.16 / 0.18 / 0.09, but only 8-16 % of the difference energy. The input-independent constant part (33-64 % of the difference energy) does not reproduce (0.04 / 0.01 / 0.04). The input-side (LN + fc1) part replicates at 0.31-0.42 but carries 4-9 % of the difference energy. | `A/skeptic_seedspec/seedspec_analyst_stride2.log` |
| Structural differences, all 3 same-init pairs (seed 0, seed 1, 2B) | Real's output varies more between frames (L0 share 0.096-0.104 vs 0.062-0.067), replicated on 26 new videos, all CIs > 0; 60-75 % of that gap is between scenes, the within-scene view-to-view gap is only +0.010 / +0.011 / +0.002 of energy. Real passes 1.2-4.2x more between-sample energy (21/21 cells). Control concentrates more fc2 energy in its top (constant) direction (0.311 / 0.301 / 0.357 vs 0.247 / 0.248 / 0.295); the analyst's "higher effective rank" is this top-1 effect. | `A/skeptic_struct_refute/`, `A/skeptic_ckpt_f5/` |
| LayerNorm gain | Its real - control direction repeats across seeds (cos 0.43 / 0.27 / 0.13) and in the 2B, but mostly as a uniform offset (real's gain rises, control's falls, 3/3). Centred correlation 0.21 / 0.19 / 0.13. | same |
| Constant component of the output | 70-86 % of L0 energy (83-92 % with pose), 59-70 % L1, 32-52 % L2; one vector describes all videos. **Not evidence of a learned shift:** an untrained projector already outputs 40-41 / 51-54 / 58-61 % constant, and a zero-init fc2 trained by AdamW on pure-noise gradients reproduces the observed shares (0.75 / 0.70 / 0.51 vs 0.745 / 0.683 / 0.486). What is learned is the magnitude (fc2 starts at 0) and the direction (cross-seed cos 0.49-0.53 vs noise-null ≤ 0.07). | `A/skeptic_const2/`, `A/skeptic_constnull/summary.txt` |
| Size of the updates | fc2 and pose `net.3` Frobenius norms match the AdamW zero-SNR prediction (travel 0.131-0.140 x sum(lr) vs 0.132 simulated; 2B 0.197-0.202 vs 0.196). The gradients are **not** zero-mean, though: fc2.bias points the same way across seeds (cos 0.47-0.53, null 0.02), equally for real and control. LN tensors travel 1.3-1.9x the zero-SNR value. **All 24 gates moved down** by 0.12-0.40 x sum(lr). A gate can move at most 0.16 % (4B) at lr 3e-5 over 106 steps, so the gates are frozen by the budget, not by a lack of signal. | `A/skeptic_ckpt/recompute_ckpt.log`, `A/skeptic_travel2/` |
| What S2 will measure | S2 gives the projector sum(lr) = 2.07e-3 vs S1's 1.62e-3 (1.28x). That ratio compares ceilings: S1 used ~13 % of its per-element ceiling, so S2 can move the projector up to ~10x as far as S1 actually did. Whether `s2_real - s2_control` measures content or seed-specific soft prompts cannot be determined without the S2 checkpoints. | `A/skeptic_s2claim_indep/s2_projection.log` |

Answer to the question:
- **At the loss level: no.** Content is readable within the real arm but not against the control, and the arms differ
  from each other by more than seed noise, in the wrong direction for a content benefit.
- **At the function level: a small reproducible difference.** It is 3-7 % of the real-minus-control energy, lies in
  the scene-dependent channel, and is consistent with a weak geometry-correlated gradient. It cannot be separated from
  the control's donor mechanics: 40 % of donors have another grid shape (resampled), and frames are cycled when image
  counts differ.
- **Interpretation:** at 106 steps and lr 3e-5 the S1 projector learned an answer-format soft prompt plus a trace of
  content, too small to predict a VSI real - control gap.
- Two gaps remain. The pose path, about 2/3 of the seed-0 L0 difference energy, is untestable at seed 1 (lost
  tensors). And with one run per cell, the interaction cannot be told from run-to-run GPU nondeterminism.

---

## 6. Latency (gate 3)

Both benches used the **base** Qwen3.5-4B in bf16 with CUT3R and zero-init projectors: **no trained weights, no LoRA**.
They ran synthetic random 256x448 frames, the map *image* in the prefix, and GPU 4. The routed configuration VSI
actually uses (facts text, OWLv2 object map, `--no-map-image`) was not timed. Sources: `REPORT.md` latency section,
`A/latency.md`, the verifier outputs cited below.

### 6.1 What each number measures

| number | what it is | verdict |
|---|---|---|
| **question-only TTFT 164 ms** (deferred bench) | `pc.answer(question, max_new_tokens=1)` on a freshly built prefix: tokenise (17-token question) + cache deepcopy + `generate`; synced | **gate criterion; passes** (< 1,000 ms) |
| **drift 1.0** (streaming bench) | mean per-frame cost of the last 10 % of blocks / the first 10 %, on the *incremental* `LiveSession` path; printed `.1f` (true value < 1.05); 512 frames = 17.1 s of video | **gate criterion; passes** (< 1.2), but on the wrong path |
| full TTFT 61,558 ms | 110 (select + geometry embed) + **61,284 (vision tower + LLM visual prefill of 3,840 tokens, 15,360 patch rows)** + 164 | diagnostic; 61.6x the budget |
| TPOT 51.1 ms (deferred) / 63.3 ms (streaming) | deferred: (16-token generate - 1-token generate) / 15, synced; streaming: 31 manual decode steps, **unsynced** | diagnostic; the 12.2 ms gap is not attributable (a missing sync would only widen it) |
| frame_ingest p50 1,160.9 / p95 1,197.6 ms | incremental path per frame (2-frame block / 2), CUT3R on every frame | diagnostic: 0.86 fps max, 34.8x slower than 30 fps |
| geometry frame 25.6 ms | median synced `push()` where CUT3R ran (deferred) | healthy (Mac: 412.6 ms) |
| offline_ttft 570,745.5 ms | the incremental path replayed (1,114.7 ms/frame), not a batched offline encode | meaningless as an offline baseline |
| peak_mem 15,336 MB | `torch.cuda.max_memory_allocated` | proves a CUDA run with the weights resident (bf16 weights alone 9.3 GB) |
| meets_live_budget False | the harness's own rule (ingest p95 < 33 ms); **not the pre-registered gate** | fails 36x; also fails the older BENCHMARKS.md §4 definition |

### 6.2 Gate 3 verdict

- **Formally PASS** under rule 5 of `SERVER_WEEKEND.md`: 164 ms < 1 s and 1.0 < 1.2. Rounding cannot flip either.
- **Conditional, not a demonstration of the live design.**
  1. **TTFT.** The question-only TTFT assumes a current prefix already exists. One build took 61.4 s, 3.6x the whole
     test stream.
     - At 1 fps the 32-keyframe HalvingSelector set changes with median gaps of 1 / 4 / 8 / 32 s over
       0-1 / 1-5 / 5-15 / 15-30 min. After the first minute a median of only 6-7 leading keyframes survive a change.
     - A background loop of 61.3 s rebuilds is therefore **never current in the first 30 min**: 0.0 % of the time,
       median lag 76-91 s.
     - Only after ~34 min do the gaps (64 s) exceed one rebuild; at 60-120 min it is current ~45 % of the time.
     - With a rebuild of ≤ 1 s it would be current 66-99 % of the time after the first minute
       (`A/skeptic_latency_halving_v2/catchup.out`, `A/skeptic_latency_prefix_churn.out`).
  2. **Drift.** It was measured on the incremental path. The deferred path's drift (CUT3R + scene map up to 400k
     points) was never computed. Nothing shows it would fail; it is **unverified**.
  3. **Question length.** The timed question is 17 tokens; VSI questions have a median of 56 (plain) or 80 (routed)
     tokens, up to 233.

### 6.3 What is a real problem

| problem | data | interpretation / status |
|---|---|---|
| **The visual prefill is ~330-780x slower than compute** | 61,284 ms for ~45 TFLOP (2N count: 38.7) -> 0.63-0.75 TFLOP/s effective, 0.064-0.078 % of H100 SXM peak, i.e. ~1,300-1,600x below peak and ~330-780x below a realistic 25-50 % MFU (the H100 variant is not recorded; PCIe gives 250-600x). One cold measurement, but the warm incremental path shows the same order, ~16 (deferred) to ~20 (incremental) ms per visual token. | Unlikely to be CPU execution (15.3 GB CUDA peak) or GPU sharing (GPU 4 was pinned; CUT3R is 115x faster per patch in the same process); neither is excluded by a direct measurement. |
| **Conv3d suspect** | Cost roughly proportional to patch rows (affine fit: 3.96 ms/row + 0.4-0.5 s per call). Patches, visual tokens, images and FLOPs are collinear (4:1) in both benches, so the vision tower cannot be separated from the LLM's visual-token prefill. | **Prime suspect, unproven.** The best-matched external report is 4.11 ms/patch (PyTorch forums: torch 2.9.0 + cuDNN 9.10, the exact `(N,3,2,16,16)` bf16 shape), which predicts 63.2 s vs 61.3 s observed. The 3.5 ms/patch figure quoted earlier (axolotl PR 3897) was torch 2.12 *without* cuDNN. A code comment says the server had torch 2.7.1 on 09-24; if still true, the cause would be cuDNN unavailable or disabled, not the 2.9.x regression. |
| **Text path is host-overhead-bound** | Question forward: 26,160 ATen ops (fp32; 26,670 bf16, ~13.1k kernel-launching), 85-88 % of them in the torch Gated DeltaNet fallback (if running). Decode step: ~4,000-4,500 ops, **only 26-29 % in the fallback**. | fla would cut the question-only TTFT roughly in half (51-70 % of it is fallback) but TPOT only ~12-42 %. Whether fla is installed on the server is unverified. |
| **The incremental path is not viable** | 1,160.9 ms/frame (0.86 fps); 28,672 visual tokens after 512 frames | a diagnostic, not the design. [U] It also ignores the configured 16,384-token cap. |
| **Report hides context** | `weekend_report.py` keeps only lines containing "ms" or "TTFT", dropping the device header and `live_fps_max` | [U] needs a one-line filter change |

---

## 7. Missing results, why, and the recovery list

### 7.1 Pre-registered rules (`SERVER_WEEKEND.md`)

| rule | needs | state | value |
|---|---|---|---|
| 1 geometry content, S2 (**key**) | `s2_control`, seed-1 pair, `p7_ablation_s2` | **not readable**: none exists (`s2_real_seed1` was in flight at the snapshot) | - |
| 2 value of the geometry path | s2_real, s2_sft | readable | -1.6 [-3.1, -0.1] |
| 3 gate 1 (lmms-eval vsibench) | gate/base, base_video, trained | **not readable**: no lmms-eval results; the local proxy needs `base_plain` (missing) | - |
| 4 gate 2 (VideoMME) | same | **not readable** | - |
| 5 gate 3 latency | latency files | readable | pass, conditional (§6) |
| 6 training vs prompting | s2_real / base_hint; routed pair | readable | +8.5 [+6.7, +10.3]; +6.7 [+5.1, +8.2] |
| 7 learning curve | `s2_real_step*.json` | **cannot exist**: no step checkpoints (`SAVE_EVERY=500` > 136 steps; `curve_steps` rounds all three to step 0) | - |
| 8 is S1 needed / size | s2_joint; s2_real_2b, s2_control_2b | **not readable**: all absent (the training arms are done) | - |
| 9 probe table | `<arm>/probe.jsonl` | **cannot exist**: the probe was committed (68ee746, 09-28 04:20 UTC) after training ended (last arms ~09-27 19:30 UTC); even with it, `PROBE_EVERY=500` > 136 steps would have probed only at the start and the end (a 2-point table) | - |

Rules 7 and 9: the commit times (2a80a5b 02:03 UTC, 68ee746 04:20 UTC on 09-28), `SAVE_EVERY` / `PROBE_EVERY` = 500
and the probe condition `step % probe_every == 0` were re-read from git and the code in the critic pass.

A partial substitute for the gate-1 proxy: the local M2 4B base in video mode (48.6 on the 60 videos) gives
S2 real - best base ≈ **+13.2 [+8.8, +17.1]**. This is cross-machine, unpaired in origin and covers 60 videos only;
it does not replace the gate (`A/skeptic_missing_p3.out`).

### 7.2 Why things are missing (verified; corrections applied)

| cause | what it explains | status |
|---|---|---|
| **`build_model` bug** (fixed in 2a80a5b, 09-28 02:03 UTC): every `--config` eval crashed at launch | all pre-fix s1_*/s2_* VSI runs, base_routed's first attempt, both S2 holdout ablations. The seed-1 VSI jobs failed 9:31 (real) and 9:43 (control) after the S1 seed-1 checkpoints' mtimes (predicted S2 arm length ~9:40) = one S2 arm + a crash at launch. | verified |
| **False gate markers** | `gate_*.done` exist but no `*results*.json` anywhere. lmms-eval 0.7.3 catches exceptions unless `--verbosity DEBUG` and exits 0; `run_gate.sh` / `gate_job` trust the exit code. gate_trained had at most ~21:40 (~22 min) between launch and the gate check, far too short for three benchmarks. The root cause is unknown and not the build_model bug (`lmms_live3r.py` never calls it). The markers make every rerun skip the gate. | verified; root cause in `gate_*.log` |
| **`timeout 6h` in `vsi_job`** | s2_real + s2_real_hint dying in the same second (09-28 22:30:57) while the orchestrator continued: plausible. It implies s1_real ran 5:59:44, but that duration is derived from the hypothesis. | **plausible, not established**; the `vsi/s2_real.log` tail settles it |
| 2B base pair failing in the same second (09-27 05:55:05) | The earlier "timeout" reading is **not supported**. At the logged 2B rate the jobs died ~5:42-5:52 after launch, and the identical 4B hint eval finished under the same cap. | cause undetermined |
| **Second launcher outside STATUS** | s2_real, s2_real_routed, s2_sft, s2_sft_hint carry absolute weight paths (the orchestrator writes relative). s2_real.json was written 33 s *after* STATUS logged its failure; s2_sft.json predates base_routed.json. PID files exist for `eval_1gpu` / `eval_4gpu`. The JSONs themselves are valid. | verified; something of that launcher may still be running |
| REPORT shows only the last 15 FAILED lines | every failure before 09-27 05:55:05 is invisible | verified |
| `base_plain` (4B, 2 modes) | 2 x 5,130 generations in one job under a 6 h cap; the first p3 attempt also left no base_routed | current blocker of p3; original cause unknown |
| `s1_real_seed1` truncated upload | §5.1 | verified |

### 7.3 Recovery list, in priority order

**Before anything:**
1. Check what is running: `s2_real_seed1` started 22:30:59 and ends by 04:30:59 UTC = 13:31 KST under its timeout;
   also `ps aux | grep eval_vsi_local` and `nvidia-smi` on the `eval_1gpu` / `eval_4gpu` pods. Two writers of one JSON
   would collide.
2. Do not re-run the unchanged orchestrator: the 6 h timeout, `role_sft` re-running p3 first, the false gate markers.
   Use the manual `vsi()` / `abl()` helpers in `A/missing_results.md` §4.1 (no timeout, STATUS-tagged), or first
   commit `timeout "${VSI_HOURS:-12}h"` plus "mark a gate step only if `*results*.json` exists".
3. Commit for the new runs: evaluate at 68ee746, or re-run both arms of any comparison at the same commit. The same
   holds for the patch-embed fix (F1): the linear form is not bit-identical in bf16, so if F1 lands before these
   evals, re-run the reference arms (`s2_real`, `s2_sft`) with it too, or evaluate the missing arms without it.

**The list.** GPU-h per VSI run are estimates (~6-7 h per 4B single-mode run on this stack, never measured cleanly).

| # | item | rule | GPU-h (est.) | M2 possible? |
|---|---|---|---|---|
| 1 | `s2_control.json` | **1 (key)** | 6-7 | no (no S2 weights local) |
| 2 | `s2_real_seed1.json` (only if it failed at 13:31 KST) | 1, seed check | 0 or 6-7 | no |
| 3 | `s2_control_seed1.json` | 1, seed check | 6-7 | no |
| 4 | `p7_ablation_s2.json` (real vs control holdout loss) | 1, loss check | ≤ 3 | **yes, once the S2 `final.pt` files are fetched** (§8.3); the S1 version ran on the M2 (§5.2); untested with LoRA weights |
| 5 | `s2_joint.json` | 8, is S1 needed | 6-7 | no |
| 6 | `s2_real_2b.json`, `s2_control_2b.json` | 8, size and 2B content | 12-14 | no |
| 7 | `p7_ablation_2b.json` | 8 | ≤ 3 | same as item 4 (needs the 2B S2 weights) |
| 8 | Gate: read `gate_base.log`; delete the markers and `gate/*` dirs; `--limit 4 --verbosity DEBUG` smoke; then the 3 runs + verdict | 3, 4 | ~0.5 + ~30-40 (guess, never completed) | crash reproduction only |
| 9 | `base_plain.json` (or `--modes oracle-video` only) | gate-1 local proxy, format rows | 12-14 (video only 6-7) | slow (2 x ~12 h) |
| 10 | `s2_real_hint.json` | clean hint effect on S2 real | 6-7 | no |
| 11 | `s1_control.json` | S1 real - control VSI | 6-7 | **yes** (60 videos, pair both arms on the M2) |
| 12 | routed extras (`s2_control_routed`, `s2_joint_routed`, 2B routed) | routed rows | 6-8 each | no |
| 13 | `s2_sft_routed.json` (never planned) | same-prompt partner for `s2_real_routed`: is the best geometry system better than routed SFT? | 6-7 | no |
| 14 | `base_hint_2b` in video mode | a clean 2B hint effect (today hint-image vs plain-video mixes two factors) | ≤ 6-7 (unmeasured) | yes, on the 60 videos |
| - | `p5_ablation_s1` | S1 loss check | done on the M2 (§5.2) | done |
| - | learning curve, probe table | 7, 9 | retraining only (next run) | no |

Items 1-7 (rules 1 and 8, both seeds, both ablations) come to **~36-48 GPU-h**: one 8-GPU node for ~7-8 h, if eight
concurrent evals do not slow each other (not measured). If the image-path fix (§8.1) lands first, the same list
should be several times cheaper, because the eval prefill runs the same patch embed.

---

## 8. Recommendations for the next server run (in sync with training)

The order matters: at today's 16 GPU-s/sample, a one-epoch arm costs ~19 node-days. The lead's reading of the S1
ablation (`A/ablation_s1_local.md`) is that the data budget has to grow roughly 50x before a null on content is
informative; that is a judgement (2 % -> 1 epoch), not a power calculation. So throughput comes first; the run matrix
is only worth launching once an epoch per arm fits in about a day.

### 8.1 Throughput first: discriminating checks, then fixes

**Checks (≤ 1 GPU-hour in total, on one idle server GPU), each with the result that decides it:**

| # | check | predicts / decides | cost |
|---|---|---|---|
| C1 | `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled)"` and `python -c "import importlib.util as u; print({m: bool(u.find_spec(m)) for m in ('fla','causal_conv1d','flash_attn','kernels')})"` | torch 2.9.x with cuDNN < 9.15, or cuDNN off, supports the Conv3d cliff. `fla` False confirms the DeltaNet fallback. | 1 min |
| C2 | Patch-embed microbench, 2,000 rows bf16: Conv3d vs its exact `F.linear` equivalent, plus a `torch.profiler` table (script in `A/throughput.md` §4) | `aten::slow_conv_dilated3d` and ~seconds for the conv vs ~ms for the linear = the image-path cause | 5 min |
| C3 | `grep -a "samp/s" outputs/weekend/{s1_real,s2_real,s2_sft}.log`; the same for `p1_4b_real_smoke_{a,c}.log` | image-path hypothesis: data < ~300 ms, geom ~100-150 ms, fb ~15,000 ms/sample. data > ~1,000 ms would point at the filesystem. (No `probe step` lines can exist: the probe postdates training, §7.1 rule 9. Any such line would mean the arms ran newer code than assumed.) | 1 min |
| C4 | `torch.profiler` over 2 micro-steps of `train.py` on 1 GPU (`--max-steps 1 --grad-accum 2`), top 20 ops by CPU and CUDA time; `nvidia-smi dmon -s u` during an arm | where the 10.7 s of own cost actually go | 30 min |

**Fixes** (effects are estimates from op counts and FLOPs unless marked):

| fix | expected effect | cost / risk |
|---|---|---|
| **F1** Run the patch embed as the exact linear (`F.linear(x, W.view(1024,-1), b)`; numerically checked here, fp32 rel. diff 2e-6), or move to a torch/cuDNN without the cliff. **Apply to training and to every eval path** (VSI harness, lmms-eval wrapper, latency). | own cost 10.7 -> ~1.2 s bottom-up, i.e. 16 -> ~1.4 GPU-s/sample (~11x); top-down range 0.5-3.2. Also shortens every VSI eval (removes the 6 h timeout problem) and the 61 s prefix rebuild. | a few lines in `Live3RModel.from_pretrained`; verify one batch's loss and one VSI video are unchanged within bf16 |
| **F2** Install flash-linear-attention and causal-conv1d; p0 asserts the imports (the transformers fallback is silent) | -~67k launches per training sample (~0.7-0.9 s); after F1 ~0.45-0.6 GPU-s/sample -> **~13-18 h per epoch per node** (meets the target only if F1 is as effective as estimated); halves the question-only TTFT | pip on 4 nodes; loss-equality check vs the torch path |
| **F3** Size the budget from the production configuration (smoke at grad-accum 16, or re-plan after the arm's first 20 steps) | uses the whole hour cap: 1.6-1.9x more samples at no cost | trivial |
| **F4** Length-bucketed global sampler (all ranks draw one length decile per step; seeded identically for real and control) | gating 0.67 -> 0.92 at 8 x 16 (~1.37x); required for any 32-rank job (0.37 -> 0.78) | sampler code; changes batch composition, so it must be the same for all arms |
| **F5** After F2: drop `--grad-checkpointing` | 1.30-1.40x fewer launches at batch 1. Extra activation ~11.5 MB/token fits for ~99 % of samples at batch 1, not for 8-sample packs. Do not combine with packing. | test memory on the longest sample (6,897 tokens) |
| **F6** CUT3R in bf16 / TF32, batched across records with equal frame counts | ~0.12 -> ~0.04 s/sample; matters only after F1 + F2 | changes geometry numerics: the same setting for the next training and all evals |
| Not recommended | geometry cache (~12 TB); more DataLoader workers (idle); packing before F1-F2 (small gain, large change; only 1.35-2.5x in the geometry arms because CUT3R is per frame) | - |

**Go / no-go before the matrix** (write it down first): after F1 (+ F2), run one 8-GPU arm for ~30 steps at the
production grad-accum and read `samp/s` and the `data / geom / fb` split. Launch the ≥ 1-epoch matrix only if it
reaches ≤ ~1 GPU-s/sample (≤ ~1 day per epoch per node) and one batch's loss matches the pre-fix path within bf16;
then size the step budget from that measured rate (F3), not from the smoke. If the rate stays above ~3 GPU-s/sample,
stop and profile (C4) instead of launching.

### 8.2 The run matrix

**Budget.** The target is ≥ 1 epoch (830k samples) per S2 arm, which needs F1 + F2 (estimated 13-18 h per epoch per
node). If F1 does not land, do not re-run the matrix at 2 % of an epoch. The material in this document shows that
2 % answers format and prior questions only.

| arm | why | seeds | priority |
|---|---|---|---|
| `s2_sft` (plain LoRA SFT, no geometry) | the baseline any geometry recipe must beat (62.3 today); a second seed gives the SFT seed noise that is missing now | 2 | 1 |
| `s2_real` (S1 -> S2, real geometry) | the recipe under test | 2 | 1 |
| `s2_control` (same, shuffled geometry) | rule 1: the value of content | 2 | 1 |
| `s2_joint` (no S1) | S1 learned mainly a format soft prompt (§5); if joint ≈ real, drop Sensenova-S1 and put its budget into S2 or into Geometry-QA S1 (plan A) | 1 | 2 |
| Geometry-QA S1 -> Sensenova S2 (plan A in `NEXT_TRAINING_AND_SYSTEM_20260927.md`) | the M2 found the projector reads content only when the questions need it; replaces Sensenova-S1 if joint ≈ real | 1-2 | 2, after the core triplet |
| 2B real / control | size question; lower priority until the 4B content question is answered. Compare against the 2B's plain *video* baseline (42.7). | 1 | 3 |

Control arm: restrict donors to same-grid-shape, same-image-count donors (today 40 % are resampled or frame-cycled),
so real vs control differs only in content. Cost: donor-buffer logic. Effect: removes the confound that blocks §5's
reading.

**Evaluations for every S2 arm.** VSI on all 288 videos:
- Run **plain and hint**, so every comparison has a same-prompt partner.
- Run **routed** with the updated fact set (below), including `s2_sft`, so the best geometry system has a routed
  SFT partner (missing today).
- Report every number on the **VSI-Debiased** split too; the gap there was larger and clearer (-3.8 [-6.0, -1.7],
  p = 0.0004, vs -1.6, p = 0.04 on the full set).
- Also run the holdout ablation (real vs shuffled vs none) for real and control, and the lmms-eval gate once the
  crash is fixed.

**Step checkpoints and probe:** `SAVE_EVERY ≈ S2_STEPS/4`, `PROBE_EVERY ≤ S2_STEPS/8`, at a commit ≥ 68ee746. These
are required for rules 7 and 9, and they are cheap.

**Routed fact configuration for trained arms** (from §4; effects in-sample unless noted):

| fact | recommendation | expected effect |
|---|---|---|
| room size | keep | +3.9 [-0.6, +8.5] vs S2 real; +6.6 vs SFT |
| absolute distance | **drop** for trained arms (keep for zero-shot) | removes -2.6 (S2 real) to -8.4 (SFT) on 27 % of distance questions; a scale anchor would not fix it |
| direction | not at `--direction-min-frames 1` (≈ 0 for trained arms); first fix and log the left/right mirror errors (per-video map transform), then pre-register k = 2 | +0.4-0.5 overall at k = 2 if the model copies at 93 % (CI touches 0) |
| fact + own-answer averaging (2-pass) | try on a held-out set before adopting | +1.15 overall in-sample, mostly room; not geometry-specific |
| count, appearance order, rel distance (logged, never attached) | do not route from this map | the logged values score below every model, zero-shot included, at every view threshold (§4 intro; single critic computation) |
| fact-conditioned SFT (plan B, I-17) | motivated here: the trained model copies blindly and never weighs the fact against its own estimate, while the two carry nearly independent errors | unknown; needs the per-sequence maps job (~1 node-day, per the plan) |

**Pre-registered comparisons (write them down before launching):**
1. `s2_real - s2_control`, both seeds, with the seed spread reported next to it. Key.
2. `s2_real - s2_sft` and `s2_control - s2_sft` (the value of a content-free injection).
3. Every comparison also on Debiased and as a route-excluded 7-type mean. Per-type claims Holm-corrected over 8 types.
4. Answer-distribution monitors per checkpoint: "2" share on counting (today 80 % geometry arms / 70 % SFT / truth
   53 %), MRA on gt ≠ 2, Spearman rank correlation per numeric type, top-3 answer share.
5. Per-question agreement between arms (today 49.1 % identical raw answers real vs sft, 82.6 % for a hint line).

**Orchestrator and report fixes** (each a few lines; they prevent this weekend's losses):
- VSI timeout configurable, 12 h.
- Gate marked done only if `*results*.json` exists; the gate smoke runs with `--verbosity DEBUG` before the full run.
- The report prints all FAILED lines since the last restart, and keeps the latency device header and fps lines.
- Manual and second-launcher runs write to STATUS.
- `latency_stream.json` saves the geometry / vision / prefill split.
- The VSI JSON records the effective prompt condition: today the routed runs carry the hint sentence with
  `format_hint = False`, which every reader has to rediscover from the code.

**Latency run** (idle GPU, after F1 / F2):
- Print the environment line (versions, fla import, GPU name).
- Component microbenchmarks.
- Deferred-path drift over ≥ 10 min of real VSI video at native resolution, with trained S2 weights + LoRA, in the
  routed configuration.
- A background-rebuild policy with questions at random times, reporting the TTFT distribution and prefix staleness.
  Consider a prefix built from the selector's append-only buffer instead of `final_selection()`: in simulation it
  stays current 86-93 % of the time after 5 min even at 61 s per rebuild (overbuffer 1), at the cost of 17-32 frames
  instead of 32.

### 8.3 What the M2 can do meanwhile (local S1 checkpoints only until the S2 weights are fetched)

| job | value | cost |
|---|---|---|
| Code for F1 (patch embed as linear, train and eval), the F2 import assert, the F3 budget re-plan, the F4 bucketed sampler, the orchestrator / report fixes of §8.2 | makes the next run possible; all testable on the M2 with the small model | 1-2 days of code |
| Reproduce the gate crash: `pip install -e . --no-deps` (the entry-point list is empty in this env), then the `--limit 4 --verbosity DEBUG` smoke with `configs/m2_zeroshot_4b.yaml` | may find the swallowed exception without waiting for the server log | minutes |
| Direction-fact mirror errors: rerun the fact logger on the 10 most-mirrored ARKitScenes videos and inspect the floor-plan transform | decides whether direction facts are worth routing | ~hours |
| S1 real vs control VSI, 60 videos, both arms on the M2 | S1 content at VSI level (±3.3 points at 60 videos) | 2 x 2.5 h |
| 2B S1 holdout ablation (`eval_geometry_ablation.py` with `m2_zeroshot_2b.yaml`) | the 2B's S1 content signal | < 1 h |
| **First:** fetch the S2 checkpoints (`s2_{real,control,sft,joint}` and the seed-1 pair; ~0.6 GB each, estimated) and run the S2 holdout ablation (`eval_geometry_ablation.py --stage sft`, real vs shuffled vs none, both arms and both seeds) | the fastest loss-level reading of rule 1, independent of the server's eval queue (the §5.2 design, now for S2); also enables M2 checks of direction-fact take-rate and 2-pass fusion (60 videos, too coarse for rule 1 at VSI level) | download + ~1 h per arm (S1 took < 1 h; LoRA untested) |

---

## 9. Open questions and the exact files / logs to request

| question | request (server paths under `/group-volume/wooyeol/vlm-live/`) |
|---|---|
| Image-path mechanism (Conv3d / cuDNN) | C1 and C2 output; `outputs/weekend/p0_<role>.log` |
| Is fla / causal-conv1d installed? | the `find_spec` line of C1 |
| Actual per-phase cost (`data / geom / fb ms/샘플`, logged since 9edc999, 09-23) | `outputs/weekend/{s1_real,s2_real,s2_sft}.log`, `p1_4b_*_smoke_{a,c}.log` (lines with `samp/s`) |
| Did every arm run all planned steps? | the same arm logs, `train_args.json` |
| Why the gate failed | `outputs/weekend/gate_base.log`, `gate_base_video.log`, `gate_trained.log` (`grep -n "Error during evaluation" -B 40`), `gate/general.env` |
| Timeout vs crash at 09-28 22:30:57 and 09-27 05:55:05 | `tail -50 outputs/weekend/vsi/{s2_real,s2_real_hint,base_hint_2b,base_plain_2b}.log` (a last `[k/288]` line with no Traceback = timeout) |
| All failures, restarts after the fix, the second launcher | `grep FAILED outputs/weekend/STATUS` (all lines), STATUS after 2026-09-28 02:03, `master.{real,control,sft,small}.log`, the `eval_1gpu` / `eval_4gpu` script or history, `ps` / `nvidia-smi` |
| Did s2_real_seed1 finish? | STATUS after 04:31 UTC (13:31 KST) 09-29; `vsi/s2_real_seed1.json` |
| Key comparison and seed noise | `vsi/s2_control.json`, `vsi/s2_control_seed1.json`, `vsi/s2_real_seed1.json`, `p7_ablation_s2.json` (§7.3) |
| S2 weights for M2-side checks and forensics | `outputs/weekend/s2_{real,control,sft,joint}/final.pt` (with sha256) |
| Intact S1 seed-1 checkpoint (pose path) | re-upload `outputs/weekend/s1_real_seed1/final.pt` with its sha256 |
| Real eval wall time per run | the last line of every finished `vsi/*.log` (`문항 5130 · 디코딩 …`) |
| Latency device, env and split | `outputs/weekend/latency_live.log` (full), `latency_stream.log`, `latency_stream.json` |
| Does Sensenova's length/image distribution match the local 1k sample? Scene overlap with VSI? | an image-count histogram of `data/sensenova.jsonl`; the source manifest with scene ids |

Questions these materials cannot answer, even with the files above:
- Whether the geometry-specific "2" shift, the scannet-route outlier or the -1.6 exceed training-seed noise. That needs
  seed replicas of `s2_real` **and** `s2_sft`.
- Whether the in-sample facts results (fusion +1.15, direction k = 2) hold on new videos.
- Whether S1's small reproducible content trace is geometry or donor mechanics. That needs a control with same-shape
  donors.

---

## Appendix A. Findings not independently verified (single analyst; not used for conclusions)

| finding | source |
|---|---|
| Object size saturated by a blind per-category prior (64.0 vs 64.7-67.4 for every 4B run) | `A/vsi_types.md` §a |
| Abs distance: training shrinks answers toward the middle (gains ≤ 1.4 m, loses > 2.5 m); s2_real rank correlation 0.67 vs s2_sft 0.72 | `A/vsi_types.md` §a |
| 60-video subset reads ~1 point high for zero-shot, 1-3.6 for trained runs; trained comparisons too noisy there (levels and the s2_real - s2_sft split now recomputed, §3.8) | `A/vsi_types.md` §e |
| The trained model reproduces room facts in its own format ("<N>.0", w x l) and answers on a coarse grid without facts ("10.0" 20 % of room, "2.0" 25 % of distance) | `A/facts_after_training.md` §2 |
| Server direction labels weaker than the M2's on the same 60 videos (68.2 % vs 76.4 %) | `A/facts_after_training.md` |
| The end-of-S1 `inj` ratios in REPORT come from one sample on rank 0 and do not reflect weight differences | `A/checkpoints.md` |
| `pose_projector.net.3.bias` equals `projectors.0.fc2.bias` in every arm (duplicated layer-0 bias). Critic check: equal to within 1.8e-8, not bit-identical, in all 5 loadable arms (consistent with two parameters receiving the same updates rather than one shared tensor) | `A/checkpoints.md`, `A/doc_check/critic_posebias.out` |
| CUT3R ~1 % of step time today, 20-30 % after F1 + F2; a geometry cache would be ~12 TB | `A/throughput.md` §2.6 |
| The incremental `LiveSession` never enforces `max_visual_tokens`; `offline_ttft_ms` should be redefined; `weekend_report.py` drops latency header lines | `A/latency.md` |
| 2B image-mode format failures by type (appear 48 %, rel dir 24 %, route 21 % with the hint; rel dist 10 %) (now recomputed: `A/doc_check/critic_facts.out`) | `A/vsi_types.md` |

---

## Verification notes (critic pass, 2026-09-29)

Scripts and outputs: `A/doc_check/critic_{vsi,facts,ckpt,ablation,unused,extra}.py` -> `.out`, plus
`critic_posebias.out`; CPU only, e.g. `PYTHONPATH=src python outputs/server_0929/analysis/doc_check/critic_vsi.py`.
Each recomputation is independent of the analysts' scripts (own loaders, own lmms-eval aggregation and bootstrap).

**Recomputed and matched** (over 250 numbers; point values within rounding, interval bounds within bootstrap noise
(≤ 0.3 points)):
- §1-2: steps x 128 and epoch shares; the per-arm samples/s and GPU-s/sample from the REPORT `elapsed` column
  (window = steps 20 -> last logged); 52-55 % of the caps; the 17.16 h checkpoint-mtime gap; the epoch targets
  (9.61 samples/s, 0.83 / 3.33 GPU-s/sample, 18.6-19.7x, 0.71 for 14 arm-stages, ~19 node-days); the S1 / S2 lr mass
  after the last logged step (0.04 % / 0.30 %).
- §3: every cell of the §3.1 table including the Debiased / pruned columns; all 13 §3.2 comparisons with a
  different bootstrap seed; the per-type SFT gains and the Debiased / pruned deltas of §3.3; the §3.4 per-type table,
  7-type mean, Debiased split, raw / normalized agreement (2,518 / 5,130; 58.6 %; 82.6-82.7 %) and the McNemar counts
  (34 vs 67, p = 0.0013; 65 vs 107, p = 0.0017); the whole §3.5 counting table, 61.8, 69.5, the 106 base answers
  below 2, the single true count below 2 and the three gt ≠ 2 deltas; the 2B format failures; the §3.8 subset numbers.
- §4: the agreement counts of §4.1; the §4.2 fact and model MRAs, copy rates (99.3 -> 97.2 %) and exact-reproduction
  rates (98.3 -> 84.4 %, 96.2 -> 71.2 %); every row of §4.3 and the 20.4 / 4.0 / 2.7 % tail shares; every row of §4.4
  and the 195 wrong / 142 mirrored facts; the §4.5 fusion scores.
- §5: tensor counts, parameter counts, dtype, finiteness and absence of LoRA keys in the 5 loadable checkpoints; the
  truncation offset (zeros from byte 346,291,712, no central directory); the 4 lost pose tensors and their shapes;
  every §5.2 quantity from the per-sample rows of both ablation JSONs.
- §6-7: the latency arithmetic (61,558 = 110 + 61,284 + 164; 0.86 fps; 34.8x; 36x; 3.6x; 16.0 ms per visual token;
  0.63-0.75 TFLOP/s = 0.064-0.078 % of SXM peak; the 12.2 ms TPOT gap); the commit times of 2a80a5b and 68ee746;
  the JSON mtimes (s2_real.json 33 s after its FAILED line; s2_sft.json before base_routed.json); `SAVE_EVERY` /
  `PROBE_EVERY` = 500 in `server_weekend.sh`.

**Changed in this pass:**
1. TL;DR 1: the budget was sized from `SMOKE_SPS` 0.30 (26.7 GPU-s/sample), not 26.2 (the smoke's own window);
   §2.2's 1.62-1.72x uses 0.30.
2. TL;DR 2, §3.3: added the counting-floor share. Against a base with 0/1 counting answers raised to 2, SFT gains
   +8.3 [+6.6, +9.8], s2_real +6.7 [+4.9, +8.4], s1_real -1.2 [-2.5, +0.2]; ~1.9 points of each gain is that prior.
3. TL;DR 6: the -10.2 comes from `s2_real_routed` (S2 real weights + hint, nothing attached), not from `s2_real`.
   "Net cost" is now qualified: vs SFT on all counting questions it is -1.5 [-3.8, +0.9], n.s.
4. §2.1: added the 2B lr mass after the last logged step (4.5 % S1, 0.7 % S2) and what the unused REPORT columns
   `skipped` (rank-0 dataset retries, equal within pairs) and `sync` mean.
5. §2.3: image-path share 65 -> 66 %; data-loader headroom ~300-400x -> ~370-420x; added the Mac training rate
   (8.9 s/sample).
6. §3.1: "decodes deterministically" softened to near-deterministic. 4B format failures are ≤ 0.04 %, not 0.0 %.
   The Debiased definition was ambiguous: Debiased = ids *not* in `pruned_ids.txt`. The two routed runs saw
   identical facts.
7. §3.3: template exposure "may" explain part of the SFT gain (unmeasured), no longer stated as fact.
8. §3.4: the p-values depend on the seed. "20 of 20 seeds" holds at 10,000 resamples only. Object size's
   significance is equally fragile. Removed [U] from the normalized agreement.
9. §3.6: the 47 % is normalized answers (raw strings 21.6 %).
10. §3.8: removed [U] from the recomputed subset numbers (the -3.3 difference keeps it).
11. §4 intro, §8.2: the logged but never attached count / order / rel-distance facts (unused material) score below
    every model at every threshold (single critic computation). Recommendation: do not route them from this map.
12. §4.1: -0.02 [-0.71, +0.63] and -0.04 [-0.79, +0.72] are the same data under two aggregations; now said.
13. §4.5: SFT room row 72.1 -> 72.0-72.2; added its copy and mean-minus-alone (+13.0 [+9.9, +16.2]) cells.
14. §6.3: "330-780x" defined (vs 25-50 % MFU; ~1,300-1,600x below peak). "Not CPU / not GPU sharing" -> "unlikely".
15. §7.1 rule 9: added `PROBE_EVERY=500` > 136 steps. Rules 7 / 9 are no longer marked as not re-derived.
16. §7.2: the seed-1 VSI failures came 9:31 / 9:43 after the S1 seed-1 checkpoints, not a flat ~9:40.
17. §7.3: the F1 numerics caveat for the recovery evals. Items 4 / 7 become M2-possible once the S2 weights are
    fetched. New item 13 `s2_sft_routed` (prompt mismatch in the best-system comparison) and item 14 (2B hint in
    video mode).
18. §8: the "~50x" is a judgement, not a power calculation. Added a go / no-go gate after F1 / F2 before the matrix,
    an effective-prompt field in the VSI JSON, and a routed `s2_sft` partner. Promoted "fetch the S2 checkpoints and
    run the S2 holdout ablation on the M2" to the first M2 job: it is the fastest route to a rule-1 reading.
19. §8.1 C3 and §9: removed the `probe step` check. The probe postdates training, so those lines cannot exist.
20. Appendix A: marked two rows as recomputed. The duplicated pose bias is equal only to within 1.8e-8.

**Material coverage.** All 9 VSI files are used, including both modes of `base_plain_2b` and the `measurements`,
`maps` and `objects` fields. All 6 checkpoints are used. Every REPORT section is used. The static reference lines
(M2 pilot "0.8B ... -0.04 ± 0.05", "S2 pilot 44.1 vs control 42.3") predate the weekend and are correctly not used
as results.

**Not verifiable from these materials:**
- Everything that needs the server logs: the per-phase split, the gate exception, timeout vs crash, and what the
  second launcher ran.
- The Mac-side numbers: the M2 subset 24.7 / 53.6, the cross-stack -0.5, the direction labels 68.2 vs 76.4 %, the
  Mac prefill and training rates.
- The simulations: selector staleness, rank gating, the AdamW null, the fusion what-ifs beyond the scores above.
- The §5.3 function-level forensics.
- The op and FLOP counts of §2.3 / §6.3 (only the arithmetic on them was checked), and the external Conv3d reports.
- The 2B step budget (57 / 71 is not reproduced by the 4B formula), the ~6-7 GPU-h per VSI run, and the S2
  checkpoint size.
