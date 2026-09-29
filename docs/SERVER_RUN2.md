# Run 2 — one training round at one epoch per arm, and the weekend recovered on the same stack

Script: `scripts/server_run2.sh`. Why: `docs/SERVER_RESULTS_20260929.md` — the weekend arms trained on ~2 % of an
epoch (16 GPU-s per sample against a planned ~0.42), plain SFT beat the geometry arm by 1.6, and the key comparison
(real − control) was never produced. The lead's call (2026-09-29): run one proper training round, take it as the
reference, recover every missing weekend result; 4B first; the existing-system (70.2) recipe arrives on 09-30 and
feeds the round after this one.

## What changed relative to the weekend

| change | why |
|---|---|
| **Patch embed as a matmul** (`patch_embed: linear`, configs `server_{4b,2b}_v2.yaml`) in training and every evaluation | the bf16 Conv3d is the prime suspect for the image path costing 4-5 ms per patch row (§2.3 of the results). The linear form is exact in fp32; in bf16 both forms sit the same distance (3e-2) from an fp32 reference, so answers differ only by bf16 noise — which is why **everything** in run 2, including the weekend checkpoints, is evaluated on this one stack |
| **flash-linear-attention** installed by p0 (`--no-deps`); d0 compares it with the torch fallback | transformers silently runs the Gated DeltaNet in pure torch without it (~67k extra kernel launches per sample) |
| **d0 diagnostic** (`scripts/diagnose_throughput.py`, ~10 min, first node) | settles both suspects; its verdict lines go to STATUS and the report; a fla mismatch bans fla on every node |
| **Budget from the production configuration**: calib = 12 S1 + 12 S2 steps at 8 GPUs × accum 16 | the weekend sized from a smoke at accum 2 that ran 1.7× slower, so every arm stopped at ~53 % of its hours |
| S1 short (5 % of an epoch = 41.5k samples), S2 **one epoch** (830k), capped by `S1_HOURS=4`, `S2_HOURS=22` | S1 learned the answer format (both arms alike); the content question needs data (the weekend S2 had 17.4k samples) |
| Control donors: buffer 64 (`--donor-buffer`) | the weekend's 8 gave a same-shape donor only 60 % of the time — a real/control confound |
| **VSI sharded 8 ways per node** + `scripts/merge_vsi_shards.py`; 12 h timeout per shard | the weekend's 6 h cap killed whole evaluations; the merge re-aggregates with lmms-eval's own function (tested: identical to one process) |
| Every trained arm evaluated **plain, + hint, + room facts** | every comparison gets a same-prompt partner (the weekend mixed prompt conditions) |
| Room facts only (`--route-facts`) | distance facts are now below the trained models; direction facts wait for the mirror fix (results §4) |
| Step checkpoints every quarter of S2, probe every eighth, resume point every ~5 % | learning curve and probe table (rules 7, 9) — the weekend had neither; a crash costs minutes, not a quarter |
| Gate: a 4-sample `--verbosity DEBUG` smoke first; a gate step is done only if lmms-eval wrote results | the weekend's three gate runs exited 0 with no results and left false markers |
| Report: run-2 comparisons, a Debiased column, every FAILED line, the d0 / budget section | results §8.2 |

## Start (4 nodes × 8 H100)

```bash
cd /group-volume/wooyeol/vlm-live && git pull && mkdir -p outputs/run2
ROLE=real     nohup bash scripts/server_run2.sh > outputs/run2/master.real.log 2>&1 &      # node 1
ROLE=control  nohup bash scripts/server_run2.sh > outputs/run2/master.control.log 2>&1 &   # node 2
ROLE=sft      nohup bash scripts/server_run2.sh > outputs/run2/master.sft.log 2>&1 &       # node 3
ROLE=recover  nohup bash scripts/server_run2.sh > outputs/run2/master.recover.log 2>&1 &   # node 4
tail -f outputs/run2/STATUS
```

Fewer nodes: roles run in order on one node, e.g. `ROLE=real,control`. The weekend's checkpoints and results are
read from `outputs/weekend` (`WK=`) and never modified. Allocation shorter than ~1.5 days: lower `S2_HOURS` (the
budget then covers less than an epoch, and the report says how much).

## Timeline (estimates; d0 and calib replace them with measurements within the first hour)

| phase | if the fixes work (~0.5 GPU-s/sample) | if they do not (16 GPU-s/sample) |
|---|---|---|
| p0 + d0 + calib | ~30 min | ~2 h |
| S1 (41.5k samples) | ~45 min | capped at 4 h (~0.5 % of an epoch) |
| S2 (1 epoch) | ~14-18 h | capped at 22 h (~1.5 % of an epoch — stop and profile instead, see below) |
| evaluations per arm (3 VSI views + curve, 8 shards) | ~1-2 h | ~20 h |
| recover node: 26 VSI runs + 4 ablations, then joint | ~5-8 h, then joint | days — reduce the list |

**Go / no-go (read STATUS within the first hour):** the `calib:` line prints the measured samples/s per node. At
≥ ~8 samples/s the plan above holds. Below ~2 samples/s the fixes did not take — stop the training roles
(`bash scripts/server_run2.sh stop`), keep `recover` running, and send `d0_throughput.txt` + `calib_s*.log`.

## Roles

| node | work |
|---|---|
| `real` | S1 real → S2 real → VSI `s2_real`, `s2_real_hint`, `s2_real_routed_room`, `s1_real`, curve `s2_real_step*` |
| `control` | the same with shuffled geometry (`s2_control*`, `s1_control`) → holdout ablations `p5_ablation_s1`, `p7_ablation_s2` (real vs control) |
| `sft` | plain SFT, no geometry → VSI `s2_sft*` → latency (gate 3) → gate smoke → gate base / base_video / trained (`s2_real`) → verdict |
| `recover` | the weekend on this stack: `wk_s2_{real,control,sft}`, `base_hint`, seed-1 pair, `wk_s2_joint`, `wk_s1_*`, hint views, `base_plain` (image + video), `base_routed_room`, the 2B (`base_*_2b`, `wk_s2_*_2b` + hint); ablations `wk_p5_ablation_s1`, `wk_p7_ablation_s2{,_seed1}`, `wk_p7_ablation_2b`; then **S2 joint** (no S1, one epoch) → VSI |

## Reading the report — pre-registered (write nothing else into the conclusions)

1. **Geometry content:** `s2_real − s2_control` (plain and hint), with the weekend pair on the same stack next to it.
   The seed noise is the weekend's `wk_s2_real_seed1 − wk_s2_real` (one epoch has no second seed yet).
2. **Value of the geometry path:** `s2_real − s2_sft`; `s2_control − s2_sft` is the value of a content-free injection.
3. **What one epoch buys:** `s2_real − wk_s2_real`, `s2_sft − wk_s2_sft`.
4. **Is S1 needed:** `s2_joint − s2_real`.
5. **Facts after training:** `s2_real_routed_room − s2_real_hint` and the same for SFT.
6. **Gates:** gate 1 (lmms-eval VSI, trained − base's best format ≥ +1.0), gate 2 (VideoMME ≥ −1.0), gate 3 (latency).
7. **Stack check:** `wk_s2_real` (new stack) − `old_s2_real` (the weekend's file) should be ≈ 0; a large value means the
   fixes changed accuracy, not only speed.
8. **Learning curve and probe:** where `shuffled − real` leaves 0 in the probe table, and whether the numeric
   answers collapse ("2" on 80 % of counting at the weekend).
Every Δ is also read on VSI-Debiased (the report's last column); per-type claims need a Holm correction over 8 types.

## What to send back

- `outputs/run2/REPORT.md`, `outputs/run2/d0_throughput.txt`, `outputs/run2/budget.env`, `grep FAILED outputs/run2/STATUS`.
- The VSI result files `outputs/run2/vsi/*.json` (not the shards) — the per-question analysis needs them.
- If anything failed: the log each FAILED line names (last 50 lines).
- Checkpoints only on request (`s2_{real,control,sft}/final.pt` with sha256): M2-side ablations use them.
