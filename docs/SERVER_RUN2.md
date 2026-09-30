# Run 2 — one training round at one epoch per arm, and the weekend recovered on the same stack

Script: `scripts/server_run2.sh`. Why: `docs/SERVER_RESULTS_20260929.md` — the weekend arms trained on ~2 % of an
epoch (16 GPU-s per sample against a planned ~0.42), plain SFT beat the geometry arm by 1.6, and the key comparison
(real − control) was never produced. The lead's call (2026-09-29): run one proper training round, take it as the
reference, recover every missing weekend result; 4B first; the existing-system (70.2) recipe arrives on 09-30/10-01 and
feeds the round after this one.

## What changed relative to the weekend

| change | why |
|---|---|
| **Patch embed as a matmul** (`patch_embed: linear`, configs `server_{4b,2b}_v2.yaml`) in training and every evaluation | the bf16 Conv3d is the prime suspect for the image path costing 4-5 ms per patch row (§2.3 of the results). The linear form is exact in fp32; in bf16 both forms sit the same distance (3e-2) from an fp32 reference, so answers differ only by bf16 noise — which is why **everything** in run 2, including the weekend checkpoints, is evaluated on this one stack |
| **flash-linear-attention** installed by p0 (`--no-deps`); d0 compares it with the torch fallback | transformers silently runs the Gated DeltaNet in pure torch without it (~67k extra kernel launches per sample) |
| **d0 diagnostic** (`scripts/diagnose_throughput.py`, ~10 min, first node) | settles both suspects; its verdict lines go to STATUS and the report; a fla mismatch bans fla on every node |
| **Budget from the production configuration**: calib = 12 S1 + 12 S2 steps at 8 GPUs × accum 16 | the weekend sized from a smoke at accum 2 that ran 1.7× slower, so every arm stopped at ~53 % of its hours |
| S1 short (5 % of an epoch = 41.5k samples), S2 **one epoch** (830k), capped by `S1_HOURS=6`, `S2_HOURS=72`; no-go below half an epoch (`GO_EPOCHS`) | S1 learned the answer format (both arms alike); the content question needs data (the weekend S2 had 17.4k samples) |
| Control donors: buffer 64 (`--donor-buffer`) | the weekend's 8 gave a same-shape donor only 60 % of the time — a real/control confound |
| **VSI sharded 8 ways per node** + `scripts/merge_vsi_shards.py`; 12 h timeout per shard | the weekend's 6 h cap killed whole evaluations; the merge re-aggregates with lmms-eval's own function (tested: identical to one process) |
| Every trained arm evaluated **plain, + hint, + room facts** | every comparison gets a same-prompt partner (the weekend mixed prompt conditions) |
| Room facts only (`--route-facts`) | distance facts are now below the trained models; direction facts wait for the mirror fix (results §4) |
| Step checkpoints every quarter of S2, probe every eighth, resume point every ~5 % | learning curve and probe table (rules 7, 9) — the weekend had neither; a crash costs minutes, not a quarter |
| Gate: a 4-sample `--verbosity DEBUG` smoke first; a gate step is done only if lmms-eval wrote results | the weekend's three gate runs exited 0 with no results and left false markers |
| Report: run-2 comparisons, a Debiased column, every FAILED line, the d0 / budget section | results §8.2 |

## Start — one command, runs to the end

```bash
cd /group-volume/wooyeol/vlm-live && git pull && mkdir -p outputs/run2
nohup bash scripts/server_run2.sh > outputs/run2/master.$(hostname -s).log 2>&1 &
tail -f outputs/run2/STATUS
```

Nothing else is needed: no `ROLE`, no check-ins, no manual stop. On **one node** (8 H100) the script runs every role
in order. The **same command on more nodes** shares the work — each node claims the next unfinished role; a node whose
heartbeat stops for 30 min loses its claims and the role resumes on the next free node. Re-running the command after
an interruption resumes (markers, `resume.pt`, finished VSI shards are kept). `bash scripts/server_run2.sh stop` ends a
node's run. The weekend's `outputs/weekend` is read, never modified.

Order (the queue): **real → control → sft → recover → joint**, after p0 / d0 / calib / d1 which run once.

| step | what | time if the fixes work (~16 samples/s per node) |
|---|---|---|
| p0, d0, calib, d1 | environment + fla; diagnostic (~10 min); 12 + 12 steps at the production settings -> budget.env; one profiled training step | ~40 min |
| real | S1 (5 % of an epoch) -> S2 (one epoch) -> VSI plain / hint / room facts, S1, learning curve | ~16 h + ~2 h |
| control | the same with shuffled geometry -> VSI -> holdout ablations S1, S2 (real vs control) | ~16 h + ~2 h |
| sft | plain SFT, no geometry, one epoch -> VSI -> latency -> gate (4-sample debug smoke first) | ~15 h + ~6 h |
| recover | the weekend on this stack (26 VSI runs, 4 ablations) | ~5-8 h |
| joint | S2 without S1, one epoch -> VSI | ~16 h + ~1 h |
| **total, one node** | | **~3.5-4 days** (4 nodes: ~1 day) |

**The go / no-go is automatic.** calib measures the real speed; if the S2 cap (`S2_HOURS=72`) then covers less than
`GO_EPOCHS=0.5` of an epoch — the weekend's speed would give 0.15 — the script writes `NO_GO` with the numbers, skips
the training roles, still runs the weekend re-evaluation and the diagnostics, and the report leads with the reason.
Between 0.5 and 1 epoch it trains what fits (e.g. 3 samples/s -> 0.89 epoch) and the report says how much. To run
after fixing the stack: `rm outputs/run2/{NO_GO,calib.done,role.*.skipped}` and start the command again.

Knobs (environment variables, all optional): `S2_EPOCHS_WANT` (1), `S2_HOURS` (72), `S1_FRAC` (0.05), `S1_HOURS`
(6), `GO_EPOCHS` (0.5), `JOINT` (1; 0 drops the joint role), `VSI_HOURS` (12 per shard), `DONOR_BUFFER` (64).
`ROLE=real,control` runs just those roles in that order (the old one-role-per-node mode, no claims).

## What happens when something fails

A failed step writes a `FAILED` line naming its log and the role is marked `failed`; everything that does not need its
output continues, and steps that do stop waiting at once ("s2_real will not come: role real failed") instead of
hanging. Tested on the M2 with stubbed training and evaluation: one node end to end, two nodes sharing the queue (each
role ran once; d0, calib and the profile once), the automatic no-go at three speeds, and a failing S2 real.

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

- `outputs/run2/REPORT.md`, `outputs/run2/d0_throughput.txt`, `outputs/run2/d1_profile/profile.txt`,
  `outputs/run2/budget.env`, `grep FAILED outputs/run2/STATUS` (and `NO_GO` if it exists).
- The VSI result files `outputs/run2/vsi/*.json` (not the shards) — the per-question analysis needs them.
- If anything failed: the log each FAILED line names (last 50 lines).
- Checkpoints only on request (`s2_{real,control,sft}/final.pt` with sha256): M2-side ablations use them.
