# Four-day run on 4 nodes x 8 H100: the Sensenova-only baseline (from 2026-09-26)

One script, one **role per node**. The nodes never talk to each other over the network — there is no multi-node
job. They coordinate through markers in the shared `outputs/weekend/` directory: a node that needs another
node's result waits for its marker. The whole run is unattended after the first ~40 minutes on each node.

**The dataset is SenseNova-SI only.** It is the in-house system's base dataset, and every later addition (the
geometry-only questions of I-14 first) will be measured against the numbers this run produces.

The file and directory names still say "weekend" (`scripts/server_weekend.sh`, `outputs/weekend/`).

## Requirement: one shared filesystem

All nodes must see the same `/group-volume/wooyeol/vlm-live` (repo, data, `outputs/`). Check before starting:

```bash
touch /group-volume/wooyeol/vlm-live/outputs/weekend/.seen_from_$(hostname)   # on each node
ls /group-volume/wooyeol/vlm-live/outputs/weekend/.seen_from_*                 # on any node: all of them listed?
```

If the nodes do not share the volume, run only `real` and `control` on two nodes that do, or one node with the
single-node plan (`ROLE=real,control,sft` sequentially — slow).

## Start

On every node:

```bash
cd /group-volume/wooyeol/vlm-live
git diff > /group-volume/wooyeol/server_changes_0926_$(hostname).patch   # keep server-side edits, if any
git stash -u && git fetch origin && git reset --hard origin/main
git log --oneline -1                                          # the commit whose message names this document
mkdir -p outputs/weekend
```

Then one role per node (the order does not matter; nodes wait for each other):

```bash
ROLE=real     nohup bash scripts/server_weekend.sh > outputs/weekend/master.real.log 2>&1 &     # node 1
ROLE=control  nohup bash scripts/server_weekend.sh > outputs/weekend/master.control.log 2>&1 &  # node 2
ROLE=sft      nohup bash scripts/server_weekend.sh > outputs/weekend/master.sft.log 2>&1 &      # node 3
ROLE=small    nohup bash scripts/server_weekend.sh > outputs/weekend/master.small.log 2>&1 &    # node 4
tail -f outputs/weekend/STATUS                                # every node writes here, lines tagged [role]
```

Stay until each node has logged **`p1.4b.<role> done`** (or `p1.2b.small done`) and its throughput line — about
40 minutes; the first node to arrive also prepares `data/sensenova.jsonl` (10–20 min) while the others wait.
Everything that can go wrong with an environment, the data, CUT3R or DDP shows up by then. `REPORT.md` is
refreshed every 30 minutes by every node.

**Fewer nodes**: a node can take several roles in order, `ROLE=control,sft`. `real` + `control` are the core;
`sft` holds the eval-data downloads, the zero-shot baseline and the gate verdict; `small` is independent.

Paths live in **`configs/server_4b.yaml`** and **`configs/server_2b.yaml`**: models under
`/group-volume/wooyeol/models/`, CUT3R in `third_party/CUT3R` + `checkpoints/`, Sensenova at
`data/sensenova_si_800k.json` + `data/sensenova_media`, `HF_HOME=/group-volume/wooyeol/hf_cache`. Whatever is
missing (the two models, CUT3R, OWLv2, the prepared annotations) is fetched by the first node that gets there.

## The plan, per node

| node / role | GPUs | work, in order | ends about |
|---|---|---|---|
| 1 `real` | 8 | S1 real (1 epoch) → S2 real (1 epoch) → VSI: plain, + hint, routed, learning curve, S1 → **seed 1**: S1 → S2 → VSI | ~60 h |
| 2 `control` | 8 | S1 control → S2 control → VSI plain + routed, S1; gate base + base video → **seed 1**: S1 → S2 → VSI | ~62 h |
| 3 `sft` | 8 | eval-data downloads (background); zero-shot VSI on all 288 videos (base, + hint, routed); **plain SFT** (no geometry); **S2 joint** (no S1); gate trained + verdict; S2 ablation; latency | ~40 h |
| 4 `small` | 4 + 4 | the **2B** model: S1 real ∥ control → S2 real ∥ control → 2B zero-shot, VSI of both arms (plain, routed), ablation | ~40 h |

Per-arm speed: one 4B arm on 8 GPUs (grad-accum 16, effective batch 128) is estimated at ~19 samples/s → an
epoch of ~829k in ~12 h (S1) / ~15 h (S2). Caps: 14 h / 18 h, stretchable to 1.25× before an arm stops itself
and saves. Everything finishes inside three days; a second training seed on the core pair is what the fourth day
buys.

### What the extra arms answer

- **Plain SFT** (`s2_sft`, no geometry path at all): what Sensenova SFT alone gives the 4B. Then
  `s2_real − s2_sft` is the value of the whole geometry path, `s2_real − s2_control` the value of its *content*,
  and `s2_control − s2_sft` what a content-free injection (a learned soft prompt) is worth.
- **S2 joint** (`s2_joint`, projector + LoRA from scratch, no S1): is the alignment stage needed?
- **Seed 1** of the real and control arms: the training-seed noise of the key number. Without it a +1 point
  real − control difference cannot be told from seed variance.
- **2B**: the same real/control pair on Qwen3.5-2B. On M2, zero-shot 2B + measurements stayed 11 points below
  4B + prompt; does SFT close the gap? (The user's stated interest: a smaller LLM if the measurements allow it.)

## The design of the core comparison

Each training stage runs a **real** arm and a **control** arm that are identical in everything — data order,
seed, steps, learning rate, effective batch 128 — except that the control arm gives every sample another
sample's geometry (`--geom-control shuffled`). Anything learnable without geometry, the control learns too. So
**real − control is what the geometry's content is worth**. The M2 pilots could not resolve this number: S1 gave
−0.04 ± 0.05 in loss, S2 gave +1.9 [−0.9, +4.4] VSI on 905 samples.

The control S2 starts from the control S1, so the control is the whole two-stage recipe without geometric content.
Trained models are evaluated on the live path: 32 uniform keyframes in image mode; plain, with the format hint,
and with the routed measurements (I-27/I-29, the best zero-shot prompting from M2 when this run was defined:
48.6 → 56.0 on 60 videos; the run keeps that configuration throughout so its routed rows compare like with like —
the relative-direction facts found on 09-26 (I-33, → 57.8) go into the next run).

## Budget

- **One epoch per arm** (`SAMPLES=epoch`) with hour caps: 4B on 8 GPUs 14 h (S1) / 18 h (S2); 2B pair on 4 + 4
  GPUs 16 h / 20 h; plain SFT and joint 18 h. If p1's measured speed says an epoch does not fit a cap, that arm's
  budget is cut to fit and the report states the number of samples used.
- **Same steps for a pair**: the real and control nodes read one shared `budget.env`, written by whichever of them
  plans first (from its own smoke speed — the hardware is the same).
- **Stops**: `--max-hours` (1.25× the cap) is a soft stop that still saves `final.pt`; `timeout` kills a hung job
  an hour after that.
- **Checkpoints**: every 500 steps (~2 h) each arm writes `step{N}.pt` (weights, for the learning curve) and
  `resume.pt` (weights, optimizer, schedule, data position — overwritten each time).
- **Overriding** — env vars at start: `SAMPLES`, `S1_HOURS`, `S2_HOURS`, `S1_HOURS_2B`, `S2_HOURS_2B`,
  `SFT_HOURS`, `SEEDS` (default 2: seeds 0 and 1), `SAVE_EVERY`.

## Reading the report — the pre-registered rules

1. **Geometry content, S2** — the key number: VSI `s2_real − s2_control`, a video-level paired bootstrap. An
   interval above 0 means the trained model uses the geometry's content. Check it against the seed-1 pair and the
   S2 ablation (holdout loss).
2. **Value of the geometry path**: `s2_real − s2_sft`. If this is ≈ 0 the geometry path adds nothing over SFT.
3. **Gate 1, spatial**: lmms-eval `vsibench`, S2 real − the base's best format ≥ +1.0.
4. **Gate 2, general**: `videomme` Δ ≥ −1.0. MMStar is reported only. (If VideoMME could not be downloaded, the
   general gate is MMStar instead, and the report says so.)
5. **Gate 3, latency**: question-only TTFT < 1 s (live design, map in the prefix); ingest drift < 1.2.
6. **Training vs prompting**: `s2_real` vs `base_hint` (M2: a 905-sample SFT lost to one line of prompt),
   `s2_real_routed` vs `base_routed`.
7. **Learning curve**: VSI at 25 / 50 / 75 / 100 % of S2. Flat or falling after 25 % = the recipe, not the amount,
   is the limit.
8. **Is S1 needed**: `s2_joint − s2_real`. **Size**: `s2_real_2b − s2_real`, and 2B real − control.

## Outputs (`outputs/weekend/`)

| file | what |
|---|---|
| `STATUS` | one line per step from every node, tagged `[role]`. `FAILED` lines name the log to look at |
| `REPORT.md` | every number in one place, refreshed every 30 min — **this is what to send** |
| `master.<role>.log` | everything a node's script printed |
| `p0_<role>.log`, `p0_shared.log`, `p1_<4b/2b>_<role>_smoke_*.log` | preflight and smoke logs per node |
| `s1_real/`, `s2_real/`, `s1_control/`, … `s2_sft/`, `s2_joint/`, `*_seed1/`, `*_2b/` | `final.pt`, `step{N}.pt`, `resume.pt` per arm |
| `<arm>.log` | training logs (loss, `inj`, samples/s, memory, `sync OK`); every attempt appended |
| `budget.env`, `budget.sft.env`, `budget.2b.env` | steps per arm and the speeds behind them |
| `p7_ablation_s2.json`, `p7_ablation_2b.json` | holdout-loss ablations |
| `vsi/<run>.json` | VSI per run, with per-question scores (the report's bootstrap reads these) |
| `gate/{base,base_video,trained}/`, `gate/general.env`, `gate_check.txt` | lmms-eval results, which general task was used, the verdict |
| `latency_stream.json`, `latency_live.log` | gate 3 |
| `*.done`, `PID.<host>`, `lock.*` | markers, the running script on each node, one-time-work locks |

## What to send back

- `outputs/weekend/REPORT.md`.
- If anything `FAILED`: those lines of `STATUS` and the last 50 lines of the log each one names.

## When something goes wrong

- **Re-run the same command on that node.** Finished steps are skipped and failed ones retry. A training arm that
  was cut off **continues from its last `resume.pt`** (at most ~2 h lost), with the same data order and steps as
  if nothing had happened. Other nodes waiting for its result keep waiting (up to 48 h) and carry on when it
  lands.
- **`p0 FAILED` on a node**: that node runs nothing else. Send the tail of `p0_<role>.log` (or `p0_shared.log`).
- **`p1 FAILED` on a node**: that node's training is broken and it stops. The other nodes are not affected.
- **A training arm `FAILED`**: the node moves on to what does not depend on it (the next seed) and a re-run
  resumes only the failed arm.
- **A bad sample**: a non-finite loss skips that sample; a non-finite gradient skips that step (no update, the
  schedule still advances). Only three skipped steps in a row stop an arm — that is divergence, not a bad sample.
- **A node died holding a lock** (`waited … lock.<name> held for 3 h` in STATUS): `rm -rf outputs/weekend/lock.<name>`
  and re-run.
- **VideoMME never arrived**: the gate runs with MMStar as the general task (`gate/general.env`). To redo it with
  VideoMME once the download has succeeded, delete `gate/general.env`, `gate_*.done` and the `gate/base*`,
  `gate/trained` directories, then re-run the `control` and `sft` roles.
- **Stop a node**: `bash scripts/server_weekend.sh stop` on that node (ends its script, torchrun, evaluations and
  downloads). Repeat on each node to stop everything.
- **Disk**: about 300 GB for VideoMME (archives plus the unpacked copy), about 120 GB for checkpoints (13 arms).

## What changed relative to NEXT_STEPS_SERVER.md steps 8–10

- **One node per arm** (8 GPUs, grad-accum 16 — the effective batch 128 of step 8) instead of two half-node arms;
  the arms of a pair run on different nodes at the same time.
- **Paths**: `configs/server_4b.yaml` / `server_2b.yaml` replace the command-line overrides.
- **Eval data**: `scripts/fetch_eval_data.py` downloads and unpacks once; several lmms-eval runs would otherwise
  race unpacking VideoMME into the same directory.
- **Resume**: `train.py --resume`; `--max-hours` decided for all DDP ranks together.
- **Bad samples**: S1 passes over text-only records; a non-finite loss or gradient skips the sample or the step;
  the failure-rate stop judges after 2,000 samples per worker.
- **Plain SFT**: `train.py --no-geometry`.
- **Gate**: `run_gate.sh ONLY=base|base_video|trained|check`, so the three measurements run on different GPUs
  and nodes.

## Deliberately not in this run

- **Geometry-only questions (I-14, `data/geomqa`)**: kept out so that this run is the Sensenova-only reference.
- **Streaming VSI** (`run_streaming_eval.sh`): the live path here is uniform keyframes; on M2 the halving selector
  cost nothing against oracle selection (49.9 vs 48.6).
- **A 32-GPU job**: cross-node NCCL would have to be set up and debugged remotely, and the natural parallelism
  is across arms, not within one.
