# Four-day run: the Sensenova-only baseline (8x H100, from 2026-09-26)

One command runs everything unattended: zero-shot baseline → S1 (real vs control) → ablation → S2 LoRA (real vs
control) → the pre-registered gate plus VSI for every arm and an S2 learning curve → one report.

**The dataset is SenseNova-SI only.** It is the in-house system's base dataset, and every later addition (the
geometry-only questions of I-14 first) will be measured against the numbers this run produces.

The file and directory names still say "weekend" (`scripts/server_weekend.sh`, `outputs/weekend/`); the plan
is now sized for four days: one epoch for S1 and one for S2.

## Start (5 minutes, then stay ~45 minutes)

```bash
cd /group-volume/wooyeol/vlm-live
git diff > /group-volume/wooyeol/server_changes_0926.patch   # keep server-side edits, if any
git stash -u && git fetch origin && git reset --hard origin/main
git log --oneline -1                                          # the commit whose message names this document
mkdir -p outputs/weekend
nohup bash scripts/server_weekend.sh > outputs/weekend/master.log 2>&1 &
tail -f outputs/weekend/STATUS
```

Stay until STATUS shows **`p1 done`** and the throughput line (about 45 minutes; up to 20 minutes more if
`data/sensenova.jsonl` still has to be prepared). Everything that can go wrong with the environment, the data,
CUT3R or DDP shows up by then. After that nothing needs a person. `outputs/weekend/REPORT.md` is refreshed every
30 minutes, so progress can be checked from anywhere.

Paths are the ones from `NEXT_STEPS_SERVER.md` and live in **`configs/server_4b.yaml`**:

- model `/group-volume/wooyeol/models/Qwen3.5-4B`
- CUT3R `third_party/CUT3R` and `checkpoints/cut3r_512_dpt_4_64.pth`
- Sensenova `data/sensenova_si_800k.json` and `data/sensenova_media`
- `HF_HOME=/group-volume/wooyeol/hf_cache`

p0 downloads whatever is missing (the model, CUT3R, OWLv2) and prepares `data/sensenova.jsonl` if it is not there.

## Timeline (estimates; the budget adapts to the speed p1 measures)

| phase | GPUs | time | what it answers |
|---|---|---|---|
| p0 preflight | – | 5–25 min | environment, tests against the real tokenizer, assets, Sensenova prepared |
| p1 smoke | 1 → 8 | ~30 min | training runs: dummy geometry / CUT3R / 8-GPU S1 / 8-GPU S2 with LoRA. Measures the speed |
| p1e eval paths | 4 | ~10 min | the ablation, local VSI (plain and routed) and lmms-eval each run on a few items |
| p2 eval data | background | 1–3 h | VSI-Bench 5.7 GB, MMStar, VideoMME 101 GB (~200 GB once unpacked) |
| p3 zero-shot VSI | 3 | ~1 h | **the baseline on all 288 videos**: base plain (image and video mode), + format hint, routed facts |
| p4 S1 | 4 + 4 | ≤ 26 h | projector alignment, real vs control, one epoch each |
| p5 S1 ablation | 1 | ~20 min | does the projector carry the geometry's *content*? |
| p6 S2 | 4 + 4 | ≤ 32 h | LoRA SFT, real vs control, one epoch each |
| p7 evaluation | 8 | 3–6 h | gate (VSI, VideoMME, MMStar), VSI for every arm, S2 learning curve, S2 ablation, latency |
| p8 report | – | seconds | `outputs/weekend/REPORT.md` |

Planned total about 65–70 hours. If the speed estimate from p1 was optimistic, each stage may run up to 1.25×
its hours before it stops itself and saves; even then the whole run stays inside four days (≈ 85 h).

## The design

Each training stage runs **two arms side by side, 4 GPUs each**. They are identical in everything: data order, seed,
steps, learning rate, effective batch 128. The one difference is that the **control** arm gives every sample
another sample's geometry (`--geom-control shuffled`).

Anything that can be learned without geometry, the control learns too: the answer format, Sensenova's answer
distribution, the injection path used as a soft prompt. So **real − control is what the geometry's content is worth**.
The M2 pilots could not resolve this number: S1 gave −0.04 ± 0.05 in loss, and S2 gave +1.9 [−0.9, +4.4] VSI on 905
samples. Here it is measured at scale.

The control S2 starts from the control S1, so the control is the whole two-stage recipe without geometric content.

The baseline side (p3) evaluates the base model on the full VSI-Bench in three ways:

- as the gate defines it (video mode is the base's best format)
- with the one-line format instruction (M2: +5.1)
- with the best zero-shot prompting found on M2 (routed room facts + object distances, I-27/I-29: 48.6 → 56.0 on 60
  videos, no training)

Trained models are evaluated on the live path: 32 uniform keyframes in image mode. Each is evaluated plain,
with the format hint (S2 real), and with the same routed facts. The report can then answer whether training
beats one line of prompt, and whether training and measurements stack. The saved S2 checkpoints nearest 25 / 50 /
75 % are evaluated as well (learning curve): does more Sensenova keep helping, or does the numeric prior take over
as it did at 905 samples?

## Budget

- **One epoch per stage** (`S1_SAMPLES=epoch`, `S2_SAMPLES=epoch`: every record of `data/sensenova.jsonl` once,
  ~829k), with hour caps of 26 (S1) and 32 (S2). If p1's measured speed says an epoch does not fit the cap, the
  budget is cut to what fits and the report states the number of samples actually used.
- **How the fit is computed**:
  - p1 measures samples/s on 8 GPUs (smokes C and D, 60 steps). One arm is estimated at half that, minus 15% for
    sharing the CPU and disk with the other arm.
  - S2 is re-planned from the speed S1 actually ran at.
  - The numbers go to `budget.env` and `budget_s2.env`, and a re-run reuses them.
- **Stops**:
  - `--max-hours` (1.25× the stage hours) is a soft stop that still saves `final.pt`.
  - `timeout` kills a hung job an hour after that.
- **Checkpoints**: every 500 steps (about 2 h) each arm writes `step{N}.pt` (weights only, for the learning curve)
  and `resume.pt` (weights, optimizer, schedule and data position, overwritten each time).
- **Overriding** — set any of these when starting, e.g. two S2 epochs:
  `S2_SAMPLES=1660000 S2_HOURS=60 nohup bash scripts/server_weekend.sh …` (also `S1_SAMPLES`, `S1_HOURS`,
  `SAVE_EVERY`).

## Reading the report — the pre-registered rules

1. **Geometry content, S1** (p5): value of content = loss[control, shuffled] − loss[real]. A 95% lower bound above
   0 means the projector carries content. M2 gave −0.04 ± 0.05.
2. **Geometry content, S2**, the key number: VSI `s2_real − s2_control`, a video-level paired bootstrap. An interval
   above 0 means the trained model uses the geometry's content. The S2 ablation (holdout loss) is the second view.
3. **Gate 1, spatial**: lmms-eval `vsibench`, S2 real − the base's best format ≥ +1.0.
4. **Gate 2, general**: `videomme` Δ ≥ −1.0. MMStar is reported only. (If VideoMME could not be downloaded, the
   general gate is MMStar instead, and the report says so.)
5. **Gate 3, latency**: question-only TTFT < 1 s (live design, map in the prefix); ingest drift < 1.2.
6. **Training vs prompting**:
   - `s2_real` vs `base_hint`: on M2 a small SFT (44.1) lost to one line of instruction (53.7).
   - `s2_real_routed` vs `base_routed`.
7. **Learning curve**: VSI at 25 / 50 / 75 / 100 % of S2. Rising = more data helps; flat or falling after 25 % =
   the recipe, not the amount, is the limit.

## Outputs (`outputs/weekend/`)

| file | what |
|---|---|
| `STATUS` | one line per step. `FAILED` lines name the log to look at |
| `REPORT.md` | every number in one place, refreshed every 30 min — **this is what to send** |
| `master.log` | everything the script printed |
| `p0_preflight.log`, `p1_smoke_[a-d].log`, `p1e_*.log` | preflight and smoke logs |
| `s1_real/`, `s1_control/`, `s2_real/`, `s2_control/` | `final.pt`, `step{N}.pt` every 500 steps, `resume.pt` |
| `s1_real.log` … | training logs (loss, `inj`, samples/s, memory, `sync OK`); every attempt appended |
| `budget.env`, `budget_s2.env` | the steps each stage was given and the speeds behind them |
| `p5_ablation_s1.json`, `p7_ablation_s2.json` | holdout-loss ablations |
| `vsi/<run>.json` | VSI per run, with per-question scores (the report's bootstrap reads these) |
| `gate/{base,base_video,trained}/`, `gate/general.env`, `gate_check.txt` | lmms-eval results, which general task was used, the verdict |
| `latency_stream.json`, `latency_live.log` | gate 3 |

## What to send back

- `outputs/weekend/REPORT.md`.
- If anything `FAILED`: those lines of `STATUS` and the last 50 lines of the log each one names.

## When something goes wrong

- **Re-run the same command.** Finished phases and jobs are skipped and failed ones retry. A training arm that
  was cut off **continues from its last `resume.pt`** (at most ~2 h lost), with the same data order and steps as
  if nothing had happened.
- **`p0 FAILED`**: nothing else runs. Send the tail of `p0_preflight.log`.
- **`p1 FAILED`**: training is broken, so nothing after it runs; the downloads continue. Send the smoke log it names.
- **`p1e FAILED`**: an evaluation path is broken, but training continues. Tell us before day 3, when p7 needs it.
- **A training arm `FAILED`**: the other arm keeps its result. A re-run resumes only the failed arm.
- **A bad sample**: a non-finite loss skips that sample; a non-finite gradient skips that step (no update, the
  schedule still advances). Only three skipped steps in a row stop an arm — that is divergence, not a bad sample.
  The report's training table shows the counts.
- **VideoMME never arrived**: the gate runs with MMStar as the general task (`gate/general.env`). To redo it with
  VideoMME once the download has succeeded, delete `gate/general.env`, `gate/*.ok` and the `gate/base*`,
  `gate/trained` directories, then re-run.
- **Stop everything**: `bash scripts/server_weekend.sh stop`. This ends the script and everything it started:
  torchrun, the evaluations and the downloads.
- **Disk**: about 300 GB for VideoMME (archives plus the unpacked copy), about 50 GB for checkpoints.

## What changed relative to NEXT_STEPS_SERVER.md steps 8–10

- **Arms**: two 4-GPU arms (grad-accum 32) replace one 8-GPU run per stage (grad-accum 16). The effective batch
  is the same, 128, and the real and control arms now run at once.
- **Paths**: `configs/server_4b.yaml` replaces the `--base-model / --geometry-checkpoint / --cut3r-repo` overrides.
  It is needed because `eval_vsi_local.py` takes the base model from the config.
- **Eval data**: `scripts/fetch_eval_data.py` downloads and unpacks it once. Several lmms-eval runs start at the
  same time, and each would otherwise unpack VideoMME into the same directory.
- **Resume**: `train.py --resume` continues from `resume.pt`; `--max-hours` is a soft time cap decided for all
  DDP ranks together.
- **Bad samples**: S1 passes over text-only records (no gradient path through the projector); a non-finite loss or
  gradient skips the sample or the step instead of ending the run; the failure-rate stop judges after 2,000
  samples per worker, not 200.
- **Gate**: `run_gate.sh` takes `ONLY=base|base_video|trained|check`, so the three measurements run on three GPUs.

## Deliberately not in this run

- **Geometry-only questions (I-14, `data/geomqa`)**: kept out so that this run is the Sensenova-only reference. They
  come next, measured against this run.
- **Streaming VSI** (`run_streaming_eval.sh`): the live path here is uniform keyframes. On M2, the halving selector
  cost nothing against oracle selection (49.9 vs 48.6).
- **A second S2 epoch**: with one epoch each the run takes about three of the four days. The learning curve says
  whether a second epoch would be worth it; it can be started as a follow-up with `S2_SAMPLES` (see Budget).
