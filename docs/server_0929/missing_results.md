# Missing results: what the pre-registered reading needs, why it is missing, how to recover it

Scope: the server run of 2026-09-26..29 (`scripts/server_weekend.sh`, roles real / control / sft / small), as seen in
the only material we have: the 09-29 09:45 KST snapshot (`REPORT.md` + 9 `vsi/*.json`) and 6 S1 `final.pt`.
All times below are **UTC** (the server clock: REPORT.md says "Generated 2026-09-29 00:31" and its archive mtime is
00:31:11 UTC). KST = UTC + 9.

Scripts (all CPU, no model loaded), outputs next to them in `outputs/server_0929/analysis/`:

| script | what it computes | output |
|---|---|---|
| `missing_audit.py` | planned vs present VSI files; every REPORT comparison recomputed with `weekend_report.paired_bootstrap` or marked missing | `missing_audit.out` |
| `timeline.py` | every timestamp we have (STATUS lines quoted in REPORT, tar mtimes of JSONs and checkpoints, fix commit) and the intervals derived from them | `timeline.out` |
| `keyset_version.py` | dates each VSI JSON to the `eval_vsi_local.py` commit whose output dict has exactly its key set | `keyset_version.out` |
| `resolution_60_vs_288.py` | paired-interval width of existing pairs at 60 videos (M2 subset) vs 288 | `resolution_60_vs_288.out` |

Everything that exists was checked for integrity: all 10 (run, mode) prediction sets have n = 5,130, 288 videos and
the identical question-id set (`missing_audit.out` §1), and all 7 comparisons the report prints were reproduced to
the digit.

---

## 1. Can rules 1-9 of docs/SERVER_WEEKEND.md be read?

| rule | needs | state | number (if readable) |
|---|---|---|---|
| 1 Geometry content, S2 (key) | `s2_control.json`; seed-1 pair `s2_real_seed1.json` + `s2_control_seed1.json`; `p7_ablation_s2.json` | **not readable** — none of the four exists (`s2_real_seed1` was still running at the snapshot) | - |
| 2 Value of the geometry path | `s2_real`, `s2_sft` | readable | s2_real − s2_sft = **−1.6 [−3.1, −0.1]** |
| 3 Gate 1, spatial (lmms-eval) | `gate/base`, `gate/base_video`, `gate/trained` results | **not readable** — no lmms-eval results anywhere; the three `.done` markers are false (§3.3). Local proxy "S2 real − base best format" also missing (`base_plain.json` absent) | - |
| 4 Gate 2, general (VideoMME) | same lmms-eval runs | **not readable** | - |
| 5 Gate 3, latency | `latency_stream.json`, `latency_live.log` | readable (in REPORT) | question-only TTFT 164 ms (< 1 s), drift 1.0 (< 1.2) |
| 6 Training vs prompting | `s2_real`/`base_hint`, `s2_real_routed`/`base_routed` | readable | +8.5 [+6.7, +10.3]; routed +6.7 [+5.1, +8.2] |
| 7 Learning curve | `s2_real_step*.json` | **cannot exist for this run** — no intermediate checkpoints were ever written (§3.4) | - |
| 8 Is S1 needed / size | `s2_joint.json`; `s2_real_2b.json`, `s2_control_2b.json` | **not readable** — all three absent | - |
| 9 Probe table (the path) | `<arm>/probe.jsonl` | **cannot exist for this run** — the probe was added to the code after all training had finished (§3.4) | - |

So of the nine rules, 2, 5 and 6 are readable; 1, 3, 4 and 8 are recoverable with evaluation-only GPU time; 7 and 9
need retraining.

## 2. Planned outputs vs present

Source: `missing_audit.out` §2-§4 and the REPORT status table.

| output | producer (role) | state | most likely cause (evidence in §3) |
|---|---|---|---|
| `vsi/base_plain.json` (4B, image + video mode) | sft / p3 | **absent**; p3 not marked done | 2 x 5,130 generations in one job under `timeout 6h` (§3.2); its FAILED line is outside the 15-line window (§3.7) |
| `vsi/base_hint.json` | sft / p3 | present (09-26 15:04, pre-09-26 eval code) | - |
| `vsi/base_routed.json` | sft / p3 | present (09-28 13:49, post-fix) | first attempt: build_model bug (§3.1) |
| `vsi/s2_real.json` | real, seed 0 | present (09-28 22:31, **not** from the orchestrator, §3.5) | orchestrator's own attempt FAILED 22:30:57 (timeout, §3.2) |
| `vsi/s2_real_routed.json` | real, seed 0 | present (09-28 20:12, not from the orchestrator) | - |
| `vsi/s2_real_hint.json` | real, seed 0 | **absent** | FAILED 22:30:57 together with s2_real (timeout, §3.2) |
| `vsi/s1_real.json` | real, seed 0 | present (09-28 22:30:41, orchestrator) | - |
| `vsi/s2_real_step*.json` | real, seed 0 | **never scheduled** | no step checkpoints (§3.4) |
| `vsi/s2_real_seed1.json` | real, seed 1 | **absent — in flight** (started 22:30:59 on run422142, GPU 0) | pre-fix attempt FAILED 09-27 19:17:14 (build_model bug) |
| `vsi/s2_control.json` **(key)** | control, seed 0 | **absent** | build_model bug (seed-0 batch ~09-27 02:3x, FAILED line outside the window); no post-fix control attempt visible |
| `vsi/s2_control_routed.json` | control, seed 0 | **absent** | same |
| `vsi/s1_control.json` | control, seed 0 | **absent** | same |
| `vsi/s2_control_seed1.json` | control, seed 1 | **absent** | FAILED 09-27 19:29:08 (build_model bug); no post-fix attempt visible |
| `vsi/s2_sft.json`, `vsi/s2_sft_hint.json` | sft | present (09-28 13:46 / 19:49, not from the orchestrator) | first attempt FAILED 09-27 10:38:16 (bug) |
| `vsi/s2_joint.json`, `vsi/s2_joint_routed.json` | sft | **absent** | FAILED 09-27 10:38:16 / 10:38:44 (bug); no post-fix attempt visible |
| `vsi/base_hint_2b.json`, `vsi/base_plain_2b.json` | small | present (09-28 13:21 / 22:04) | first attempt FAILED 09-27 05:55:05, both in the same second (timeout, §3.2) |
| `vsi/s2_real_2b[_routed].json`, `vsi/s2_control_2b[_routed].json` | small | **absent** | build_model bug (~09-26 23:55, outside the window); no post-fix attempt visible |
| `p7_ablation_s2.json` | sft | **absent** | FAILED 09-27 10:38:16 (bug: eval_geometry_ablation builds with the minimal namespace) |
| `p7_ablation_2b.json` | small | **absent** | same bug (outside the window) |
| `p5_ablation_s1.json` | **no role** | **never scheduled** — REPORT has a section for it, `server_weekend.sh` has no job that writes it | - |
| `gate/base`, `gate/base_video`, `gate/trained` | control, control, sft | **absent**, but `gate_*.done` all exist | lmms-eval exits 0 on an exception (§3.3) |
| `gate_check.txt` verdict | sft | FAILED 09-27 10:59:28 (FileNotFoundError: no results in gate/base) | consequence of the above |
| `latency_stream.json`, `latency_live.log` | sft | present | - |
| `<arm>/probe.jsonl` | train_arm | **never produced** | §3.4 |
| `<arm>/step{N}.pt` | train_arm | **never produced** | §3.4 |

S1 VSI is scheduled for seed 0 only (`s1_real`, `s1_control`); S1 seed 1 and 2B S1 have no VSI job by design.

## 3. Why things failed — evidence

### 3.1 The build_model bug (fixed in 2a80a5b, 09-28 02:03 UTC) — all pre-fix `--config` evaluations

Every `--config` evaluation with LoRA on crashed in `build_model` (`args.no_geometry`), per the commit message: all
`s1_*`/`s2_*` VSI runs, `base_routed`, and both holdout ablations. The timing agrees: the seed-1 VSI jobs failed
9:30:33 (real) and 9:42:51 (control) after the seed-1 S1 checkpoints were written (`timeline.out`) — that is one S2
arm (136 steps at the logged 7.1 h per 100 steps = 9:39:21) followed by a crash at launch; `s2_sft`, `s2_joint` and
`p7_ablation_s2` failed in the same second (10:38:16) and `s2_joint_routed` 28 s later (it starts after `s2_joint`
returns). This bug does **not** explain anything after 09-28 02:03 and does not touch the lmms-eval gate (the
wrapper `src/live3r/eval/lmms_live3r.py` never calls `build_model`).

### 3.2 `timeout 6h` in `vsi_job` — the most likely cause of the post-fix failures

One VSI run = 5,130 questions on one GPU. The evidence that this takes about 6 h on this server:

- **real node, 09-28**: `s2_real` and `s2_real_hint` FAILED in the same second, 22:30:57; they were launched together
  with `s1_real` (`role_real`, seed 0: `vsi_set 0 s2_real` ∥ `vsi_job 1 s2_real_hint` ∥ `vsi_set 2 s1_real`), and the
  orchestrator's `s1_real.json` (relative weight path = the orchestrator's `$OUT`) was written at 22:30:41. Two jobs
  dying in the same second while the orchestrator lives on (seed 1 started at 22:30:59) is the signature of a
  timeout; a launch at 16:30:57 makes `s1_real` a 5:59:44 run that finished 16 s inside the limit. A crash cannot
  explain it: the jobs were started together with `s1_real`, hours earlier.
- **small node, 09-27**: `base_hint_2b` and `base_plain_2b` FAILED in the same second, 05:55:05, i.e. launched
  together at 23:55:05 if killed at 6 h; the logged 2B S2 rate puts the end of 2B S2 at ~00:07 (±~15 min), consistent.
  Both were `--model` runs (no `--config`), so the build_model bug cannot be the cause.
- **outside the orchestrator**: `s2_sft.json` → `s2_sft_hint.json` = 6:02:42 (the two run back-to-back on one GPU in
  `role_sft`) — a single run that would itself have been killed by `timeout 6h`.

Consequences: (a) any re-run through the unchanged script will lose runs at random at 6 h, (b) a two-mode run
(`base_plain`, 2 x 5,130 generations) can never finish, (c) `role_sft` re-runs p3 first (`p3.done` is absent), so a
re-run of that role burns 6 h on `base_plain` before it reaches `s2_joint`, the gate and the ablation. The kill itself
cannot be confirmed without `vsi/s2_real.log`: a timeout leaves the last per-video progress line (`[k/288] … 누적 N문항`)
and no Traceback.

### 3.3 The gate markers are false positives

`gate_base.done`, `gate_base_video.done`, `gate_trained.done` exist (REPORT status table) but no `*results*.json`
exists under `gate/base`, `gate/base_video` or `gate/trained` (REPORT: "No gate results yet"; `check_gate.py`:
FileNotFoundError). Mechanism: lmms-eval 0.7.3 catches every exception of an evaluation and only logs it
(`.refs/lmms-eval/lmms_eval/__main__.py` l.610-616: `traceback.print_exc()` + "Error during evaluation" +
`results_list.append(None)`, re-raised only with `--verbosity DEBUG`), so the process exits 0; `run_gate.sh`
(`set -e`) and `gate_job` trust the exit code and mark the step done.
Timing confirms that nothing was evaluated: `gate_trained` was launched no earlier than 10:38:16 on 09-27 and was
marked done before the gate check at 10:59:28 — at most 21:12 for vsibench + videomme + mmstar. On the control
node the whole seed-0 evaluation batch (both gate runs included) sits between the estimated end of S2 control
(~02:31) and the estimated start of S1 control seed 1 (~02:21) — i.e. it lasted about zero minutes (±~20 min
estimate from logged step rates, `timeline.out`).
**Root cause: cannot be determined from these materials** (it is not the build_model bug). The swallowed traceback is
in `outputs/weekend/gate_base.log`, `gate_base_video.log`, `gate_trained.log` (grep "Error during evaluation").
Candidates to check in that order: model registration (`--model live3r` resolves through the
`lmms_eval.models` entry point, which exists only after `pip install -e .`; in this machine's env the entry-point
list is empty), the `Qwen3_5` base-class import (`lmms_eval.models.simple.qwen3_5`, emits a missing
`qwen_vl_utils` warning here), and the vsibench/videomme task data paths under `HF_HOME`.

### 3.4 Rules 7 and 9 were impossible by construction

- Learning curve: `curve_steps` asks for the checkpoints nearest 25/50/75 % of `S2_STEPS` in multiples of
  `SAVE_EVERY`. With `S2_STEPS=136` (2B: 71) and `SAVE_EVERY=500`, `int(136*f/500+0.5)*500 = 0` for all three
  fractions (computed with the script's own awk expression), so no `step{N}.pt` was saved and no curve job was
  scheduled. `SAVE_EVERY=34` would have given steps 34/68/102.
- The tiny step count follows from the budget: `14 h × 3600 × 0.30 samples/s × 0.9 / 128 = 106` S1 steps and
  `18 h × … = 136` S2 steps (reproduced exactly) — 0.02 epoch, because the smoke measured 0.30 samples/s against the
  planned ~19.
- Probe: `--probe-ann/--probe-every` were added to `train_arm` in 68ee746 (09-28 04:20 UTC); the last training arm
  ended ~09-27 19:29. No arm ever ran with the probe (and `PROBE_EVERY=500` > 136 steps would not have fired
  anyway). Recoverable only by retraining.

### 3.5 A second launcher produced half of the post-fix results, outside STATUS

- Weight paths: `eval_vsi_local.py` stores `args.weights` verbatim. `s1_real.json` has
  `outputs/weekend/s1_real/final.pt` (the orchestrator's relative `$OUT`); `s2_real`, `s2_real_routed`, `s2_sft`,
  `s2_sft_hint` have `/group-volume/wooyeol/vlm-live/outputs/weekend/...` (absolute).
- Timing contradictions for a single orchestrator launch: `s2_sft.json` (13:46:59) was written 2:53 **before**
  `base_routed.json` (13:49:52) although `role_sft` finishes p3 before any `s2_sft` eval; `s2_sft_hint` ran 6:02:42;
  `base_hint_2b` and `base_plain_2b` (launched together in `role_small`) finished 8:43:39 apart; `s2_real.json` was
  written 33 s **after** STATUS logged `FAILED VSI s2_real`.
- REPORT lists PID files for `eval_1gpu` and `eval_4gpu` next to 16 `run4xxxxx-vlm-live` hosts.
- Code version: every post-09-27 JSON has the 35-key output of `eval_vsi_local.py` at dc46758..68ee746
  (`keyset_version.py`); the `--config` ones must therefore come from 2a80a5b, ed91f27, e59e3ca or 68ee746.
  `base_hint.json` has the 24-key output of the run's starting code.

Implications: the REPORT's failure list does not cover these runs, so a job of that launcher may still be running
or may have failed silently (for example an `s2_control` run). Check before re-running, otherwise two processes
write the same JSON. The existing JSONs are valid regardless (complete, same questions, same weights file).

### 3.6 `s1_real_seed1/final.pt` is truncated

The gzip stream ends early (EOFError after 335,544,320 decompressed bytes); the extracted file has the full size
(380,308,683) but is zero from byte 346,291,712 on and has no zip central directory (`zipfile.is_zipfile` False).
Parsing the local headers: 24 of 28 storages are complete, including all 21 deepstack-projector tensors; lost are
`pose_projector.net.1.{weight,bias}` (2560x768, 2560) and `pose_projector.net.3.{weight,bias}` (2560x2560, 2560).
With `expose_pose_token: true` the model uses them, so the file is unusable for evaluation (usable for a
weight-level comparison of the projectors only). The other five checkpoints load on CPU, all finite, fp32
(95.1 M parameters for 4B, 68.7 M for 2B).

### 3.7 The REPORT hides the earliest failures

`weekend_report.py` prints `failed[-15:]`; the 15 shown start at 09-27 05:55:05. Every seed-0 VSI crash, the first
`base_plain` / `base_routed` failures, the 2B S2 evaluation crashes and `p7_ablation_2b` are outside the window.
`grep FAILED outputs/weekend/STATUS` shows them.

## 4. Recovery list, in priority order

### 4.0 Before anything

1. **What is still running.** `s2_real_seed1` started 22:30:59 alone on run422142 GPU 0; under `timeout 6h` it ends by
   04:30:59 UTC = **13:31 KST today** (done, or FAILED). Also check the `eval_1gpu` / `eval_4gpu` launcher:
   `ps aux | grep eval_vsi_local`, `nvidia-smi`, `ls -la --time-style=full-iso outputs/weekend/vsi/`.
2. **Send back the logs** that settle §3: `grep FAILED outputs/weekend/STATUS`, `tail -50` of `vsi/s2_real.log`,
   `vsi/s2_real_hint.log`, `vsi/s2_control.log`, `gate_base.log`, `gate_trained.log`, and `cat gate/general.env`.
3. **Do not re-run through the unchanged orchestrator.** Two reasons: `timeout 6h` (§3.2) and the ordering (`role_sft`
   re-runs p3 first; `role_control` starts seed 1 only after its seed-0 batch, which includes two lmms-eval gate runs
   of up to 14 h each once the false markers are deleted). Either commit a one-line change (proposed:
   `timeout "${VSI_HOURS:-12}h"` in `vsi_job`, and in `gate_job` mark only when
   `find "$OUT/gate/$s" -name '*results*.json'` finds a file), or run the jobs by hand as below.
4. **Commit for the new runs.** The existing JSONs came from one of 2a80a5b..68ee746. Since then only the
   scale-anchor path (off without a `scale_anchor` config option) and new eval flags that default to off have
   changed, so HEAD should behave the same for plain / hint / routed; the strict choice is to evaluate at 68ee746,
   or to re-run both arms of a comparison at the same commit.

### 4.1 Manual commands (the exact `vsi_job` / `ablation` command lines, no timeout, STATUS-tagged)

```bash
cd /group-volume/wooyeol/vlm-live
export PYTHONPATH=$PWD/src HF_HOME=/group-volume/wooyeol/hf_cache TOKENIZERS_PARALLELISM=false
O=outputs/weekend; C4=configs/server_4b.yaml; C2=configs/server_2b.yaml
vsi() { local g=$1 n=$2; shift 2; [[ -f $O/vsi/$n.json ]] && { echo "$n exists"; return; }
  echo "$(date '+%F %T') [manual] VSI $n started on $(hostname -s) GPU $g" >> $O/STATUS
  ( CUDA_VISIBLE_DEVICES=$g python scripts/eval_vsi_local.py "$@" --videos 288 --seed 1 --out $O/vsi/$n.json \
      > $O/vsi/$n.log 2>&1 && echo "$(date '+%F %T') [manual] VSI $n done" >> $O/STATUS \
      || echo "$(date '+%F %T') [manual] FAILED VSI $n - $O/vsi/$n.log" >> $O/STATUS ) & }
abl() { local g=$1 n=$2 st=$3 cfg=$4 w=$5 c=$6; [[ -f $O/$n.json ]] && return
  ( CUDA_VISIBLE_DEVICES=$g python scripts/eval_geometry_ablation.py --config $cfg --stage $st --weights $w \
      --control-weights $c --ann data/sensenova.holdout.jsonl --media-root data/sensenova_media --n 500 \
      --out $O/$n.json > $O/$n.log 2>&1 || echo "$(date '+%F %T') [manual] FAILED $n - $O/$n.log" >> $O/STATUS ) & }
```

The report needs no change to see these: it globs `vsi/*.json` and reads `p5/p7_ablation_*.json`. VSI has no
`.done` markers — `vsi_job` skips a run iff its JSON exists; to redo one, delete `vsi/<name>.json`.

### 4.2 The list

GPU-hour estimates: a full VSI run is ~6-7 GPU-h on one H100 (§3.2: 5:59:44 for `s1_real` with three jobs on the
node; 6:02:42 for `s2_sft_hint`); the 2B was not faster (its base runs took > 6 h), which suggests the time is not
spent in the LLM (answers are capped at 16 new tokens). Routed runs add OWLv2 on 32 keyframes per video: 6-8 GPU-h (not measured). Ablation: at most 3 GPU-h (its
timeout; no completed run to measure). lmms-eval gate runs: never completed, 14 h timeouts each; the numbers
below are guesses.

| # | item | rules | command | GPU-h | M2 instead? |
|---|---|---|---|---|---|
| 1 | `s2_control.json` | **1 (key)** | `vsi 0 s2_control --config $C4 --weights $O/s2_control/final.pt --modes oracle-image` | 6-7 | no (no S2 weights local) |
| 2 | `s2_real_seed1.json` | 1 (seed check) | nothing if it lands by 13:31 KST (the JSON is written only at the end, so no partial file); else `vsi 1 s2_real_seed1 --config $C4 --weights $O/s2_real_seed1/final.pt --modes oracle-image` | 0 or 6-7 | no |
| 3 | `s2_control_seed1.json` | 1 (seed check) | `vsi 2 s2_control_seed1 --config $C4 --weights $O/s2_control_seed1/final.pt --modes oracle-image` | 6-7 | no |
| 4 | `p7_ablation_s2.json` | 1 (loss check) | `abl 3 p7_ablation_s2 sft $C4 $O/s2_real/final.pt $O/s2_control/final.pt` | ≤ 3 | no |
| 5 | `s2_joint.json` | 8 (is S1 needed) | `vsi 4 s2_joint --config $C4 --weights $O/s2_joint/final.pt --modes oracle-image` | 6-7 | no |
| 6 | `s2_real_2b.json`, `s2_control_2b.json` | 8 (size, 2B content) | `vsi 5 s2_real_2b --config $C2 --weights $O/s2_real_2b/final.pt --modes oracle-image`; same for `s2_control_2b` on GPU 6 | 12-14 | no |
| 7 | `p7_ablation_2b.json` | 8 (2B loss check) | `abl 7 p7_ablation_2b sft $C2 $O/s2_real_2b/final.pt $O/s2_control_2b/final.pt` | ≤ 3 | no |
| 8 | gate: debug, then `gate/{base,base_video,trained}` + verdict | 3, 4 | see 4.3 | ~0.5 + ~30-40 | reproduce the crash only |
| 9 | `base_plain.json` | 3 (local proxy: base best format), report's format-confound rows | `vsi 0 base_plain --model /group-volume/wooyeol/models/Qwen3.5-4B --modes oracle-image,oracle-video` (or `--modes oracle-video` only: enough for `base_best` if video > image, as on M2 48.6 vs 24.7 and the server 2B 42.7 vs 24.5) | 12-14 (video only 6-7) | possible but 2 x ~12 h |
| 10 | `s1_control.json` | report "S1: real − control" (no numbered rule) | `vsi 1 s1_control --config $C4 --weights $O/s1_control/final.pt --modes oracle-image` | 6-7 | **yes** (§5) |
| 11 | `p5_ablation_s1.json` | report section (never scheduled) | `abl 2 p5_ablation_s1 align $C4 $O/s1_real/final.pt $O/s1_control/final.pt` | ≤ 3 | **yes** (§5) |
| 12 | `s2_real_hint.json` | report "S2 real: + hint − plain" | `vsi 3 s2_real_hint --config $C4 --weights $O/s2_real/final.pt --format-hint --modes oracle-image` | 6-7 | no |
| 13 | routed extras: `s2_control_routed`, `s2_joint_routed`, `s2_real_2b_routed`, `s2_control_2b_routed` | report routed rows | as the plain line plus `--scene-map --no-map-image --object-map --detector owlv2 --detector-device cuda --route-facts --route-objects --min-frames 3 --detect-every 1` | 6-8 each | no |
| 14 | learning curve | 7 | retrain S2 real with `SAVE_EVERY=34` (→ steps 34/68/102), then 3 VSI runs | ~78 (8 GPUs x ~9.7 h) + ~20 | no — do it in the next run |
| 15 | probe table | 9 | retrain at ≥ 68ee746 with `PROBE_EVERY` ≤ S2_STEPS/8 | same as 14 | no — next run |

Items 1-7 (the pre-registered rules 1 and 8, both seeds, both ablations) are ~36-48 GPU-h (the upper end if
`s2_real_seed1` has to be redone): one 8-GPU node, ~7-8 h wall if eight concurrent evaluations do not slow each other down (not measured: with three per node a run took
~6 h; watch the per-video lines in `vsi/<name>.log` after 30 min and extrapolate).

### 4.3 The gate

```bash
cat $O/gate/general.env                                # expect GENERAL=videomme (p2_videomme is done)
grep -n "Error during evaluation" -B 40 $O/gate_base.log | head -80    # the swallowed exception
rm -f $O/gate_base.done $O/gate_base_video.done $O/gate_trained.done  # false markers
rm -rf $O/gate/base $O/gate/base_video $O/gate/trained
# smoke with exceptions re-raised, a few docs:
CUDA_VISIBLE_DEVICES=7 python -m lmms_eval --model live3r --model_args \
  config=$C4,keyframe_budget=32,eval_path=live3r,enable_thinking=False,use_geometry=False \
  --tasks vsibench --batch_size 1 --limit 4 --verbosity DEBUG --output_path /tmp/gate_smoke
# once the smoke writes a *results*.json, the three measurements on three GPUs:
GEN=$(cat $O/gate/general.env)
gate() { local g=$1 s=$2; shift 2
  ( CUDA_VISIBLE_DEVICES=$g env "$@" ONLY=$s bash scripts/run_gate.sh $C4 $O/s2_real/final.pt $O/gate > $O/gate_$s.log 2>&1
    find $O/gate/$s -name '*results*.json' | grep -q . && touch $O/gate_$s.done \
      || echo "$(date '+%F %T') [manual] FAILED gate $s (no results json) - $O/gate_$s.log" >> $O/STATUS ) & }
gate 5 base $GEN; gate 6 base_video; gate 7 trained $GEN
# after all three:
env $GEN ONLY=check bash scripts/run_gate.sh $C4 $O/s2_real/final.pt $O/gate > $O/gate_check.txt 2>&1 && touch $O/gate_check.done
```

## 5. What the M2 can do with the checkpoints we have

Local and valid: S1 `final.pt` of `s1_real`, `s1_control`, `s1_control_seed1`, `s1_real_2b`, `s1_control_2b`.
No S2 checkpoint is local, so **nothing in rules 1 and 8 can run on the M2**. `configs/m2_zeroshot_4b.yaml` /
`m2_zeroshot_2b.yaml` build the same structure as `server_4b.yaml` / `server_2b.yaml` (the server-only keys
`stride`, `freeze`, `expose_pose_token`, `mlp_ratio`, `attn_implementation` equal the defaults in
`src/live3r/config.py`) with LoRA off, which for an S1 checkpoint is exact (the S1 LoRA would be zero-init).

| M2 job | command (on MPS, later — not run here) | time | value / limits |
|---|---|---|---|
| S1 real − control, VSI (item 10) | `eval_vsi_local.py --config configs/m2_zeroshot_4b.yaml --weights outputs/server_0929/ckpt/s1_{real,control}/final.pt --modes oracle-image --videos 60 --seed 1` (the M2 subset: 60 videos, 1,151 questions, verified) | 2 x 2.5 h | ± ~3.3 points (s1_real − base_hint: ±1.4 at 288 videos, ±3.3 at 60). Pair both arms on the M2 — do not pair an M2 run with a server JSON (bf16 path differences change answers) |
| p5_ablation_s1 (item 11) | `eval_geometry_ablation.py --config configs/m2_zeroshot_4b.yaml --stage align --weights …/s1_real/final.pt --control-weights …/s1_control/final.pt --ann data/sensenova_1k/sn1k.holdout.jsonl --media-root data/sensenova_1k/media --n 95` | < 1 h (guess) | 95 samples, not the server holdout (a random 2,000 records, not local); S1 saw 13,568 of 830,135 records, so overlap is small but not excluded. A 1-sample smoke of this path exists in this directory (`ablation_s1_4b_smoke.json`, written by another analysis) |
| 2B S1 real − control (not planned) | same with `m2_zeroshot_2b.yaml` and the 2B S1 files | 2 x ≤ 2.5 h | extra |
| S1 control seed noise (not planned) | `s1_control_seed1` vs `s1_control` | 2 x 2.5 h | the S1 real seed-1 file is unusable (§3.6) |
| gate crash reproduction | `pip install -e . --no-deps` in the env (registers the entry point), then the smoke of 4.3 with `configs/m2_zeroshot_4b.yaml` | minutes | may reproduce the server exception without waiting for its log |

If S2 checkpoints were brought over (~0.6 GB each, estimated from 95 M projector + 61 M LoRA parameters at fp32),
the M2 could do the key pair on 60 videos in 2 x 2.5 h — but at 60 videos the paired interval is 2.3-2.8x wider
than at 288 (s2_real − s2_sft: ±1.5 at 288, ±4.1 at 60; `resolution_60_vs_288.out`), and the effects seen so far
are 0.4-1.6 points. Rule 1 needs the server's full run.

## 6. What these materials cannot settle, and what would

| question | what settles it |
|---|---|
| Were `s2_real`/`s2_real_hint` (22:30:57) and the 2B base pair (05:55:05) killed by the timeout? | `tail` of their `vsi/*.log`: last `[k/288]` line, no Traceback |
| Why did the gate runs fail? | `gate_base.log` / `gate_trained.log` ("Error during evaluation" + traceback) |
| Why did `base_plain` fail the first time (09-26)? | `grep FAILED outputs/weekend/STATUS`, `vsi/base_plain.log` |
| Were the control / sft / small roles restarted after the fix, and is anything of theirs still running? | STATUS lines after 2026-09-28 02:03, `PID.*`, `ps` on the nodes |
| What did the `eval_1gpu` / `eval_4gpu` launcher run, with which command and timeout? | its script / shell history and logs |
| Does `s2_real_seed1` finish inside 6 h? | STATUS after 04:31 UTC (13:31 KST) |
| Real per-run eval times (to size the recovery) | the timing line `문항 5130 · 디코딩 …s · oracle-image …s/문항` at the end of every finished `vsi/*.log` |
