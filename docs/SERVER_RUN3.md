# Run 3 — our implementation of the reference design, trained on the VG-LLM data (no CoT: live)

Script: `scripts/server_run3.sh`. Why: `docs/REFERENCE_70_ANALYSIS.md` — the 70.83 system's score comes from data
(2.03 M samples of VSI-domain QA), resolution (long side 512) and a simple input fusion trained at that scale. We may
use the data; we do not take their code. So this run trains **our implementation** of the design on **their data
mix** without its chain-of-thought part (the live system answers at once - the lead's call, 2026-09-30), against a
no-geometry arm at equal data, on our live evaluation path.

## 2026-10-13 (evening) — the live path holds: plain 72.7; next: the strategies on it and the gate (`ROLE=strategies,plain`)

**Live result (`ROLE=live`, all 5,130 questions):** the trained plain model with frames chosen as the video streams
and the LLM capped at 128 frames scores **72.7**, +1.4 [+0.2, +2.7] over its offline baseline (71.3, 32 oracle
frames) and above the reference's 70.83. Choosing causally costs −0.8 [−2.2, +0.6] at 32 frames; 32 → 128 frames
+2.2 [+0.9, +3.6] (abs distance +4.0 and object size +1.3 significant); 128 → 256 +0.3 (128 is enough). gated −
plain stays at −0.2 [−1.4, +1.1] at 128 frames: the geometry trades appearance order (−4.7 [−9.1, −0.7]) against rel
direction (+3.2, n.s.). **The live design for VSI: plain SFT model, 1 fps frames capped at 128, causal selection.**

**The gate never ran:** the smoke (`--limit 4`) included VSI-Bench, whose lmms-eval aggregation raises `KeyError`
without every question type - fixed: the smoke runs MMStar only (`SPATIAL=`), the full gate is unchanged. The gated
arm's own gate reached 4,659 / 9,330 requests and stopped with its node (no error).

**Caution - two prompt forms.** Run 3 trained, and was first evaluated (plain 71.3, gated 71.0), at `5d963f6`, where
every image block was followed by "\n"; the live rows above used today's form without it (`image_newline: false`,
3da9a51). So "live 32 − oracle 32" and "live 128 − oracle 32" mix the frame choice with the format. `ROLE=format`
(~30 min) separates them: the 32-frame baseline in today's form (`plain_oracle32_now`) and 128 live frames in the
trained form (`plain_live128_nl`, `configs/server_4b_v3_plain_nl.yaml`). (Run 3 also had 200 LoRA modules, not 248:
`in_proj_a/b` came later - irrelevant to evaluation, the added adapters start at zero.)

**Next, `ROLE=format,strategies,plain`** (one command, in that order; `format` first):
1. `strategies` (~2-3 h, 8 GPUs): the zero-shot strategies on the live design (plain, 128 live frames) - the format
   line alone (the reference for the routed runs), + the prior hints, + object distances, + directions (the I-49
   instance rule), and the whole zero-shot stack (room, distances, directions, hints). The measurements use
   `configs/server_4b_v3_plain_facts.yaml` (the same model; CUT3R only for the measurements, with the frame-count
   scale calibration) and OWLv2, fetched once into `checkpoints/` if missing. Ends with a bundle.
2. `plain` (~7 h on 3 GPUs; training and VSI are done and skipped): the gate smoke, then the base, base video and
   plain gates in parallel and the verdict (`gate_check_plain.txt`). Ends with the final bundle.

```
cd /group-volume/wooyeol/vlm-live && git pull
OUT=outputs/run3_live_2x8_20260930 ROLE=format,strategies,plain nohup bash scripts/server_run3.sh \
  > outputs/run3_live_2x8_20260930/master.next.$(hostname -s).log 2>&1 &
```

Send `results_to_send.tar.gz` when STATUS says `role.strategies done` (the strategy results) and again at the end
(with the gate).

## 2026-10-13 — the result, and next: the trained arms on the live path (`ROLE=live`)

**Result (REPORT.md of 10-13; all 288 videos / 5,130 questions; 32 uniform frames as images at 192 tokens a frame -
the reference's protocol):** plain **71.3**, gated **71.0** against the reference's 70.83. gated − plain −0.2
[−1.6, +1.1]: the geometry path adds nothing at this data scale (their own runs: 70.83 vs 70.89). Room facts now cost
on the trained arms (gated −0.8 [−1.6, −0.1], its room size 75.3 → 67.6; plain 71.3 → 70.6): the trained models size
rooms better than the measurement does (MRA ~66). The video path loses 6.1 to images. The gated learning curve still
rises (68.9 / 69.9 / 70.6 / 71.0 at the quarters).

**Next, `ROLE=live` (evaluation only, ~2-3 h on one 8-GPU node):** the plain and gated final checkpoints on the live
path - frames chosen as the video streams (halving over the 1 fps grid; the budget is the cap, so nothing depends on
the video's length - BRAINSTORM I-48), the LLM capped at 32, 128 and 256 frames (256 = every 1 fps frame of a VSI
video). It answers what choosing causally costs at 32 frames, what more frames buy (the reference: 72.18 at 1 fps),
and whether gated − plain moves with more frames. REPORT.md is then regenerated with the new rows and comparisons,
and everything to send back is bundled into `results_to_send.tar.gz` (the report, STATUS, the arms' VSI results with
their per-question answers, and the gate logs).

```
cd /group-volume/wooyeol/vlm-live && git pull
OUT=outputs/run3_live_2x8_20260930 ROLE=live nohup bash scripts/server_run3.sh \
  > outputs/run3_live_2x8_20260930/master.live.$(hostname -s).log 2>&1 &
tail -f outputs/run3_live_2x8_20260930/STATUS
```

It can start now, next to the running `plain+base+gated` process (that one only waits on the gated arm's VideoMME gate
on GPU 7; the live role uses all 8 GPUs and shares GPU 7 with it). No longer needed: control and gated_wn (the key
comparison is answered) - `touch outputs/run3_live_2x8_20260930/role.{control,gated_wn}.skipped` keeps an auto run
from resuming them. Send back `outputs/run3_live_2x8_20260930/results_to_send.tar.gz`. It carries `gate_smoke.log`:
the lmms-eval gate fails every time ("wrote no results"), and its exception is in that log.

## What is new (all ours, written from the understood design)

| piece | what | where |
|---|---|---|
| gated input fusion | a merger (CUT3R's last decoder layer, 2×2 merge, MLP 3,072 → 4,096 → d_llm) gives one geometry token per visual token; a per-token gate `g = sigmoid(w·[N(v); N(geo)] + b)` mixes it in, `(1 − g)·N(v) + g·N(geo)` replaces the visual embedding at the LLM input. One stage with LoRA | `src/live3r/fusion/gated.py`, `fusion.mode: gated_input` |
| data registry and mixture | the 17 annotation files on `/group-volume/VG-LLM_data` with their media roots; mixes `ref70` (the 70.83 run's 2,029,000 records), `all6`, and **`live`** = ref70 without the six CoT variants (1,443,777 records: spar_234k, llava_hound_64k, vlm3r_scannet / scannetpp / arkit, sensenova_800k, sqa3d, scanqa); one dataset over several sources, fixed subsets and repeats (`name%N`) | `configs/data/vgllm.yaml`, `src/live3r/data/mixture.py`, `train.py --mix` |
| record forms | `images` as the image key; system messages dropped; videos (files or frame directories - llava_hound is 10-56 JPEG frames per clip) given as 4-8 frames (one per 2 s) as images — the live path's form; a `<think>…</think>` block in any answer stripped to the final answer (no reasoning is trained) | `src/live3r/data/datasets.py`, `data.video_as_images` |
| SPAR marks | `spar_info` points and boxes drawn by our code onto the frames both the VLM and CUT3R see: a fixed frame order per question type (as the reference's team describes its rule), types the reference never draws kept unmarked, only the kind the question names, 20 px on the original image, labels on the video and map types; checked against their renderings of two records; a record of an unknown type whose placement is not certain is left out | `src/live3r/data/marks.py`, `data.marks: draw` |
| no evaluation scene in training | the data pass found one VSI-Bench scene (`scene0685_00`, ScanNet) in the SPAR sets; every record whose media paths name a VSI-Bench scene is left out | `prepare_mix.py --exclude-scenes` |
| resolution | the VLM input at long side 512, the rule CUT3R's input already follows, so the two token grids coincide (144 tokens for 16:9 frames, 192 for 4:3 — the reference's operating point) | `data.vlm_long_side`, `VisionSpec.long_side` |
| preparation | each source once as JSONL + offset index (lazy loading; a 740 MB JSON array per rank would not fit), a 300-record holdout per source, a media check | `scripts/prepare_mix.py` |
| data pass | fields, media forms, marks, answer kinds, and the scene ids in the paths against VSI-Bench's 288 scenes | `scripts/inspect_reference_data.py` |

**What the data pass showed (2026-09-30, `outputs/reference_data_srv/`):** all 17 files as registered, media 100 %
found except vsi_590k (not in any mix); spar_234k is 100 % `spar_info` (points/boxes in ~95 %, the rest only a type);
the vlm3r sets are 32 frames with VSI-style one-word answers; sqa3d / scanqa 32 frames with short answers; the CoT sets
are `<think>…</think>` answers after a system message. Scene overlap with VSI-Bench: one ScanNet scene in SPAR, none
elsewhere.

## Arms (same data order, same steps, same seed)

| arm | config | what it answers |
|---|---|---|
| **gated** | `server_4b_v3.yaml`: gated_input, LayerNorm on both streams, gate weights from N(0, 0.02) (g spread 0.14-0.86 at step 0, as the reference's) | the design as the reference trains it |
| **plain** | `server_4b_v3_plain.yaml`: no geometry (fusion none), same LoRA / data / resolution | **gated − plain** = what the geometry path adds at this data scale (the key comparison) |
| **control** | gated with shuffled geometry (`--geom-control shuffled`) | gated − control = the value of the geometry's content |
| **gated_wn** | `server_4b_v3_wn.yaml`: the reference's wonorm variant - the output mixes the raw visual token with the normalised geometry | the other forward the 70.83 run may have used (its train.log says wonorm, its config gated_token; their two branches scored 70.86 / 70.83) |
| base | no training, same resolution | the reference point for every arm |

Recipe (the reference's, from its adapter, trainer state and its team's answers): LoRA r16 α32 on every linear
layer of the LLM (the same 12 names: 248 modules on the 4B, none in the vision tower), the merger's LayerNorm on each
768-d token before the 2×2 merge (`fusion.merger_norm: before`; 3,072 → 4,096 → 2,560), the gate from N(0, 0.02)
(`gate_init_std`), CUT3R under bf16 autocast (`geometry.options.autocast`), the VLM input shrunk to long side 512 only
when larger and CUT3R fed the same image, training text without the newline after an image (`data.image_newline:
false`), lr 5e-5 LoRA / 1e-5 merger + gate, AdamW (0.9, 0.999), the VSI prompt "These are frames of a video.\n{q}\n
{post}" (lmms-eval's default, also the form of the vlm3r training records), global batch 64
(8 GPUs × accum 8, or 4 × 16), cosine, warmup 3 %, weight decay 0.01, one full pass of the `live` mix (no time cap).

Step-0 check on the M2 (`scripts/check_fusion_init.py`, 4B, 12 Sensenova holdout samples, random merger):

| start | held-out loss | mean abs Δ vs base | first answer token differs |
|---|---|---|---|
| base | 1.707 | 0 | 0 % |
| gated (norm, b = 0) | 1.514 | 0.520 | 75 % |
| norm, b = −4 | 1.500 | 0.366 | 58 % |
| gated_raw (no norm, b = −4) | 1.723 | 0.038 | 8 % |
| no norm, b = 0 | 2.013 | 0.532 | 33 % |

The LayerNorm makes the visual tokens 5.6× louder (norm 9.0 → 50.6): the reference-style start is far from the base
model in behaviour, yet its loss on spatial QA is lower before any training (0.8B: 2.68 → 2.40 with the norm alone).
The raw start keeps the base model. Which start trains to the better model is what gated vs gated_raw measures.

## Start — one command, runs to the end

```bash
cd /group-volume/wooyeol/vlm-live && git pull && mkdir -p outputs/run3
nohup bash scripts/server_run3.sh > outputs/run3/master.$(hostname -s).log 2>&1 &
tail -f outputs/run3/STATUS
```

Run the same command on **each of the two nodes**. A node may have 8 or 4 GPUs: it uses the GPUs it was given
(the scheduler's `CUDA_VISIBLE_DEVICES`, else all it sees) with the global batch kept at 64. As run 2: the nodes share
the roles (claims and heartbeats on the shared volume, takeover after 30 min without a heartbeat); re-running resumes;
`bash scripts/server_run3.sh stop` ends a node's run. Queue: **gated → plain → base → control → gated_wn**, after
the one-time steps:

1. **p0** on every node (packages, flash-linear-attention unless run 2 banned it, the tests, the annotation files).
2. **data** (once, CPU): the data pass → `outputs/run3/reference_data/DATA.md`; the `live` mix → `data/vgllm/` and
   `outputs/run3/PREPARED.md` (tens of minutes; per source: kept, held out, dropped and why, marks, reasoning stripped).
3. **calib** (once): 12 steps of the gated arm at the production settings (with 3 dumped samples: check the prompt
   forms and a drawn mark) → `budget.env`: the steps of one pass. d1: one profiled step.

On two nodes: gated and plain start together (the key comparison first); the node that finishes first runs base (hours)
and then control; the other takes gated_wn.

**What to expect (their own runs, `docs/REFERENCE_70_ANALYSIS.md` §9):** the same data with VGGT scored 71.42; CUT3R
scored the same as no geometry in their CoT runs (70.83 vs 70.89). So **gated ≈ 70-70.5 and gated − plain ≈ 0** are the
reproduction's expected outcome; the run tells whether our implementation reaches that level and whether CUT3R helps
in our form. **The 09-30 attempt** (2 × 8 GPUs) stopped in p0: collecting `tests/test_sim3.py` failed on the new container.
The suite passes under its transformers (5.18, checked on the M2); p0 now runs the tests one node at a time without
writing bytecode into the shared checkout, and the FAILED line carries the error itself.

| one arm, one pass of `live` (≈1.4 M samples) | 8 GPUs | 4 GPUs |
|---|---|---|
| at the reference's speed (6.48 samples/s per 8 GPUs) | ≈ 60 h | ≈ 120 h |
| all five roles on two nodes | ≈ 5-6 days | ≈ 10-11 days |

8 GPUs per node halves everything; the script runs either way (and a mix of one 8- and one 4-GPU node).

Evaluation per arm: VSI on the live path (32 uniform keyframes as images) plain and with the room facts; gated also
in video mode (the offline protocol) and at every quarter checkpoint; the lmms-eval gate (VideoMME Δ ≥ −1.0 is our
general-ability target) per arm against base runs at this resolution; latency of the gated design; the holdout
ablation gated vs control.

Open details of the reference (template, resolution rule, CUT3R feature, other runs' scores - incl. its no-CoT
"all" mix, our expected level): requested in [`docs/REFERENCE_70_REQUEST.md`](REFERENCE_70_REQUEST.md); answers that
arrive before the start are folded into the configs.

## Knobs

`MIX` (live | ref70 | all6 | any registry spec, e.g. `live,sqa3d%50`), `TRAIN_HOURS` (0 = no cap; > 0 caps each arm
at the calibrated speed, with `GO_EPOCHS` 0.2 as the no-go line), `ROLE` (a subset in order, no claims), `PROBE_SRC`
(vlm3r_scannet), `R2` (run 2's outputs, for the cross-run rows), `NGPU` (dry runs without nvidia-smi only).

## Reading (REPORT.md)

1. **gated − plain** (key): positive with its interval above 0 = the geometry path pays at this data scale.
2. **gated − control**: the content of the geometry, not the extra path.
3. **gated vs the reference's 70.8** (their protocol differs: video path; our `gated_video` row is the closest).
4. **gate**: VideoMME within −1.0 of the base = general ability kept.
5. **gated − gated_raw**: whether the start matters after training.
6. Rows against run 2 mix data, design and resolution - context, not a controlled comparison.

What to send back: `outputs/run3/REPORT.md`, `STATUS`, `PREPARED.md`, `reference_data/DATA.md`, `budget.env`,
`calib.log` (head: the dumped samples), and any FAILED log it names.
