# Run 3 — our implementation of the reference design, trained on the VG-LLM data

Script: `scripts/server_run3.sh`. Why: `docs/REFERENCE_70_ANALYSIS.md` — the 70.83 system's score comes from data
(2.03 M samples of VSI-domain QA), resolution (long side 512) and a simple input fusion trained at that scale. We may
use the data; we do not take their code. So this run trains **our implementation** of the design on **their data
mix**, against a no-geometry arm at equal data, on our live evaluation path.

## What is new (all ours, written from the understood design)

| piece | what | where |
|---|---|---|
| gated input fusion | a merger (CUT3R's last decoder layer, 2×2 merge, MLP 3,072 → 4,096 → d_llm) gives one geometry token per visual token; a per-token gate `g = sigmoid(w·[N(v); N(geo)] + b)` mixes it in, `(1 − g)·N(v) + g·N(geo)` replaces the visual embedding at the LLM input. One stage with LoRA | `src/live3r/fusion/gated.py`, `fusion.mode: gated_input` |
| data registry and mixture | the 17 annotation files on `/group-volume/VG-LLM_data` with their media roots; mixes `ref70` (the 70.83 run's 2,029,000 records) and `all6`; one dataset over several sources, fixed subsets and repeats (`name%N`) | `configs/data/vgllm.yaml`, `src/live3r/data/mixture.py`, `train.py --mix` |
| record forms | `images` as the image key; videos (files or frame directories) given as 4-8 frames (one per 2 s) as images — the live path's form; records with `spar_info` marks | `src/live3r/data/datasets.py`, `data.video_as_images`, `data.marks` |
| resolution | the VLM input at long side 512, the rule CUT3R's input already follows, so the two token grids coincide (144 tokens for 16:9 frames, 192 for 4:3 — the reference's operating point) | `data.vlm_long_side`, `VisionSpec.long_side` |
| preparation | each source once as JSONL + offset index (lazy loading; a 740 MB JSON array per rank would not fit), a 300-record holdout per source, a media check | `scripts/prepare_mix.py` |
| data pass | fields, media forms, marks, answer kinds, and the scene ids in the paths against VSI-Bench's 288 scenes | `scripts/inspect_reference_data.py` |

**Marks.** Questions of `spar_info` records refer to points or boxes drawn on the frames. Until our code draws them
(it needs the record format - the data pass reads it), those records are left out at preparation (`data.marks: skip`
→ `prepare_mix.py --drop-marked`); `PREPARED.md` says how many.

## Arms (same data order, same steps, same seed)

| arm | config | what it answers |
|---|---|---|
| **gated** | `server_4b_v3.yaml`: gated_input, LayerNorm on both streams, g = 0.5 at step 0 | the design as the reference trains it |
| **plain** | `server_4b_v3_plain.yaml`: no geometry (fusion none), same LoRA / data / resolution | **gated − plain** = what the geometry path adds at this data scale (the key comparison) |
| **control** | gated with shuffled geometry (`--geom-control shuffled`) | gated − control = the value of the geometry's content |
| **gated_raw** | `server_4b_v3_raw.yaml`: no norm, g = 0.018 at step 0 | whether the start matters: near the base model vs the reference-style start |
| base | no training, same resolution | the reference point for every arm |

Recipe (the reference's): LoRA r16 α32 (our target modules), lr 5e-5 LoRA / 1e-5 merger + gate, global batch 64
(8 GPUs × accum 8), cosine, warmup 3 %, weight decay 0.01, one pass of the mix or what `TRAIN_HOURS` allows.

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

As run 2: on one node everything runs in order; the same command on more nodes shares the roles (claims and
heartbeats on the shared volume, takeover after 30 min without a heartbeat); re-running resumes; `bash
scripts/server_run3.sh stop` ends a node's run. Queue: **gated → plain → control → base → gated_raw**, after the
one-time steps:

1. **p0** on every node (packages, flash-linear-attention unless run 2 banned it, the tests, the annotation files).
2. **data** (once, CPU): the data pass → `outputs/run3/reference_data/DATA.md`; the mix → `data/vgllm/` and
   `outputs/run3/PREPARED.md` (tens of minutes).
3. **calib** (once): 12 steps of the gated arm at the production settings (with 3 dumped samples: check the prompt
   forms) → `budget.env`: steps = one pass, or what `TRAIN_HOURS` (72) allows at the measured speed; **NO_GO** when
   that is under `GO_EPOCHS` (0.2) of a pass (training skipped, base still evaluated). d1: one profiled step.

With four nodes: gated, plain, control and base start together; base finishes in hours and its node takes gated_raw.

| duration (per node) | at 6.48 samples/s (the reference's speed) | at 2 samples/s |
|---|---|---|
| one pass of ref70 without marked records (≈1.5-1.9 M) | ≈ 65-80 h → capped at 72 h | 72 h = ≈ 0.27 M (≈ 15 % of a pass) |

Evaluation per arm: VSI on the live path (32 uniform keyframes as images) plain and with the room facts; gated also
in video mode (the offline protocol) and at every quarter checkpoint; the lmms-eval gate (VideoMME Δ ≥ −1.0 is our
general-ability target) per arm against base runs at this resolution; latency of the gated design; the holdout
ablation gated vs control.

## Knobs

`MIX` (ref70 | all6 | any registry spec, e.g. `all6,sqa3d%50`), `TRAIN_HOURS` (72), `GO_EPOCHS` (0.2), `ROLE`
(a subset in order, no claims), `PROBE_SRC` (vlm3r_scannet), `R2` (run 2's outputs, for the cross-run rows).

## Reading (REPORT.md)

1. **gated − plain** (key): positive with its interval above 0 = the geometry path pays at this data scale.
2. **gated − control**: the content of the geometry, not the extra path.
3. **gated vs the reference's 70.8** (their protocol differs: video path; our `gated_video` row is the closest).
4. **gate**: VideoMME within −1.0 of the base = general ability kept.
5. **gated − gated_raw**: whether the start matters after training.
6. Rows against run 2 mix data, design and resolution - context, not a controlled comparison.

What to send back: `outputs/run3/REPORT.md`, `STATUS`, `PREPARED.md`, `reference_data/DATA.md`, `budget.env`,
`calib.log` (head: the dumped samples), and any FAILED log it names.
