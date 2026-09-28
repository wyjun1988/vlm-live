# What the M2 findings mean for training, and the system questions (2026-09-27)

Written while the four-node Sensenova baseline runs (docs/SERVER_WEEKEND.md). Numbers: docs/BRAINSTORM.md §0b.

## 0. Where we stand

| | VSI (60 videos, live path) |
|---|---|
| base 4B, its best format | 48.6 |
| + format instruction + routed room facts, distances (≥ 3 views), directions (≥ 1 view) — **no training** | **57.8** |
| S2 LoRA pilot on 905 Sensenova samples (M2), for contrast | 44.1 (control 42.3) |
| in-house offline system (VGGT + 4B, full SFT), the target to keep live | 73.3 |

The prompting family is at its ceiling on this subset (I-34): every VSI type that geometry answers better than
the frames is routed. The remaining ~12–15 points are the training side's.

## 1. Training — what is running, what the findings say, what to queue

### Running (server, 4 nodes)
Sensenova-only, one epoch per arm: S1 real ∥ control → S2 real ∥ control (two seeds), plain SFT (no geometry),
S2 joint (no S1), the 2B pair, learning curve, gate. It answers, in order of importance:
1. does the *trained* model use geometry content (S2 real − control on VSI)?
2. what does Sensenova SFT alone give the 4B (plain SFT vs the base's 48.6 — and vs the prompt's 57.8)?
3. do training and prompting stack (`s2_real_routed − s2_real`, and both against `base_routed`)?
4. is S1 needed (joint vs real), and does 2B close the gap?

### What this week's M2 results say about training
- **The projector can be made to read geometry, weakly** (I-14): with questions that have no shortcut, the
  S1 loss depends on the geometry's content (+0.168 ± 0.064) and the decoded number moves for camera displacement.
  On Sensenova the same stage learned only the answer format. So the *data*, not the architecture, decided what
  S1 learned. Orientation was not learned at all in one epoch of 0.8B S1.
- **Prompted facts are worth 9 points and the model copies them 93% of the time** (I-27/29/33). Training must
  not unlearn that: an SFT model that becomes confident in its own numbers stops copying, and the routed facts
  stop paying. The server's `s2_real_routed − s2_real` row is the first check.
- **The size question is a copy-rate question** (I-22/I-33 on 2B): the 2B follows the same direction fact half
  as often. Any smaller-model plan needs training that teaches fact use, not more facts.

### Queue for the server, after the baseline report (in this order)
A. **Geometry QA as the alignment stage** — S1 on geometry questions generated from Sensenova's own image
   sequences (`make_geometry_qa.py --records data/sensenova.jsonl`, ~830k sequences → millions of questions;
   cap at the S1 budget), then S2 on Sensenova. Against the baseline's Sensenova-only S1 → S2. Decides whether
   an aligned projector helps the LLM stage. Adds nothing to inference cost.
B. **Fact-conditioned SFT (I-17)** — S2 with the situation prompt present in training: the same routed facts the
   live system produces (room size, distances, directions from the maps built on the training sequences), present
   for a random half of the samples so the model learns both to use them and to answer without them. Aim: make
   training and prompting stack instead of compete. Needs the maps computed once per training sequence
   (CUT3R + OWLv2 over 830k sequences — a pre-processing job of its own, ~1 day on a node).
C. **Numeric calibration (I-18)** only if the baseline's S2 shows the 905-sample collapse ("1.1" for every
   distance) at scale: up-weight numeric answers, or lean on B (the model copies a measurement rather than a prior).
D. **Orientation-carrying geometry tokens** only if A shows scale-only again: today's projector maps each frame's
   tokens independently and adds the pose token to layer 0; relative orientation between frames has no direct
   path. Candidate: inject per-keyframe *relative* pose (Δyaw, Δposition to the previous keyframe) as a few
   explicit tokens — the quantity the direction facts carry in text, in embedding form.

### Methods not planned, and why
- Teacher distillation from the in-house 73.3 system: excluded by decision.
- Thinking at question time: excluded by measurement (I-21: neither latency nor accuracy).
- Full fine-tuning instead of LoRA: possible on 8 H100s, but the general-ability gate (VideoMME ≥ −1.0) is the
  constraint, and LoRA is the safer first answer to it. Revisit if the S2 LoRA plateaus well below 73.3.
- Reward-based tuning (numeric rewards on VSI-like questions): a real option for calibration after B; not before
  we know whether SFT alone collapses numbers at scale.

## 2. System — measured tonight, and designs

*Updated 2026-09-28 with the results; the numbers are in docs/BRAINSTORM.md §7 (rows 24b–30) and docs/M2_LOCAL_20260924.md §10.*

### 2a. Accumulating CUT3R before the question: is the result the same? — **measured: no (scale drift)**
In every VSI run so far the encoder ingested only the 32 keyframes. Live, it ingests every frame (or every k-th)
and the maps accumulate all of them; the state at a keyframe has seen 30× more frames. Two effects pull against
each other: more points and views (better maps), and state dilution/drift (I-05). Running now on the adopted
configuration (57.8): geometry every 30th frame (~1 fps) and every 10th frame (~3 fps), maps built from all of
it, the LLM unchanged. Read with `vsi_measurement_table.py`: room-area MRA against the ground truth, distance MRA,
direction accuracy — the same three facts, so a change is attributable to the encoder's state. If the 1-fps maps
are worse, the live system needs windowed state resets or I-11 (register per-frame geometry ourselves); if
better, accumulation is free accuracy.

**Measured (09-27/28).** Geometry every 30th frame (~1 fps): 56.9 vs 57.8 with the keyframes only (−0.9 [−2.2,
+0.4]). The two effects are both real: the direction facts get *better* with more views (accuracy 76.4 → 83.4 %),
the metric facts get *worse* because the state's scale inflates with the number of frames ingested — room area
measured/true 1.03 (keyframes) → 1.18–1.23 (~1 fps) → 1.39 (~3 fps, 54.7 on 20 videos), area MRA 65.8 → 55.5 →
30.0. Windowed state resets (I-36) recover half the scale (ratio 1.10, 40/3 → 56.9) but re-align to drifted poses
and lose the direction gain (83.4 → 77.7; 80/6 → 63.1, 55.3). So the fix must rescale without touching
orientation: a scale anchor at the encoder's output (I-37; camera height / floor depth against the first frames),
measured with an offline lab that replays variants on captured raw streams (`scripts/scale_anchor_lab.py`).

**Resolved (09-28 evening).** The per-frame floor / camera-height anchors carry no signal (factor–ideal correlation
≈ 0 over 24 variants; applied per frame they break the layout: 49.5). The drift does follow the number of frames
ingested (corr +0.44): a two-parameter calibration, log(measured/true area) = 0.3224 log n − 1.3781, applied as one
factor per video to the metric facts at read time (geometry untouched) gives room MRA 55.5 → 63.8 split-half, and the
4B run at ~1 fps scores **58.0** — +1.1 [+0.3, +2.1] over the uncorrected stream, parity with keyframes only (57.8,
+0.2 [−1.3, +1.9]) while keeping the accumulation's direction gain (+5.8 on the type). The live design therefore
keeps the state accumulating at ~1 fps with the calibrated read-time factor; the per-video estimator that could
replace the calibration (CUT3R's self-view vs world-view distance ratio, `SelfViewScale`) is being measured.

### 2b. Long video — **measured by proxy: the facts compensate only partly, and need the scale fix first**
VSI videos are 20 s – 3.6 min; the live setting is 10 minutes and more. What breaks first is the LLM's view:
32 keyframes over 10 minutes is one frame per 19 s. The maps do not have that problem — they see every frame.
Proxy tonight: the LLM sees **8** keyframes (as if the video were 4× longer than the budget allows) with and
without the routed facts. The gap "facts − no facts at budget 8" against the same gap at 32 is how much the
measurement path compensates for a shrinking view. Design for the real thing (I-02): when frames are evicted
from the LLM's window, what they contributed stays in the maps and is routed as text at question time — this is
already how the system works; the open part is text *memory* beyond the maps (events, "last seen" per object),
which needs a long/streaming benchmark (OVO-Bench or a long-video spatial set) — none is local.

**Measured (09-27).** LLM at 8 keyframes: format hint 47.3 → facts from a keyframe-detected map 48.3 → facts from a
map built online over the whole stream (detection every 30th frame, geometry ~1 fps) 48.8 (+1.5 [−0.8, +3.9]).
The online map fixes coverage (direction facts on 35 → 87 % of the questions, distances 4 → 71 %), but dense
detections dilute the view-count filter (`view_gap` now counts separated views) and inherit the scale drift of
§2a. With the LLM at 32 keyframes the combined online configuration scores 55.1 vs 57.8: on 1–3-minute videos the
keyframes already see everything, so the accumulated map only adds its errors. The long-video case is exactly
where the map must carry the load, which makes the scale anchor (§2a) the prerequisite.

### 2c. Other encoders — **first bake-off done: CUT3R wins clearly**
The registry reserves `anchor3r`, `lingbot`, `vggt_window`; only `cut3r` and `dummy` are implemented. The routed-
measurement harness is now an **encoder bake-off harness that needs no training**: any encoder that gives poses
and point maps plugs into SceneMap/ObjectMap, and its quality is read directly as room-area MRA, distance MRA and
direction accuracy against VSI ground truth (today: 76% direction at ≥ 1 view, room area MRA 65.8). First
candidate: a sliding-window VGGT (the in-house system's encoder, `vggt_window`), because it removes the recurrent-
state question of §2a at the cost of a window; then any streaming reconstruction model with the same outputs.
This is a day of adapter work per encoder plus one 60-video run each, on the M2.

**Measured (09-27).** `vggt_window` (VGGT-1B on a sliding window of 8 keyframes, windows chained by their shared
cameras, 0.8B as the LLM since the facts do not depend on it): direction accuracy 56.7 % vs CUT3R's 76.4 % on the
same questions, room-area MRA 9.5 vs 65.8, scale-free 16.2 vs 69.0. Each window's normalised scale does not chain
consistently (VGGT/CUT3R area ratio IQR 0.24–3.85). Keep CUT3R; a candidate has to beat these three numbers in
the same harness before any training.

### 2d. Dynamic scenes
Everything assumes a static scene: CUT3R's state, the point maps, the instance merging. A moving person or a
moved chair produces two instances or a smeared one. The pieces that exist: per-instance first/last-seen keyframe
and per-frame timestamps. Design (not measurable on VSI): (i) flag an instance whose new observation lies outside
its merge radius as *moved* and keep both positions with times; (ii) recency weighting when answering "where is
X" (latest observation wins, older ones decay); (iii) state windows for the encoder so an old configuration does
not anchor the coordinates. Needs a dynamic benchmark before any of it is worth building; the timestamps (I-04)
are the prerequisite and are already in the maps.

## 3. Decisions for you
1. After the baseline report: run A (geometry-QA S1 → Sensenova S2)? It reuses the server script's roles.
2. B (fact-conditioned SFT) needs the pre-processing job (maps over 830k sequences). Start it in parallel with A?
3. Encoder bake-off on the M2 (§2c): worth a day per encoder now, or after the training results?
