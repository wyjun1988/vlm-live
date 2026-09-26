# Brainstorm — how Live3R gets spatial knowledge into a live answer

Started 2026-09-24. This is the idea pool: everything proposed so far (by the user and by Claude), with the
evidence behind it and its status, so that experiments do not get tangled. Promising ideas get a test; results go
back into the idea's entry. Measured numbers live in `docs/M2_LOCAL_20260924.md`; decisions in
`docs/DIRECTION_20260923.md`.

Status: **adopted** · **testing** · **proposed** · **parked** · **tested: positive / negative / null**.
(U#n) = the user's brainstorm item n (2026-09-24).

---

## 0. Where we stand (2026-09-24 evening)

Goal: Qwen3.5-4B answering VSI-style spatial questions **live** (first token < 1 s after the question) at about
**70**, with general ability kept (VideoMME drop <= 1.0). LLM trained with LoRA only; CUT3R vanilla and frozen.
The internal offline system (VGGT + Qwen3.5-4B, lots of data) is at 73.3.

| What | Measured (M2, 4B, VSI 60 videos / 1,151 questions unless noted) |
|---|---|
| Our input path vs official | identical (logit Δ 0.0); CUT3R per-frame reproduction == original `forward_recurrent` (Δ 0.0) |
| Cost of live keyframe selection | none — causal halving 49.9 vs offline uniform 48.6 (Δ 95% [-1.7, +4.9]) |
| Live latency | question-only TTFT **0.23 s** with the visual prefix precomputed (map or not) |
| Base 4B | video mode 48.6 · image mode 24.7 (answer-format failures) · **image + format instruction 53.7** |
| Best zero-shot so far | **56.0** — routed room facts (I-27) + routed object distances from >= 3 views (I-29); +7.3 [+3.6, +11.2] over the best-format base, no training |
| Format instruction | a pure format fix: recovers 585 lost answers; among already well-formed answers the changes net to zero (no leak) |
| Geometry token path | S1 (projector only): learns answer format, geometry value 0 · S2 LoRA (905 samples): +1.9 [-0.9, +4.4] vs control, numeric answers collapse to the data prior, 44.1 overall |
| Literature | same data with vs without the 3D encoder: VLM-3R +3.2, VG-LLM +0.9 — the data itself gives ~+20 |
| Scene map image / measured facts (zero-shot) | net 0 either way — measured room area +13..+19 on room size, but the added geometric context costs -6..-10 on appearance order / direction |

Reading: the live constraint is cheap *for VSI* (static scene, whole-video questions); the open problem is
putting spatial knowledge into the answer. New embeddings need the LLM to learn to read them (lots of data);
text the VLM already reads works zero-shot where it is relevant (room size), but always-on geometric context
costs on other question types — so *what* to show, and when, matters (I-27).

---

## 0b. Scoreboard — everything tried, and what it gave (2026-09-26)

VSI, same 60 videos / 1,151 questions, **image mode** (the live path), Qwen3.5-4B, zero-shot unless marked.
Intervals are video-level bootstrap, lmms-eval aggregation.

### Adopted — the configuration that works (cumulative)

| Step | Score | Δ | 95% CI |
|---|---|---|---|
| Base, image mode | 24.7 | — | |
| Base, video mode (its best format) | 48.6 | — | *gate-1 baseline* |
| + format instruction (one line, in the prefix) | 53.7 | +5.1 | [+1.7, +8.7] |
| + room measurements routed to room-size questions (I-27) | 55.4 | +1.7 | [+0.6, +2.8] |
| + object distances from >= 3 views (I-29) | 56.0 | +0.6 | [+0.2, +1.0] |
| + relative directions from three tracked positions, >= 1 view (I-33) | **57.8** | +1.8 | [+0.9, +2.8] |
| **total over the gate-1 baseline** | | **+9.2** | [+5.0, +13.4] |

No training. One LLM pass, cached prefix unchanged, a few dozen tokens at question time, question-only TTFT 0.23 s.
Same-day nulls (09-26): appearance order from the tracker (I-32), keyframe time labels (I-04), thinking at question
time (I-21). The rule that held every time: attach geometry only for what the frames do not already say.

### Rejected (kept in the code, off by default)

| Tried | Result | Why it failed |
|---|---|---|
| Top-down map image (I-08) | −0.3 | costs order/direction as much as the area gains |
| Measured facts always in the prompt (I-07) | −0.4 img / +0.8 vid | same, without a picture → route instead |
| Object map as text, VLM grounding, oracle names (I-09) | **−13.7** | hallucinated objects, one bed → 7 instances |
| Object map as text, OWLv2 fixed vocabulary (I-09) | **−9.9** | fewer hallucinations, counts still wrong |
| The 4B writing its own cognitive map first (I-20) | **−4.8** | adds only its own errors |
| Every computable distance, min_frames=1 (I-29 run 1) | −0.7 vs I-27 | 1-view 3D estimates are bad |
| Object size from measured extent (I-30) | model 66.7 vs ours 40.3 | category prior beats measurement (L6) |
| Repeat the instruction after the question (I-31) | 4B −0.7, 0.8B −1.0 | one copy is enough at both sizes |

### Training path, for contrast (Sensenova 905 samples, LoRA + projector)

| | Score |
|---|---|
| S2 real geometry | 44.1 |
| S2 control (shuffled geometry) | 42.3 |
| geometry's own contribution | +1.9 [−0.9, +4.4] — not significant |

A small SFT lands **11.9 below** prompting the base: it fixes the answer format but collapses numeric answers onto
the training prior. Scale is the missing ingredient (literature: ~+20 from data, +1..3 from the 3D encoder).

### Model size

| | instruction only | + routed measurements | gain | copies the measurement (room / distance) |
|---|---|---|---|---|
| Qwen3.5-4B | 53.7 | **56.0** | +2.3 [+1.0, +3.6] | 98% / 98% |
| Qwen3.5-2B | 40.3 | **42.7** | +2.4 [+0.7, +4.0] | 93% / 98% |
| Qwen3.5-0.8B | 23.4 | **28.8** | +5.4 [+4.0, +6.8] | 42% / 97% |

**Measurements do not buy back model size.** 2B + measurements (42.7) is still 11.0 [−16.0, −6.4] below 4B with a
prompt alone (53.7); 4B + measurements is 13.3 above 2B + measurements. The 0.8B's larger gain is headroom, not
capability — it started at 5.7 on room size.

Where the smaller models actually lose (vs 4B, routed): appearance order **10.1 / 6.8 vs 66.9**, relative
direction 35.5 / 39.8 vs 54.6, relative distance 46.5 / 33.1 vs 64.0. These are perception and cross-frame
tracking, which a number in the prompt cannot supply. Where a measurement is handed over, 2B uses it almost as
well as 4B (93% vs 98%), so arithmetic is not the bottleneck at 2B. (2B is *better* than 4B at route planning,
37.8 vs 27.0 — the one type where the 4B is oddly weak.)

---

## 1. Live architecture and memory

### I-01 Background situation prompt, question-only answering — **adopted**
Keep the prompt (keyframes + everything derived from geometry) prefilled in the background; when a question
arrives only its tokens are processed. Measured 0.23 s TTFT on M2 (4B); prefix rebuild 11–12 s on M2,
estimated 0.1–0.4 s on H100 (refresh every second). The geometry-derived text sits after the keyframes so a
text-only change recomputes just the tail.

### I-02 (U#1) CUT3R at ~1 fps + text memory compression beyond the keyframe budget — **proposed**
CUT3R keeps interpreting the space at ~1 fps. When more than 32 frames have been seen, what the evicted part of
the stream contributed is extracted once through a downstream step and kept as **text**; at question time the
keyframes + that text go in together.
- Why: the LLM's view is capped (32 keyframes → 1 frame per 19 s on a 10-minute stream); text memory is cheap
  in tokens and the VLM reads it zero-shot.
- Already partly there: the scene/object maps accumulate over *every* geometry frame, not just the keyframes.
- Open: what to write down (objects, positions, first/last seen, room layout, events) and how to merge/forget.
- Test: on VSI, build the maps from ~1 fps geometry (60–180 frames/video) while the LLM sees 32 keyframes;
  compare with maps built from the 32 keyframes only.

### I-03 Always include the latest frame at question time — **proposed**
Halving keeps evenly spaced frames, so the last few seconds are nearly absent. For "what is in front of me now",
append the current frame when the question arrives (~117 tokens: tens of ms on H100).

### I-04 Timestamps in the situation prompt — **zero-shot form tested: null (2026-09-26)**
Image-mode keyframes carry no time (order is only implicit). Add keyframe times and per-object first/last-seen
times to the text. Also the basis for recency and decay (I-26).
- Zero-shot form: every keyframe labelled `Frame k at t s:` (`--frame-labels`), on top of the format instruction
  (53.7) → **54.3, +0.7 [−2.4, +3.6]**. Not a gain: the labels change answers of every type (435 of 1,151), and
  the changes cancel — route planning +5.4 and counting +3.2 against relative distance −6.4 and room size −2.3.
  The interval is four times wider than the routed-facts ones because the labels perturb everything rather than
  adding one targeted fact. Timestamps stay relevant for the live system (recency, I-26), not as a VSI lever.

### I-05 CUT3R forgetting and drift over long streams — **proposed (risk)**
The CUT3R state is a fixed 768-token memory overwritten every frame, so old content dilutes and poses drift.
In evaluation CUT3R only sees 32 sparse keyframes, which hides this; live at 10 fps it sees thousands of frames,
and the maps inherit any drift because they live in the first frame's coordinates.
- Test: ScanNet sequences with ground-truth poses — pose / scale error vs stream length; map consistency when
  the camera revisits a place. Mitigations: windowed state reset + re-anchoring, or I-11.

### I-06 Halving-aware incremental prefill — **parked**
Append keyframes to the LLM cache as they arrive and rebuild only on evictions. Not needed on H100 (full
rebuild is fast enough); relevant for an on-device demo.

---

## 2. Getting spatial information out of CUT3R

### I-07 Measured facts as text — **tested: null overall (helps one type)**
Room width × length (area), height, camera path length from CUT3R point maps + poses (upright, wall-aligned).
Image mode, 60 videos: facts-only 53.3 vs instruction-only 53.7 (−0.4 [−2.5, +1.5]). Per type vs
instruction-only: **room size +12.6**, counting +3.5, but appearance order −6.1, relative direction −6.5,
relative distance −3.5. Video mode: 51.7 vs 51.0 (+0.8 [−1.5, +3.1]); room size +19.0, relative direction −7.4,
route planning −5.4. So the cost seen with the map image (I-08) is not about the picture: geometric text in
front of the question costs on order / direction questions. Known error: one ARKit room measured 34 m² vs 21.3 m²
true (points seen through doors).
→ I-27 (select facts at question time).

### I-08 Top-down map image — **tested: null**
Net −0.3 [−2.8, +2.4] against the same instruction without the map. Appearance order / direction drop −7..−10;
the facts-only run (I-07) shows a similar drop without any picture, so it is the extra geometric context in
front of the question, not (only) the numbered camera path in the image.

### I-09 Object-level cognitive map as text — **testing**
Detect objects on keyframes, lift each box to 3D with the CUT3R point map, merge across frames, and write per
object: count, floor-plan position, first-seen frame, plus distances between objects. Targets counting,
distances, direction, appearance order — "Thinking in Space": a correct cognitive map gives +20..32% on relative
distance.
- Prototype: the VLM's own grounding with the **oracle** vocabulary (object names from the questions) —
  **tested: negative.** Same 20 videos, image mode: 41.0 vs instruction-only 54.7 (**−13.7**); counting
  63.5 → 38.6, absolute distance 37.9 → 23.9, relative distance 64.5 → 40.3. Why (example scene0435_02):
  (1) asked for a list of names, the VLM "finds" most of them — 3 chairs, 3 doors, 3 lamps, mostly "first seen
  in frame 1"; (2) one bed became 7 instances (partial-view centroids > 0.8 m apart); (3) the model **trusts the
  text** — it copies wrong counts and distances just as it copied the correct room area. Lesson: a wrong
  measurement is worse than none; text memory must be precise or absent.
- OWLv2 (open-vocabulary, fixed indoor vocabulary — question-agnostic): **tested: negative.** 44.8 vs 54.7
  (**−9.9**) on the same 20 videos. Fewer hallucinations than VLM grounding (relative distance 51.6 vs 40.3),
  but counting 63.5 → 38.9 and absolute distance 37.9 → 28.4 — wrong counts/distances still get copied.
- What a v2 would need → I-28.

### I-10 (U#3) Object-aware extraction from CUT3R — **proposed**
Let object recognition shape *what* is extracted from CUT3R, not just be merged afterwards: pool CUT3R point maps
and/or decoder tokens inside each object's box/mask → per-object 3D extent, size, position (text path) or
per-object geometry tokens (token path).
- The current ObjectMap already pools point maps per box (centroid); extent → object size is the next step.

### I-11 (U#2) Extract before CUT3R accumulates; keep a separate memory — **proposed**
The trained downstream reads CUT3R embeddings that already mix in the accumulated state, which may not help as
the state dilutes (I-05). Add a shortcut: take per-frame information before it is folded into the state (e.g.
the encoder tokens, self-view point maps) and accumulate it in our own memory.
- Our scene/object maps already accumulate outside CUT3R, but their coordinates come from CUT3R's pose (state).
- Test with I-05's drift measurement: do self-view geometry + our own registration hold up better?

### I-12 Geometry token path (projector + DeepStack injection) — **tested: small / demoted**
S1: 0 (format only). S2 LoRA pilot: +1.9 [−0.9, +4.4]. Literature: +1..3 with 200–300k samples. Kept as an
auxiliary, always measured against a control projector trained on shuffled geometry.

### I-13 3D positions as M-RoPE positions for vision tokens — **parked**
Video-3D LLM / C²RoPE style. Needs training; conflicts with the pretrained (t, h, w) layout.

---

### I-27 Select which facts to show when the question arrives — **tested: positive (adopted)**
Everything is still computed in the background, but only the facts relevant to the question are attached to it
(e.g. room measurements for a room-size question, object distances for a distance question). Costs a few dozen
question-time tokens (milliseconds), avoids the −6 on order / direction from always-on text (I-07), and is
live-valid (the question is known at that moment). Rule-based routing first; could be learned later.
- Simulated from the facts-only and instruction-only runs (room facts only on room-size questions; the facts
  text already sat right before the question, where routing would put it): **image 55.3 vs 53.7, +1.6
  [+0.4, +2.8]; video 53.3 vs 51.0, +2.4 [+1.0, +3.8].** First significant zero-shot gain from CUT3R geometry.
  vs the best-format base (48.6): image +6.6 [+2.9, +10.4].
- **Real run (image mode, 60 videos): 55.4 — +1.7 [+0.6, +2.8] over instruction-only, +6.8 [+3.0, +10.5] over
  the best-format base.** Prefix = keyframes + format instruction; the measured room facts are attached only to
  room-size questions when they arrive (a keyword rule hits exactly the 288 room-size questions). Every other
  type is identical to instruction-only, as designed; room size 53.2 → 67.0. Current best live-valid zero-shot
  configuration.
- Next: object facts through the same routing (I-28).

### I-28 Object facts v2 — precise or absent — **tested: null**
From I-09's failures: (a) no counts in the text (the model counts better from the images: 63.5 vs 38.9 with
our counts); (b) at question time, look up only the objects the question names (fuzzy match to the background
map's labels) and attach just their positions and a **closest-point** distance computed from the lifted point
sets (VSI measures closest points; centroid distances overestimate); (c) only instances seen in >= 2 keyframes;
(d) merge per label with size-aware radii (a bed or sofa spans > 1 m) or 3D box overlap, not a 0.8 m centroid
threshold; (e) attach nothing when unsure. Needs I-27's routing.
- Result (image mode, 60 videos; OWLv2 fixed vocabulary on CPU, every 2nd keyframe; on top of I-27): 55.2 vs
  I-27 55.4 (−0.2 [−0.7, +0.4]). Absolute distance +1.7 (37.6 → 39.3), relative distance −3.0 (64.0 → 61.0);
  other types untouched.
- Diagnosis (15 videos, attached text logged per question):
  - Absolute distance: a measurement was attached to only **18%** (9/49) of the questions — most named objects
    were missing from the fixed vocabulary or seen in < 2 keyframes. Where attached, the measurement's own MRA was
    **0.50** (the model alone: ~0.38; measured/true median 0.84) and the model copied it **100%**. → the gain is
    capped by coverage, not by the model.
  - Relative distance: attached to 47% (28/59), but the closest object implied by the measurements was right only
    **12/28 (43%)** — worse than the model's own ~64% — and the model followed it (11/28). → our measurements are
    not fine enough to rank objects by distance.
- Next (I-29): raise coverage for absolute distance; stop attaching relative-distance rankings (or only with a
  clear margin).

### I-29 Measure what the question names, at question time — **tested: positive (adopted)**
I-28's absolute-distance measurements were better than the model's own guesses (MRA 0.50 vs 0.38) but attached to
only 18% of questions. Detect the question's object names on the stored keyframes when the question arrives
(OWLv2 on 16 frames in one batch: ~0.1 s on H100; slow on M2), lift and measure, attach if both are seen in >= 2
keyframes. Keeps TTFT near the budget on H100; trade-off to measure. Drop relative-distance rankings unless the
margin between options is large (> 30%).
- Coverage diagnosis (I-28 data, abs-distance questions): of the object mentions, 70% were in the map,
  7% were missing from the OWLv2 vocabulary (closet, computer mouse, washer, cutting board — now added),
  22% were in the vocabulary but never detected. Both objects present in only 45% of questions; the
  "seen in >= 2 keyframes" filter then cut that to the 18% actually attached.
- **Run 1** (all 32 keyframes detected, `min_frames=1`, absolute distance only, 60 videos): overall 54.7 —
  *worse* than I-27's 55.4; absolute distance 37.6 → **32.0**. Coverage rose to 83% but quality collapsed.
- **The number of views is a reliability signal** (same run, from the log; the model copies the measurement,
  so the predicted type score tracks reality — predicted 31.5 vs actual 32.0):

  | min_frames | coverage | MRA(measured) | MRA(model, same questions) | predicted abs-distance score |
  |---|---|---|---|---|
  | 1 | 83% | 31.7 | 39.0 | 31.5 |
  | 2 | 51% | 44.0 | 39.8 | 39.7 |
  | 3 | 31% | **51.4** | 36.6 | **42.2** |
  | none (I-27) | 0% | — | — | 37.6 |

  A single-view instance is a bad 3D estimate (one partial view of an object, no triangulation); three or more
  views beat the model by 15 MRA points.
- **Run 2 (`min_frames=3`): overall 56.0 — +0.6 [+0.2, +1.0] over I-27, +2.3 [+1.0, +3.6] over instruction-only,
  +7.3 [+3.6, +11.2] over the best-format base.** Absolute distance 37.6 → **42.1** (predicted 42.2).
  Extending the log to 6 confirms 3 is the optimum: coverage 83/51/31/20/12/8% at thresholds 1-6, predicted type
  score 31.5/39.7/**42.2**/40.2/39.6/39.4 — quality saturates around 50 MRA while coverage keeps falling.
- Current best live-valid zero-shot configuration: cached prefix (keyframes + format instruction); at question
  time attach room measurements to room-size questions and a closest-point distance to distance questions when
  both objects were seen in >= 3 keyframes. One LLM pass; routing is a string match (0.2 us); +43 tokens.
- → L5.

### I-30 Object size from the lifted point sets — **tested: negative (do not attach)**
VSI asks "the longest dimension of the X in centimeters" (1/8 of the score, currently 65.3). We already hold a
3D point set per object instance, so the extent is `max(ptp(points))` — no new machinery, and the same
view-count gate (I-29 / L5) decides whether to attach it. Risk: our point sets come from box interiors (the
central 50% of each box), so extents are systematically *under*-estimated; check the bias before attaching,
and consider using the full box for the extent while keeping the centre for position.
- Implemented with a separate full-box point set and a 2-98 percentile extent, then measured on 12 videos
  (43 size questions): **worse than the model at every threshold.** Model alone 66.7 MRA; measured 36.6 / 40.3 /
  39.5 at min_frames 1/2/3, predicted type score 38.4 / 48.4 / 54.4 — all below 66.7. The bias flipped from the
  expected under-estimate to a **+26..39% over-estimate** (measured/truth median 1.26-1.39): the box contains
  floor and wall around the object, and a percentile extent does not remove it (door 118 vs 135 cm is good, but
  table 166 vs 107 and sofa 229 vs 181 are not).
- Not run at full scale — the diagnosis already shows it would cost ~1.4 points overall. `--size-facts` stays in
  the code, off by default.
- → L6.

### I-32 Appearance order from the object tracker — **tested: null (2026-09-26)**
The routed object map (I-29) already records the keyframe in which each tracked instance was first seen. VSI's
appearance-order questions ask exactly that for four named categories. Route it: when such a question arrives,
look up the four names; if all four were tracked (view-count filter, distinct first frames), attach "first seen:
X (frame 2), Y (frame 5), …; so the order is X, Y, …". Nothing is attached when any name is missing — a partial
order would mislead (L6: precise or absent). Coverage: 557/618 of these questions name only categories in the
detector's vocabulary; how many pass the view filter is what the run measures. Counting is logged the same way
(instance count per threshold vs the truth) but not attached — the tracker over-splits.
- Run: routed configuration (56.0) + `--order-facts`, 60 videos → **56.1, +0.2 [+0.0, +0.6]**. Facts attached to
  17 of 148 appearance questions (11% coverage at ≥ 3 views); only 2 answers changed (both to correct).
- Why it cannot pay: where the tracker covers, its order is **less** accurate than the model's own answer
  (64.7 vs 82.4 on the same 17 questions; at ≥ 1 view 32.9 vs 73.2). The VLM reads temporal order straight from
  the frame sequence — it is the one VSI quantity the images carry directly (L6 again: attach geometry only where
  the images do not already say it). Counting the same way: the map's instance count scores 24–42 MRA against the
  model's 60 at every threshold — never attach counts.

### I-33 Relative direction from three tracked positions — **tested: positive (adopted, 2026-09-26)**
VSI's relative-direction questions ("standing by A facing B, is C to the left/right[/back | quadrant]?") are a
function of three positions the object map already has. Compute the signed angle in the wall-aligned floor plan
(counter-clockwise = left) and turn it into the question's own label set. Diagnostic (logged at every view-count
threshold from one routed run, nothing attached):

| ≥ views | coverage | measured | model on the same questions | projected gain on the type |
|---|---|---|---|---|
| 1 | 77% | **76.4** | 53.5 | +17.6 |
| 2 | 40% | 88.9 | 59.3 | +11.8 |
| 3 | 19% | 92.3 | 56.4 | +6.9 |

By difficulty at ≥ 1 view: easy 91 vs 60, medium 80 vs 59, hard (4 quadrants) 63 vs 44. Unlike a metric distance,
the sign of an angle survives the centroid error of a single view, so the threshold for directions is separate
(`--direction-min-frames 1`, distances stay at 3). Roughly +2 overall if the model copies the fact as it does the
distances. This is the category the model is weakest at among the choice questions, and the one the tracker is
best at — the complement of I-32.
- Attach run: routed (56.0) + `--direction-facts --direction-min-frames 1`, 60 videos → **57.8, +1.8 [+0.9, +2.8]**,
  all of it on relative direction (54.6 → 69.2, +14.7; easy +7, medium +12, hard +11 net correct answers). Facts
  attached to 157 of 204 direction questions; the model chose the attached direction on 93% of them and was right
  on 73% (its own answers on the same questions: 53.5). Total over the gate-1 baseline: 48.6 → 57.8, +9.2
  [+5.0, +13.4], still zero-shot. Adopted into the routed configuration; the running server baseline keeps the
  pre-direction configuration for internal consistency, the next run includes it.

### Lessons so far (zero-shot, 4B)
- **L6 Attach geometry only for scene-specific quantities; the model's prior already wins on canonical ones.**
  Object size is a property of the object category — a door is ~135 cm, a sofa ~180 cm — and the pretrained model
  knows it (66.7 MRA) better than we can measure it (40.3). Room area and inter-object distance are properties of
  *this* scene, which no prior can supply, and there measurement wins (+13.8 and +4.5). Before building any new
  measurement, ask whether the answer is knowable from the object category alone.
- **L1 The model copies numbers from the prompt.** Correct → big gains (room size +13..+19); wrong → big losses
  (counts −25). Any text memory must be precise, or absent.
- **L2 Always-on geometric context costs other question types** (order / direction −6..−10). Show facts only to
  the questions that need them (I-27: +1.6 / +2.4 simulated).
- **L3 A 4B model's self-generated map does not help** (I-20: −4.8). Measurements must come from geometry, not
  from the same model looking at the same frames.
- **L5 Trust a measurement by how many views it came from.** Attaching every measurement we can compute is
  worse than attaching none (−5.6 on absolute distance); attaching only the well-observed ones beats the model.
  The view count is a free, general confidence signal — the same idea should gate room facts and any future
  measurement.
- **L4 Coverage and precision are separate problems.** Absolute distances measured from CUT3R beat the model's
  guesses where we had them (MRA 0.50 vs 0.38) but covered 18% of questions; rankings by distance were worse than
  the model (43% vs ~64%). Attach a measurement only for question types where it is known to beat the model.

---

## 3. Training objectives and data

### I-14 (U#4) Questions that can only be answered with geometry — **tested: positive but weak (0.8B S1, 2026-09-26)**
Generate new training questions whose answers come from geometry, e.g. "how did the camera move between frame 1
and frame 5", "how far is the point marked here", "which of these two frames was taken closer to the door".
Answers computed from CUT3R's own outputs — no human labels, no teacher model. The format shortcut becomes
impossible (a content-free signal cannot answer them). Directly fixes why S1 learned only format.
- Data: multi-image Sensenova records (153 in the 1k preview have 16–28 images) or any unlabelled video.
- **Built** (`src/live3r/data/geometry_qa.py`, `scripts/make_geometry_qa.py`). Five question kinds, all computed
  from the encoder's own poses / point maps: camera displacement between two named images, turn direction
  (left / right / same), which image was taken closest to another, total path length, room area/height.
  Reads videos or existing image-sequence records (the server's Sensenova corpus), writes the SenseNova record
  format the trainer already consumes. ~5 s per sequence on M2.
- Two design rules carried over from the prompting experiments:
  * **balanced answers** — frame pairs are sampled so the answers cover the range evenly, or the model learns the
    mode instead of reading (the 905-sample S2 collapsed absolute distance onto "1.1");
  * **explicit reject bands** — a turn between 15 and 40 degrees, a displacement under 0.30 m, or a
    multiple-choice pair within 0.50 m is dropped rather than guessed; a room height outside 2–5 m is dropped
    because hand-held scans rarely see the ceiling (real ScanNet scans gave 1.7 m).
- **Label quality, validated against VSI ground truth** (room area, 60 scenes): MRA 65.8, median measured/true
  1.07, 92% within 50%. The tail is over-estimates where the reconstruction sees through a doorway. Pose-derived
  labels cannot be checked against the world here (no ground-truth poses locally) but are self-consistent with
  the geometry input, which is what alignment needs.
- **Generated locally (2026-09-25)**: 3,651 questions from the 452 VSI videos that are not among the 60 eval
  scenes (8 frames each; 452 per kind for displacement / turn / closest / path / area, 269 room heights that
  passed the band). `data/geomqa/` (gitignored), 150 held out. The holdout is per record, so other questions
  about the same video are in training — for the decisive test, split by video instead.
- **Decisive test (M2, 0.8B, S1 projector-only, 3,417 records / 1 epoch / 428 steps, lr 3e-5; control arm with
  the same data order and `--geom-control shuffled`; 227 held-out questions from 30 videos never seen in training):**
  - `shuffled − real` = **+0.168 ± 0.064** (Sensenova pilot: −0.026 ± 0.047) → the projector's output depends
    on the geometry's content.
  - value of content = control·shuffled − real = **+0.052 ± 0.020** (Sensenova: −0.041 ± 0.052); real beats the
    control on 65.6% of samples.
  - Per kind (shuffled − real): camera displacement +0.13, path length +0.25, room area +0.72, room height +0.13;
    **turn direction +0.005 and closest image −0.006 — nothing.** Scale and distance are read from the geometry
    tokens; orientation is not (at this size and stage).
  - The training curves say the same: the control follows the real arm's loss almost exactly (aligned gap
    −0.027 over 428 steps), so the two projectors learned the same format and differ only in what they read.
  - **Generated answers** (`scripts/geomqa_generate_eval.py`, greedy, same 227 questions; each projector under
    its own / another record's / no geometry): the decoded answer changes with the geometry **only for camera
    displacement** — 17 of 52 answers change, MRA 35.8 with the record's geometry vs 29.0 with another's
    (+7.9 ± 7.3), and the answers spread over 9 values where the control says "0.5 m" to 50 of 60 questions. For
    path length, room area and room height the decoded number is identical under swapped geometry (0 of 30, 0 of
    30, 0 of 18 change): the loss gap there is a likelihood shift that never flips the argmax. The real projector
    still beats the control on those kinds (+9 to +13 MRA) — through a better learned prior over typical values,
    not by reading the record. Turn direction 35.0 vs chance 33; closest image 41.7, unchanged under swap.
  - **Verdict**: the shortcut is gone (the loss depends on content, the control does not follow), and the
    projector reads scale — but at 0.8B, projector-only, one epoch, that reading is weak: it moves likelihoods
    everywhere and decoded answers for one kind. Whether S2 (the LLM learning to use what the projector conveys)
    or a 4B turns it into answers is the next question, and the one the server can settle.
- **Implication for the server**: the first S1 that provably carries content. The natural follow-up after the
  Sensenova baseline is S1 on geometry QA (generated from Sensenova's own image sequences with
  `make_geometry_qa.py --records`) → S2 on Sensenova, against the Sensenova-only S1 → S2 pair.

### I-15 Geometry captioning alignment (LLaVA stage-1 analogue) — **proposed**
Frozen LLM, train only the projector to make CUT3R tokens *describable* ("the camera moved 1.2 m forward",
"the room is about 5 × 4 m"). Aligned tokens become readable by the frozen LLM → closer to zero-shot use.
Overlaps I-14 (I-14 is the QA form, I-15 the caption form).

### I-16 Large-scale SFT in the live input format — **planned (server)**
Sensenova 830k (+ more) with LoRA, image-mode keyframes + situation prompt, with control arms. The literature's
~+20 comes from here. Risk: the numeric-prior collapse seen at 905 samples (I-18).

### I-17 Train with the situation prompt — **proposed**
GPT4Scene: training with the BEV/marker prompts improves the model even when the prompts are removed at
inference. Our analogue: train with the measured facts / object map in the prompt.

### I-18 Keep numeric answers calibrated — **proposed**
The 905-sample SFT made absolute distance mostly "1.1" (prediction/truth 0.69). Options: data diversity;
up-weight numeric answers; lean on measured facts (I-07/I-09) so the model copies a measurement instead of a
prior.

---

## 4. Reasoning

### I-19 (U#5) Spatial thinking: look at the geometry first — **proposed**
Thinking is off today only because thinking was reported not to work well below 9B. Training (or prompting) the
model to first lay out the geometry, then answer, may raise accuracy. Evidence: "Thinking in Space" — linguistic
chain-of-thought did not help, but generating a cognitive map first did (+10% relative distance).
- Tension: thinking at question time costs seconds of tokens (live TTFT).

### I-20 Background thinking — the VLM writes the cognitive map before the question — **tested: negative**
Combine I-01 + I-02 + I-19: in the background the VLM looks at the keyframes and writes a cognitive map /
scene summary (objects, grid positions); it becomes part of the cached situation prompt. The "thinking first"
happens before the question, so question time does not change.
- Zero-shot test (no CUT3R): one generation per video, then the map text in the prompt for every question.
- Compare with I-09 (CUT3R-measured object map): self-estimated vs measured geometry.
- Result (20 videos, image mode): 49.9 vs instruction-only 54.7 (**−4.8**); relative distance −14.5, route
  −8.3, room size −7.5. The 4B's own map adds no new information, only its own errors. "Thinking in Space"
  saw +10% with Gemini-1.5 Pro — a much stronger model; at 4B, thinking-first does not help without training
  (keeps I-19's training variant open, closes the zero-shot one).

### I-21 Thinking on at question time (4B) — **tested: incompatible with live (2026-09-26)**
Latency cost; only if I-20 shows thinking itself is what helps.
- Zero-shot probe: thinking on (`<think>` open, 512-token budget, answer read after `</think>`; a budget that
  runs out counts as wrong — the honest cost), format instruction, 20 videos / 333 questions, against 54.7 on the
  same videos → **the 4B closed its reasoning on 11 of 333 questions** (mean 509 of 512 tokens): 96.7% empty
  answers, score ~3. Its reasoning is orderly ("locate the nightstand … the TV is on a dresser to the left …") and
  simply long. Whether it would be *right* given room is unanswered at this budget (a 2,048-token probe on 5
  videos is queued); the live question is answered: > 500 reasoning tokens per question is seconds on an H100
  (the 0.8B did not close within 96 tokens either), against a 1-second answer. Thinking at question time is out;
  I-19's remaining form is *trained* short geometric reasoning, or none.

---

## 5. Model size

### I-22 Smaller LLM with an explicit object map — **tested: gain is bigger, but the gap is not closed**
If measurements (I-27/I-29) carry the spatial reasoning, the LLM mostly reads and compares numbers →
0.8B/2B may suffice. The detector (OWLv2, 622 MB) is cheaper than LLM size.

| 60 videos, image mode | instruction only | + routed measurements | gain |
|---|---|---|---|
| Qwen3.5-4B | 53.7 | **56.0** | +2.3 [+1.0, +3.6] |
| Qwen3.5-0.8B | 23.4 | **28.8** | **+5.4 [+4.0, +6.8]** |

- **Measurements help the small model more than twice as much** (room size 5.7 → 34.2, absolute distance
  1.2 → 16.2) — the direction the "shrink the LLM" idea needs.
- **But 0.8B + measurements (28.8) is still 24.8 below 4B with a prompt alone (53.7).** The 0.8B's weakness is
  not arithmetic: appearance order 6.8 vs 66.9, relative direction 39.8 vs 54.6 — it cannot read the scene.
  Measurements cannot supply that.
- **Where the small model loses the measurement we hand it:** it copies a distance 98% of the time (same as the
  4B) but the room area only 53%, because 13/60 room-size answers are format failures ("Based on the visual
  ev...", truncated) — the 0.8B fails the answer format 15-18% of the time even with the instruction, vs 0.2%
  for the 4B. So part of the remaining gap is instruction-following, not perception → I-31.
- **2B (added 2026-09-25): 40.3 → 42.7, +2.4 [+0.7, +4.0].** It copies a handed-over measurement almost as well
  as the 4B (93% / 98%), so arithmetic is not its bottleneck — but it is still 11.0 below the 4B with a prompt
  alone, and its appearance-order score is 10.1 vs the 4B's 66.9.
- **Verdict: measurements do not buy back LLM size.** The gap between sizes is perception and cross-frame
  tracking, not calculation, and numbers in the prompt cannot supply it. Keep 4B for quality; revisit the small
  sizes only after training, where they have the most headroom.

---

### I-31 Repeat the format instruction after the question — **tested: negative on both sizes**
The instruction lives in the cached prefix, thousands of tokens before the answer; small models drift back to
explaining. Repeating it directly after the question costs ~12 tokens at question time (still one pass, prefix
untouched). Tested on 0.8B (15-18% format failures, so the headroom looked large) and on 4B (0.2%, to check for
a regression).
- **0.8B: 23.4 → 22.4, and format failures got *worse*, 17.5% → 24.2%.** 746 of 1,151 answers changed, so the
  repeated instruction is not a small nudge — it shifts the whole answer distribution (e.g. a size answer 16 → 40,
  a distance 0.22 → 0.27). The instruction is not "too far away"; a second copy just competes with the question
  for the small model's attention.
- **4B: 53.7 → 53.0**, even though format failures went 0.2% → 0.0%. The instruction was already doing its job;
  a second copy only perturbs answers that were fine.
- Lesson: one copy of the instruction, in the prefix, is enough at both sizes. The small model's format failures
  are a capability limit, not a placement problem — fixing them needs training (or constrained decoding), not
  prompt surgery. This also bounds how much of the 0.8B's gap is "formatting": very little.

---

## 6. Evaluation and gates

### I-23 Gate-1 baseline includes prompting — **decided**
The base's best format includes the format instruction (53.7, image mode) — verified to be a pure format fix,
not a leak. A trained model must beat what a prompt alone achieves.

### I-24 No question information in live-valid results — **rule**
Oracle vocabularies (object names from the questions) are allowed only for upper-bound probes and must be
labelled as such. Anything claimed for the live system must be question-agnostic.

### I-25 Temporal and streaming evaluation — **planned**
VSI barely tests time (appearance order, route planning). "What just changed / what happens next" needs a
streaming benchmark (OVO-S-Bench, the agreed secondary metric).

### I-26 Dynamic scenes — **proposed**
CUT3R and the maps assume a static scene; moving people/objects mix past and present positions. Needs
timestamps (I-04), recency weighting / decay, possibly state windows (I-05).

---

## 7. Test queue (M2)

| # | Test | Ideas | Status |
|---|---|---|---|
| 1 | Measured facts as text, no map image (image + video mode, 60 videos) | I-07, I-27 | done: null always-on; **+1.6 / +2.4 routed (simulated)** |
| 2 | Object map, VLM grounding + **oracle** vocabulary (image mode, first 20 videos) | I-09 | done: **−13.7** (hallucinated / over-split objects) |
| 3 | Object map, **OWLv2 + fixed indoor vocabulary** (question-agnostic) | I-09, I-10, I-24 | done: **−9.9** |
| 4 | Background cognitive map written by the VLM itself (zero-shot, no CUT3R) | I-20, I-19 | done: **−4.8** |
| 4b | I-27 routed room facts, real run (image, 60 videos) | I-27 | done: **55.4, +1.7 [+0.6, +2.8]** |
| 5 | Same prompts on 0.8B | I-22 | after 3 |
| 6 | CUT3R drift vs stream length (needs ScanNet GT poses — data not local yet) | I-05, I-11 | proposed |
| 7 | Object facts v2: question-time lookup of the named objects, closest-point distances, no counts | I-28 | done: null (−0.2) — abs distance good but 18% coverage; rel distance misleading |
| 8 | Question-time detection of the named objects (coverage), abs distance only | I-29 | done: **56.0, +0.6 [+0.2, +1.0]**; min_frames=3 optimal |
| 9 | Object size from the lifted point sets (extent), routed like I-29 | I-10, I-30 | done: **negative** (model 66.7 vs measured 40.3) — not attached |
| 10 | The whole routed configuration on 0.8B | I-22 | done: +5.4 (bigger gain) but 24.8 below 4B |
| 11 | Repeat the format instruction after the question (0.8B and 4B) | I-31 | done: **negative both** (0.8B 22.4, 4B 53.0) |
| 12 | The whole routed configuration on 2B | I-22 | done: 40.3 → **42.7** (+2.4); still 11.0 below 4B+prompt |
| 13 | Geometry-only training questions: generate, then S1 + ablation | I-14 | generated (3,651); training after the Sensenova baseline |
| 15 | I-32 appearance order from the tracker's first-seen frames, routed (4B, 60 videos) | I-32, I-29 | done: **null** (56.1, +0.2 [+0.0, +0.6]; 11% coverage, tracker order worse than the model where it covers) |
| 16 | I-04 keyframe time labels in image mode (4B, 60 videos) | I-04 | done: **null** (54.3, +0.7 [−2.4, +3.6]; per-type gains and losses cancel) |
| 17 | **I-14 decisive test**: S1 real vs control on the geometry QA (0.8B, 3,417 records, video-level holdout 234), ablation | I-14 | done: **positive** — shuffled − real +0.168 ± 0.064, value of content +0.052 ± 0.020; numeric kinds only, orientation not read |
| 18 | I-21 thinking on at question time (4B, 20 videos, 512-token budget) | I-21, I-19 | done: **live-incompatible** — 11/333 closed within 512 tokens, score ~3; 2,048-token accuracy probe on 5 videos queued |
| 19 | I-33 relative direction from three tracked positions: diagnostic, then attached at ≥ 1 view (4B, 60 videos) | I-33, I-29 | done: **57.8, +1.8 [+0.9, +2.8]** — adopted; total over the base's best format +9.2 [+5.0, +13.4] |
| 20 | I-21b thinking with a 2,048-token budget (4B, 5 videos): does reasoning help accuracy at all? | I-21, I-19 | queued |
| 14 | **Sensenova-only baseline on 4 nodes x 8 H100**: zero-shot VSI (full); S1 and S2 (one epoch each) real vs control, two seeds; plain SFT (no geometry); S2 joint (no S1); the 2B pair; S2 learning curve; gate | I-12, I-16, I-18, I-22 | four-day run from 09-26 (docs/SERVER_WEEKEND.md) |
