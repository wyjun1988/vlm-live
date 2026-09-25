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

### I-04 Timestamps in the situation prompt — **proposed**
Image-mode keyframes carry no time (order is only implicit). Add keyframe times and per-object first/last-seen
times to the text. Also the basis for recency and decay (I-26).

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

### I-30 Object size from the lifted point sets — **proposed**
VSI asks "the longest dimension of the X in centimeters" (1/8 of the score, currently 65.3). We already hold a
3D point set per object instance, so the extent is `max(ptp(points))` — no new machinery, and the same
view-count gate (I-29 / L5) decides whether to attach it. Risk: our point sets come from box interiors (the
central 50% of each box), so extents are systematically *under*-estimated; check the bias before attaching,
and consider using the full box for the extent while keeping the centre for position.

### Lessons so far (zero-shot, 4B)
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

### I-14 (U#4) Questions that can only be answered with geometry — **proposed (strong)**
Generate new training questions whose answers come from geometry, e.g. "how did the camera move between frame 1
and frame 5", "how far is the point marked here", "which of these two frames was taken closer to the door".
Answers computed from CUT3R's own outputs — no human labels, no teacher model. The format shortcut becomes
impossible (a content-free signal cannot answer them). Directly fixes why S1 learned only format.
- Data: multi-image Sensenova records (153 in the 1k preview have 16–28 images) or any unlabelled video.

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

### I-21 Thinking on at question time (4B) — **parked**
Latency cost; only if I-20 shows thinking itself is what helps.

---

## 5. Model size

### I-22 Smaller LLM with an explicit object map — **proposed**
If measurements (I-07/I-09) carry the spatial reasoning, the LLM mostly reads and compares numbers →
0.8B/2B may suffice. The detector (OWLv2, 622 MB) is cheaper than LLM size. Test after I-09 works on 4B.

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
| 9 | Object size from the lifted point sets (extent), routed like I-29 | I-10, I-30 | next |
| 10 | The whole routed configuration on 0.8B | I-22 | after 9 |
