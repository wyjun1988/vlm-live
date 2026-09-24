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
relative distance −3.5. So the cost seen with the map image (I-08) is not about the picture: geometric text in
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
- Prototype (queued): the VLM's own grounding with the **oracle** vocabulary (object names from the questions) —
  an optimistic upper bound, not a live-valid result.
- Next: OWLv2 (open-vocabulary detector, downloaded) with a fixed indoor vocabulary — question-agnostic.

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

### I-27 Select which facts to show when the question arrives — **proposed**
Everything is still computed in the background, but only the facts relevant to the question are attached to it
(e.g. room measurements for a room-size question, object distances for a distance question). Costs a few dozen
question-time tokens (milliseconds), avoids the −6 on order / direction from always-on text (I-07), and is
live-valid (the question is known at that moment). Rule-based routing first; could be learned later.

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

### I-20 Background thinking — the VLM writes the cognitive map before the question — **testing**
Combine I-01 + I-02 + I-19: in the background the VLM looks at the keyframes and writes a cognitive map /
scene summary (objects, grid positions); it becomes part of the cached situation prompt. The "thinking first"
happens before the question, so question time does not change.
- Zero-shot test (no CUT3R): one generation per video, then the map text in the prompt for every question.
- Compare with I-09 (CUT3R-measured object map): self-estimated vs measured geometry.

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
| 1 | Measured facts as text, no map image (image + video mode, 60 videos) | I-07 | image done (null overall, room size +12.6); video running |
| 2 | Object map, VLM grounding + **oracle** vocabulary (image mode, first 20 videos) | I-09 | queued |
| 3 | Object map, **OWLv2 + fixed indoor vocabulary** (question-agnostic) | I-09, I-10, I-24 | next |
| 4 | Background cognitive map written by the VLM itself (zero-shot, no CUT3R) | I-20, I-19 | next |
| 5 | Same prompts on 0.8B | I-22 | after 3 |
| 6 | CUT3R drift vs stream length (needs ScanNet GT poses — data not local yet) | I-05, I-11 | proposed |
| 7 | Question-time fact selection (room facts only for room-size questions, etc.) | I-27 | after 1–4 |
