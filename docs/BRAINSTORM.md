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
| CUT3R fed ~1 fps instead of the 32 keyframes (I-05) | −0.9 | angles improve (83 % direction facts), metric scale inflates ×1.18 |
| Windowed state resets 40/3 · 80/6 (I-36) | −0.9 · −2.5 | half the scale back, the direction gain lost |
| Per-frame camera-height scale anchor (I-37 v1) | **−8.3** | rescaling each frame by its own factor breaks the layout; the signal is wrong without a floor in view |
| One reasoning hint per type, all types (I-40) | −0.5 | prior hints +6.3 / +1.5 / +1.1 on room / size / distance, procedural hints −6.5 / −4.1 / −2.7 on direction / closest / route — keep the priors only |
| The model places the named objects first (I-41) | **−4.4** | its own placement is wrong often enough to cost on every relational type; same failure as I-20 (−4.8) |
| "Trust the measurements" sentence, 2B (I-42) | +0.1 | copy rate 53 → 49 %: a small model's fact use is not an instruction problem |
| Utonia (PTv3 + ScanNet-20 probe) counts and room area on the CUT3R reconstruction (I-47) | not run: counts MRA 0.26-0.35 vs the model's 0.64; room 0.47-0.55 vs our room fact 0.66 | no instance head: clusters merge (chairs around a table) or split (fragments, drift); CUT3R sees the floor partially |

### Adopted for the streaming path (~1 fps accumulation)

| Step | Score | Δ | 95% CI |
|---|---|---|---|
| keyframes only (the table above) | 57.8 | — | |
| CUT3R fed ~1 fps, maps from every frame (I-05) | 56.9 | −0.9 | [−2.2, +0.4] |
| + frame-count scale calibration on the metric facts at read time (I-37) | 58.0 | +1.1 vs 56.9 | [+0.3, +2.1] |
| + a scale-prior hint for room size, object size and distance questions (I-40b) | 58.3 | +0.3 vs 58.0 | [−0.0, +0.6] |
| + every frame at 1 fps to the LLM, the measurements still from the 32 frames (I-45d, 2026-10-04) | **61.8** | +3.5 vs 58.3 | [+0.3, +6.7] |
| the same chosen causally - frames and measurement frames by halving over the 1 fps grid: **the live number** (I-48, 2026-10-05) | 59.4 | −2.4 vs 61.8 | [−4.4, −0.4] |
| + the LLM capped at 128 frames (I-48c) and the most confident instance per named object, ambiguous directions not attached (I-49, 2026-10-06): **live** | **60.6** | +1.0 vs 59.6 | [+0.2, +2.1] |

Directions 69.2 → 75.0 from the accumulation; room 67.0 → 66.2 and abs distance 42.1 → 39.8 remain the cost of the
drift; size 65.3 → 66.8 from the prior. vs keyframes only +0.5 [−0.9, +2.1]: the best measured configuration, and
the maps are the ones a long video needs.
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

**Run 3 (2026-10-13, the VG-LLM `live` mix, 1.42 M samples, one pass; all 288 videos, 32 uniform frames as images):**
plain SFT **71.3**, gated input fusion (CUT3R) **71.0** - the reference's 70.83 reproduced with our implementation.
gated − plain −0.2 [−1.6, +1.1]: at this data scale the 3D encoder adds nothing in the LLM input, and the routed room
facts cost on the trained arms (−0.7 / −0.8: they size rooms better than the measurement). Data was the ingredient;
for the live system CUT3R's VSI value is now an open question (next: the trained arms on the live path, `ROLE=live`).
**On the live path (2026-10-13):** plain with frames chosen as the video streams and the LLM capped at 128 frames
scores **72.7**, +1.4 [+0.2, +2.7] over its offline 71.3 (32 → 128 frames +2.2; 128 → 256 +0.3; choosing causally at
32 −0.8, n.s.); gated − plain −0.2 at 128 too (appearance order −4.7, significant). The zero-shot frame findings
(I-45, I-48) carried over to the trained model; the measurement facts (I-27..I-49) did not (room facts −0.7 / −0.8).
**The zero-shot strategies on the trained model (2026-10-14, plain at 128 live frames, each vs the format line):**
prior hints −0.0, object distances −0.2 [−0.3, −0.1] (abs distance −1.4: the model's own estimate beats the
measurement), directions −0.0, the whole stack −1.0 [−1.5, −0.5] (room facts −6.3 on room size). The trained form
("\n" after images) evaluates 1.1 lower than the evaluation form. The gate passed (VSI 70.67 live-32 via lmms-eval,
Video-MME 62.74, MMStar 62.64; the base reference is format-handicapped - see docs/SERVER_RUN3.md 10-14).

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
- Long video, first measurement (row 24): with 8 keyframes instead of 32 the 4B loses 6.4 points (53.7 → 47.3)
  even on these 1–3 min videos, and facts computed from the same 8 frames give nothing back (47.2) — a map that
  sees only what the LLM sees is as starved as the LLM.
- Long video, second measurement (row 24b): maps from ~1 fps geometry, LLM at 8 keyframes → 48.3 vs 47.3 without
  facts (+1.0 [−0.8, +2.8]). Where facts exist they work (direction +8.0), but the object detector still ran only
  on the 8 keyframes, so directions were covered for 35% of questions (77% at 32 keyframes) and distances for 4%.
  The geometry is not the limit (room facts as good as the 1 fps run); the object map is, and it is tied to the
  LLM's keyframe budget. For the live system detection has to run on the geometry stream, not on the LLM's
  keyframes — an online object map (row 27): then the maps' coverage no longer depends on how much of the
  video the LLM gets to see.
- Long video, third measurement (row 27a): the online map (detection on every 30th frame and every keyframe,
  median 82 detection frames per video, 52,893 detections lifted vs 17,134 in the 32-keyframe run) at 8 keyframes
  → **48.8**: +0.5 [−1.3, +2.3] over keyframe detection, +1.5 [−0.8, +3.9] over no facts; direction coverage at
  ≥ 1 view 35% → 87% (above the 32-keyframe run's 77%) and the direction type 55.9 → 60.5. So the maps' coverage
  is now independent of the LLM's view — but the facts recover ~1.5 of the 6.4 points lost, because the rest is
  the LLM's own loss on counting (−20), appearance order (−9), relative distance (−16) and size (−5), which no
  fact addresses. Two things to fix in the dense map: the view-count filter (at ~1 fps "3 views" are often three
  near-duplicate frames — distance MRA at ≥ 3 views 35.2 vs 51.4 with keyframe views; the count should require
  time separation, or the threshold has to rise: with dense detection the measured distance improves
  monotonically with the view count, 8.5 → 42.2 MRA from ≥ 1 to ≥ 6 views, and only at ≥ 6 (48% coverage) does it
  beat the model on the same questions, +3.1 on the type), and the scale drift of the accumulated state (I-05),
  which the dense map inherits (median measured/true distance 0.31 at ≥ 1 view → 0.87 at ≥ 6).
- At 32 keyframes (row 27b) the same dense map scores **55.4 vs 57.8** (−2.4 [−4.0, −0.7]): direction coverage
  77% → 86% but accuracy 76.4 → 75.6 (net 0), distances −8.0 from the diluted view count, room −10.8 from the
  ~1 fps drift. So dense detection is necessary for long video but not sufficient: it needs the drift control and a
  view count that means separate views (row 30). The view count now uses real frame indices and `--view-gap`
  (frames apart) so that "3 views" can mean three views at least a second apart.
- **Combined (row 30): 55.1**, −2.7 [−4.6, −0.9] vs keyframes only and no better than the dense map without the
  fixes (55.4). A 1-second view separation is not enough: the dense map's distances beat the model only at ≥ 5–6
  separated views (47–54% coverage, +3–4 on the type) and stay under-scaled (median measured/true 0.68 at ≥ 3
  views) — many oblique, partial observations bias the merged point sets low. Directions at 86% coverage reach
  69.9 (keyframes-only 76.4), the resets' cost. **State of the accumulation question**: on 1–3-minute videos,
  keyframes-only geometry with keyframe detection (57.8) is the best measured configuration; every variant that
  ingests more frames pays 2–3 points for scale drift and lower-quality observations, and buys coverage these
  videos do not need. For the live system (10+ minutes) the coverage will matter and the price has to come down:
  drift control that does not cost orientation (point-cloud alignment or direct scale correction of the state),
  observation-quality weighting instead of view counts, and a long-video benchmark to measure it on.
- Zero-shot form: every keyframe labelled `Frame k at t s:` (`--frame-labels`), on top of the format instruction
  (53.7) → **54.3, +0.7 [−2.4, +3.6]**. Not a gain: the labels change answers of every type (435 of 1,151), and
  the changes cancel — route planning +5.4 and counting +3.2 against relative distance −6.4 and room size −2.3.
  The interval is four times wider than the routed-facts ones because the labels perturb everything rather than
  adding one targeted fact. Timestamps stay relevant for the live system (recency, I-26), not as a VSI lever.

### I-05 CUT3R forgetting and drift over long streams — **measured: scale drift (2026-09-27)**
The CUT3R state is a fixed 768-token memory overwritten every frame, so old content dilutes and poses drift.
In evaluation CUT3R only sees 32 sparse keyframes, which hides this; live at 10 fps it sees thousands of frames,
and the maps inherit any drift because they live in the first frame's coordinates.
- Test: ScanNet sequences with ground-truth poses — pose / scale error vs stream length; map consistency when
  the camera revisits a place. Mitigations: windowed state reset + re-anchoring, or I-11.
- **Measured without GT poses, through the facts** (row 23): CUT3R fed every 30th frame (~1 fps; 3–4× the
  frames of the keyframe-only runs) with the maps built from all of it, on the 57.8 configuration → 56.9
  (−0.9 [−2.4, +0.8]). Scale-free facts got **better**: direction accuracy at ≥ 1 view 76.4 → 83.4, the type 69.2
  → 75.0 (+5.8). Metric facts got **worse**: room-area median measured/true 1.07 → 1.18, area MRA 65.8 → 55.5
  (type −10.8), distance MRA at ≥ 3 views 51.4 → 43.2 (type −2.3). Relative layout holds and improves with more
  views; the metric scale inflates ~10% over ~100 ingested frames. That is the recurrent state's scale drift,
  and the live system (thousands of frames) will see more of it.
- **Trend (row 26, the same 20 videos)**: room-area measured/true **1.03 with keyframes only → 1.23 at ~1 fps →
  1.39 at ~3 fps**; area MRA 57.5 → 52.0 → 30.0; overall 59.1 → 56.9 → 54.7 (−4.3 [−6.5, −2.4] at 3 fps). The
  inflation grows with the number of frames the state has ingested. Direction accuracy on these 20 videos is the
  same in all three runs (85.4 on 41 questions) — the +7 seen on 60 videos came from the other 40. At live frame
  rates the metric facts need a drift control: state resets (I-36), or a separate short-window state for the
  metric path while the long state serves directions.
- **I-36, windowed state (reset every 40 ingested frames, 3 frames re-fed, rigid chaining), same ~1 fps input**
  (row 28): overall 56.9 (= no resets; −0.9 [−2.6, +1.0] vs keyframes only). Metric facts recover part of the
  way — room-area ratio 1.18 → 1.10 (keyframes 1.07), area MRA 55.5 → 58.7 (65.8), distance ratio at ≥ 3 views
  0.82 → 0.86 (0.99) — but the direction gain is lost: measured accuracy 83.4 → 77.7 (keyframes 76.4), the type
  75.0 → 71.7. The long recurrent baseline is what improved the relative layout; cutting it every 40 frames gives
  that back while only halving the drift. Neither accumulation variant beats keyframes-only.
- **80 / 6** (row 29) is worse: 55.3. Its scale is as good (room ratio 1.12, distance ratio at ≥ 3 views 0.93,
  the best of the accumulated variants) but direction accuracy falls to 63.1% — the fresh window is aligned to
  the *drifted* poses of the old state, and more shared cameras average in more of that drift as an orientation
  error. Rigid pose-chaining is the weak link. Better: align windows on their overlapping point clouds, or
  correct the state's scale directly (the same surface re-observed at a growing scale is detectable against the
  accumulated map), or keep two states — a long one for directions, a short one for metric facts.

### I-06 Halving-aware incremental prefill — **parked**
Append keyframes to the LLM cache as they arrive and rebuild only on evictions. Not needed on H100 (full
rebuild is fast enough); relevant for an on-device demo.

### I-50 Voice front-end: a spoken Korean question in, its text out — **built; 4.5 % of real Korean turns truncated at a 0.80 s mean wait (2026-10-13)**
User request 2026-10-12; the survey and the choices are in docs/VOICE_FRONTEND.md. `live3r.serve.voice`: Silero VAD
v6.2 -> Smart Turn v3.2 scores each pause 0.2 s in -> the turn commits after 0.7 s of silence when it says finished,
1.5 s when not -> Qwen3-ASR-1.7B transcribes (speculatively, from the scoring point) -> the reader (I-38).
- `scripts/voice_eot_eval.py` on eot-bench `ko` (377 real human-to-agent turns): eot-bench's own protocol reproduces
  the published Smart Turn numbers (9.7 % cut-offs at 0.6 s, as published; 710 vs 734 ms at 5 %; the VAD baseline
  exactly), so the preprocessing is right. End to end with our VAD, counted per turn: **4.5 % truncated at a mean
  wait of 0.80 s** - 0.9 % of the turns under 8 s of speech (0.1-0.5 mid-turn pauses each), 10.3 % of the longer ones
  (2.5 pauses). The action delay is the knob (0.6 s: 8.0 % at 0.71 s; 0.8 s: 2.9 % at 0.88 s); the threshold is flat
  over 0.4-0.8. A plain 1.0 s silence timeout gives 3.4 % at 1.03 s: the detector buys ~0.2 s at equal truncation.
- A text end-of-turn model (Namo-v1-Korean) on the reference transcript up to each pause - its upper bound - adds
  nothing measurable, alone or fused (best 6.9 % vs 8.8 % at a 0.7 s wait, in-sample over a large grid): in these
  calls people pause after complete sentences and go on. Not attached.
- `scripts/voice_asr_eval.py`: Qwen3-ASR-1.7B vs the benchmark's reference transcript, CER 10.3 % (7.5 % without
  numbers, Latin letters or truncated references). The reference is machine-made and often the wrong side ("메일이"
  vs "매일에", "룸 이용료" vs "이용류"), and the model keeps fillers and stutters the reference drops, so the true
  error is lower. 0.44 s per question on the M2.
- `scripts/voice_demo.py --reader`: 7 synthetic Korean questions (macOS `say`, pauses up to 0.9 s, one after "-면"):
  7 turns, no truncation, transcripts exact but one character (변 -> 편); the 4B reads 7/7 (type and roles).
Open: our own spoken spatial questions (the eot-bench turns are reservation calls), the ASR pass on the H100.

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
- **On the 2B** (same run): 43.5 vs 42.7, +0.9 [+0.2, +1.6]. The facts are identical (same tracker), but the 2B
  chose the attached direction on only **50%** of the 157 questions (4B: 93%), so the same fact is worth half as
  much — a fact the model has to map onto a lettered option is harder to copy than a number it repeats (2B copied
  room/distance numbers 93–98% of the time). Measurements still do not buy back model size (I-22).

### I-34 Route-planning turns and closest-object rankings from the map — **diagnostic negative (2026-09-27)**
The last two VSI types the map could speak to, logged at every threshold from one routed run (nothing attached):
- **Closest-object ranking** (172 questions): the map's argmin over the four options is right 40.7% at ≥ 1 view
  (model on the same questions 63.0), 54.2% at ≥ 2 (59.3), 66.7% at ≥ 3 with 9% coverage (66.7). Never better
  than the model — I-28's rejection holds with the improved estimator. Ranking four noisy distances compounds
  their errors; reporting one (I-29) does not.
- **Route planning** (34 questions in the subset): the turn sequence along the path (the direction math applied
  leg by leg) needs every waypoint tracked — 15% coverage — and was right on 1 of those 5 (the model: 3 of 5).
  The questions' path abstraction ("go forward until the trash bin is on your right") is not the map's straight
  legs, and the category is 3.8% of the benchmark anyway.
So **57.8 is the ceiling of the routed-measurement family** on this subset: every VSI type that geometry can
answer better than the frames (room size, absolute distance, relative direction) is already covered, and the
rest (order, counting, size, closest-object, route) are either read from the frames already or beyond what
tracked centroids can give.

### I-35 The routed-measurement harness as an encoder bake-off — **first result: CUT3R wins clearly (2026-09-27)**
Every fact the live system attaches is scored against VSI ground truth (room area MRA, distance MRA, direction
accuracy at each view threshold) — so any encoder that yields poses and point maps can be compared with CUT3R
without training a single parameter: swap the encoder behind SceneMap/ObjectMap, run the 60 videos, read the
measurement table. First comparison encoder: **VGGT on a sliding window** (`geometry.name: vggt_window`,
`configs/m2_zeroshot_4b_vggt.yaml`): each new keyframe re-runs VGGT-1B on the last 8 frames; consecutive windows
are chained into one frame by the cameras they share (a similarity transform, `sim3.py`; the global scale is set
by the first two-frame window — VGGT's scale is normalised, not metric). ~8 s per window on the M2. Read-out:
direction accuracy (scale-free) head to head with CUT3R's 76.4% at ≥ 1 view; room area and distances after ratio
normalisation (shape quality) and the median ratio itself (how far from metric). What it decides: whether the
streaming, metric, recurrent encoder (CUT3R) is leaving accuracy on the table versus a joint multi-view model that
would need a scale source live (a metric depth model, or CUT3R itself).
- **Result** (60 videos; the 0.8B as the LLM since the facts do not depend on it and VGGT + 4B swaps the M2):

  | fact | CUT3R (keyframes) | VGGT window 8, chained |
  |---|---|---|
  | direction accuracy at ≥ 1 view (coverage) | **76.4** (77%) | 56.7 (51%) |
  | the 99 direction questions both cover | **72** right | 54 right |
  | room area MRA / median measured-true ratio / scale-free MRA | **65.8** / 1.07 / **69.0** | 9.5 / 0.54 / 16.2 |
  | distance MRA at ≥ 3 views, scale-normalised (coverage) | **51.4** (31%) | 37.3 (8%) |
  | detections lifted into the map | 17,134 | 7,950 |

  The per-video VGGT/CUT3R area ratio has median 0.50 and IQR 0.24–3.85: the per-window normalised scale does
  not chain consistently through the shared cameras, so the global map is a patchwork of scales and even the
  scale-free facts degrade. A metric, recurrent encoder is the right map builder here — and it is the streaming
  one. Caveats: window 8, chordal-mean chaining, fp32 on MPS, no offline joint-32 upper bound. The harness is
  ready for the next encoder (an adapter + one 60-video run).

### I-36 Windowed CUT3R state (reset + overlap, rigid chaining) — **tested: no gain (2026-09-27)**
Drop the recurrent state every N ingested frames, re-feed the last k frames to the fresh state, align the new
window to the global frame rigidly through the shared cameras (`geometry.options.reset_every / overlap`).
- 40 / 3 at ~1 fps: 56.9 (= no resets); room-area ratio 1.18 → 1.10, but the direction facts' accuracy 83.4 → 77.7 %.
- 80 / 6: 55.3; direction 63.1 % — chaining to drifted poses through more shared cameras inherits their
  orientation error.
- Lesson: resets trade the accumulation gain (angles) for half the scale. Not the fix.

### I-37 Metric scale correction — **v1 (per-frame anchor) strongly negative; frame-count calibration at read time: 58.0 = parity with keyframes at ~1 fps (2026-09-28)**
The state's scale inflates with the frames ingested (I-05: room area ×1.03 / ×1.18–1.23 / ×1.39 at keyframes /
~1 fps / ~3 fps) while the layout improves. v1: per frame, camera height above the frame's own floor (2nd
percentile of its points along up) vs the first 10 frames; points and camera divided by the ratio about the origin.
- 60 videos at ~1 fps, 4B: **49.5** vs 56.9 without (−7.4 [−9.3, −5.4]); room MRA 55.5 → 14.3 (median ratio 0.70 —
  over-shrunk, 37 % of videos within 50 %), **scale-free MRA 65 → 22**, direction facts 83.4 → 70.7 %, abs distance
  39.7 → 31.6.
- Two failures, both instructive. (1) The per-frame signal is wrong often: a frame without the floor in view
  (wall, table top) puts the low percentile above the floor, the camera looks lower, the factor inflates; a
  hand-held camera also moves up and down. (2) **Rescaling each frame by its own factor destroys the layout**:
  frames scaled differently shift relative to each other about the origin, so the room's shape and the angles
  between objects seen in different frames break. Angles do not need a scale at all.
- v2 lab (16:18, `scripts/capture_geometry.py` + `scripts/scale_anchor_lab.py` on the captured raw streams; the
  capture reproduces the reference run's maps exactly): **every floor-reading variant carries no signal.** Camera
  height or floor depth, windows 5–15, support test, gate — the correlation across videos between the factor applied
  and the ideal one (sqrt(truth / measured area)) is −0.11…+0.13; the best variant equals `none` (55.0 vs 55.5)
  while skipping half the frames; the global application never beats 48. The per-frame floor says nothing about how
  far the state has inflated.
- What does: **the number of frames ingested** (`scripts/scale_calibration.py`; corr(log area ratio, log n) = +0.44
  over the 60 videos, n = 51–250). log(measured/true) = 0.3224 log n − 1.3781; applied as one factor per video to
  the facts at read time, **split-half room-area MRA 55.5 → 63.8** (ceiling, the median-ratio oracle, 65.2). The same
  curve applied per frame to the points scores 25.2 — so the drift is not a uniform inflation about the origin, and
  no per-frame rescaling of the geometry can be right. `FrameCountScale` (`scale_anchor: {mode: frame_count, a, b}`,
  read-time application via `CUT3RStream.metric_scale` → `SceneMap.facts(scale)` and the attached distances) is the
  4B run (`configs/m2_zeroshot_4b_anchor_v2.yaml`; caveat: a, b fitted on these 60 videos, 2 parameters — in-sample
  room MRA 66.0 vs 63.8 split-half). By length, median measured/true: short videos (< 70 frames, n = 8) 1.00 → 1.03,
  mid (70–149, n = 38) 1.17 → 0.99, long (≥ 150, n = 14) 1.30 → 0.92; videos within 50 % 85 → 90 %.
- **Result (18:41): 58.0.** vs the same ~1 fps stream without the correction (56.9) **+1.1 [+0.3, +2.1]** — room type
  56.2 → 65.7 (+9.5), everything else unchanged (abs distance −0.5, noise). vs keyframes only (57.8) +0.2 [−1.3, +1.9]:
  room 65.7 vs 67.0, abs distance 39.3 vs 42.1 (the object positions still carry the within-video drift), direction
  75.0 vs 69.2 (+5.8, the accumulation gain kept). Maps: room MRA 66.0, median ratio 0.97; the factors applied ran
  0.82–1.06 (median 0.94). So **the live configuration — the state accumulating at ~1 fps, as a stream must — now
  matches the best offline configuration** instead of paying 0.9 for it, and the maps it builds are the ones a long
  video needs. Adopted for the ~1 fps path; the estimator that could beat a calibration (self-view ratio) is next.
- **Out of sample (09-29, 03:06): the calibration holds on videos it was not fitted on.** Streaming stack on the
  seed-2 set (60 videos, 45 not in the fitting set): room-area median ratio 1.10 → 0.96 on the 45 new videos, MRA
  54.2 → 63.3, videos within 50 % 84 → 96 % (all 60: 58.7 → 65.7). Two parameters, and the drift's dependence on
  the frame count is a property of the encoder, not of the fitting videos.
- **Streaming vs keyframes on the seed-2 videos (04:27): 55.3 vs 56.1, −0.8 [−1.9, +0.5]**; room at parity (both
  65.7, ratio 0.96 vs 1.03), abs distance −1.0, **direction −3.4** — the direction facts were *less* accurate from the
  accumulated maps here (64.7 % vs 69.6 % at ≥ 1 view, coverage 70 vs 74 %), the opposite of seed 1 (83.4 vs 76.4).
  **Pooled over the 105 distinct videos: streaming stack − keyframes-only = −0.1 [−1.1, +1.1].** The honest
  statement: the configuration a live stream must run (state accumulating at ~1 fps) is at **parity** with the best
  offline configuration; the calibration is what buys the parity (without it, −0.9 and −1.1 on the two sets); the
  "accumulation improves angles" effect of seed 1 does not replicate and was noise.
- **CUT3R's own two point maps carry no signal either (19:40).** The head predicts the frame's points in the camera
  frame (`pts3d_in_self_view`) and in the state's world frame; through the predicted pose their distance ratio is
  **1.00 on every frame of every video** (`SelfViewScale`, relative or absolute, w5/w15: factor 1.00, corr −0.06…+0.01).
  The "monocular" head is conditioned on the same state, so its scale inflates together with the world map — the
  model is internally consistent about a wrong scale, and nothing it outputs reveals the drift. The calibration
  (frame count) stays; the per-video estimator would have to come from outside the model (object-size priors from
  the detector — doors, chairs — or a metric depth model on a keyframe).

### I-38 The model reads the question first — routing for any phrasing, any language — **4B: reads perfectly (2026-09-28)**
The routed facts were keyed on VSI's templates with regular expressions — exact on the benchmark, useless for a real
question. `live3r.serve.interpret`: the model returns one line of JSON (type, objects, roles, English rendering),
which is rendered back to a VSI-style template so the existing fact router runs unchanged; the rules are the
fallback when the JSON does not parse (`scripts/interpret_check.py`, `eval_vsi_local.py --interpret llm`).
- 4B, 160 VSI questions (16 per type): type 100 %, roles 99 %, **routes identically to the rules on 100 %**; the
  two "misses" are the model being more right than the regex ("facing the door and to the left of the trash bin" →
  facing *door*; "the farthest sofa" → *sofa*).
- 28 paraphrased / Korean questions (hand-written readings, no template): **type 100 %, roles 100 %** — e.g. "냉장고
  옆에 서서 문을 바라보면 침대는 왼쪽이야 오른쪽이야?" → rel_direction {standing refrigerator, facing door, target
  bed, easy}. 96 % of replies parsed; the rest fell back to the rules and were still right.
- Cost: 4.9 s per reading on the M2 (≈100 JSON tokens, 4B) — an H100 makes that ~0.3–0.5 s, still a large share of
  the 1 s budget. **The 0.8B cannot do it zero-shot**: type 54 % / roles 29 % on VSI (it copies the prompt's
  "count / chair" example for order, size and route questions), 43 % / 32 % on the paraphrases, 2.0 s per reading.
  So the levers are: a compact output format for the 4B, and — the case the user's LoRA suggestion fits — a reading
  LoRA on the 0.8B trained from (question → JSON) pairs, which the 4B's own readings can generate (the check's JSON
  keeps every reply). The zero-shot 4B itself needs no training for this.
- **Compact format (20:03):** the full reply is 74 tokens (4.9 s on the M2); the compact one (no object list, English
  rendering only for non-English questions) **29 tokens, 2.4 s**, VSI agreement type 99 % / roles 98 % / identical
  routing 99 %, paraphrases 93 % / 86 %. Its misses were structural — no rel_distance example in the compact prompt
  (read as abs_distance / order), Korean object names ("bedside", 책상 → table), easy vs medium choices — so the
  prompt gained a rel_distance and a Korean easy-direction example. **Re-check (21:38): 100 % type / 99 % roles /
  identical routing 100 % on VSI, 100 % / 100 % on the paraphrases, still 29 tokens** (the two remaining "misses"
  are the model reading better than the regex). The compact format is now the default of the llm backend. On an
  H100 with the prefix cached, ~30 output tokens is what lets the reading fit the 1 s budget.
- What it buys is not a VSI point (the templates were already read exactly) but the live product: questions in any
  wording or language route to the same facts.

### I-39 Learn the scaffold — the pre-prompt as a training target — **planned (user direction, 2026-09-28)**
Everything the harness attaches at question time (format instruction, routed facts, and now the type hints of I-40)
is a hand-written prefix the base model reads zero-shot; the 4B follows a direction fact 93 % of the time, the 2B
50 %. Two ways to make the model itself better at using it: (1) **fact-conditioned SFT** — train with the same
routed facts in the prompt so the answer learns to rely on them (the "B" item of the 09-27 plan; it needs maps over
the training videos); (2) **prompt tuning** — learn the pre-prompt's embeddings (a soft prefix, a few hundred
parameters × hidden) on the same data, which touches no LLM weight beyond LoRA and can be swapped per question
type. Order: measure I-40/I-41 first (do hand-written scaffolds move anything?), then train the one that does.

### I-40 One reasoning hint per question type, no measurement — **tested: priors help, procedures hurt (2026-09-28)**
The interpreter knows the type and the named objects, so a hint can go to that type only, a sentence long, after
the cached prefix. Base + format hint (53.7) with a hint for every type, no geometry: **53.1, −0.5 [−2.6, +1.4]** —
but the types split cleanly:
- hints that supply a **prior** helped: room size +6.3 ("estimate the floor area from the room's width and length"),
  object size +1.5 ("judged against typical objects - a door is about 200 cm"), abs distance +1.1 (the same scale
  anchors in meters);
- hints that prescribe a **procedure** hurt: relative direction −6.5 ("first fix your pose ..."), closest object −4.1,
  route −2.7.
So the model does not lack a reasoning scaffold — telling it how to reason makes it worse — but it does use a stated
scale prior, and it does so on exactly the types where measurement was weakest or unavailable (sizes, I-30). Default
hint set is now the three priors (`--type-hints room_size,size,abs_distance`).
- **I-40b (23:39): the three priors on the streaming stack (~1 fps + facts + calibration, 58.0) → 58.3**, +0.3
  [−0.0, +0.6]: size +1.5 (the same gain as without facts — the size type has no measurement, the prior is all it
  gets), abs distance +0.6, room +0.5 (the prior adds a little even next to the measured room facts). vs keyframes
  only 57.8: +0.5 [−0.9, +2.1]. Small, free (one sentence, three types) — kept in the streaming stack. **On the
  seed-2 videos the size gain did not reproduce (67.6 vs 67.9 without the hints); pooled, the hints are ≈ 0.**
  Treat them as harmless, not as a gain.

### I-41 The model places the named objects first (focused two-turn) — **tested: negative, −4.4 (2026-09-28)**
I-20 (whole-scene cognitive map first) scored −4.8. The user's proposal narrowed: before the question, one or two
sentences on where only the objects the question names are, relative to each other and to the camera's path; the
text goes in front of the question. **49.3 vs 53.7, −4.4 [−6.6, −2.2]**, 1,088 descriptions: closest object −8.7,
appearance order −8.8, direction −7.2, counting −5.6, size −2.0; distances and room unchanged. So I-20 did not fail
for describing *everything* — it fails whenever the model describes at all: its own placement ("the door is to the
left of the mirror") is wrong often enough that reading it back costs more than it gives, and it drags the answer
towards the description even on the types the description does not address (order, counting). Together with I-20,
I-09 (−13.7) and I-21 (thinking, null): **the model must not write about the scene before answering**; what helps
is a measured fact or a stated prior, never its own narration.

### I-42 "Trust the measurements" hint for the 2B — **tested: null (2026-09-29, 01:01)**
The 2B follows an attached direction fact 53 % of the time (the 4B 93 %). One sentence after the facts — "the
measurements above come from a 3D reconstruction of the video; trust them over your own estimate" — on the 2B facts
run (43.5): **43.6, +0.1 [−0.6, +0.8]**; copy rate 53 → 49 %, direction −0.9, room +1.5. Instruction does not make a
small model use the facts; that is capacity or training. On the 157 direction questions with a fact, the 4B copies
93 % and answers 80 % right against a fact that is right 83 % — it is already at the facts' ceiling. This is the
precise case for I-39: teach fact use by training, not by prompt, and it matters for the small models only.

### I-44 Visual-token gain — **idea (2026-09-30), from the fusion's step-0 check**
Checking where our gated input fusion starts (`scripts/check_fusion_init.py`, docs/SERVER_RUN3.md): a LayerNorm on
the visual tokens at the LLM input makes them 5.6× louder (4B: norm 9.0 → 50.6; 0.8B 5.5 → 32) and changes 58-75 %
of the first answer tokens - yet the held-out Sensenova loss *drops* before any training (4B 1.71 → 1.50 at a 2 %
geometry mix; 0.8B 2.68 → 2.40), on 12-16 samples. If real, the base model under-weights its visual tokens for spatial
QA, and a single scalar gain on the visual embeddings is a zero-shot lever. Test on the M2: VSI (60 videos, keyframes)
with the visual embeddings × {1, 2, 3, 5.6} and LayerNorm'd, per type; the loss effect first on 200 holdout samples.
Watch the general side too (a louder image may drown the question).

### I-45 Every frame at 1 fps in the LLM, not 32 keyframes — **tested: +4.4 [+0.8, +8.0] (corrected 10-04); adopted with I-45d**
> Correction (2026-10-04): the first vanilla 1 fps runs had all 31 questions of the longest video (scannetpp/38d58a7a31,
> 219 frames) answered "!!!!" - a transient failure of the long sequential MPS run (the same video run alone answers
> normally; the CUT3R runs never broke). With that video re-run alone and spliced in: format hint only **58.1 vs 53.7,
> +4.4 [+0.8, +8.0]**; with the prior hints 59.1 vs 54.8, +4.3 [+0.7, +7.7]. The numbers below are the first run's.
The reference team's numbers (docs/REFERENCE_70_ANALYSIS.md §9): their trained 70.83 model scored **72.18** when fed
every frame at 1 fps (up to 512) instead of 32 uniform frames; and our own 8-keyframe run (I-36) lost 6.4. Live, the
frames accumulate anyway (cached prefix). `eval_vsi_local.py --budget-fps 1 --budget-cap 256`: per video, its length in
seconds frames (median 81.5, 24-219; 2 of 60 videos under 32), as images, base 4B + format hint, the 60 seed-1 videos:
**56.4 vs 53.7, +2.7 [−1.1, +6.4]**. Per type: appearance order **+11.5**, relative direction **+10.6**, route +5.4,
counting +1.4, size +1.0, abs distance +0.4; relative distance **−4.7**, room size **−4.2**. Format failures 2.7 %.
The cross-frame types gain as the reference's did (theirs: rel dir +5.5); the metric/relational ones that our routed
facts already cover lose a little. → I-45b: the adopted streaming stack (58.3) with every frame at 1 fps.

**I-45b** (the stack + every frame at 1 fps): **59.5 vs 58.3, +1.2 [−2.6, +4.8]** - the best zero-shot score so far,
not significant. Order +11.5, route +5.4, rel distance +4.1, counting +3.4, size +2.6; but **rel direction −7.8**
(75.0 → 67.1), room −5.0, abs distance −4.4 - exactly the types the facts serve. The facts got worse, not the
model's use of them: direction facts right **83.4 → 71.6 %** (the model follows them 93 → 89 %), room-fact MRA
0.658 → 0.633, the scale factor 0.944 → 0.884 (median). In this run CUT3R ingested the every-30th-frame stream plus
the 1 fps keyframes - ~2 fps, outside the ~1 fps regime the calibration and the direction tracking were tuned on.
Against 1 fps alone (56.4) the stack still adds +3.2 (room +12.2, rel distance +8.7). → I-45c: CUT3R fed only the 1
fps keyframes.

**I-45c** (CUT3R on the 1 fps keyframes only, ~1 fps): **59.9 vs 58.3, +1.6 [−2.3, +5.5]** - the best zero-shot score.
But the facts did not come back: direction facts 73.4 % (I-45b 71.6, the 32-frame stack 83.4), room-fact MRA 0.620,
the scale factor now 0.978 as calibrated. So the ingest rate was not the cause; what both 1 fps runs share is the
frame set the object map detects and tracks on (every 2nd 1 fps keyframe instead of the 32 uniform frames).
→ **I-45d**: decouple - the LLM gets every frame at 1 fps, the measurements are made exactly as in the 58.3 run
(`--facts-budget 32`: CUT3R every 30th frame + the 32 uniform frames, detection on those 32; StreamingSession's
`facts_selector`). If the facts come back to their 32-frame quality the expected score is ~61.5.

**I-45d: 61.8 vs 58.3, +3.5 [+0.3, +6.7]** - significant, and the best zero-shot score. The attached facts are
identical to the 58.3 run's on all 1,151 questions (direction facts 83.4 %, room-fact MRA 0.658, scale 0.944), so the
gain is the frames alone: order +11.5, route +5.4, rel distance +4.1, counting +3.4, size +2.6, abs distance +1.4,
rel direction +0.8, room −1.5. Against 1 fps alone (56.4) the facts add **+5.4 [+2.3, +9.8]** (room +15.7, rel
direction +10.6, rel distance +8.7): the two levers stack. Cost: ~2.6x the visual tokens (median 82 frames x 128).
Rule for the live design: **every frame at 1 fps goes to the LLM; the measurements are made on a sparse, fixed
frame set** - a denser detection/tracking set made the direction facts worse (83 → 72 %). (The 32 measurement
frames are uniform over the whole video here, an oracle choice; live they need a causal selector, e.g. halving.)

### I-46 CUT3R removed: vanilla Qwen3.5 with the same prompt — **CUT3R's own share is +2.7 to +3.5 (2026-10-04)**
The stack's prompt minus its CUT3R facts (format instruction + the three prior hints), same 60 videos:

| Qwen input | vanilla + format | + prior hints (no CUT3R) | + CUT3R facts (the stack) | CUT3R's share |
|---|---|---|---|---|
| 32 uniform frames | 53.7 | 54.8 | 58.3 | **+3.5 [+1.8, +5.3]** |
| every frame at 1 fps | 58.1 | 59.1 | **61.8** | **+2.7 [+0.9, +4.6]** |
| 1 fps − 32 | +4.4 [+0.8, +8.0] | +4.3 [+0.7, +7.7] | +3.5 [+0.3, +6.7] | |

CUT3R changes only the types it attaches facts to - at 1 fps rel direction +10.6, room size +10.3, abs distance +0.8,
every other answer identical. **More frames to the LLM (59.1, no CUT3R) is worth more than CUT3R facts on 32 frames
(58.3)**; the two add (61.8). For the live design: the frame budget of the LLM is the bigger lever, the CUT3R facts a
consistent +3 on top.

### I-47 Utonia on the CUT3R reconstruction: counts and floor area as facts — **worse than the model on both, not attached (2026-10-04)**
Utonia (Pointcept PTv3, 2026, one encoder for all point clouds) with its released ScanNet-20 linear probe, as the
user's floorplan-vlm project wraps it for MPS (`encoders/utonia_encoder.py`, `zeroshot/scene_text`), labels every
point of our own streaming reconstruction: the captured CUT3R stream (stride 30 + the 32 keyframes,
`outputs/geom_capture/stride30`), each frame's least confident 30 % dropped, colored from the video, the frame-count
scale (I-37), z-up from the cameras' mean up vector. Objects per class two ways: every connected cluster (8 cm
voxels, ≥ 50 points), and floorplan-vlm's own rules as they are (≥ 80 points and ≥ 0.3 m across; doors and windows
≥ 0.5 m wide). Room area three ways: occupied 10 cm floor cells, the wall-aligned 2-98 % box of the floor points,
and that box over all points (the room fact's rule). 3-17 s a scene on the M2 (`scripts/utonia_scene_probe.py`,
`scripts/utonia_probe_report.py`).

| 60 videos, MRA (VSI's metric for numbers) | Utonia | the model (61.8 run) | our room fact |
|---|---|---|---|
| counts: the 74 of 125 counting questions on a ScanNet-20 class | 0.255 every cluster · 0.350 floorplan rules | **0.642** | |
| room size: 60 questions | 0.547 floor cells · 0.472 floor box | 0.647 | **0.658** |

Every class with n ≥ 4 is worse under both rules, except doors under floorplan's rules (0.75 vs 0.55 on 12
questions - the best of 18 class × rule cells, too few to act on). The labels look right (wall, floor, cabinet,
chair, table in plausible shares; a bathroom gets its toilet, sink and tub); the instances are wrong, both ways.
Chairs around a table merge into one cluster (classrooms: 23 → 4, 36 → 4, 16 → 4), while CUT3R's drift draws walls
and objects twice (2 chairs → 5-6). floorplan's size rules trade over-counts for under-counts (39 / 20 → 22 / 33).
(A repeat of the every-cluster pass gave 0.277: Utonia on MPS is not bit-stable.)

Is it Utonia or the reconstruction? The 20 VSI ARKitScenes scenes that have a ground-truth scan in floorplan-vlm
(`storage/data/gt/arkitscenes`, none among the 60) were probed on the scan and on a fresh CUT3R capture of the same
videos (`capture_geometry.py --only`), same questions:

| 20 ARKitScenes videos, MRA | counts (20 q): every cluster | floorplan rules | room (20 q): floor cells | floor box | box of all points |
|---|---|---|---|---|---|
| ground-truth scan | 0.200 | 0.195 | 0.565 | **0.630** (1.03 × truth) | 0.285 (1.52 ×) |
| CUT3R stream | 0.355 | 0.250 | 0.405 | 0.245 (0.58 ×) | **0.595** (0.84 ×) |

Counting fails on the clean scan too (14 over / 3 under): denser points leave more fragments, and the scan reaches
into the neighbouring spaces. Utonia ships a semantic probe, not an instance head, and counting needs instances. The
floor labels are right - the floor box of a complete scan is the best room estimate here - but CUT3R sees the floor
partially and unevenly, so on our stream the floor box collapses and the box of all points (the rule we already use)
stays best. **Nothing attached**: a wrong count gets copied (L1), and no room estimate beats ours. What would change
it: an instance head on Utonia's features (training), or a reconstruction that sees the whole floor.

The ceiling of such a head (parked - a training job, not zero-shot): perfect counts on the ScanNet-20 classes would
take the 61.8 run to 64.0 (+2.3, 70 questions), perfect distances between ScanNet-20 objects as well to 65.4 (+3.6,
103 questions); counts 10 % off everywhere +1.9, 25 % off +0.5. Relative direction and relative distance questions
name objects outside ScanNet-20 94-100 % of the time - the vocabulary limits Utonia more than its instances do.

### I-48 The 61.8 stack made live: causal frame and measurement selection, LLM frame caps — **live 59.4; a 128-frame cap is free, 64 costs −0.7 (2026-10-05)**
The 61.8 run chose its 32 measurement frames uniformly over the whole video (an oracle: it needs the length), and
CUT3R ingested them on top of the 1 fps frames. Live versions (`--modes live-image --facts-selector live`): the LLM's
frames and the measurement frames by halving over the 1 fps grid (`GridSelector(HalvingSelector)`) - every 1 fps
frame until the budget, then spread over everything seen; the budgets are the cap and 32, so nothing depends on the
video's length. CUT3R ingests exactly the 1 fps frames in all three runs, so their measurements are identical and
the caps change only the LLM's frames (the cap binds on 37 / 12 of the 60 videos at 64 / 128; median 81 frames).

| 60 videos | VSI | vs |
|---|---|---|
| the 61.8 run (oracle frames and measurement frames) | 61.8 | |
| (a) live, uncapped | 59.4 | −2.4 [−4.4, −0.4] vs 61.8 |
| (b) live, the LLM capped at 64 frames | 58.7 | −0.7 [−2.0, +0.6] vs (a) |
| (c) live, the LLM capped at 128 frames | 59.6 | +0.2 [−0.4, +0.8] vs (a) |

**The frame budget is cheap**: 128 frames (~15k visual tokens at ≤ 128 a frame) lose nothing on VSI's 1-4 minute videos, 64 lose 0.7
(n.s.). **The measurements are not**: (a) loses on the types the facts serve - rel direction −5.6, room size −4.0 -
because the facts degraded: direction facts 83.4 → 72.3 % right (coverage 77 → 78 %), room-fact MRA 66.0 → 60.7
(scale 0.97 → 1.07 × truth, scale-free 65.0 → 61.8). The other types moved only with the frames' new positions
(counting −3.0 on 16 changed answers, order −2.0 on 6: the noise of moving the same number of frames).

72 % is also the I-45b/c number, which we put down to the dense detection set; here detection is sparse (32 frames)
and it is still 72 %. What the 83 % runs had and every 72 % run lacked: the measured frames were **extra** CUT3R
inputs, each ingested within a second after a 1 fps frame, and only 32 of them. Hypothesis: a frame CUT3R sees right
after a nearby view is placed more precisely, and the 32 extra views also sharpen the map (the scale-free room MRA
fell without them). The frame-count scale calibration was fitted in the 83 % regime (32 more frames a video), hence
1.07. Live test: measurement candidates half a second off the 1 fps grid, so each is an extra CUT3R input as in
the oracle run.

**I-48b, the test (`--facts-offset 0.5`, cap 128): 59.5, −0.1 [−1.9, +1.8] vs (c) - the hypothesis does not hold.**
Direction facts 76.2 % (coverage 72 %), room-fact MRA 60.8 (scale 0.93, scale-free 62.5). What the 83 % really was:
the whole gap sits in 2-3 object triples per run. VSI asks each (A, B, C) triple 6-9 times (three difficulties, the
roles rotated), and every flipped triple has an object our map found several times (3 lamps in one run, 2 in the
other; 7-9 doors) - the most-observed instance changes with the frame set and the triple flips as a block (the
layouts themselves agree up to a rotation, 0.14 m rms; nothing is mirrored). So the oracle's 83 % was in part a
favourable draw, and oracle vs live on 60 videos is mostly this luck, not a systematic cost. Attaching a direction
only when every instance combination agrees is no fix: 18-29 % coverage at 49-97 % right, erratic across runs.

### I-49 Which "lamp": the instance behind a named object — **most confident detection, ambiguous ones not attached: live 60.6, +1.0 [+0.2, +2.1] (2026-10-06)**
I-48 traced the oracle-live gap to triples flipping with the map's choice among duplicate instances ("3 lamps" when
VSI's question means one). Data run: the live 59.6 configuration with every object observation dumped
(`eval_vsi_local.py --dump-objects`: label, frame, detector score, floor-plan position, point samples) for the 32
measurement frames (set a) and for 32 frames one encoder step later (set b, each with its own CUT3R output);
`scripts/instance_rules_lab.py` replays the online merging (202/204 direction and 176/188 distance facts reproduced)
and scores instance rules on both sets.

- Merging instances by position makes things worse (from the listings, then from the dumps: never-co-observed
  instances within 1.5 m, mean or cluster centroids): the extra "lamps" are mostly other things detected under the
  same name, not the same lamp drawn twice. Suppressing an instance whose points overlap another label's both ways
  (a cabinet front that is also "door") did not help either.
- Choosing by detector confidence does: the instance holding the most confident single detection (score_max) fixes
  the failing triples (set a +18 facts, no video worse) but swaps the error elsewhere on set b (4 videos better,
  3 worse). Adding abstention - no fact when an alternative instance scoring >= 0.8 x the chosen one's top detection
  flips the answer - is the only variant that gains on both sets:

| direction facts, expected accuracy (model alone 65 % where nothing is attached) | set a | set b |
|---|---|---|
| most-viewed instance (until now) | 71.7 | 72.0 |
| score_max | 80.5 | 73.5 |
| **score_max, ambiguous not attached (0.8)** | **78.3** (65 % covered, 85.6 % right) | **78.1** (53 %, 89.8 %) |

Validation (`--instance-rule score_max --direction-abstain 0.8`, everything else as I-48c): **60.6, +1.0 [+0.2, +2.1]**
- rel direction 69.8 → 76.4, abs distance 41.3 → 43.0 (the distances also between the chosen instances), every
other answer identical; direction facts 85.6 % right at 65 % coverage, exactly as replayed. vs the 61.8 oracle run
−1.2 [−2.9, +0.7]: direction is level (76.4 vs 75.8), room size remains (−4.0: the live map and its scale calibration,
I-48), the rest is the noise of moved frames. The rule was picked on these 60 videos (set b is only a partial
check), so the +1.0 wants confirming on other videos.

### I-51 Questions mid-stream, at the moment they become answerable — **proposed (2026-10-15); the tracker's first-seen times are not yet good enough to time them**
Every live evaluation so far asks at the end of the video, as VSI does. A live system is asked during the stream,
about things seen a while ago. Six of VSI's eight types have answers that do not change once the named objects
have appeared (abs / rel distance, rel direction, route, size, appearance order); counting and room size depend on
how much was seen and stay end-only. Protocol: ask each such question at the first time all its objects have been
seen (+ a margin), the same ground truth; compare with the end-of-video answer; the delta against "time since
seen" is the memory curve within a video. Longer memory needs OVO-S-Bench (`docs/BENCHMARKS.md` §2).

The timestamps would come from the tracker's first detections. Checked offline against VSI's appearance-order
ground truth on the 60 videos (the I-49 dumps: detections on the 32 measurement frames): all four objects found in
83 % of the 148 order questions, the exact order right in 22.8 % (chance 4 %), pairwise 64 % (chance 50 %) - too
coarse to time questions by. Before the protocol: detection at 1 fps for the timestamps (not 32 frames) and a
"seen in >= 2 frames" rule, then this check again; the bar is pairwise >= 85 %.

### I-52 Camera motion between two photographs as a sentence — MMSI-Bench — **tested on the base: camera types 27.5 → 45.5, +18.0 [+9.6, +26.3] (2026-10-16, v4)**
MMSI-Bench's camera questions (Cam–Cam 93, Motion (Cam.) 74) ask how the camera moved between two photographs;
open 7-8B models answer at chance and GPT-5 scores 32.4 on the motion type. A two-view reconstruction returns
exactly that, so the geometry encoder's poses of the two images become a sentence (`live3r.serve.camera_facts`:
relative pose in the first camera's frame - to the right / forward / up, the distance, the yaw and tilt; and, for
questions that define "+Y up, −Z forward, right-handed", the same in that vocabulary) attached to those two types
(`lmms_live3r` model_arg `camera_facts=True`; `scripts/mmsi_probe_local.py` on the M2).

Base Qwen3.5-4B with the format line, zero-shot: Motion (Cam.) 25.7 → 48.6 (v1) → **50.0** (v2: turns from 5
degrees, no direction words under 0.3 m, the axis form); Cam–Cam 29.0 → 29.0 → **36.6**; the 167 together 27.5 →
37.7 → **42.5** (+15.0 [+6.6, +22.8]). All 1,000 questions: the base 30.8 (Qwen2.5-VL-72B 30.7); image labels −0.5,
long side 768 −0.1 - nothing else moved it. The sign conventions were settled with synthetic crops (a crop from the
right of a frame = a pan right: yaw +9.8°; from the left −8.5°; a centre zoom "moved forward"; from the top pitch
up). Lesson L6 holds here too: the fact carries what two photographs cannot show. VGGT for the same facts
(computed alone, read by the 4B): 36.5, +9.0 [+1.8, +16.8] - Motion (Cam.) 35.1 vs CUT3R's 50.0, Cam–Cam 37.6 vs
36.6: CUT3R's recurrent regime wins the consecutive-frame pairs, the two are level on wide baselines; CUT3R stays.
The 2B base: camera types 24.0 → 34.7 (+10.8 [+2.4, +19.2]), Motion (Cam.) 20.3 → 43.2 - the fact is worth about
as much at 2B as at 4B (28 of 167 answers with the fact still unparsable prose, so more in truth).
**v4 (10-16): 27.5 → 45.5, +18.0 [+9.6, +26.3]** - the question read before the fact is written: its direction
(second → first for nine questions, by sentence structure - a word-order rule sent 41 backwards and lost 10 points on
Cam–Cam) and the turn's owner (the "perspective ... changed to obtain" phrasing counts the view's turn, the opposite
word: +3.2). Cam–Cam 29.0 → 41.9, Motion (Cam.) 50.0; turn agreement 6 of 6. Lesson L9: a measurement is only as
good as the reading of what is asked - direction, frame and whose motion - and the reading is cheaper to fix than
the measurement.
Open: the same on run 3's plain model (does SFT keep it, as it did not for the VSI facts?), the word-phrased turn
questions (both encoders disagree with the human on 10 of 11 while the synthetic check and the coordinate-form
questions agree with our sign), and the Cam–Obj / Cam–Reg types with object positions from the registered points.

### Lessons so far (zero-shot, 4B)
- **L9 A measurement is only as good as the reading of the question** (I-52, 2026-10-16): the same CUT3R poses were
  worth +15 or +18 on MMSI's camera types depending on whether the fact followed the asked direction and named the
  turn as the question does; a wrong reading (word order for direction) cost 10 points. Read direction, frame and
  whose motion before writing the sentence; that reading is cheaper to fix than the encoder.
- **L8 Measurement facts are a zero-shot lever only** (2026-10-14). On the untrained model they were worth +3 to +9;
  on the model trained on 1.4 M VSI-domain samples every one of them is ≤ 0 (distances −0.2, the stack −1.0) because
  the trained model's own estimates are better than what we measure from CUT3R. What survived SFT: the frame budget
  (1 fps, cap 128: +2.2) and causal selection (−0.8, n.s.). CUT3R's VSI value after SFT is ~0 as tokens and as text.
- **L7 Count fact changes by the object triples they flip, not by questions** (I-48, 2026-10-05). VSI asks one
  (A, B, C) triple 6-9 times, so one wrong instance choice moves the direction score by ~5 points; an object
  detected several times is where it happens. On the 60-video set a change of the measurement frames alone moved
  the overall score by ~2 (61.8 vs 59.4-59.6): treat differences of that size in the fact-based types as noise
  until they hold on other videos.
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

### I-21 Thinking on at question time (4B) — **tested: incompatible with live, and no accuracy gain (2026-09-26)**
Latency cost; only if I-20 shows thinking itself is what helps.
- Zero-shot probe: thinking on (`<think>` open, 512-token budget, answer read after `</think>`; a budget that
  runs out counts as wrong — the honest cost), format instruction, 20 videos / 333 questions, against 54.7 on the
  same videos → **the 4B closed its reasoning on 11 of 333 questions** (mean 509 of 512 tokens): 96.7% empty
  answers, score ~3. Its reasoning is orderly ("locate the nightstand … the TV is on a dresser to the left …") and
  simply long. Whether it would be *right* given room is unanswered at this budget (a 2,048-token probe on 5
  videos is queued); the live question is answered: > 500 reasoning tokens per question is seconds on an H100
  (the 0.8B did not close within 96 tokens either), against a 1-second answer.
- **2,048-token probe** (5 videos, 101 questions): still only 23 closed (mean 1,808 tokens, median at the cap);
  on those 23 the thinking answer scores **67.8 vs 73.5 without thinking** on the same questions (relative distance
  88.9 = 88.9, counting 64.0 = 64.0, appearance order 25 vs 50, size 60 vs 67.5). No type where it closed got
  better. So zero-shot thinking at 4B neither fits the latency nor helps where it finishes — the "Thinking in
  Space" finding (linguistic CoT does not help) reproduced at this size. Thinking at question time is out; I-19's
  remaining form is *trained* short geometric reasoning, or none.

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
| 20 | I-21b thinking with a 2,048-token budget (4B, 5 videos): does reasoning help accuracy at all? | I-21, I-19 | done: **no** — 23/101 closed; on those, 67.8 vs 73.5 without thinking |
| 21 | I-33 direction facts on the 2B (60 videos) | I-33, I-22 | done: 43.5 vs 42.7, +0.9 [+0.2, +1.6]; the 2B follows the fact 50% of the time (4B 93%) |
| 22 | I-34 closest-object rankings and route-planning turns from the map (diagnostic, 4B, 60 videos) | I-34, I-28 | done: **negative** — rankings 40.7–66.7 never above the model; routes 15% coverage, 1/5 right |
| 23 | accumulation: CUT3R fed every 30th frame (~1 fps; maps from all of it) vs keyframes only, on the 57.8 configuration | I-02, I-05 | done: **56.9, −0.9 [−2.4, +0.8]** — directions +5.8 (83.4% measured), room −10.8 and distance −2.3 from a ~10% scale inflation |
| 26 | accumulation at ~3 fps (every 10th frame): does the scale drift grow? | I-05 | done (20 videos): **yes** — room-area ratio 1.03 → 1.23 → 1.39 for keyframes / 1 fps / 3 fps; overall 54.7 vs 59.1 (−4.3 [−6.5, −2.4]) |
| 28 | I-36 windowed CUT3R state (reset 40 / overlap 3, rigid chaining) at ~1 fps | I-36, I-05 | done: 56.9 (= no resets) — metric facts half recovered (room ratio 1.10, area MRA 58.7), direction gain lost (83.4 → 77.7) |
| 29 | I-36 with longer windows (reset 80 / overlap 6) | I-36 | done: **55.3** (−2.5 [−4.3, −0.9] vs keyframes; −1.6 vs 40/3) — scale similar (room ratio 1.12, distance ratio 0.93) but direction accuracy 63.1% (40/3: 77.7): re-aligning to drifted poses through more shared cameras inherits their orientation error |
| 24 | sparse view AND sparse map: 8 keyframes for the LLM, geometry on those 8 only, with / without routed facts | I-02 | done: hint-only **47.3** (vs 53.7 at 32 keyframes, −6.4 [−9.8, −2.6]); + facts 47.2 (−0.1 [−1.8, +1.7]) — facts built from 8 frames cover 30% of direction questions (77% at 32) and add nothing |
| 24b | long-video proxy done right: LLM sees 8 keyframes, maps from every 30th frame, routed facts | I-02 | done: **48.3** vs 47.3 without facts (+1.0 [−0.8, +2.8]); direction +8.0 where facts exist — but detection still ran on the 8 keyframes only (direction coverage 35%, distances 4%): the object map, not the geometry, is the bottleneck |
| 27a | detection decoupled from the LLM's keyframes: OWLv2 on the geometry frames (~1 fps, median 82 detection frames per video), online object map, LLM at 8 keyframes | I-02 | done: **48.8** — +0.5 [−1.3, +2.3] vs keyframe detection, +1.5 [−0.8, +3.9] vs no facts; direction coverage 35% → 87% (type +4.6); distance facts worse (view count no longer means separate views) |
| 27b | the same with the LLM at 32 keyframes — does dense detection lift the adopted 57.8? | I-02 | done: **55.4**, −2.4 [−4.0, −0.7] — direction coverage 77% → 86% but accuracy 76.4 → 75.6 (net 0); distances −8.0 (three near-duplicate "views"), room −10.8 (~1 fps drift) |
| 30 | dense detection + time-separated view counting (`--view-gap 30`) + 40/3 resets, 32 keyframes — the combined fix | I-02, I-05, I-36 | done: **55.1**, −2.7 [−4.6, −0.9] vs keyframes only; −0.3 vs the dense map without the fixes. Distances beat the model only at ≥ 5–6 separated views; direction accuracy 69.9 (resets' cost); room ratio 1.10 |
| 31 | I-37 v1 camera-height scale anchor, per-frame rescaling, ~1 fps | I-37, I-05 | done: **49.5**, −7.4 [−9.3, −5.4] vs no anchor — room MRA 14.3 (ratio 0.70), scale-free 21.7, direction facts 70.7 %: per-frame rescaling breaks the layout |
| 32 | I-37 v2: raw-stream capture (60 videos) + offline lab over floor-level / camera-height × window × support × gate, per-frame vs global application | I-37 | done: **no signal** — factor–ideal correlation −0.11…+0.13 for all 24 variants; best per-frame MRA 55.0 = none 55.5; global ≤ 48. Frame-count calibration instead: split-half MRA 63.8 (ceiling 65.2) |
| 33 | I-37 frame-count calibration applied at read time (`frame_count`, a 0.3224 / b −1.3781), 4B, ~1 fps, routed + directions | I-37 | done: **58.0**, +1.1 [+0.3, +2.1] vs no correction (room 56.2 → 65.7); +0.2 [−1.3, +1.9] vs keyframes (direction +5.8, abs −2.8, room −1.3). Map room MRA 66.0, ratio 0.97. The ~1 fps stream is at parity with keyframes |
| 34 | re-capture with the self-view points; `SelfViewScale` (relative / absolute, w5 / w15) in the lab next to the calibration | I-37 | done: **no signal** — self-view / world-view distance ratio 1.00 on every frame (corr −0.06…+0.01); the calibration's in-sample global MRA 66.0 confirmed in the same table |
| 35 | I-38 question reading by the model (JSON → template), 4B and 0.8B, 160 VSI + 28 paraphrased/Korean | I-38 | done — 4B: type 100 % / roles 99 % / identical routing 100 % on VSI, 100 % / 100 % on the paraphrases, 4.9 s per reading (M2); **0.8B: 54 % / 29 % and 43 % / 32 %** — a small reader needs the LoRA, the 4B does not |
| 36 | I-38 compact reply format vs full, 4B, accuracy and output tokens | I-38 | done: full 74 tokens / 4.9 s (100 % / 99 %); compact **29 tokens / 2.4 s**, 99 % / 98 % on VSI (routing 99 % identical), 93 % / 86 % on the paraphrases |
| 37 | I-40 type hints (all types), 4B, base + format hint, no geometry | I-40 | done: **53.1**, −0.5 [−2.6, +1.4] vs 53.7 — room +6.3, size +1.5, abs +1.1 / direction −6.5, rel dist −4.1, route −2.7: prior hints help, procedural hints hurt |
| 38 | I-41 describe the named objects first, 4B, base + format hint, no geometry | I-41, I-20 | done: **49.3**, −4.4 [−6.6, −2.2] vs 53.7 — closest −8.7, order −8.8, direction −7.2, count −5.6; the self-description family is closed |
| 39 | I-40 + measured facts + "use the measurement" hint on the **2B** — does a hint raise the copy rate (50 %)? | I-40, I-22 | planned (tomorrow) |
| 40 | I-38 compact prompt v2 (+ rel_distance and Korean easy-direction examples), 4B re-check | I-38 | done: **100 % / 99 % / routing 100 %** on VSI, **100 % / 100 %** paraphrases, 29 tokens, 2.6 s — compact is the default now |
| 41 | I-40b the three prior hints (room_size, size, abs_distance) on the streaming stack (~1 fps + facts + calibration, 58.0) | I-40, I-37 | done: **58.3**, +0.3 [−0.0, +0.6] — size +1.5, abs +0.6, room +0.5; +0.5 [−0.9, +2.1] vs keyframes. Adopted |
| 42 | I-42 "trust the measurements" hint, 2B + routed facts (43.5, copy rate 53 %) | I-42, I-22 | done: **43.6**, +0.1 [−0.6, +0.8]; copy rate 49 % — null. Fact use in small models needs training (I-39) |
| 43 | out-of-sample check of the adopted stacks on 60 *other* videos (seed 2): streaming stack (58.3 config) vs keyframes-only adopted (57.8 config) | I-37, I-40 | done: **55.3 vs 56.1, −0.8 [−1.9, +0.5]**; calibration holds (45 new videos: room ratio 1.10 → 0.96, MRA 54.2 → 63.3); direction −3.4 (facts 64.7 vs 69.6 %) — seed 1's direction gain was noise. **Pooled 105 videos: −0.1 [−1.1, +1.1] = parity** |
| 44 | I-44 visual-token gain: held-out loss (200 samples) then VSI (60 videos, keyframes) with the visual embeddings × {1, 2, 3, 5.6} and LayerNorm'd | I-44 | ready (`--visual-scale`), after I-45b |
| 45 | I-45 every frame at 1 fps in the LLM (`--budget-fps 1`), base + format hint, seed 1 | I-45 | done: **56.4 vs 53.7, +2.7 [−1.1, +6.4]**; order +11.5, rel dir +10.6, route +5.4; rel dist −4.7, room −4.2 |
| 46 | I-45b the adopted streaming stack (58.3) with every frame at 1 fps | I-45 | done: **59.5, +1.2 [−2.6, +4.8]** vs 58.3; facts degraded (direction facts 83 → 72 %, CUT3R at ~2 fps) |
| 47 | I-45c as I-45b with CUT3R fed only the 1 fps keyframes (~1 fps, the calibrated regime) | I-45 | done: **59.9, +1.6 [−2.3, +5.5]**; facts still degraded (direction 73 %) - not the ingest rate |
| 48 | I-45d every frame at 1 fps for the LLM, the measurements from the 58.3 run's 32 frames (`--facts-budget 32`) | I-45 | done: **61.8, +3.5 [+0.3, +6.7]** vs 58.3 - adopted; facts identical, the frames alone |
| 49 | I-46 vanilla Qwen (no CUT3R), the stack's prompt minus the facts, 32 frames and 1 fps | I-46 | done: 54.8 / 59.1 (1 fps spliced, see I-45's correction); CUT3R's share +3.5 / +2.7 |
| 50 | I-47 Utonia + ScanNet-20 probe on the CUT3R stream: counts and room area vs the model's answers (60 videos); 20 ARKitScenes ground-truth scans vs CUT3R | I-47 | done: counts MRA 0.26 / 0.35 vs 0.64, room 0.55 / 0.47 vs our fact 0.66 - not attached; the clean scan does not fix counting (0.20) |
| 51 | I-48 the 61.8 stack with causal selection (live): uncapped, the LLM capped at 64 and 128 frames | I-48 | done: **59.4**, −2.4 [−4.4, −0.4] vs 61.8 (direction facts 83 → 72 %, room facts 66 → 61); cap 128 +0.2, cap 64 −0.7 |
| 52 | I-48b live, cap 128, the measurement frames half a second off the 1 fps grid (extra CUT3R inputs, as the oracle's) | I-48 | done: **59.5**, −0.1 [−1.9, +1.8] vs cap 128 - no; the gap is 2-3 object triples flipping with the instance choice |
| 53 | I-49 data: the live cap-128 run with every object observation dumped (`--dump-objects`), two frame sets; instance rules replayed offline (`scripts/instance_rules_lab.py`) | I-49 | done: 59.6 (= I-48c); score_max + abstention 0.8 the only rule up on both sets (+6.6 / +6.1 expected direction accuracy) |
| 54 | I-49 validation: `--instance-rule score_max --direction-abstain 0.8` | I-49 | done: **60.6, +1.0 [+0.2, +2.1]**; rel direction +6.7, abs distance +1.6, nothing else changed |
| 55 | I-50 voice front-end, end of turn on eot-bench `ko` (377 real turns): the benchmark's protocol, our VAD per turn, a text detector on the reference words | I-50 | done: protocol reproduces the published Smart Turn numbers; **4.5 % of turns truncated at a 0.80 s mean wait** (0.9 % under 8 s); the text detector adds nothing |
| 56 | I-50 Qwen3-ASR-1.7B on the same turns; synthetic Korean questions through the whole chain to the reader | I-50 | done: CER 10.3 % vs a machine-made reference (7.5 % filtered, the model often the right side); demo 7/7 turns, reader 7/7 |
| 25 | I-35 encoder bake-off: VGGT window 8 behind the same maps (directions attached, metric facts logged) | I-35 | done: **CUT3R wins** — direction 76.4 vs 56.7 (72 vs 54 on common questions), room area MRA 65.8 vs 9.5 (scale-free 69.0 vs 16.2) |
| 14 | **Sensenova-only baseline on 4 nodes x 8 H100**: zero-shot VSI (full); S1 and S2 (one epoch each) real vs control, two seeds; plain SFT (no geometry); S2 joint (no S1); the 2B pair; S2 learning curve; gate | I-12, I-16, I-18, I-22 | four-day run from 09-26 (docs/SERVER_WEEKEND.md) |
