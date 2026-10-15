# MMSI-Bench — what it measures, where our model will fail, what to change (2026-10-15)

MMSI-Bench (ICLR 2026, arXiv 2505.23764) is the second accuracy target next to VSI-Bench. It is a different animal:
two photographs taken from far-apart viewpoints, often outdoors or egocentric, and the question is about what moved
between them. VSI-Bench is one continuous indoor video. What carried us to 72.7 on VSI (frames at 1 fps, causal
selection) does not transfer by itself; data does.

## 1. The benchmark

- **1,000 four-option questions** (answers A-D balanced: 265 / 250 / 255 / 230), 1,990 images, 2.55 images a question;
  the ten basic types use **two images**, multi-step reasoning (MSR) up to ten. Difficulty easy / medium / hard
  334 / 333 / 333.
- **Types and counts:** MSR 198 · Obj–Obj 94 · Cam–Cam 93 · Cam–Obj 86 · Obj–Reg 85 · Cam–Reg 83 · Reg–Reg 81 ·
  Motion (Obj) 76 · Motion (Cam) 74 · Appearance 66 · Measurement 64.
- **Sources:** ScanNet, Matterport3D (indoor), nuScenes, Waymo (driving), Ego4D (egocentric), AgiBot-World (robot),
  DTU (objects), DAVIS. Our training data is indoor only.
- **Scores:** human 97.2, chance 25. Open 7-8B models sit at chance (Qwen2.5-VL-7B 25.9, InternVL3-8B 25.7); the
  exception is SenseNova-SI-1.2-InternVL3-8B at 42.6 (their recipe: 8 M samples, with perspective-taking data as the
  credited ingredient). Proprietary: Gemini-3-pro 49.2, GPT-5 41.9, o3 41.0.
- **Where everyone fails:** camera motion (GPT-5 32.4), appearance - the same object across views (36.4), Obj–Obj
  (35.1), Reg–Reg (32.1). Easiest: measurement (60.9), Cam–Reg (51.8-62.6). The largest error share is
  "overlap-matching and scene-reconstruction errors", then situation-transformation (perspective) errors.
- **What did not help in the paper:** zero-shot CoT (helped GPT-4o only, hurt the others), correspondence-line visual
  prompts (mixed), spatial fine-tuning at modest scale (Spatial-MLLM 27.7 vs base 26.5). The authors' conclusion:
  data quality and diversity, not model size.
- **Evaluation path (lmms-eval `mmsi_bench`):** post-prompt "Answer with the option's letter from the given choices
  directly. Enclose the option's letter within ``", 2,048 new tokens, temperature 0; the extractor takes backticks,
  then braces, then a bare letter - our one-letter answers score. Our wrapper puts the images first and the question
  after (`_question_with_placeholders`), the form of the multi-view SPAR training records.

## 2. Why 72.7 on VSI does not carry over

| | VSI-Bench | MMSI-Bench |
|---|---|---|
| input | one room, continuous video; adjacent frames overlap, correspondences come free | two shots with a large viewpoint jump; finding the overlap is the task |
| asked | the room's layout | how the camera moved between the shots, is it the same object, what is where if I stand there |
| domain | indoor scans | indoor + driving + egocentric + robot |
| our training | a near-perfect match (Sensenova 58 %, vlm3r, ScanQA / SQA3D) | SPAR's multi-view types (15 % of the mix) only; **no camera-motion supervision**, nothing on appearance, regions or outdoors |

Expectation for run 3's plain model: a few points over the base on Obj–Obj, Cam–Obj and measurement; chance on
camera motion, appearance and the outdoor questions; **28-35 overall**. The per-type scores of the current
measurement (lmms-eval prints all eleven) will say.

## 3. What to change, in order

**A. Data - the one lever the literature agrees on**
1. Two registered sets we never used, on the server already: `vsti_camera` (66k: "how far did the camera move
   between frame i and j of 32", metres) and `vsti_camera_obj` (48k: camera-to-object distance in frame k). The only
   camera-motion supervision we hold. Video form and numeric answers, so a partial match to MMSI's two-image
   direction questions - but free to add to run 4's mix.
2. Upweight SPAR's multi-view types (`name%N` in the mix): `position_matching` (the same object across views -
   MMSI's largest error source), `spatial_imagination_*_mv` (perspective taking), `obj_spatial_relation_*_mv`.
3. Perspective-taking data at scale - what took SenseNova from 25.7 to 42.6. Public candidates to check (existence,
   license): the full SPAR-7M (we hold VG-LLM's 234k subset), MindCube's training split, SenseNova-SI-8M if released.
4. Outdoor / egocentric domains come only with 3; without it those questions stay at chance.

**B. Input settings - evaluation only, no training**
- Resolution: MMSI's photos are large (driving 1600×900, Ego4D 1080p); we shrink to long side 512. Appearance and
  measurement need detail - evaluate at 768 and 1024 (a mismatch with training, but one run each).
- Image labels ("Image 1:", "Image 2:"): the questions refer to images by order (the I-04 idea, cheap).
- No CoT / thinking (the paper's finding, and ours on VSI).

**C. Geometry - worth re-testing here**
- On VSI the trained model needed no CUT3R: a continuous video gives it the correspondences. MMSI's dominant failure
  is the correspondence and the camera-pose change between two shots - exactly what a two-view reconstruction returns.
- Probe on the trained plain model: CUT3R (or VGGT - better suited to two wide-baseline views than a recurrent video
  encoder; the I-35 bake-off was on video) on the two images → relative pose (translation direction, rotation) as a
  sentence for the Cam–Cam and camera-motion questions (167); object distances from the registered points for
  measurement / Obj–Obj. The routed-facts machinery exists; a new fact type and MMSI's phrasing are the work.
- For the live demo this is a strength in any case: a live system reads its own camera motion from CUT3R's
  trajectory instead of inferring it from pixels.

**D. Caveats**
- Contamination: MMSI's ScanNet questions may come from scenes in our training sets (SPAR, vlm3r, ScanQA, SQA3D) -
  scene-level exposure, the same for the in-house system; read the ScanNet-sourced types with that in mind.
- Model size is not the lever (the authors' finding); 4B stays.

## 4. Plan

1. Per-type scores of the current measurement and of the base (which model and path - to confirm).
2. Three evaluation-only runs on the plain model: resolution 768 / 1024, image labels, CUT3R relative-pose facts on
   the camera types.
3. Run 4's mix: `live_vsi` + `vsti_camera` + `vsti_camera_obj`, SPAR multi-view upweighted; MMSI per type before
   and after.
4. Perspective-taking data (A.3): availability and license - a user step, like VSI-590K.

Sources: [arXiv 2505.23764](https://arxiv.org/abs/2505.23764) · [the leaderboard](https://github.com/InternRobotics/MMSI-Bench) ·
[project page](https://runsenxu.com/projects/MMSI_Bench/) · [SenseNova-SI (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Cai_Scaling_Spatial_Intelligence_with_Multimodal_Foundation_Models_CVPR_2026_paper.pdf) ·
[SPAR-7M](https://arxiv.org/abs/2503.22976v1) · the HF dataset statistics of `RunsenXu/MMSI-Bench`.
