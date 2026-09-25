"""Training questions that can only be answered from the geometry (I-14 in docs/BRAINSTORM.md).

Why: S1 (projector only) and the small S2 both learned the *answer format* instead of reading the geometry —
the answer loss had a cheaper shortcut, and a control projector trained on shuffled geometry did just as well.
A question like "how far did the camera move between image 2 and image 6?" has no such shortcut: a content-free
signal cannot produce the number, and no amount of object-category prior helps, because the answer is a property
of *this* recording.

The labels are computed from the geometry encoder's own output (camera poses and point maps) — no human labels,
no teacher model. The model is therefore trained to *report what the geometry tokens say*, which is exactly the
alignment that was missing. (Errors in CUT3R become the target too; that is acceptable for alignment, and S2 on
real spatial data corrects the mapping to the world.)

Two properties matter as much as the questions themselves, both learned the hard way on M2:
  * **Answers must be spread out.** If most displacements were ~1 m the model would learn that prior instead of
    reading (the 905-sample S2 collapsed absolute distance onto "1.1"). `balanced_sample` picks frame pairs so
    the answers cover the available range evenly.
  * **Ambiguous cases must be dropped, not guessed.** A 25-degree turn is neither "left" nor "same"; a
    multiple-choice question whose two best options differ by 5 cm teaches noise. Every generator has an
    explicit reject band.

Everything here is pure geometry (numpy) so it can be tested against a known synthetic trajectory.
"""

from __future__ import annotations

import math
import random

import numpy as np

# Reject bands — a question is only emitted when the answer is unambiguous.
MIN_DISPLACEMENT_M = 0.30     # below this, "how far did the camera move" is noise
TURN_SAME_DEG = 15.0          # |yaw| below this is "the same way"
TURN_CLEAR_DEG = 40.0         # |yaw| above this is a clear left/right; in between -> drop
MIN_CHOICE_MARGIN_M = 0.50    # the best and second-best option must differ by this much
# A hand-held scan usually never looks at the ceiling, so the reconstruction's height is a floor-to-highest-point
# span, not a room height (M2: 1.7 m on real ScanNet rooms). Outside this band we decline to label it rather than
# teach a wrong number - the same "precise or absent" rule the prompting experiments arrived at (L1/L5).
ROOM_HEIGHT_BAND_M = (2.0, 5.0)


def camera_frame(c2ws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-to-world matrices [N,4,4] -> (positions [N,3], yaw degrees [N], world->map rotation [3,3]).

    The map frame is gravity-aligned the same way as `SceneMap`: "up" is the mean of the cameras' up axes
    (OpenCV cameras look along +z with +y down), so x/y are floor-plan axes and z is height. Yaw is the heading
    of the camera's forward axis in that floor plane, measured counter-clockwise, so a *decreasing* yaw is a
    turn to the right.
    """
    c2ws = np.asarray(c2ws, dtype=np.float64)
    up = -c2ws[:, :3, 1].mean(0)
    up /= np.linalg.norm(up) + 1e-9
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ax = np.cross(up, ref)
    ax /= np.linalg.norm(ax) + 1e-9
    ay = np.cross(up, ax)
    R = np.stack([ax, ay, up])
    pos = c2ws[:, :3, 3] @ R.T
    fwd = c2ws[:, :3, 2] @ R.T                      # camera forward in the map frame
    yaw = np.degrees(np.arctan2(fwd[:, 1], fwd[:, 0]))
    return pos, yaw, R


def yaw_delta(yaw_from: float, yaw_to: float) -> float:
    """Signed turn in degrees, wrapped to (-180, 180]. Positive = counter-clockwise = to the left."""
    return (yaw_to - yaw_from + 180.0) % 360.0 - 180.0


def balanced_sample(pairs: list[tuple[int, int, float]], k: int, bins: int = 6,
                    rng: random.Random | None = None) -> list[tuple[int, int, float]]:
    """Pick k (i, j, value) triples whose values cover the range evenly.

    Without this the sampler follows the trajectory's own distribution — mostly short hops — and the model can
    score well by always answering the mode. Values are bucketed into `bins` equal-width bins over the observed
    range and drawn round-robin from the non-empty bins.
    """
    rng = rng or random.Random(0)
    if not pairs or k <= 0:
        return []
    vals = [v for _, _, v in pairs]
    lo, hi = min(vals), max(vals)
    width = (hi - lo) / bins if hi > lo else 1.0
    buckets: dict[int, list] = {}
    for p in pairs:
        b = min(bins - 1, int((p[2] - lo) / width)) if hi > lo else 0
        buckets.setdefault(b, []).append(p)
    for b in buckets:
        rng.shuffle(buckets[b])
    out: list[tuple[int, int, float]] = []
    order = sorted(buckets)
    while len(out) < k and any(buckets[b] for b in order):
        for b in order:
            if buckets[b] and len(out) < k:
                out.append(buckets[b].pop())
    return out


def _label(i: int) -> str:
    return f"image {i + 1}"


def displacement_questions(pos: np.ndarray, k: int, rng: random.Random) -> list[dict]:
    """How far apart were the camera positions when two images were taken."""
    n = len(pos)
    cand = [(i, j, float(np.linalg.norm(pos[j] - pos[i])))
            for i in range(n) for j in range(i + 1, n)]
    cand = [c for c in cand if c[2] >= MIN_DISPLACEMENT_M]
    out = []
    for i, j, d in balanced_sample(cand, k, rng=rng):
        out.append({
            "kind": "camera_displacement",
            "question": f"How far apart were the camera positions when {_label(i)} and {_label(j)} were taken, "
                        "in meters?",
            "answer": f"{d:.1f}",
            "value": d,
        })
    return out


def turn_questions(pos: np.ndarray, yaw: np.ndarray, k: int, rng: random.Random) -> list[dict]:
    """Which way the camera turned between two images (left / right / the same way)."""
    n = len(yaw)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            d = yaw_delta(yaw[i], yaw[j])
            if abs(d) < TURN_SAME_DEG:
                cand.append((i, j, d, "the same way"))
            elif abs(d) > TURN_CLEAR_DEG:
                cand.append((i, j, d, "left" if d > 0 else "right"))
            # the band in between is genuinely ambiguous -> dropped
    out = []
    by_answer: dict[str, list] = {}
    for c in cand:
        by_answer.setdefault(c[3], []).append(c)
    for lst in by_answer.values():
        rng.shuffle(lst)
    # equal numbers of left / right / same, so the answer itself carries no information
    per = max(1, k // max(1, len(by_answer)))
    for ans, lst in by_answer.items():
        for i, j, d, _ in lst[:per]:
            out.append({
                "kind": "camera_turn",
                "question": f"Going from {_label(i)} to {_label(j)}, did the camera turn left, turn right, or "
                            "keep facing the same way?",
                "answer": ans,
                "value": float(d),
            })
    rng.shuffle(out)
    return out[:k]


def closest_image_questions(pos: np.ndarray, k: int, rng: random.Random, n_options: int = 4) -> list[dict]:
    """Which of several images was taken closest to a given one — needs every camera position at once."""
    n = len(pos)
    if n < n_options + 1:
        return []
    out = []
    anchors = list(range(n))
    rng.shuffle(anchors)
    for a in anchors:
        others = [i for i in range(n) if i != a]
        rng.shuffle(others)
        opts = others[:n_options]
        d = sorted((float(np.linalg.norm(pos[i] - pos[a])), i) for i in opts)
        if d[1][0] - d[0][0] < MIN_CHOICE_MARGIN_M:      # too close to call
            continue
        opts_sorted = sorted(opts)
        out.append({
            "kind": "closest_image",
            "question": f"Of {', '.join(_label(i) for i in opts_sorted)}, which one was taken closest to where "
                        f"{_label(a)} was taken?",
            "answer": _label(d[0][1]),
            "value": d[0][0],
        })
        if len(out) >= k:
            break
    return out


def path_length_question(pos: np.ndarray) -> list[dict]:
    if len(pos) < 2:
        return []
    total = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    if total < MIN_DISPLACEMENT_M:
        return []
    return [{
        "kind": "path_length",
        "question": "Following the images in order, about how far did the camera travel in total, in meters?",
        "answer": f"{total:.1f}",
        "value": total,
    }]


def room_questions(facts: dict) -> list[dict]:
    """Room measurements, from the same reconstruction (scene-specific, so no prior can answer them)."""
    if not facts:
        return []
    out = [{
        "kind": "room_area",
        "question": "About how large is this room, in square meters?",
        "answer": f"{facts['area_m2']:.0f}",
        "value": facts["area_m2"],
    }]
    if ROOM_HEIGHT_BAND_M[0] <= facts.get("height_m", 0) <= ROOM_HEIGHT_BAND_M[1]:
        out.append({
            "kind": "room_height",
            "question": "About how high is the ceiling in this room, in meters?",
            "answer": f"{facts['height_m']:.1f}",
            "value": facts["height_m"],
        })
    return out


def build_questions(c2ws: np.ndarray, facts: dict | None = None, per_kind: int = 2,
                    seed: int = 0) -> list[dict]:
    """All geometry-only questions for one image sequence, from its camera poses (+ optional room facts)."""
    rng = random.Random(seed)
    pos, yaw, _ = camera_frame(c2ws)
    return (displacement_questions(pos, per_kind, rng)
            + turn_questions(pos, yaw, per_kind, rng)
            + closest_image_questions(pos, per_kind, rng)
            + path_length_question(pos)
            + room_questions(facts or {}))


def to_record(rec_id: str, images: list[str], qa: dict, source: str = "geometry_qa") -> dict:
    """One question -> a training record in the SenseNova format the trainer already reads."""
    prefix = "".join("<image>" for _ in images)
    return {
        "id": rec_id,
        "image": list(images),
        "conversations": [
            {"from": "human", "value": prefix + qa["question"]},
            {"from": "gpt", "value": qa["answer"]},
        ],
        "data_source": source,
        "kind": qa["kind"],
    }
