"""Camera motion between views as a sentence, from the geometry encoder's poses (MMSI-Bench probe, 2026-10-15).

MMSI-Bench's hardest types ask how the camera moved between two photographs (camera-camera, camera motion); open
models answer at chance there. A two-view reconstruction returns exactly that - the relative pose - so this turns
CUT3R's (or VGGT's) camera-to-world matrices of two images into words the question router can attach, as the VSI
room / distance / direction facts were (docs/MMSI_BENCH_ANALYSIS_20261015.md §3C).

Conventions: OpenCV camera axes (+x right, +y down, +z forward); the pose of view b is expressed in view a's frame.
The encoder's metric scale on two wide-baseline photos is not calibrated, so the directions and the turn angle are
the information, the distance only a rough size.
"""

from __future__ import annotations

import math

import numpy as np

PREFIX = "Measured from a 3D reconstruction of the two images: from the first view to the second, the camera "


def relative_pose(c2w_a, c2w_b) -> dict:
    """Camera b in camera a's frame: translation (metres, a's axes) and the yaw / pitch of b's forward axis.
    yaw > 0 = turned right, pitch > 0 = tilted up."""
    a, b = np.asarray(c2w_a, dtype=np.float64), np.asarray(c2w_b, dtype=np.float64)
    rel = np.linalg.inv(a) @ b
    t, f = rel[:3, 3], rel[:3, 2]
    f = f / (np.linalg.norm(f) + 1e-12)
    yaw = math.degrees(math.atan2(f[0], f[2]))
    pitch = math.degrees(-math.asin(max(-1.0, min(1.0, float(f[1])))))
    return {"right": float(t[0]), "down": float(t[1]), "forward": float(t[2]),
            "distance": float(np.linalg.norm(t)), "yaw_deg": yaw, "pitch_deg": pitch}


def camera_motion_text(rel: dict, min_share: float = 0.25, min_move: float = 0.05, name_move: float = 0.3,
                       min_turn: float = 5.0, axes_form: bool = False) -> str:
    """The sentence: the directions whose component is at least `min_share` of the distance - named only when the
    distance is at least `name_move` (below it the distance alone: small two-view translations are noise), the turn
    and the tilt when at least `min_turn` degrees (MMSI's first-person pairs turn by ~10 degrees). `axes_form` adds
    the same motion in the vocabulary of MMSI's coordinate-system questions (+Y up, -Z forward, right-handed)."""
    d = rel["distance"]
    words = []
    for key, pos, neg in (("right", "to the right", "to the left"), ("forward", "forward", "backward"),
                          ("down", "down", "up")):
        v = rel[key]
        if d >= name_move and abs(v) >= min_share * d:
            words.append(pos if v > 0 else neg)
    if d < min_move:
        move = "stayed in place"
    elif words:
        move = f"moved {' and '.join(words)} (about {d:.1f} m)"
    else:
        move = f"moved about {d:.1f} m"
    yaw, pitch = rel["yaw_deg"], rel["pitch_deg"]
    turn = (f"turned {'right' if yaw > 0 else 'left'} by about {abs(yaw):.0f} degrees" if abs(yaw) >= min_turn
            else "kept its heading")
    tilt = (f" and tilted {'up' if pitch > 0 else 'down'} by about {abs(pitch):.0f} degrees"
            if abs(pitch) >= min_turn else "")
    text = f"{PREFIX}{move} and {turn}{tilt}."
    if axes_form:
        text += " " + axes_sentence(rel, min_share, name_move, min_turn)
    return text + "\n"


def axes_sentence(rel: dict, min_share: float = 0.25, name_move: float = 0.3, min_turn: float = 5.0) -> str:
    """The motion in a right-handed camera frame with +Y up and -Z forward (MMSI's coordinate-system questions):
    right = +X, forward = -Z, up = +Y; a turn to the right is a negative angle about Y, a tilt up a positive angle
    about X."""
    d = rel["distance"]
    moves = []
    for key, pos, neg in (("right", "positive X", "negative X"), ("forward", "negative Z", "positive Z"),
                          ("down", "negative Y", "positive Y")):
        v = rel[key]
        if d >= name_move and abs(v) >= min_share * d:
            moves.append(pos if v > 0 else neg)
    rots = []
    if abs(rel["yaw_deg"]) >= min_turn:
        rots.append(f"a {'negative' if rel['yaw_deg'] > 0 else 'positive'} angle around the Y axis")
    if abs(rel["pitch_deg"]) >= min_turn:
        rots.append(f"a {'positive' if rel['pitch_deg'] > 0 else 'negative'} angle around the X axis")
    parts = []
    if moves:
        parts.append("a move in the " + " and ".join(moves) + " direction" + ("s" if len(moves) > 1 else ""))
    if rots:
        parts.append("a rotation by " + " and ".join(rots))
    body = " and ".join(parts) if parts else "no move and no rotation"
    return f"In a right-handed camera coordinate system with +Y up and -Z forward, that is {body}."


def camera_motion_from_images(geom, images, image_size: int, unit: int, device) -> dict | None:
    """Run the geometry stream over the images in order (a fresh stream) and return relative_pose(first, last);
    None when the encoder gives no poses. `images`: PIL images or [H, W, 3] uint8 arrays."""
    import torch

    from ..data.vision import geometry_frame

    had = getattr(geom, "decode_points", None)
    if had is not None:
        geom.decode_points = True               # the pose head runs only with the point decode
    geom.reset()
    poses = []
    try:
        with torch.no_grad():
            for img in images:
                arr = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img)
                raw = torch.from_numpy(np.ascontiguousarray(arr))
                out = geom.ingest(geometry_frame(raw, image_size, unit).unsqueeze(0).to(device))
                c2w = (getattr(out, "extra", None) or {}).get("c2w")
                if c2w is None:
                    return None
                poses.append(c2w[0].float().cpu().numpy())
    finally:
        geom.reset()
        if had is not None:
            geom.decode_points = had
    return relative_pose(poses[0], poses[-1]) if len(poses) >= 2 else None
