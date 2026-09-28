"""Metric scale anchor for a recurrent encoder whose scale drifts with stream length (I-05 / I-37).

Measured 2026-09-27: CUT3R fed only 32 keyframes measures rooms at 1.03x their true area; fed every 30th frame
1.23x; every 10th frame 1.39x. The relative layout keeps improving with more frames, only the metric scale
inflates - so the fix must rescale without re-anchoring orientation (state resets cost the direction facts).

A scale-carrying quantity is measured per frame from the frame's own points and pose, in the (possibly drifted)
world units, and compared with its value over the first frames:

  camera_height  the camera's height above the floor seen in the frame. Constant for a camera carried at a steady
                 height; fooled when the operator crouches or stands up.
  floor_level    the depth of the floor below the world origin (the first camera). The floor does not move, and the
                 origin is where a uniform inflation of the state's frame is anchored, so the depth grows exactly
                 with the scale. Needs the first camera well above the floor (`min_ref`).

Both need a floor in view, and a frame that shows only walls or a table top puts the low percentile above the
floor: the camera looks lower than it is and the anchor inflates the output (the first smoke video, 2026-09-28:
5.25 -> 6.93 m2 against a 4.2 m2 truth). Two guards:
  support  the fraction of the frame's confident points that count as floor evidence must reach this value
           (ungated: points within `band` x camera height above the floor percentile - a sparse tail of wall
           bottoms or noise is not a floor);
  gate     once the reference exists, the floor is searched only within +-gate (relative) of where the current
           scale estimate predicts it - a table top 0.7 m above the floor is outside a 25% gate and the frame is
           skipped, while the drift, which is slow, stays inside.
The correction is 1 / clamp(median over `window` recent frames / reference); points and the camera position are
divided by it about the origin. Pure numpy; the encoder adapter applies it. Parameters are chosen with
scripts/scale_anchor_lab.py, which replays the variants on captured raw streams against VSI's room areas.
"""

from __future__ import annotations

from collections import deque

import numpy as np

MODES = ("camera_height", "floor_level")


class ScaleAnchor:
    def __init__(self, mode: str = "camera_height", ref_frames: int = 10, window: int = 5, floor_pct: float = 2.0,
                 conf_quantile: float = 0.3, support: float = 0.0, band: float = 0.08, gate: float = 0.0,
                 min_ref: float = 0.3, ref_quantile: float = 0.5, clamp: tuple[float, float] = (0.5, 2.0)) -> None:
        """ref_quantile: which quantile of the first frames' measurements is the reference. 0.5 = median; higher
        prefers the frames that saw deeper, i.e. the true floor rather than a table top, when the scan starts
        pointing at furniture (the floor is the lowest surface; only noise is lower, and the support test drops
        sparse tails)."""
        if mode not in MODES:
            raise ValueError(f"unknown scale anchor mode {mode!r} (one of {MODES})")
        self.mode = mode
        self.ref_frames, self.window = int(ref_frames), int(window)
        self.floor_pct, self.conf_quantile = float(floor_pct), float(conf_quantile)
        self.support, self.band, self.gate, self.min_ref = float(support), float(band), float(gate), float(min_ref)
        self.ref_quantile = float(ref_quantile)
        self.clamp = (float(clamp[0]), float(clamp[1]))
        self.reset()

    def reset(self) -> None:
        self._downs: list[np.ndarray] = []
        self._ref: list[float] = []
        self._recent: deque = deque(maxlen=self.window)
        self.reference: float | None = None
        self.disabled = False      # the reference is too small to divide by (first camera at floor level)
        self.factor = 1.0          # the last correction applied (1/ratio)
        self.frames = 0
        self.skipped = 0           # frames without a floor in view (support / gate)
        self.history: list[float] = []   # the factor returned for every frame after the reference exists

    def up(self) -> np.ndarray:
        u = -np.mean(np.stack(self._downs), axis=0)
        return u / (np.linalg.norm(u) + 1e-9)

    def ratio_estimate(self) -> float:
        """The current scale estimate (recent median / reference), 1.0 before anything was measured."""
        if self.reference is None or not self._recent:
            return 1.0
        return float(np.median(self._recent)) / self.reference

    def floor(self, z: np.ndarray, cam_z: float) -> tuple[float, float]:
        """(floor level, support) from the heights `z` of a frame's confident points along up.

        Ungated: the low percentile of all points, and the fraction of points within `band` x (camera height above
        that level) of it. Gated (reference known, gate > 0): the low percentile of the points within +-gate of the
        floor level the current scale estimate predicts, and the fraction of all points in that band."""
        if self.gate > 0 and self.reference is not None and not self.disabled:
            est = self.ratio_estimate() * self.reference
            pred = -est if self.mode == "floor_level" else cam_z - est
            half = self.gate * est
            sel = (z >= pred - half) & (z <= pred + half)
            if sel.sum() < 50:
                return pred, 0.0
            return float(np.percentile(z[sel], self.floor_pct)), float(sel.mean())
        level = float(np.percentile(z, self.floor_pct))
        width = self.band * max(0.3, cam_z - level)
        return level, float(np.mean((z >= level) & (z < level + width)))

    def measure(self, c2w: np.ndarray, pts: np.ndarray, conf: np.ndarray | None) -> float | None:
        """This frame's scale-carrying quantity in its own (drifted) units; None when the frame cannot be used."""
        self._downs.append(np.asarray(c2w[:3, 1], dtype=np.float64))
        up = self.up()
        p = pts.reshape(-1, 3)
        ok = np.isfinite(p).all(1)
        if conf is not None:
            c = conf.reshape(-1)
            fin = np.isfinite(c)
            if fin.any():
                ok &= c >= np.quantile(c[fin], self.conf_quantile)
        p = p[ok]
        if len(p) < 200:
            return None
        cam_z = float(np.asarray(c2w[:3, 3], dtype=np.float64) @ up)
        level, sup = self.floor(p @ up, cam_z)
        if sup < self.support or sup <= 0.0:
            self.skipped += 1
            return None
        v = cam_z - level if self.mode == "camera_height" else -level
        return v if v > 0 else None

    def update(self, c2w: np.ndarray, pts: np.ndarray, conf: np.ndarray | None) -> float:
        """Observe one frame (raw encoder output) and return the factor to multiply its points and camera position
        by. 1.0 while the reference is being built; the last factor is held on frames that cannot be measured."""
        self.frames += 1
        v = self.measure(c2w, pts, conf)
        if v is None:
            return self.factor
        if self.reference is None:
            self._ref.append(v)
            if len(self._ref) >= self.ref_frames:
                self.reference = float(np.quantile(self._ref, self.ref_quantile))
                self.disabled = self.reference < self.min_ref
            return 1.0
        if self.disabled:
            return 1.0
        self._recent.append(v)
        ratio = float(np.median(self._recent)) / self.reference
        ratio = min(max(ratio, self.clamp[0]), self.clamp[1])
        self.factor = 1.0 / ratio
        self.history.append(self.factor)
        return self.factor

    @staticmethod
    def apply(factor: float, c2w: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scale a camera pose and points about the world origin (rotation untouched)."""
        out = np.array(c2w, dtype=np.float64, copy=True)
        out[:3, 3] *= factor
        return out, pts * factor


class CameraHeightAnchor(ScaleAnchor):
    """The first version (I-37, chain 13 on 2026-09-28): camera height, no support test, no reference floor."""

    def __init__(self, **kw) -> None:
        kw.setdefault("mode", "camera_height")
        kw.setdefault("min_ref", 0.0)
        super().__init__(**kw)


def build_anchor(spec) -> ScaleAnchor | None:
    """From the geometry config's `scale_anchor` option: None/"none" -> no anchor; a mode name -> that mode with
    the defaults; a dict -> ScaleAnchor keyword arguments (mode, window, support, ...)."""
    if spec in (None, "", "none", False):
        return None
    if isinstance(spec, str):
        return ScaleAnchor(mode=spec)
    if isinstance(spec, dict):
        return ScaleAnchor(**spec)
    raise ValueError(f"scale_anchor must be a mode name or a dict of ScaleAnchor arguments, got {spec!r}")
