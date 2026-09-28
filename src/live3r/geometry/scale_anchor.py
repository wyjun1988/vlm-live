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

MODES = ("camera_height", "floor_level")          # ScaleAnchor; build_anchor also knows frame_count and self_view


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

    def global_factor(self) -> float:
        """One factor for the whole stream so far: the geometric mean of the per-frame factors (1.0 before any).
        This is the correction applied at read time to the metric facts - never to the geometry itself: frames
        rescaled by different factors shift relative to each other and break the layout (I-37 v1, 2026-09-28)."""
        if not self.history:
            return 1.0
        return float(np.exp(np.mean(np.log(self.history))))

    @staticmethod
    def apply(factor: float, c2w: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scale a camera pose and points about the world origin (rotation untouched)."""
        out = np.array(c2w, dtype=np.float64, copy=True)
        out[:3, 3] *= factor
        return out, pts * factor


class FrameCountScale:
    """The drift as a function of the frames ingested, calibrated once (scripts/scale_calibration.py, 2026-09-28):
    log(measured / true room area) = a * log(n) + b over the 60 evaluation videos at ~1 fps (n = 51..250,
    corr +0.44). One factor per stream, 1 / sqrt(exp(a * log n + b)), applied to the metric facts at read time;
    split-half room-area MRA 55.5 -> 63.8 (ceiling 65.2). No per-frame signal is read at all - this is the
    calibration the floor / camera-height anchors failed to beat (factor-ideal correlation ~0). Same interface as
    ScaleAnchor so the encoder adapter and the lab treat it alike."""

    def __init__(self, a: float = 0.3224, b: float = -1.3781, min_frames: int = 1) -> None:
        self.a, self.b, self.min_frames = float(a), float(b), int(min_frames)
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.skipped = 0
        self.factor = 1.0
        self.history: list[float] = []

    def ratio(self, n: int) -> float:
        """The predicted measured/true area ratio after n frames."""
        return float(np.exp(self.a * np.log(max(1, n)) + self.b))

    def update(self, c2w=None, pts=None, conf=None) -> float:
        self.frames += 1
        if self.frames >= self.min_frames:
            self.factor = 1.0 / np.sqrt(self.ratio(self.frames))
            self.history.append(self.factor)
        return self.factor

    def global_factor(self) -> float:
        return self.factor

    apply = staticmethod(ScaleAnchor.apply)


class SelfViewScale:
    """The state's scale read from CUT3R itself: the head predicts each frame's points twice - in the camera's own
    frame (`pts3d_in_self_view`, a monocular, metric-trained estimate) and in the world frame that the recurrent
    state defines. Brought into the same frame through the predicted pose, their per-pixel distance ratio is the
    state's scale relative to the frame's own; its median over confident pixels is one number per frame that needs
    no floor and no assumption about the camera. relative=True measures it against the first `ref_frames` (the
    early stream is metric to ~3%), else the monocular scale is trusted outright. Per-frame values are only
    aggregated (`global_factor`), never applied to the points."""

    uses_self_view = True

    def __init__(self, ref_frames: int = 10, window: int = 5, conf_quantile: float = 0.3, min_depth: float = 0.2,
                 relative: bool = True, clamp: tuple[float, float] = (0.5, 2.0)) -> None:
        self.ref_frames, self.window, self.conf_quantile = int(ref_frames), int(window), float(conf_quantile)
        self.min_depth, self.relative, self.clamp = float(min_depth), bool(relative), (float(clamp[0]), float(clamp[1]))
        self.reset()

    def reset(self) -> None:
        self.frames = self.skipped = 0
        self.factor = 1.0
        self.reference: float | None = None
        self._ref: list[float] = []
        self._recent: deque = deque(maxlen=self.window)
        self.history: list[float] = []
        self.ratios: list[float] = []      # the raw per-frame state/self scale, for inspection

    def measure(self, c2w, pts, conf, pts_self, conf_self) -> float | None:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        q = np.asarray(pts_self, dtype=np.float64).reshape(-1, 3)
        if len(p) != len(q):
            return None
        ok = np.isfinite(p).all(1) & np.isfinite(q).all(1)
        c = conf_self if conf_self is not None else conf
        if c is not None:
            c = np.asarray(c, dtype=np.float64).reshape(-1)
            fin = np.isfinite(c)
            if fin.any():
                ok &= c >= np.quantile(c[fin], self.conf_quantile)
        m = np.asarray(c2w, dtype=np.float64)
        pc = (p - m[:3, 3]) @ m[:3, :3]                       # world -> this camera's frame
        dw, ds = np.linalg.norm(pc, axis=1), np.linalg.norm(q, axis=1)
        ok &= (dw > self.min_depth) & (ds > self.min_depth)
        if ok.sum() < 200:
            return None
        return float(np.median(dw[ok] / ds[ok]))

    def update(self, c2w, pts, conf, pts_self=None, conf_self=None) -> float:
        self.frames += 1
        s = self.measure(c2w, pts, conf, pts_self, conf_self) if pts_self is not None else None
        if s is None or s <= 0:
            self.skipped += 1
            return self.factor
        self.ratios.append(s)
        if self.relative:
            if self.reference is None:
                self._ref.append(s)
                if len(self._ref) >= self.ref_frames:
                    self.reference = float(np.median(self._ref))
                return 1.0
            s = s / self.reference
        self._recent.append(s)
        r = min(max(float(np.median(self._recent)), self.clamp[0]), self.clamp[1])
        self.factor = 1.0 / r
        self.history.append(self.factor)
        return self.factor

    def global_factor(self) -> float:
        return float(np.exp(np.mean(np.log(self.history)))) if self.history else 1.0

    apply = staticmethod(ScaleAnchor.apply)


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
        spec = {"mode": spec}
    if isinstance(spec, dict):
        kw = dict(spec)
        mode = kw.pop("mode", "camera_height")
        if mode == "frame_count":
            return FrameCountScale(**kw)
        if mode == "self_view":
            return SelfViewScale(**kw)
        return ScaleAnchor(mode=mode, **kw)
    raise ValueError(f"scale_anchor must be a mode name or a dict of arguments, got {spec!r}")
