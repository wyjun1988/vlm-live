"""Metric scale anchor for a recurrent encoder whose scale drifts with stream length (I-05 / I-37).

Measured 2026-09-27: CUT3R fed only 32 keyframes measures rooms at 1.03x their true area; fed every 30th frame
1.23x; every 10th frame 1.39x. The relative layout keeps improving with more frames, only the metric scale
inflates - so the fix must rescale without re-anchoring orientation (state resets cost the direction facts).

The anchor here is the camera's height above the floor. A hand-held or head-mounted camera stays at roughly the
same height; if the encoder's output puts it 20% higher than in the first frames, the output has grown by 20%.
Per ingested frame: floor = a low percentile of the frame's own points along "up" (the mean camera-down direction
so far), height = camera above that floor, ratio = running median of recent heights / the reference height of the
first frames. Points and camera position are divided by the ratio about the world origin (the first camera), which
is where a uniform inflation of the state's frame is undone. Pure numpy; the encoder adapter applies it.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class CameraHeightAnchor:
    def __init__(self, ref_frames: int = 10, window: int = 5, floor_pct: float = 2.0, conf_quantile: float = 0.3,
                 clamp: tuple[float, float] = (0.5, 2.0)) -> None:
        self.ref_frames, self.window = int(ref_frames), int(window)
        self.floor_pct, self.conf_quantile, self.clamp = floor_pct, conf_quantile, clamp
        self.reset()

    def reset(self) -> None:
        self._downs: list[np.ndarray] = []
        self._ref_heights: list[float] = []
        self._recent: deque = deque(maxlen=self.window)
        self.reference: float | None = None
        self.factor = 1.0          # the last correction applied (1/ratio)
        self.frames = 0

    def up(self) -> np.ndarray:
        u = -np.mean(np.stack(self._downs), axis=0)
        return u / (np.linalg.norm(u) + 1e-9)

    def height(self, c2w: np.ndarray, pts: np.ndarray, conf: np.ndarray | None) -> float | None:
        """Camera height above this frame's own floor, in the frame's (possibly drifted) units."""
        self._downs.append(np.asarray(c2w[:3, 1], dtype=np.float64))
        up = self.up()
        p = pts.reshape(-1, 3)
        ok = np.isfinite(p).all(1)
        if conf is not None:
            c = conf.reshape(-1)
            ok &= c >= np.quantile(c[np.isfinite(c)], self.conf_quantile) if np.isfinite(c).any() else ok
        p = p[ok]
        if len(p) < 200:
            return None
        floor = np.percentile(p @ up, self.floor_pct)
        return float(np.asarray(c2w[:3, 3]) @ up - floor)

    def update(self, c2w: np.ndarray, pts: np.ndarray, conf: np.ndarray | None) -> float:
        """Observe one frame (raw encoder output) and return the factor to multiply its points and camera
        position by so that the camera stays at the reference height. 1.0 while the reference is being built."""
        self.frames += 1
        h = self.height(c2w, pts, conf)
        if h is None or h <= 0:
            return self.factor
        if self.reference is None:
            self._ref_heights.append(h)
            if len(self._ref_heights) >= self.ref_frames:
                self.reference = float(np.median(self._ref_heights))
            return 1.0
        self._recent.append(h)
        ratio = float(np.median(self._recent)) / self.reference
        ratio = min(max(ratio, self.clamp[0]), self.clamp[1])
        self.factor = 1.0 / ratio
        return self.factor

    @staticmethod
    def apply(factor: float, c2w: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scale a camera pose and points about the world origin (rotation untouched)."""
        out = np.array(c2w, dtype=np.float64, copy=True)
        out[:3, 3] *= factor
        return out, pts * factor
