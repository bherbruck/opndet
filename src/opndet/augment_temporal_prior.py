"""Per-sample synthesis of the 4-ch model's prior channel.

Produces a stride-4 heatmap that mimics what the host-side TailAccumulator
emits at deployment: bbox-shaped Gaussians at offset positions (where objects
*were* on previous frames), confidence-weighted, with realistic noise sources
(detection drops, false positives, spawn-zone hints, cold-start).

Critical correctness invariant: stamps are NEVER at current-frame GT centers
(except in the rare stationary-motion case). The synth loop iterates k in
[N..1] inclusive — k>=1 is the structural guarantee. See temporal-prior-
synthesis-spec.md §"Critical correctness assertions" for why this matters.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


_DEFAULTS = {
    "n_max": 8,
    "stride": 4,
    "motion_axis_aligned_prob": 0.70,
    "motion_diagonal_prob": 0.25,
    "motion_zero_prob": 0.05,
    "motion_speed_range": (2.0, 15.0),
    "motion_diagonal_jitter_deg": 10.0,
    "confidence_range": (0.5, 0.95),
    "object_drop_prob": 0.05,
    "false_positive_prob": 0.10,
    "false_positive_count_range": (1, 3),
    "false_positive_amplitude_range": (0.3, 0.5),
    "spawn_zone_prob": 0.10,
    "spawn_zone_amplitude_range": (0.3, 0.5),
    "zero_prior_prob": 0.05,
    "gaussian_sigma_factor": 4.0,
}


def _t(v):
    return tuple(v) if isinstance(v, list) else v


class TemporalPriorSynth:
    def __init__(self, config: dict | None = None, seed: int | None = None):
        c = dict(_DEFAULTS)
        if config:
            for k, v in config.items():
                if k in c:
                    c[k] = _t(v)
        self.cfg = c
        self.stride = int(c["stride"])
        self.rng = np.random.default_rng(seed)

        p_sum = c["motion_axis_aligned_prob"] + c["motion_diagonal_prob"] + c["motion_zero_prob"]
        assert abs(p_sum - 1.0) < 1e-6, f"motion probs must sum to 1.0, got {p_sum}"

    def __call__(
        self,
        boxes: np.ndarray,
        H: int,
        W: int,
        force_motion: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Return prior heatmap of shape (H//stride, W//stride), float32 in [0,1].

        boxes: (N,4) xyxy in input pixel coords (post-letterbox). Pass the
        same box list that will become GT for the current frame — the synth
        will OFFSET them per-tail-frame so they never land on current GT.
        """
        Hs, Ws = H // self.stride, W // self.stride
        prior = np.zeros((Hs, Ws), dtype=np.float32)

        if self.rng.random() < self.cfg["zero_prior_prob"]:
            return prior

        N = int(self.rng.integers(0, self.cfg["n_max"] + 1))

        if N == 0:
            if self.rng.random() < self.cfg["spawn_zone_prob"]:
                self._overlay_spawn_zone(prior)
            return prior

        motion = force_motion if force_motion is not None else self._sample_motion()
        fade_step = 1.0 / N

        for k in range(N, 0, -1):
            for box in boxes:
                if self.rng.random() < self.cfg["object_drop_prob"]:
                    continue
                offset = self._shift_box(box, dx=-k * motion[0], dy=-k * motion[1])
                if not self._box_center_in_frame(offset, H, W):
                    continue
                conf = float(self.rng.uniform(*self.cfg["confidence_range"]))
                amp = max(conf - (k - 1) * fade_step, 0.0)
                if amp <= 0.0:
                    continue
                self._stamp_gaussian(prior, self._to_stride_coords(offset), amp)

        if self.rng.random() < self.cfg["false_positive_prob"]:
            lo, hi = self.cfg["false_positive_count_range"]
            n_fp = int(self.rng.integers(lo, hi + 1))
            for _ in range(n_fp):
                fp_box = self._random_box(Hs, Ws, mean_size_from=boxes)
                fp_amp = float(self.rng.uniform(*self.cfg["false_positive_amplitude_range"]))
                self._stamp_gaussian(prior, fp_box, fp_amp)

        if self.rng.random() < self.cfg["spawn_zone_prob"]:
            self._overlay_spawn_zone(prior)

        np.clip(prior, 0.0, 1.0, out=prior)
        return prior

    def _sample_motion(self) -> tuple[float, float]:
        r = self.rng.random()
        speed = float(self.rng.uniform(*self.cfg["motion_speed_range"]))
        if r < self.cfg["motion_axis_aligned_prob"]:
            d = self.rng.choice(4)
            dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][d]
            return (dx * speed, dy * speed)
        if r < self.cfg["motion_axis_aligned_prob"] + self.cfg["motion_diagonal_prob"]:
            base = float(self.rng.choice([45.0, 135.0, 225.0, 315.0]))
            angle = base + float(self.rng.normal(0.0, self.cfg["motion_diagonal_jitter_deg"]))
            rad = math.radians(angle)
            return (math.cos(rad) * speed, math.sin(rad) * speed)
        return (0.0, 0.0)

    @staticmethod
    def _shift_box(box: np.ndarray, dx: float, dy: float) -> np.ndarray:
        return np.array([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy], dtype=np.float32)

    @staticmethod
    def _box_center_in_frame(box: np.ndarray, H: int, W: int) -> bool:
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        return 0.0 <= cx < W and 0.0 <= cy < H

    def _to_stride_coords(self, box: np.ndarray) -> np.ndarray:
        return box / self.stride

    def _stamp_gaussian(self, prior: np.ndarray, box: np.ndarray, amplitude: float) -> None:
        """Anisotropic 2D Gaussian at box center, sigma = edge / sigma_factor.
        Max-merge into prior in-place. Box is in stride coords.
        """
        sf = float(self.cfg["gaussian_sigma_factor"])
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        sigma_x = max(1.0, (x2 - x1) / sf)
        sigma_y = max(1.0, (y2 - y1) / sf)
        rx = int(3 * sigma_x) + 1
        ry = int(3 * sigma_y) + 1
        Hs, Ws = prior.shape
        x0 = max(0, int(cx - rx))
        x1_ = min(Ws, int(cx + rx) + 1)
        y0 = max(0, int(cy - ry))
        y1_ = min(Hs, int(cy + ry) + 1)
        if x1_ <= x0 or y1_ <= y0:
            return
        xx, yy = np.meshgrid(np.arange(x0, x1_, dtype=np.float32),
                             np.arange(y0, y1_, dtype=np.float32))
        g = (amplitude * np.exp(
            -0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2)
        )).astype(np.float32)
        region = prior[y0:y1_, x0:x1_]
        np.maximum(region, g, out=region)

    def _random_box(self, Hs: int, Ws: int, mean_size_from: np.ndarray) -> np.ndarray:
        if mean_size_from.shape[0] > 0:
            w = float(np.median(mean_size_from[:, 2] - mean_size_from[:, 0])) / self.stride
            h = float(np.median(mean_size_from[:, 3] - mean_size_from[:, 1])) / self.stride
            w = max(2.0, w)
            h = max(2.0, h)
        else:
            w = float(self.rng.uniform(2, max(3, Ws / 8)))
            h = float(self.rng.uniform(2, max(3, Hs / 8)))
        cx = float(self.rng.uniform(w * 0.5, Ws - w * 0.5))
        cy = float(self.rng.uniform(h * 0.5, Hs - h * 0.5))
        return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)

    def _overlay_spawn_zone(self, prior: np.ndarray) -> None:
        Hs, Ws = prior.shape
        edge = self.rng.choice(4)  # 0=top, 1=bottom, 2=left, 3=right
        amplitude = float(self.rng.uniform(*self.cfg["spawn_zone_amplitude_range"]))
        if edge in (0, 1):
            band = int(self.rng.integers(max(2, Hs // 6), max(3, Hs // 3) + 1))
            for d in range(band):
                row_amp = amplitude * (1.0 - d / band)
                idx = d if edge == 0 else Hs - 1 - d
                np.maximum(prior[idx, :], row_amp, out=prior[idx, :])
        else:
            band = int(self.rng.integers(max(2, Ws // 6), max(3, Ws // 3) + 1))
            for d in range(band):
                col_amp = amplitude * (1.0 - d / band)
                idx = d if edge == 2 else Ws - 1 - d
                np.maximum(prior[:, idx], col_amp, out=prior[:, idx])


def upsample_prior(prior: np.ndarray, H: int, W: int) -> np.ndarray:
    """Stride-4 prior -> full input resolution via nearest (opset-13 friendly)."""
    return cv2.resize(prior, (W, H), interpolation=cv2.INTER_NEAREST)
