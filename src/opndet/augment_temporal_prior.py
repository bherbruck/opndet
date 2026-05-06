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
    # Per-frame motion in input pixels.
    "motion_speed_range": (2.0, 15.0),
    # Optional override: per-frame motion as FRACTION of input width. When set
    # (any value > 0 in either bound), this overrides motion_speed_range.
    # E.g. [0.005, 0.03] = 0.5%-3% of frame width per frame. At W=512, that's
    # 2.6-15.4 px/frame, matching the pixel default.
    "motion_speed_frac_range": (0.0, 0.0),
    "motion_diagonal_jitter_deg": 10.0,
    "confidence_range": (0.5, 0.95),
    "object_drop_prob": 0.05,        # per-frame-per-object miss
    "object_skip_prob": 0.0,         # per-object total exclusion (brand-new object)
    # Edge-margin exclusions (as fraction of input H or W). A box whose center
    # falls within `H*margin_top` of the top edge gets NO prior stamps — models
    # "just appeared from off-frame" objects. Set to 0.0 to disable.
    "margin_top": 0.0,
    "margin_bottom": 0.0,
    "margin_left": 0.0,
    "margin_right": 0.0,
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
        return_trails: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[list[tuple[float, float]]]]:
        """Return prior heatmap of shape (H//stride, W//stride), float32 in [0,1].

        boxes: (N,4) xyxy in input pixel coords (post-letterbox). Pass the
        same box list that will become GT for the current frame — the synth
        will OFFSET them per-tail-frame so they never land on current GT.

        return_trails: if True, also return a list[list[(cx, cy)]] of stamp
        center positions per eligible object in INPUT pixel coords, ordered
        from oldest (k=N) to newest (k=1). Used for visualization to draw
        the trail line directly instead of reconstructing from heatmap.
        """
        Hs, Ws = H // self.stride, W // self.stride
        prior = np.zeros((Hs, Ws), dtype=np.float32)
        trails: list[list[tuple[float, float]]] = []

        if self.rng.random() < self.cfg["zero_prior_prob"]:
            return (prior, trails) if return_trails else prior

        N = int(self.rng.integers(0, self.cfg["n_max"] + 1))

        if N == 0:
            if self.rng.random() < self.cfg["spawn_zone_prob"]:
                self._overlay_spawn_zone(prior)
            return (prior, trails) if return_trails else prior

        motion = force_motion if force_motion is not None else self._sample_motion(W)
        fade_step = 1.0 / N

        eligible = self._filter_eligible(boxes, H, W)
        per_obj_trails: list[list[tuple[float, float]]] = [[] for _ in eligible]

        for k in range(N, 0, -1):
            for obj_i, box in enumerate(eligible):
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
                cx_in = (float(offset[0]) + float(offset[2])) * 0.5
                cy_in = (float(offset[1]) + float(offset[3])) * 0.5
                per_obj_trails[obj_i].append((cx_in, cy_in))

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
        if return_trails:
            trails = [t for t in per_obj_trails if len(t) > 0]
            return prior, trails
        return prior

    def _sample_motion(self, W: int = 512) -> tuple[float, float]:
        r = self.rng.random()
        # Use frac_range (relative to W) when explicitly set; otherwise fall
        # back to the absolute pixel range. Any nonzero value in frac_range
        # turns it on.
        frac_lo, frac_hi = self.cfg["motion_speed_frac_range"]
        if frac_hi > 0.0:
            speed = float(self.rng.uniform(frac_lo, frac_hi)) * W
        else:
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

    def _filter_eligible(self, boxes: np.ndarray, H: int, W: int) -> list[np.ndarray]:
        """Drop boxes that should get no prior at all: edge-margin exclusions
        (just-appeared / about-to-disappear) and per-object skip (brand-new
        objects mixed in with established ones).

        Margin test is intersection-based (permissive): a box is excluded if
        ANY part of it overlaps the margin band. E.g. margin_top=0.05 on a
        384px input excludes any box with y1 < 19.2.
        """
        m_top = float(self.cfg["margin_top"]) * H
        m_bot = float(self.cfg["margin_bottom"]) * H
        m_left = float(self.cfg["margin_left"]) * W
        m_right = float(self.cfg["margin_right"]) * W
        skip_p = float(self.cfg["object_skip_prob"])
        out = []
        for box in boxes:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            if m_top > 0 and y1 < m_top:
                continue
            if m_bot > 0 and y2 > H - m_bot:
                continue
            if m_left > 0 and x1 < m_left:
                continue
            if m_right > 0 and x2 > W - m_right:
                continue
            if skip_p > 0 and self.rng.random() < skip_p:
                continue
            out.append(box)
        return out

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
