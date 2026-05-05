"""Deployment-side TailAccumulator: O(1)-memory rolling prior produced on the
host, fed as the 4th input channel to a 4-ch (temporal) opndet model.

Numpy-only (no torch). Runs in the inference application loop, between the
detection step and the next preprocess step. Per-frame cost is small:
fade is a single np.maximum, stamping is a small Gaussian gather per
detection, optional spawn-zone overlay is a max-merge.

Usage sketch:

    acc = TailAccumulator((96, 128), n_frames=8, stamp_threshold=0.4,
                          spawn_mask=mask, spawn_amplitude=0.4)
    for frame in stream:
        prior = acc.acc                   # (Hs, Ws) float32 in [0,1]
        prior_full = upsample_to(H, W)    # nearest, ONNX-13 friendly
        x = concat(rgb, prior_full)       # (4, H, W)
        dets = run_model(x)
        acc.update(dets)
"""
from __future__ import annotations

import numpy as np


class TailAccumulator:
    def __init__(
        self,
        shape: tuple[int, int],
        n_frames: int = 8,
        stamp_threshold: float = 0.4,
        spawn_mask: np.ndarray | None = None,
        spawn_amplitude: float = 0.4,
        gaussian_sigma_factor: float = 4.0,
        stride: int = 4,
    ):
        Hs, Ws = int(shape[0]), int(shape[1])
        self.shape = (Hs, Ws)
        self.acc = np.zeros((Hs, Ws), dtype=np.float32)
        self.fade_step = 1.0 / max(1, int(n_frames))
        self.stamp_threshold = float(stamp_threshold)
        self.spawn_amplitude = float(spawn_amplitude)
        self.sigma_factor = float(gaussian_sigma_factor)
        self.stride = int(stride)
        if spawn_mask is not None:
            sm = np.asarray(spawn_mask, dtype=np.float32)
            if sm.shape != self.shape:
                raise ValueError(f"spawn_mask shape {sm.shape} != accumulator shape {self.shape}")
            self.spawn_mask = sm
        else:
            self.spawn_mask = None

    def reset(self) -> None:
        self.acc.fill(0.0)

    def update(self, detections) -> np.ndarray:
        """detections: iterable of (box_xyxy_input_coords, score). Boxes in
        input-pixel coords (matching the model's input H/W); they are converted
        to stride-coords internally for stamping into self.acc.
        """
        np.maximum(self.acc - self.fade_step, 0.0, out=self.acc)

        for box, score in detections:
            score = float(score)
            if score < self.stamp_threshold:
                continue
            self._stamp_gaussian(box, score)

        if self.spawn_mask is not None:
            np.maximum(self.acc, self.spawn_mask * self.spawn_amplitude, out=self.acc)

        np.clip(self.acc, 0.0, 1.0, out=self.acc)
        return self.acc

    def _stamp_gaussian(self, box, amplitude: float) -> None:
        s = self.stride
        x1, y1, x2, y2 = float(box[0]) / s, float(box[1]) / s, float(box[2]) / s, float(box[3]) / s
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        sigma_x = max(1.0, (x2 - x1) / self.sigma_factor)
        sigma_y = max(1.0, (y2 - y1) / self.sigma_factor)
        rx = int(3 * sigma_x) + 1
        ry = int(3 * sigma_y) + 1
        Hs, Ws = self.shape
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
        region = self.acc[y0:y1_, x0:x1_]
        np.maximum(region, g, out=region)
