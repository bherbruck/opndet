from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def decode(out: np.ndarray, img_h: int, img_w: int, stride: int, threshold: float = 0.3) -> list[Detection]:
    """Decode opndet-bbox output tensor to list of detections. No NMS.

    out: [5, H', W'] post-sigmoid, peak-suppressed (model output).
        channels: [obj, cx_rel, cy_rel, w_norm, h_norm]
    """
    assert out.ndim == 3 and out.shape[0] == 5
    obj, cx, cy, wn, hn = out
    ys, xs = np.nonzero(obj > threshold)
    if len(ys) == 0:
        return []
    scores = obj[ys, xs]
    cx_rel = cx[ys, xs]
    cy_rel = cy[ys, xs]
    w = wn[ys, xs] * img_w
    h = hn[ys, xs] * img_h
    cx_img = (xs + cx_rel) * stride
    cy_img = (ys + cy_rel) * stride
    x1 = cx_img - w * 0.5
    y1 = cy_img - h * 0.5
    x2 = cx_img + w * 0.5
    y2 = cy_img + h * 0.5
    return [
        Detection(float(a), float(b), float(c), float(d), float(s))
        for a, b, c, d, s in zip(x1, y1, x2, y2, scores)
    ]


def decode_batch(out: np.ndarray, img_h: int, img_w: int, stride: int, threshold: float = 0.3) -> list[list[Detection]]:
    """out: [B, 5, H', W']."""
    assert out.ndim == 4 and out.shape[1] == 5
    return [decode(out[i], img_h, img_w, stride, threshold) for i in range(out.shape[0])]
