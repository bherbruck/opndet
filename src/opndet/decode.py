from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass
class OBBDetection:
    """Oriented bounding box. (cx, cy, w, h) define the enclosing AABB the OBB
    is inscribed in (matches the encode-side reconstruction); theta is the
    rotation in radians applied around the AABB center to recover the OBB.

    `to_corners()` returns the 4 rotated corners (px) for visualization.
    """
    cx: float
    cy: float
    w: float
    h: float
    theta: float
    score: float

    def to_corners(self) -> np.ndarray:
        """Reconstruct OBB corners from the predicted (AABB, θ).

        The encoder stores the OBB's enclosing AABB plus θ; we invert
        (aw, ah, θ) → (w_obb, h_obb) via the 2x2 system
            aw = w*|cos θ| + h*|sin θ|
            ah = w*|sin θ| + h*|cos θ|
        Singular when |cos 2θ| ≈ 0 (θ ≈ ±45°); in that fallback we draw the
        AABB itself rotated by θ — visually fine, just not metrically tight.
        """
        c = abs(math.cos(self.theta))
        s = abs(math.sin(self.theta))
        det = c * c - s * s
        if abs(det) > 1e-3:
            w_obb = ( c * self.w - s * self.h) / det
            h_obb = (-s * self.w + c * self.h) / det
            w_obb = max(0.0, w_obb)
            h_obb = max(0.0, h_obb)
        else:
            w_obb, h_obb = self.w, self.h
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        half_w, half_h = w_obb * 0.5, h_obb * 0.5
        local = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h],
        ], dtype=np.float32)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        return local @ R.T + np.array([self.cx, self.cy], dtype=np.float32)


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
    """out: [B, C, H', W']. Polymorphic on C:
      C=5 with cxy/wh layout (legacy bbox)            → channel order (obj, cx_rel, cy_rel, w_norm, h_norm)
      C=5 with ltrb layout (-pro Phase 1)             → channel order (obj, l, t, r, b)  → decode_ltrb
      C=7 with OBB layout (-pro Phase 4b)             → 7 channels; AABB extracted from ltrb,
                                                        angle dropped (eval/calib see AABB only).

    Heuristic for C=5: Phase 1 -pro presets emit ltrb in [0, 1]; legacy bbox emits cxy in [0, 1] too,
    so the bytes look identical. We dispatch via decode() (cxy/wh) by default for C=5 — this matches
    historical behavior. -pro callers should use decode_ltrb_batch / decode_obb_batch directly when
    they need the correct semantics. eval/calibrate route through here regardless of head variant;
    for OBB-head models we extract the enclosing AABB and ignore the rotation channels.
    """
    assert out.ndim == 4
    C = out.shape[1]
    if C == 7:
        # OBB head: build AABB-only Detection list from ltrb subset; drop angle.
        ltrb_only = out[:, :5, :, :]
        return [decode_ltrb(ltrb_only[i], img_h, img_w, stride, threshold) for i in range(out.shape[0])]
    assert C == 5, f"decode_batch expects 5 or 7 output channels; got {C}"
    return [decode(out[i], img_h, img_w, stride, threshold) for i in range(out.shape[0])]


def decode_ltrb(out: np.ndarray, img_h: int, img_w: int, stride: int, threshold: float = 0.3) -> list[Detection]:
    """Decode `-pro` output tensor (ltrb regression). No NMS.

    out: [5, H', W'] post-sigmoid, peak-suppressed (model output).
        channel order: [obj, l, t, r, b]. l/t/r/b are image-normalized distances
        in [0,1] from each cell *center* (ix+0.5, iy+0.5)*stride to the predicted
        box's left/top/right/bottom edges respectively.

    Decoded box per peak cell:
        cx_px = (ix + 0.5) * stride
        cy_px = (iy + 0.5) * stride
        x1 = cx_px - l * img_w
        y1 = cy_px - t * img_h
        x2 = cx_px + r * img_w
        y2 = cy_px + b * img_h
    """
    assert out.ndim == 3 and out.shape[0] == 5
    obj, l, t, r, b = out
    ys, xs = np.nonzero(obj > threshold)
    if len(ys) == 0:
        return []
    scores = obj[ys, xs]
    cx_px = (xs + 0.5) * stride
    cy_px = (ys + 0.5) * stride
    x1 = cx_px - l[ys, xs] * img_w
    y1 = cy_px - t[ys, xs] * img_h
    x2 = cx_px + r[ys, xs] * img_w
    y2 = cy_px + b[ys, xs] * img_h
    return [
        Detection(float(a), float(b_), float(c), float(d), float(s))
        for a, b_, c, d, s in zip(x1, y1, x2, y2, scores)
    ]


def decode_ltrb_batch(out: np.ndarray, img_h: int, img_w: int, stride: int, threshold: float = 0.3) -> list[list[Detection]]:
    """out: [B, 5, H', W']."""
    assert out.ndim == 4 and out.shape[1] == 5
    return [decode_ltrb(out[i], img_h, img_w, stride, threshold) for i in range(out.shape[0])]


def decode_obb(out: np.ndarray, img_h: int, img_w: int, stride: int, threshold: float = 0.3) -> list[OBBDetection]:
    """Decode `-pro` OBB output tensor. No NMS.

    out: [7, H', W'] post-sigmoid+tanh, peak-suppressed.
        channels: [obj, l, t, r, b, sin2θ, cos2θ]. l/t/r/b in [0,1] image-norm
        distances from cell center to enclosing-AABB edges; sin2θ/cos2θ in
        [-1, 1] (post-Tanh).

    Returns list of OBBDetection. (cx, cy, w, h) define the AABB; theta is
    derived from 0.5 * atan2(sin2θ, cos2θ), wrapped to [0, π).
    """
    assert out.ndim == 3 and out.shape[0] == 7
    obj, l, t, r, b, s2, c2 = out
    ys, xs = np.nonzero(obj > threshold)
    if len(ys) == 0:
        return []
    scores = obj[ys, xs]
    cx_px = (xs + 0.5) * stride
    cy_px = (ys + 0.5) * stride
    x1 = cx_px - l[ys, xs] * img_w
    y1 = cy_px - t[ys, xs] * img_h
    x2 = cx_px + r[ys, xs] * img_w
    y2 = cy_px + b[ys, xs] * img_h
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = np.maximum(x2 - x1, 0.0)
    h = np.maximum(y2 - y1, 0.0)
    theta = 0.5 * np.arctan2(s2[ys, xs], c2[ys, xs])
    theta = np.mod(theta, math.pi)
    return [
        OBBDetection(float(cx_), float(cy_), float(w_), float(h_), float(th), float(s))
        for cx_, cy_, w_, h_, th, s in zip(cx, cy, w, h, theta, scores)
    ]


def decode_obb_batch(out: np.ndarray, img_h: int, img_w: int, stride: int, threshold: float = 0.3) -> list[list[OBBDetection]]:
    """out: [B, 7, H', W']."""
    assert out.ndim == 4 and out.shape[1] == 7
    return [decode_obb(out[i], img_h, img_w, stride, threshold) for i in range(out.shape[0])]
