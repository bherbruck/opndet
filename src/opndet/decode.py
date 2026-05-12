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
    """Oriented bounding box. (cx, cy, w, h, theta) — direct rotated rect:
    cx/cy is the OBB's centroid in pixels, w/h are its OWN dims (NOT the
    enclosing AABB), theta is the rotation of the major axis in radians.

    `to_corners()` returns the 4 rotated corners (px) for visualization.
    """
    cx: float
    cy: float
    w: float
    h: float
    theta: float
    score: float

    def to_corners(self) -> np.ndarray:
        """Build OBB corners directly from (cx, cy, w, h, θ)."""
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        half_w, half_h = self.w * 0.5, self.h * 0.5
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
    for OBB-head models (6-ch direct cxywhθ) we derive a non-rotated bounding rect from
    (cx, cy, w, h) just for AABB-shaped Detection rows used by score-vs-IoU eval paths.
    """
    assert out.ndim == 4
    C = out.shape[1]
    if C == 6:
        # 6-ch OBB head (cxywhθ direct). Build axis-aligned Detection from
        # (cx, cy, w, h) only — the angle channel is read separately by
        # OBB-aware eval paths via decode_obb_batch.
        out_dets: list[list[Detection]] = []
        for b in range(out.shape[0]):
            obj, cxo, cyo, wn, hn, _tn = out[b]
            ys, xs = np.nonzero(obj > threshold)
            if len(ys) == 0:
                out_dets.append([])
                continue
            scores = obj[ys, xs]
            cx = (xs + cxo[ys, xs]) * stride
            cy = (ys + cyo[ys, xs]) * stride
            w  = wn[ys, xs] * img_w
            h  = hn[ys, xs] * img_h
            x1 = cx - w * 0.5
            y1 = cy - h * 0.5
            x2 = cx + w * 0.5
            y2 = cy + h * 0.5
            out_dets.append([
                Detection(float(a), float(b_), float(c), float(d), float(s))
                for a, b_, c, d, s in zip(x1, y1, x2, y2, scores)
            ])
        return out_dets
    assert C == 5, f"decode_batch expects 5 or 6 output channels; got {C}"
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
    """Decode 6-ch OBB output. Direct (cx, cy, w, h, θ). No AABB.

    out: [6, H', W'] post-sigmoid + peak-suppressed obj.
        channels: [obj, cx_offset, cy_offset, w_norm, h_norm, θ_norm]
            cx_offset = (cx_px - ix*stride) / stride       in [0, 1]
            cy_offset = (cy_px - iy*stride) / stride       in [0, 1]
            w_norm    = w_obb / img_w                      in [0, 1]
            h_norm    = h_obb / img_h                      in [0, 1]
            θ_norm    = θ / π                              in [0, 1)
    """
    assert out.ndim == 3 and out.shape[0] == 6
    obj, cxo, cyo, wn, hn, tn = out
    ys, xs = np.nonzero(obj > threshold)
    if len(ys) == 0:
        return []
    scores = obj[ys, xs]
    cx = (xs + cxo[ys, xs]) * stride
    cy = (ys + cyo[ys, xs]) * stride
    w  = wn[ys, xs] * img_w
    h  = hn[ys, xs] * img_h
    theta = tn[ys, xs] * math.pi  # already in [0, π) by construction
    return [
        OBBDetection(float(cx_), float(cy_), float(w_), float(h_), float(th), float(s))
        for cx_, cy_, w_, h_, th, s in zip(cx, cy, w, h, theta, scores)
    ]


def decode_obb_batch(out: np.ndarray, img_h: int, img_w: int, stride: int, threshold: float = 0.3) -> list[list[OBBDetection]]:
    """out: [B, 6, H', W']."""
    assert out.ndim == 4 and out.shape[1] == 6
    return [decode_obb(out[i], img_h, img_w, stride, threshold) for i in range(out.shape[0])]


def gt_obbs_from_targets(pos: np.ndarray, obb: np.ndarray, img_h: int, img_w: int, stride: int) -> list[np.ndarray]:
    """Reconstruct per-image GT OBBs from encoded targets (5-ch direct).
        pos: [B, 1, H', W']  (or [B, H', W'])  binary GT-cell mask
        obb: [B, 5, H', W']  (cx_off, cy_off, w_norm, h_norm, θ_norm) at GT cells
    Returns list[B] of [N, 5] arrays = (cx_px, cy_px, w_obb, h_obb, θ_rad).
    """
    if pos.ndim == 4:
        pos = pos[:, 0]
    out: list[np.ndarray] = []
    for b in range(pos.shape[0]):
        ys, xs = np.nonzero(pos[b] > 0.5)
        if len(ys) == 0:
            out.append(np.zeros((0, 5), dtype=np.float32))
            continue
        cxo = obb[b, 0, ys, xs]
        cyo = obb[b, 1, ys, xs]
        wn  = obb[b, 2, ys, xs]
        hn  = obb[b, 3, ys, xs]
        tn  = obb[b, 4, ys, xs]
        cx = (xs + cxo) * stride
        cy = (ys + cyo) * stride
        w  = wn * img_w
        h  = hn * img_h
        theta = tn * math.pi
        out.append(np.stack([cx, cy, w, h, theta], axis=-1).astype(np.float32))
    return out


@dataclass
class SegBlob:
    cx: float          # centroid x, image px
    cy: float          # centroid y, image px
    area_px: float     # foreground pixel count of this blob (the convex object's area)
    peak: float        # max dome value inside the blob (≈1.0 for a clean detection)
    x1: float          # tight AABB of the blob (informational)
    y1: float
    x2: float
    y2: float


def decode_seg(dome: np.ndarray, threshold: float = 0.5, min_area: int = 4) -> list[SegBlob]:
    """Decode one dense dome map [H, W] (the bbox-*-seg output channel) into instances.

    Unlike the detector heads (which bake peak-suppression into the graph so NO
    postprocessing runs), a dense seg dome IS a segmentation map — extracting the
    per-object area/center is inherently postprocessing: threshold → connected
    components → per-blob centroid + pixel count. The dome hits exactly 0 between
    touching objects, so plain 4-connectivity components already separate them; no
    NMS, no watershed needed for convex blobs.

    threshold: foreground cut on the dome (0.5 = "inside the object by >half-depth").
               Lower it to recover the full object area; raise it to get just cores.
    min_area : drop blobs smaller than this many px (denoise).
    Returns blobs sorted by descending peak.
    """
    import cv2
    H, W = dome.shape
    fg = (dome >= float(threshold)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=4)
    out: list[SegBlob] = []
    for i in range(1, n):  # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                 cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        peak = float(dome[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i].max())
        cx, cy = float(cents[i, 0]), float(cents[i, 1])
        out.append(SegBlob(cx, cy, float(area), peak, float(x), float(y), float(x + w), float(y + h)))
    out.sort(key=lambda b: -b.peak)
    return out


def decode_seg_batch(out: np.ndarray, threshold: float = 0.5, min_area: int = 4) -> list[list[SegBlob]]:
    """out: [B, 1, H, W] dome maps → per-image instance lists (see decode_seg)."""
    if out.ndim == 4:
        out = out[:, 0]
    return [decode_seg(out[b], threshold=threshold, min_area=min_area) for b in range(out.shape[0])]
