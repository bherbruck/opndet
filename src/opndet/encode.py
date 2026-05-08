from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from opndet.config import ModelConfig


# ---- OBB representation helpers (ROADMAP §1.8 Phase 4b) -------------------
# Internal canonical: (cx, cy, w_major, h_minor, theta) with theta in [0, pi).
# w >= h by construction. theta is the rotation of the major axis from +x.


def corners_to_obb(corners: np.ndarray) -> tuple[float, float, float, float, float]:
    """4 corners (px, [4,2]) → (cx, cy, w_major, h_minor, theta in [0, pi))."""
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w_r, h_r), ang_deg = cv2.minAreaRect(pts)
    # cv2.minAreaRect returns angle in (-90, 0] (legacy) or [0, 90) depending on
    # OpenCV version. Canonicalize to (w >= h, theta in [0, pi)).
    w, h = float(w_r), float(h_r)
    theta = math.radians(float(ang_deg))
    if h > w:
        w, h = h, w
        theta += math.pi / 2.0
    theta = theta % math.pi
    return float(cx), float(cy), w, h, theta


def obb_to_corners(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """OBB → 4 corners (float32, [4,2]). theta in radians, major axis rotation."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    half_w, half_h = w * 0.5, h * 0.5
    local = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ], dtype=np.float32)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return local @ R.T + np.array([cx, cy], dtype=np.float32)


def obb_to_aabb(cx: float, cy: float, w: float, h: float, theta: float) -> tuple[float, float, float, float]:
    """OBB → enclosing AABB (x1, y1, x2, y2). The AABB the rotated rect inscribes."""
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    aw = w * cos_t + h * sin_t
    ah = w * sin_t + h * cos_t
    return cx - aw * 0.5, cy - ah * 0.5, cx + aw * 0.5, cy + ah * 0.5


def yolo_obb_line_to_obb(line: str, img_w: int, img_h: int) -> tuple[int, float, float, float, float, float] | None:
    """Parse one YOLOv8-OBB line into (class_id, cx, cy, w, h, theta).
    Format: 'class_id x1 y1 x2 y2 x3 y3 x4 y4' (8 normalized floats, four corners).
    Returns None on malformed input.
    """
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    try:
        class_id = int(parts[0])
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(4, 2)
    except ValueError:
        return None
    coords[:, 0] *= img_w
    coords[:, 1] *= img_h
    cx, cy, w, h, theta = corners_to_obb(coords)
    return class_id, cx, cy, w, h, theta


def gaussian_radius(w: float, h: float, min_overlap: float = 0.7) -> float:
    """CornerNet radius heuristic: smallest r s.t. shifted box still has IoU >= min_overlap."""
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(b1 * b1 - 4 * a1 * c1)) / 2
    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(b2 * b2 - 4 * a2 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 * b3 - 4 * a3 * c3)) / 2
    return max(1.0, min(r1, r2, r3))


def _draw_gaussian(hm: np.ndarray, cx: int, cy: int, sigma: float) -> None:
    h, w = hm.shape
    rad = int(3 * sigma)
    x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
    y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma * sigma))
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)


def _render_dist_target(boxes: np.ndarray, Hp: int, Wp: int, s: int) -> np.ndarray:
    """Per-pixel distance target at output resolution. For each GT, renders the inscribed
    ellipse with values that ramp linearly from 1 at center to 0 at the boundary, then
    aggregates with elementwise max across objects. The convex prior: midplanes between
    touching objects naturally drop to ~0 because both objects' ramps decay there.
    """
    dist = np.zeros((Hp, Wp), dtype=np.float32)
    if len(boxes) == 0:
        return dist
    yy, xx = np.mgrid[0:Hp, 0:Wp].astype(np.float32)
    for x1, y1, x2, y2 in boxes:
        bw = max(0.0, x2 - x1) / s
        bh = max(0.0, y2 - y1) / s
        if bw < 2.0 or bh < 2.0:
            continue
        cx_g = (x1 + x2) * 0.5 / s
        cy_g = (y1 + y2) * 0.5 / s
        a = bw * 0.5; b = bh * 0.5
        # ((x-cx)/a)^2 + ((y-cy)/b)^2 = 1 at boundary, 0 at center
        r = np.sqrt(((xx - cx_g) / a) ** 2 + ((yy - cy_g) / b) ** 2)
        obj = np.clip(1.0 - r, 0.0, 1.0)
        np.maximum(dist, obj, out=dist)
    return dist


def encode_targets(
    boxes: np.ndarray,
    cfg: ModelConfig,
    min_sigma: float = 1.0,
    dist_head: bool = False,
) -> dict[str, torch.Tensor]:
    """Encode list of (x1,y1,x2,y2) pixel boxes into dense GT tensors.

    Returns dict with:
      hm   : [1, H', W']     gaussian heatmap targets in [0,1]
      cxy  : [2, H', W']     cell-relative center offset GT (only valid where pos)
      wh   : [2, H', W']     image-normalized w,h GT (only valid where pos)
      pos  : [1, H', W']     1.0 at positive cells (peak), 0 elsewhere — for size loss masking
      dist : [1, H', W']     (only if dist_head=True) inscribed-ellipse linear ramp, [0,1]
    """
    H, W = cfg.img_h, cfg.img_w
    s = cfg.stride
    Hp, Wp = H // s, W // s
    hm = np.zeros((Hp, Wp), dtype=np.float32)
    cxy = np.zeros((2, Hp, Wp), dtype=np.float32)
    wh = np.zeros((2, Hp, Wp), dtype=np.float32)
    pos = np.zeros((Hp, Wp), dtype=np.float32)

    if len(boxes) > 0:
        for x1, y1, x2, y2 in boxes:
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw < 1.0 or bh < 1.0:
                continue
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            cx_g = cx / s
            cy_g = cy / s
            ix = int(cx_g)
            iy = int(cy_g)
            if ix < 0 or iy < 0 or ix >= Wp or iy >= Hp:
                continue
            r_px = gaussian_radius(bw, bh)
            sigma = max(min_sigma, r_px / s / 3.0)
            _draw_gaussian(hm, ix, iy, sigma)
            cxy[0, iy, ix] = cx_g - ix
            cxy[1, iy, ix] = cy_g - iy
            wh[0, iy, ix] = bw / W
            wh[1, iy, ix] = bh / H
            pos[iy, ix] = 1.0

    out = {
        "hm": torch.from_numpy(hm).unsqueeze(0),
        "cxy": torch.from_numpy(cxy),
        "wh": torch.from_numpy(wh),
        "pos": torch.from_numpy(pos).unsqueeze(0),
    }
    if dist_head:
        d = _render_dist_target(boxes, Hp, Wp, s)
        out["dist"] = torch.from_numpy(d).unsqueeze(0)
    return out


def encode_targets_ltrb(
    boxes: np.ndarray,
    cfg: ModelConfig,
    min_sigma: float = 1.0,
) -> dict[str, torch.Tensor]:
    """ltrb variant of encode_targets used by `-pro` presets.

    Single positive cell per GT (the center cell), as in encode_targets. The
    regression target switches semantics from (cx_offset, cy_offset, w_norm,
    h_norm) to (l, t, r, b): image-normalized distances from the cell *center*
    (ix+0.5, iy+0.5)*stride to the four box edges. Each in [0, 1] (clamped).

    Returns dict with:
      hm   : [1, H', W']  gaussian heatmap targets in [0,1]
      ltrb : [4, H', W']  image-normalized (l, t, r, b) distances; valid where pos
      pos  : [1, H', W']  1.0 at positive (center) cells
    """
    H, W = cfg.img_h, cfg.img_w
    s = cfg.stride
    Hp, Wp = H // s, W // s
    hm = np.zeros((Hp, Wp), dtype=np.float32)
    ltrb = np.zeros((4, Hp, Wp), dtype=np.float32)
    pos = np.zeros((Hp, Wp), dtype=np.float32)

    if len(boxes) > 0:
        for x1, y1, x2, y2 in boxes:
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw < 1.0 or bh < 1.0:
                continue
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            cx_g = cx / s
            cy_g = cy / s
            ix = int(cx_g)
            iy = int(cy_g)
            if ix < 0 or iy < 0 or ix >= Wp or iy >= Hp:
                continue
            r_px = gaussian_radius(bw, bh)
            sigma = max(min_sigma, r_px / s / 3.0)
            _draw_gaussian(hm, ix, iy, sigma)
            cx_cell = (ix + 0.5) * s
            cy_cell = (iy + 0.5) * s
            ltrb[0, iy, ix] = float(np.clip((cx_cell - x1) / W, 0.0, 1.0))
            ltrb[1, iy, ix] = float(np.clip((cy_cell - y1) / H, 0.0, 1.0))
            ltrb[2, iy, ix] = float(np.clip((x2 - cx_cell) / W, 0.0, 1.0))
            ltrb[3, iy, ix] = float(np.clip((y2 - cy_cell) / H, 0.0, 1.0))
            pos[iy, ix] = 1.0

    return {
        "hm": torch.from_numpy(hm).unsqueeze(0),
        "ltrb": torch.from_numpy(ltrb),
        "pos": torch.from_numpy(pos).unsqueeze(0),
    }


def _draw_rotated_gaussian(hm: np.ndarray, cx: int, cy: int, sigma_x: float, sigma_y: float, theta: float) -> None:
    """Rotated elliptical Gaussian on a stride-cell grid. Center (cx, cy) in cell coords;
    sigma_x/sigma_y in cells; theta = rotation of MAJOR axis (sigma_x) from +x axis.
    Vectorized over a (2*rad+1)^2 patch.
    """
    h, w = hm.shape
    rad = int(3 * max(sigma_x, sigma_y))
    if rad < 1:
        rad = 1
    x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
    y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dx = (xs - cx).astype(np.float32)
    dy = (ys - cy).astype(np.float32)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # Rotate (dx, dy) into the OBB's local frame (major along x_local).
    xp =  dx * cos_t + dy * sin_t
    yp = -dx * sin_t + dy * cos_t
    g = np.exp(-(xp * xp / (2.0 * sigma_x * sigma_x) + yp * yp / (2.0 * sigma_y * sigma_y)))
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)


def encode_targets_obb(
    obbs: np.ndarray,
    cfg: ModelConfig,
    min_sigma: float = 1.0,
    aspect_round_thresh: float = 1.15,
) -> dict[str, torch.Tensor]:
    """Encode list of OBBs (cx, cy, w_major, h_minor, theta) px+rad into dense GT.

    Reg target is (l, t, r, b) image-frame distances to the OBB's enclosing AABB,
    plus (sin2θ, cos2θ) angle channels in [-1, 1]. The 2θ encoding handles the
    180° wrap of OBB orientation.

    obbs: [N, 5] of (cx, cy, w, h, theta).
    Returns dict with:
      hm    : [1, H', W']  rotated elliptical Gaussian heatmap
      obb   : [6, H', W']  (l, t, r, b, sin2θ, cos2θ)
      pos   : [1, H', W']  1.0 at positive (center) cells
      angle_mask : [1, H', W']  1.0 at non-round positives (aspect > thresh), 0 elsewhere
    """
    H, W = cfg.img_h, cfg.img_w
    s = cfg.stride
    Hp, Wp = H // s, W // s
    hm = np.zeros((Hp, Wp), dtype=np.float32)
    reg = np.zeros((6, Hp, Wp), dtype=np.float32)
    pos = np.zeros((Hp, Wp), dtype=np.float32)
    ang_mask = np.zeros((Hp, Wp), dtype=np.float32)

    if len(obbs) > 0:
        for cx_px, cy_px, bw, bh, theta in obbs:
            bw = float(bw)
            bh = float(bh)
            if bw < 1.0 or bh < 1.0:
                continue
            cx_g = float(cx_px) / s
            cy_g = float(cy_px) / s
            ix = int(cx_g)
            iy = int(cy_g)
            if ix < 0 or iy < 0 or ix >= Wp or iy >= Hp:
                continue
            r_px = gaussian_radius(bw, bh)
            base_sigma = max(min_sigma, r_px / s / 3.0)
            # Circular cls heatmap: shape info lives in the reg head (sin2θ/cos2θ),
            # NOT in the heatmap target. Elongated targets confuse peak-pick because
            # neighbors along the major axis get high supervision values and steal
            # the local-max winner. Box rotation still flows through reg channels.
            _draw_rotated_gaussian(hm, ix, iy, base_sigma, base_sigma, 0.0)
            x1, y1, x2, y2 = obb_to_aabb(cx_px, cy_px, bw, bh, float(theta))
            cx_cell = (ix + 0.5) * s
            cy_cell = (iy + 0.5) * s
            reg[0, iy, ix] = float(np.clip((cx_cell - x1) / W, 0.0, 1.0))
            reg[1, iy, ix] = float(np.clip((cy_cell - y1) / H, 0.0, 1.0))
            reg[2, iy, ix] = float(np.clip((x2 - cx_cell) / W, 0.0, 1.0))
            reg[3, iy, ix] = float(np.clip((y2 - cy_cell) / H, 0.0, 1.0))
            two_theta = 2.0 * float(theta)
            reg[4, iy, ix] = math.sin(two_theta)
            reg[5, iy, ix] = math.cos(two_theta)
            pos[iy, ix] = 1.0
            if max(bw, bh) / max(min(bw, bh), 1e-6) >= aspect_round_thresh:
                ang_mask[iy, ix] = 1.0

    return {
        "hm": torch.from_numpy(hm).unsqueeze(0),
        "obb": torch.from_numpy(reg),
        "pos": torch.from_numpy(pos).unsqueeze(0),
        "angle_mask": torch.from_numpy(ang_mask).unsqueeze(0),
    }


def collate_targets(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}
