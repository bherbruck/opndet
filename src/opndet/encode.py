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


def _draw_obj_blob(hm: np.ndarray, ix: int, iy: int, bw_px: float, bh_px: float, theta: float,
                   stride: int, min_sigma: float, blob_frac: float, target_shape: str = "gaussian") -> None:
    """Draw the objectness target blob at cell (ix, iy).

    target_shape == "ellipse" → a *domed ellipse* fitted to the box: 1.0 at the
        center, linearly ramping to 0 at the box's inscribed-ellipse boundary,
        0 in the box corners and beyond, rotated by theta. Single-peaked (peak
        suppression unaffected) and — unlike a Gaussian whose tail never reaches
        0 — it hits exactly 0 at the object edge, so it does NOT bleed into the
        gaps between touching objects. (Same family as _render_dist_target,
        generalized to rotation, painted per-box.) `blob_frac` ignored here.

    Otherwise a Gaussian:
      blob_frac <= 0  → legacy CornerNet: isotropic, σ = IoU-shift radius / 3 —
                        a tight bump usually much smaller than the object.
      blob_frac > 0   → oriented elliptical Gaussian, σ ≈ blob_frac·{w,h}/stride.
                        Object-shaped-ish but the tail bleeds into gaps — keep modest.
    """
    if target_shape == "ellipse":
        a = bw_px / (2.0 * stride)
        b = bh_px / (2.0 * stride)
        _draw_rotated_dome(hm, ix, iy, a, b, float(theta))
    elif blob_frac > 0.0:
        sx = max(min_sigma, blob_frac * bw_px / stride)   # along the box's local x (theta dir)
        sy = max(min_sigma, blob_frac * bh_px / stride)
        _draw_rotated_gaussian(hm, ix, iy, sx, sy, float(theta))
    else:
        sigma = max(min_sigma, gaussian_radius(bw_px, bh_px) / stride / 3.0)
        _draw_gaussian(hm, ix, iy, sigma)


def _draw_rotated_dome(hm: np.ndarray, cx: int, cy: int, a: float, b: float, theta: float,
                       ramp_px: float = 0.0) -> None:
    """Domed ellipse on a stride-cell grid (semi-axes a,b in cells, major along theta), 0
    outside the ellipse, max-aggregated onto hm.

    ramp_px == 0 (default): proportional ramp — 1.0 at (cx,cy), linear to 0 at the boundary
        (single-peaked: the center cell is the unique max).
    ramp_px  > 0: FLAT-TOP plateau — 1.0 across the interior, linear falloff to 0 only over
        the last ~ramp_px cells before the boundary. Whole interior is the max (no single
        peak); the edge dropoff is the only gradient. Still hits exactly 0 at the boundary →
        touching objects' domes don't bleed across the contact line."""
    h, w = hm.shape
    a = max(1.0, float(a))
    b = max(1.0, float(b))
    rad = int(math.ceil(max(a, b))) + 1
    x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
    y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dx = (xs - cx).astype(np.float32)
    dy = (ys - cy).astype(np.float32)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    xp = dx * cos_t + dy * sin_t      # along the box's local x (major / theta dir)
    yp = -dx * sin_t + dy * cos_t
    r = np.sqrt((xp / a) ** 2 + (yp / b) ** 2)   # 0 at center, 1 at the ellipse boundary
    if ramp_px > 0:
        # radial distance (cells) from this point to the boundary along its ray through center:
        # boundary is at r=1, so dist = |（xp,yp)|·(1−r)/r  (rotation preserves distance).
        rho = np.sqrt(xp * xp + yp * yp)
        dist = np.where(r > 1e-6, rho * (1.0 - r) / np.maximum(r, 1e-6), np.float32(1e9))
        g = np.clip(dist / float(ramp_px), 0.0, 1.0).astype(np.float32)
        g[r > 1.0] = 0.0
    else:
        g = np.clip(1.0 - r, 0.0, 1.0).astype(np.float32)
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)


def _box_near_edge(x1: float, y1: float, x2: float, y2: float, W: int, H: int, margin: float) -> bool:
    """True if the box comes within `margin` (fraction of the image) of any edge —
    i.e. it's clipped or near-clipped. Used to fall back from the ellipse target to
    a plain Gaussian for edge objects (rendering a clean egg-ellipse for the
    visible chunk of a half-egg would put the peak at the wrong place)."""
    if margin <= 0.0:
        return False
    return x1 < margin * W or y1 < margin * H or x2 > (1.0 - margin) * W or y2 > (1.0 - margin) * H


def _peak_moat(hm: np.ndarray, pos: np.ndarray, r: int, margin: float) -> None:
    """Carve a moat around every positive (peak) cell: each NON-peak cell within
    L∞ radius `r` is depressed to <= (peak_value - margin), clamped >= 0.

    Why: PeakSuppress keeps the *unique* window-max — two byte-equal adjacent
    cells both pass (each IS the max in its own window), giving "two touching
    detections". The natural Gaussian target only differs by exp(-1/(2σ²)) ≈ 0.97
    between a peak and its neighbour, and a soft target like the ellipse dome can
    be high right next to the peak too — so a converged model legitimately
    reproduces a near-tie. Forcing a >= margin gap into the GT means the fitted
    model lands the neighbour comfortably below the peak → PeakSuppress fully
    zeros it. Peaks don't depress each other (two genuinely-adjacent objects keep
    both peaks); a non-peak cell is clamped by whichever in-range peak leaves it
    lowest. Mutates `hm` in place. `r` should be the model's peak_kernel // 2.
    """
    if margin <= 0.0 or r < 1:
        return
    H, W = hm.shape
    ys, xs = np.nonzero(pos)
    is_peak = pos > 0
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        cap = max(0.0, float(hm[cy, cx]) - margin)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        win = hm[y0:y1, x0:x1]
        free = ~is_peak[y0:y1, x0:x1]
        np.minimum(win, np.where(free, cap, win), out=win)


def _apply_peak_moat(hm: np.ndarray, pos: np.ndarray, cfg: object) -> None:
    m = float(getattr(cfg, "hm_peak_moat", 0.0) or 0.0)
    if m > 0.0:
        _peak_moat(hm, pos, int(getattr(cfg, "peak_kernel", 5)) // 2, m)


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
            shape = getattr(cfg, "hm_target", "gaussian")
            if shape == "ellipse" and _box_near_edge(x1, y1, x2, y2, W, H, float(getattr(cfg, "hm_ellipse_edge_margin", 0.1))):
                shape = "gaussian"
            _draw_obj_blob(hm, ix, iy, bw, bh, 0.0, s, min_sigma, getattr(cfg, "hm_blob_frac", 0.0), shape)
            cxy[0, iy, ix] = cx_g - ix
            cxy[1, iy, ix] = cy_g - iy
            wh[0, iy, ix] = bw / W
            wh[1, iy, ix] = bh / H
            pos[iy, ix] = 1.0
        _apply_peak_moat(hm, pos, cfg)

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
            shape = getattr(cfg, "hm_target", "gaussian")
            if shape == "ellipse" and _box_near_edge(x1, y1, x2, y2, W, H, float(getattr(cfg, "hm_ellipse_edge_margin", 0.1))):
                shape = "gaussian"
            _draw_obj_blob(hm, ix, iy, bw, bh, 0.0, s, min_sigma, getattr(cfg, "hm_blob_frac", 0.0), shape)
            cx_cell = (ix + 0.5) * s
            cy_cell = (iy + 0.5) * s
            ltrb[0, iy, ix] = float(np.clip((cx_cell - x1) / W, 0.0, 1.0))
            ltrb[1, iy, ix] = float(np.clip((cy_cell - y1) / H, 0.0, 1.0))
            ltrb[2, iy, ix] = float(np.clip((x2 - cx_cell) / W, 0.0, 1.0))
            ltrb[3, iy, ix] = float(np.clip((y2 - cy_cell) / H, 0.0, 1.0))
            pos[iy, ix] = 1.0
        _apply_peak_moat(hm, pos, cfg)

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
    """Encode list of OBBs into dense GT for the YOLO-style 6-channel head.

    Output reg channels (5 total, all sigmoid range [0, 1]):
        cx_offset = (cx_px - ix*stride) / stride       in [0, 1]  cell-relative
        cy_offset = (cy_px - iy*stride) / stride       in [0, 1]
        w_norm    = w_obb / img_w                      in [0, 1]
        h_norm    = h_obb / img_h                      in [0, 1]
        theta_norm = theta / π                         in [0, 1)  (sigmoid output × π = θ)

    NO enclosing-AABB representation anywhere. Direct (cx, cy, w, h, θ).
    Wrap-handling is the loss's job (ProbIoU treats each box as a 2D Gaussian
    and is wrap-continuous via the rotated-box covariance).

    obbs: [N, 5] of (cx, cy, w, h, theta) in image-pixel units, θ in radians ∈ [0, π).
    Returns dict with:
      hm         : [1, H', W']  circular Gaussian heatmap (cls)
      obb        : [5, H', W']  (cx_off, cy_off, w_norm, h_norm, θ_norm) at GT cells
      pos        : [1, H', W']  1.0 at GT center cells
      angle_mask : [1, H', W']  1.0 at non-round positives (aspect > thresh)
    """
    H, W = cfg.img_h, cfg.img_w
    s = cfg.stride
    Hp, Wp = H // s, W // s
    hm = np.zeros((Hp, Wp), dtype=np.float32)
    reg = np.zeros((5, Hp, Wp), dtype=np.float32)
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
            shape = getattr(cfg, "hm_target", "gaussian")
            if shape == "ellipse":
                # edge check on the OBB's axis-aligned envelope
                ct, st = math.cos(float(theta)), math.sin(float(theta))
                aabb_w = abs(bw * ct) + abs(bh * st)
                aabb_h = abs(bw * st) + abs(bh * ct)
                ex1, ey1 = float(cx_px) - aabb_w / 2, float(cy_px) - aabb_h / 2
                ex2, ey2 = float(cx_px) + aabb_w / 2, float(cy_px) + aabb_h / 2
                if _box_near_edge(ex1, ey1, ex2, ey2, W, H, float(getattr(cfg, "hm_ellipse_edge_margin", 0.1))):
                    shape = "gaussian"
            _draw_obj_blob(hm, ix, iy, bw, bh, float(theta), s, min_sigma, getattr(cfg, "hm_blob_frac", 0.0), shape)
            reg[0, iy, ix] = float(np.clip(cx_g - ix, 0.0, 1.0))
            reg[1, iy, ix] = float(np.clip(cy_g - iy, 0.0, 1.0))
            reg[2, iy, ix] = float(np.clip(bw / W, 0.0, 1.0))
            reg[3, iy, ix] = float(np.clip(bh / H, 0.0, 1.0))
            theta_wrapped = float(theta) % math.pi
            reg[4, iy, ix] = float(np.clip(theta_wrapped / math.pi, 0.0, 0.99999))
            pos[iy, ix] = 1.0
            if max(bw, bh) / max(min(bw, bh), 1e-6) >= aspect_round_thresh:
                ang_mask[iy, ix] = 1.0
        _apply_peak_moat(hm, pos, cfg)

    return {
        "hm": torch.from_numpy(hm).unsqueeze(0),
        "obb": torch.from_numpy(reg),
        "pos": torch.from_numpy(pos).unsqueeze(0),
        "angle_mask": torch.from_numpy(ang_mask).unsqueeze(0),
    }


def encode_targets_seg(cfg: object, obbs: np.ndarray | None = None,
                       masks: list | None = None) -> dict[str, torch.Tensor]:
    """Dense per-pixel dome target for the segmentation head (`bbox-*-seg`).

    Rendered at (cfg.img_h, cfg.img_w) // seg_stride (`cfg.seg_stride`, default 1).

    `cfg.seg_dome_ramp_px` (default 0) picks the dome PROFILE:
      0  : proportional ramp — 1.0 at the deepest interior point, ~linear to 0 at the
           boundary, spread over the object's whole inscribed radius. Single-peaked.
      >0 : FLAT-TOP plateau — 1.0 across the interior, linear falloff to 0 only over the
           last ~ramp_px cells (at seg res) before the boundary. The interior carries no
           gradient; the thin edge dropoff is the only ramp. Trivial to learn ("am I inside
           the egg") and `seg_fg_thresh` becomes ~irrelevant (any cut in (0,1) gives the
           same footprint). Still hits exactly 0 at the boundary, so touching-but-not-
           overlapping objects keep a 0 valley between them and separate cleanly.

    `cfg.seg_instance_gap_px` (default 0, masks source only): erode each instance mask by
      ~gap/2 px before the distance transform → the GT has a guaranteed ≥gap-wide 0-corridor
      between any two touching instances, so the model learns to keep them apart and a plain
      threshold + connected-components decode (no watershed) separates them. Costs a ~gap/2-px
      shrink of each object's apparent area (small vs the edge ramp; bias is toward under-).

    Sources, in priority order:
      - `masks`: list of HxW binary instance masks (at image res). Each → (optional erode) →
        its L2 distance-transform, then `dt/dt.max()` (proportional) or `clip(dt/ramp_px,0,1)`
        (flat-top). Handles arbitrary convex shapes; max-aggregated across instances ⇒
        exactly 0 between two touching ones (no bleed across the contact line).
      - `obbs`: [N,5] of (cx,cy,w,h,θ) in image px ⇒ the elliptical dome
        (`_draw_rotated_dome`, same two profiles) per box, at seg res. The fallback when
        only OBB sidecars exist (an egg's egg-ellipse is a decent stand-in for its mask).
    Returns {"dome": Tensor[1, Hd, Wd]} in [0,1].
    """
    st = max(1, int(getattr(cfg, "seg_stride", 1)))
    ramp = float(getattr(cfg, "seg_dome_ramp_px", 0.0) or 0.0)
    ramp_d = max(ramp / st, 1e-3) if ramp > 0.0 else 0.0     # ramp width in seg-res cells
    gap = float(getattr(cfg, "seg_instance_gap_px", 0.0) or 0.0)
    gap_r = int(round(gap / st / 2.0)) if gap > 0.0 else 0   # erosion radius in seg-res cells
    Hd, Wd = int(cfg.img_h) // st, int(cfg.img_w) // st
    dome = np.zeros((Hd, Wd), dtype=np.float32)
    if masks is not None and len(masks) > 0:
        import cv2 as _cv2
        gap_kernel = np.ones((2 * gap_r + 1, 2 * gap_r + 1), np.uint8) if gap_r > 0 else None
        for mk in masks:
            m = (np.asarray(mk) > 0).astype(np.uint8)
            if m.shape != (Hd, Wd):
                m = _cv2.resize(m, (Wd, Hd), interpolation=_cv2.INTER_NEAREST)
            m = np.ascontiguousarray(m)
            if gap_kernel is not None:
                m = _cv2.erode(m, gap_kernel)
            if int(m.sum()) == 0:
                continue
            dt = _cv2.distanceTransform(m, _cv2.DIST_L2, 5)
            if ramp_d > 0.0:
                np.maximum(dome, np.clip(dt / ramp_d, 0.0, 1.0).astype(np.float32), out=dome)
            else:
                mx = float(dt.max())
                if mx > 0.0:
                    np.maximum(dome, (dt / mx).astype(np.float32), out=dome)
    elif obbs is not None and len(obbs) > 0:
        for cx, cy, w, h, theta in obbs:
            if float(w) < 1.0 or float(h) < 1.0:
                continue
            _draw_rotated_dome(dome, int(round(float(cx) / st)), int(round(float(cy) / st)),
                               (float(w) * 0.5) / st, (float(h) * 0.5) / st, float(theta), ramp_px=ramp_d)
    return {"dome": torch.from_numpy(dome).unsqueeze(0)}


def collate_targets(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}
