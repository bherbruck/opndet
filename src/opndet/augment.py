from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugConfig:
    # photometric
    brightness: float = 0.4         # uniform shift in [-x, +x] of [0,1] range
    contrast: float = 0.4           # multiplicative in [1-x, 1+x]
    gamma: tuple[float, float] = (0.6, 1.6)
    hue: int = 20                   # degrees
    saturation: float = 0.5
    grayscale_prob: float = 0.2
    blur_prob: float = 0.1
    noise_sigma: float = 0.02
    # geometric
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate90_prob: float = 0.5
    scale_jitter: tuple[float, float] = (0.7, 1.3)
    translate_frac: float = 0.1
    # mosaic (handled in dataset, not in aug fn — but config flag lives here)
    mosaic_prob: float = 0.0
    # cutout / random erase
    cutout_prob: float = 0.0        # chance of applying cutout to an image
    cutout_count: int = 3           # how many holes per application
    cutout_size_frac: tuple[float, float] = (0.05, 0.20)  # hole side as frac of img dim
    # bbox visibility — drop boxes with <min_visible_frac of original area visible
    min_visible_frac: float = 0.5
    # hard-negative paste (mined via `opndet mine-negatives`)
    hard_negative_pool: str | None = None      # dir of patch pngs OR mining out_dir
    hard_negative_prob: float = 0.0            # per-image inject probability
    hard_negative_count: int = 1               # patches per injection
    hard_negative_max_tries: int = 16          # how hard to try to find a non-GT spot
    # composite
    enabled: bool = True


def _photometric(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    img = img.astype(np.float32) / 255.0

    if cfg.brightness > 0:
        b = rng.uniform(-cfg.brightness, cfg.brightness)
        img = np.clip(img + b, 0, 1)

    if cfg.contrast > 0:
        c = rng.uniform(1 - cfg.contrast, 1 + cfg.contrast)
        m = img.mean()
        img = np.clip((img - m) * c + m, 0, 1)

    if cfg.gamma is not None:
        g = rng.uniform(*cfg.gamma)
        img = np.clip(img ** g, 0, 1)

    img = (img * 255).astype(np.uint8)

    if cfg.hue > 0 or cfg.saturation > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        if cfg.hue > 0:
            dh = int(rng.integers(-cfg.hue, cfg.hue + 1))
            hsv[..., 0] = (hsv[..., 0] + dh) % 180
        if cfg.saturation > 0:
            ds = rng.uniform(1 - cfg.saturation, 1 + cfg.saturation)
            hsv[..., 1] = np.clip(hsv[..., 1] * ds, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if rng.random() < cfg.grayscale_prob:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if rng.random() < cfg.blur_prob:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    if cfg.noise_sigma > 0:
        noise = rng.normal(0, cfg.noise_sigma * 255, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def _cutout(img: np.ndarray, boxes: np.ndarray, cfg: AugConfig, rng: np.random.Generator,
            obbs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Paste random rectangles of mean-gray over the image, then drop boxes whose
    visible area falls below cfg.min_visible_frac. Visibility is computed
    analytically as 1 - sum(box-hole intersections) / box_area; assumes holes
    don't overlap each other significantly within any single box (true for the
    typical small N_holes regime).

    obbs: optional [N, 5] (cx, cy, w, h, θ) OBB GT — same N as boxes. The keep
    mask applied to boxes is also applied to obbs (since obbs[i] corresponds to
    boxes[i]). Cutout occludes pixels but doesn't change orientation, so OBB
    angle/dim values pass through unchanged for surviving rows.
    """
    h, w = img.shape[:2]
    pad_value = 114
    holes = np.zeros((cfg.cutout_count, 4), dtype=np.float32)
    for k in range(cfg.cutout_count):
        sf = rng.uniform(*cfg.cutout_size_frac)
        ch = max(1, int(round(sf * h)))
        cw = max(1, int(round(sf * w)))
        y0 = int(rng.integers(0, max(1, h - ch)))
        x0 = int(rng.integers(0, max(1, w - cw)))
        img[y0:y0 + ch, x0:x0 + cw] = pad_value
        holes[k] = (x0, y0, x0 + cw, y0 + ch)

    if boxes.shape[0] == 0:
        return img, boxes, obbs

    bx = boxes
    bw = (bx[:, 2] - bx[:, 0]).clip(min=0)
    bh = (bx[:, 3] - bx[:, 1]).clip(min=0)
    box_area = bw * bh

    # pairwise intersection area: [N, K]
    ix1 = np.maximum(bx[:, None, 0], holes[None, :, 0])
    iy1 = np.maximum(bx[:, None, 1], holes[None, :, 1])
    ix2 = np.minimum(bx[:, None, 2], holes[None, :, 2])
    iy2 = np.minimum(bx[:, None, 3], holes[None, :, 3])
    inter = (ix2 - ix1).clip(min=0) * (iy2 - iy1).clip(min=0)
    obscured = inter.sum(axis=1)
    visible_frac = np.where(box_area > 0, 1.0 - obscured / np.maximum(box_area, 1e-9), 1.0)
    keep = visible_frac >= cfg.min_visible_frac
    obbs_out = obbs[keep] if obbs is not None and obbs.shape[0] == boxes.shape[0] else obbs
    return img, boxes[keep], obbs_out


def _geometric(img: np.ndarray, boxes: np.ndarray, cfg: AugConfig, rng: np.random.Generator,
               obbs: np.ndarray | None = None,
               mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Geometric augmentation. Transforms boxes AND OBBs through hflip/vflip/rotate90,
    and (if given) the seg instance-label-map `mask` in lockstep (NEAREST, so instance
    ids don't get interpolated together).

    OBB θ transforms (rectangle has π-symmetry, so we wrap mod π):
      - hflip: θ → (π - θ) mod π   (mirror reflects major-axis direction)
      - vflip: θ → (π - θ) mod π   (same as hflip — mirror is the same up to rect symmetry)
      - rot90 CCW (k times): θ → (θ + k·π/2) mod π
    OBB w, h are unchanged by all three (rotation/mirror preserves intrinsic dims).
    """
    import math
    h, w = img.shape[:2]

    if rng.random() < cfg.hflip_prob:
        img = img[:, ::-1].copy()
        if mask is not None:
            mask = mask[:, ::-1].copy()
        if boxes.shape[0]:
            x1 = w - boxes[:, 2]
            x2 = w - boxes[:, 0]
            boxes = np.stack([x1, boxes[:, 1], x2, boxes[:, 3]], axis=1)
        if obbs is not None and obbs.shape[0]:
            obbs = obbs.copy()
            obbs[:, 0] = w - obbs[:, 0]
            obbs[:, 4] = (math.pi - obbs[:, 4]) % math.pi

    if rng.random() < cfg.vflip_prob:
        img = img[::-1].copy()
        if mask is not None:
            mask = mask[::-1].copy()
        if boxes.shape[0]:
            y1 = h - boxes[:, 3]
            y2 = h - boxes[:, 1]
            boxes = np.stack([boxes[:, 0], y1, boxes[:, 2], y2], axis=1)
        if obbs is not None and obbs.shape[0]:
            obbs = obbs.copy()
            obbs[:, 1] = h - obbs[:, 1]
            obbs[:, 4] = (math.pi - obbs[:, 4]) % math.pi

    if rng.random() < cfg.rotate90_prob:
        k = int(rng.choice([1, 2, 3]))
        img = np.rot90(img, k=k).copy()
        if mask is not None:
            mask = np.rot90(mask, k=k).copy()
        cur_w, cur_h = w, h
        if boxes.shape[0]:
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
        obb_cx = obb_cy = obb_theta = None
        if obbs is not None and obbs.shape[0]:
            obb_cx = obbs[:, 0].copy()
            obb_cy = obbs[:, 1].copy()
            obb_theta = obbs[:, 4].copy()
        for _ in range(k):
            if boxes.shape[0]:
                cx, cy = cy, cur_w - cx
                bw, bh = bh, bw
            if obb_cx is not None:
                obb_cx, obb_cy = obb_cy, cur_w - obb_cx
                obb_theta = (obb_theta + math.pi / 2) % math.pi
            cur_w, cur_h = cur_h, cur_w
        if boxes.shape[0]:
            boxes = np.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], axis=1)
        if obb_cx is not None:
            obbs = obbs.copy()
            obbs[:, 0] = obb_cx
            obbs[:, 1] = obb_cy
            obbs[:, 4] = obb_theta
        w, h = cur_w, cur_h

    # --- scale + translate (affine, NO rotation) -----------------------------
    # OBB-safe: a uniform scale + translation leaves θ unchanged; w,h scale by s;
    # centers shift. Boxes that end up (mostly) outside the frame are dropped via
    # min_visible_frac (clipped-area / original-area). No-op when scale_jitter is
    # (1,1) AND translate_frac is 0. (Unlike rotate90, this one is fine for OBB —
    # it's why the OBB presets can opt into it without the θ-augmenter caveat.)
    lo, hi = cfg.scale_jitter
    do_scale = lo != 1.0 or hi != 1.0
    do_trans = cfg.translate_frac > 0.0
    if do_scale or do_trans:
        ch, cw = img.shape[:2]
        s = float(rng.uniform(lo, hi)) if do_scale else 1.0
        tx = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * cw) if do_trans else 0.0
        ty = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * ch) if do_trans else 0.0
        if s != 1.0 or tx != 0.0 or ty != 0.0:
            ccx, ccy = cw * 0.5, ch * 0.5
            M = np.array([[s, 0.0, ccx - s * ccx + tx],
                          [0.0, s, ccy - s * ccy + ty]], dtype=np.float32)
            bval = 114 if img.dtype == np.uint8 else 114.0 / 255.0
            img = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(bval, bval, bval))
            if mask is not None:
                # uint16 for warpAffine (CV_32S isn't supported); instance ids fit easily
                mask = cv2.warpAffine(mask.astype(np.uint16), M, (cw, ch), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int32)
            keep = None
            if boxes.shape[0]:
                bx = boxes.astype(np.float64).copy()
                bx[:, [0, 2]] = s * (bx[:, [0, 2]] - ccx) + ccx + tx
                bx[:, [1, 3]] = s * (bx[:, [1, 3]] - ccy) + ccy + ty
                # area of the *transformed* box (pre-clip) — i.e. what "fully visible"
                # means after the scale. Comparing clipped-area to this (not to the
                # pre-scale area) correctly keeps a zoomed-OUT box (smaller but whole)
                # and drops a zoomed-IN box that slid mostly off the frame.
                pre_area = (bx[:, 2] - bx[:, 0]).clip(min=0) * (bx[:, 3] - bx[:, 1]).clip(min=0)
                cl = bx.copy()
                cl[:, [0, 2]] = cl[:, [0, 2]].clip(0, cw)
                cl[:, [1, 3]] = cl[:, [1, 3]].clip(0, ch)
                new_area = (cl[:, 2] - cl[:, 0]).clip(min=0) * (cl[:, 3] - cl[:, 1]).clip(min=0)
                keep = (pre_area <= 1e-9) | (new_area / np.maximum(pre_area, 1e-9) >= cfg.min_visible_frac)
                boxes = cl[keep].astype(np.float32)
            if obbs is not None and obbs.shape[0]:
                ob = obbs.astype(np.float64).copy()
                ob[:, 0] = s * (ob[:, 0] - ccx) + ccx + tx
                ob[:, 1] = s * (ob[:, 1] - ccy) + ccy + ty
                ob[:, 2] *= s
                ob[:, 3] *= s
                # θ (col 4) unchanged. drop in sync with `boxes` (= the OBB's AABB
                # envelope for OBB models → same N) when available.
                if keep is not None and keep.shape[0] == ob.shape[0]:
                    ob = ob[keep]
                obbs = ob.astype(np.float32)

    return img, boxes, obbs, mask


def _hard_negative_paste(
    img: np.ndarray, boxes: np.ndarray, pool: list[np.ndarray],
    cfg: AugConfig, rng: np.random.Generator,
) -> np.ndarray:
    """Paste random pool patches into non-GT regions of img.

    No labels are added: the pasted region is explicit "DON'T FIRE" supervision.
    Each paste hunts for a window that doesn't overlap any GT bbox; if no spot
    is found in cfg.hard_negative_max_tries attempts, the paste is skipped
    silently (image fully covered by GTs is the only realistic skip case).
    """
    if not pool:
        return img
    H, W = img.shape[:2]
    for _ in range(int(cfg.hard_negative_count)):
        patch = pool[int(rng.integers(0, len(pool)))]
        ph, pw = patch.shape[:2]
        if ph >= H or pw >= W:
            continue
        spot = None
        for _t in range(int(cfg.hard_negative_max_tries)):
            y = int(rng.integers(0, H - ph + 1))
            x = int(rng.integers(0, W - pw + 1))
            if boxes.shape[0] == 0:
                spot = (y, x); break
            # rejection: any GT bbox overlap with the paste rect
            px1, py1, px2, py2 = x, y, x + pw, y + ph
            ix1 = np.maximum(boxes[:, 0], px1)
            iy1 = np.maximum(boxes[:, 1], py1)
            ix2 = np.minimum(boxes[:, 2], px2)
            iy2 = np.minimum(boxes[:, 3], py2)
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            if (inter <= 0).all():
                spot = (y, x); break
        if spot is None:
            continue
        y, x = spot
        # patch is BGR (loaded via cv2.imread); training img is RGB. Convert.
        if patch.ndim == 3 and patch.shape[2] == 3:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        else:
            patch_rgb = patch
        img[y:y + ph, x:x + pw] = patch_rgb
    return img


def make_augment(cfg: AugConfig, hn_pool: list[np.ndarray] | None = None):
    if not cfg.enabled:
        return None

    pool = hn_pool or []

    def aug(img: np.ndarray, boxes: np.ndarray, obbs: np.ndarray | None = None,
            mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        rng = np.random.default_rng()
        img = _photometric(img, cfg, rng)
        img, boxes, obbs, mask = _geometric(img, boxes, cfg, rng, obbs=obbs, mask=mask)
        if cfg.cutout_prob > 0 and rng.random() < cfg.cutout_prob:
            # cutout only blanks the IMAGE (occlusion robustness) — the seg label map keeps
            # the object there on purpose (teaches "object is here even if locally occluded").
            img, boxes, obbs = _cutout(img, boxes, cfg, rng, obbs=obbs)
        if pool and cfg.hard_negative_prob > 0 and rng.random() < cfg.hard_negative_prob:
            img = _hard_negative_paste(img, boxes, pool, cfg, rng)
        return img, boxes, obbs, mask

    return aug
