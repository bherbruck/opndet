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


def _box_orig_areas(boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    w = (boxes[:, 2] - boxes[:, 0]).clip(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clip(min=0)
    return w * h


def _filter_visible(boxes: np.ndarray, orig_areas: np.ndarray, min_frac: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return boxes
    new_w = (boxes[:, 2] - boxes[:, 0]).clip(min=0)
    new_h = (boxes[:, 3] - boxes[:, 1]).clip(min=0)
    new_area = new_w * new_h
    keep = (orig_areas <= 0) | (new_area / np.maximum(orig_areas, 1e-9) >= min_frac)
    return boxes[keep]


def _cutout(img: np.ndarray, boxes: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Paste random rectangles of mean-gray over the image, then drop boxes whose
    visible area falls below cfg.min_visible_frac.
    """
    h, w = img.shape[:2]
    if boxes.shape[0] > 0:
        orig_areas = _box_orig_areas(boxes)
        # track per-box "visible" pixel count by maintaining a binary visibility mask per box
        vis_mask = np.ones((boxes.shape[0], h, w), dtype=bool)
    else:
        orig_areas = np.zeros(0, dtype=np.float32)
        vis_mask = None

    pad_value = 114
    for _ in range(cfg.cutout_count):
        sf = rng.uniform(*cfg.cutout_size_frac)
        ch = max(1, int(round(sf * h)))
        cw = max(1, int(round(sf * w)))
        y0 = int(rng.integers(0, max(1, h - ch)))
        x0 = int(rng.integers(0, max(1, w - cw)))
        img[y0:y0 + ch, x0:x0 + cw] = pad_value
        if vis_mask is not None:
            for i in range(boxes.shape[0]):
                bx1, by1, bx2, by2 = [int(round(v)) for v in boxes[i]]
                ix0 = max(bx1, x0); iy0 = max(by1, y0)
                ix1 = min(bx2, x0 + cw); iy1 = min(by2, y0 + ch)
                if ix1 > ix0 and iy1 > iy0:
                    vis_mask[i, iy0:iy1, ix0:ix1] = False

    if vis_mask is None or boxes.shape[0] == 0:
        return img, boxes
    keep_idx = []
    for i in range(boxes.shape[0]):
        bx1, by1, bx2, by2 = [int(round(v)) for v in boxes[i]]
        bx1 = max(0, min(w, bx1)); by1 = max(0, min(h, by1))
        bx2 = max(0, min(w, bx2)); by2 = max(0, min(h, by2))
        if bx2 <= bx1 or by2 <= by1:
            continue
        visible = vis_mask[i, by1:by2, bx1:bx2].sum()
        orig = orig_areas[i]
        if orig <= 0 or visible / orig >= cfg.min_visible_frac:
            keep_idx.append(i)
    return img, boxes[keep_idx] if keep_idx else np.zeros((0, 4), dtype=np.float32)


def _geometric(img: np.ndarray, boxes: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]

    if rng.random() < cfg.hflip_prob:
        img = img[:, ::-1].copy()
        if boxes.shape[0]:
            x1 = w - boxes[:, 2]
            x2 = w - boxes[:, 0]
            boxes = np.stack([x1, boxes[:, 1], x2, boxes[:, 3]], axis=1)

    if rng.random() < cfg.vflip_prob:
        img = img[::-1].copy()
        if boxes.shape[0]:
            y1 = h - boxes[:, 3]
            y2 = h - boxes[:, 1]
            boxes = np.stack([boxes[:, 0], y1, boxes[:, 2], y2], axis=1)

    if rng.random() < cfg.rotate90_prob:
        k = int(rng.choice([1, 2, 3]))
        img = np.rot90(img, k=k).copy()
        if boxes.shape[0]:
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            for _ in range(k):
                cx, cy = cy, w - cx
                bw, bh = bh, bw
                w, h = h, w
            boxes = np.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], axis=1)
        else:
            for _ in range(k):
                w, h = h, w

    return img, boxes


def make_augment(cfg: AugConfig):
    if not cfg.enabled:
        return None

    def aug(img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        img = _photometric(img, cfg, rng)
        img, boxes = _geometric(img, boxes, cfg, rng)
        if cfg.cutout_prob > 0 and rng.random() < cfg.cutout_prob:
            img, boxes = _cutout(img, boxes, cfg, rng)
        return img, boxes

    return aug
