from __future__ import annotations

import math

import numpy as np
import torch

from opndet.config import ModelConfig


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


def encode_targets(
    boxes: np.ndarray,
    cfg: ModelConfig,
    min_sigma: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Encode list of (x1,y1,x2,y2) pixel boxes into dense GT tensors.

    Returns dict with:
      hm   : [1, H', W']     gaussian heatmap targets in [0,1]
      cxy  : [2, H', W']     cell-relative center offset GT (only valid where pos)
      wh   : [2, H', W']     image-normalized w,h GT (only valid where pos)
      pos  : [1, H', W']     1.0 at positive cells (peak), 0 elsewhere — for size loss masking
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

    return {
        "hm": torch.from_numpy(hm).unsqueeze(0),
        "cxy": torch.from_numpy(cxy),
        "wh": torch.from_numpy(wh),
        "pos": torch.from_numpy(pos).unsqueeze(0),
    }


def collate_targets(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}
