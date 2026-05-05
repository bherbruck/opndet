from __future__ import annotations

import cv2
import numpy as np
import torch

from opndet.decode import decode_batch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm(img_t: torch.Tensor) -> np.ndarray:
    """[C,H,W] normalized tensor -> [H,W,3] uint8 RGB. Slices to first 3
    channels so 4-ch (temporal) tensors render their RGB only."""
    if img_t.shape[0] > 3:
        img_t = img_t[:3]
    arr = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _draw(img: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], thick: int = 1) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    return out


def _conf_color(conf: float) -> tuple[int, int, int]:
    """Green at high conf, yellow at mid, red at low. RGB."""
    if conf >= 0.7:
        return (0, 255, 0)            # bright green — confident TP
    if conf >= 0.5:
        return (180, 255, 0)          # yellow-green
    if conf >= 0.3:
        return (255, 220, 0)          # yellow
    if conf >= 0.15:
        return (255, 130, 0)          # orange
    return (255, 30, 30)               # red — low-conf extra


def _draw_pred(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, conf: float) -> None:
    color = _conf_color(conf)
    thick = 2 if conf >= 0.5 else 1
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    label = f"{conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    ty = y1 - 2 if y1 - 2 - th >= 0 else y1 + th + 2
    cv2.rectangle(img, (x1, ty - th - 1), (x1 + tw + 2, ty + 1), color, -1)
    cv2.putText(img, label, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)


@torch.no_grad()
def render_predictions(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    gt_boxes: list[np.ndarray],
    img_h: int,
    img_w: int,
    stride: int,
    threshold: float = 0.05,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Run model on a batch of normalized images; return [N, 3, H, W] uint8 grid-ready array.

    Pred boxes are color-coded by confidence (green=high, red=low) with a numeric label so
    you can visually identify "over-detection at low confidence" issues. GT boxes drawn in
    bright red, thickness 2. Default threshold lowered to 0.05 to surface borderline preds.
    """
    model.eval()
    out = model(imgs.to(device))
    out_t = out["output"] if isinstance(out, dict) else out
    out_np = out_t.cpu().numpy()
    dets_per = decode_batch(out_np, img_h, img_w, stride, threshold=threshold)

    rendered = []
    for i in range(imgs.shape[0]):
        rgb = _denorm(imgs[i])
        # GT in solid magenta — visually distinct from the colored pred gradient.
        rgb = _draw(rgb, gt_boxes[i], color=(255, 0, 255), thick=2)
        for d in dets_per[i]:
            x1, y1 = int(round(d.x1)), int(round(d.y1))
            x2, y2 = int(round(d.x2)), int(round(d.y2))
            _draw_pred(rgb, x1, y1, x2, y2, float(d.score))
        rendered.append(rgb.transpose(2, 0, 1))
    return np.stack(rendered, axis=0)
