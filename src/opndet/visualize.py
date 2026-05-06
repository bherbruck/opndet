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


def _prior_overlay(rgb: np.ndarray, prior_full: np.ndarray, max_alpha: float = 0.5) -> np.ndarray:
    """Blend a JET-colormapped prior heatmap onto the RGB image.

    Per-pixel alpha = clip(prior, 0, 1) * max_alpha. Zero-prior pixels keep
    the original RGB unchanged; hot pixels get up to max_alpha colored tint.
    Boxes are drawn AFTER this so they always sit on top of the overlay.
    """
    p = np.clip(prior_full.astype(np.float32), 0.0, 1.0)
    if p.max() <= 0.0:
        return rgb
    h = (p * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    alpha = (p * max_alpha)[..., None]
    out = rgb.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_prior_trails(
    rgb: np.ndarray,
    prior_full: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    min_amp: float = 0.10,
) -> np.ndarray:
    """For each connected blob of prior > min_amp, draw a single AA line from
    the trail's TAIL (oldest visible position, farthest from peak) to its
    HEAD (current position, brightest pixel). One line per trail; head gets
    a slightly larger dot than the tail to show motion direction.

    No attempt to recover individual stamp positions — overlapping Gaussians
    merge into one elongated blob, and the head→tail vector along that blob
    is what conveys the actual motion.
    """
    p = prior_full.astype(np.float32)
    if p.max() <= min_amp:
        return rgb
    blob_mask = (p > min_amp).astype(np.uint8)
    n_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(blob_mask, connectivity=8)
    if n_labels <= 1:
        return rgb
    out = rgb.copy()
    for label_id in range(1, n_labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
            continue
        # Head = argmax of prior amplitude within this blob
        amps = p[ys, xs]
        head_i = int(np.argmax(amps))
        head_x, head_y = int(xs[head_i]), int(ys[head_i])
        # Tail = pixel in blob farthest from head
        d2 = (xs.astype(np.int64) - head_x) ** 2 + (ys.astype(np.int64) - head_y) ** 2
        tail_i = int(np.argmax(d2))
        tail_x, tail_y = int(xs[tail_i]), int(ys[tail_i])
        if (tail_x, tail_y) != (head_x, head_y):
            cv2.line(out, (tail_x, tail_y), (head_x, head_y), color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (tail_x, tail_y), 1, color, -1)
        cv2.circle(out, (head_x, head_y), 2, color, -1)
    return out


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
    # Cap dets per image — same rationale as eval.py's max_dets_per_image: an
    # untrained model (esp 4-ch with warm priors) can decode hundreds of cells
    # per image at low thresholds, and rendering thousands of cv2.rectangle +
    # putText calls is dominant cost on cold-start vis. Top-K by score.
    max_dets = 200
    for i in range(len(dets_per)):
        if len(dets_per[i]) > max_dets:
            dets_per[i] = sorted(dets_per[i], key=lambda d: -d.score)[:max_dets]

    has_prior = imgs.shape[1] >= 4
    rendered = []
    for i in range(imgs.shape[0]):
        rgb = _denorm(imgs[i])
        if has_prior:
            prior_full = imgs[i, 3].detach().cpu().numpy()
            rgb = _prior_overlay(rgb, prior_full, max_alpha=0.5)
            rgb = _draw_prior_trails(rgb, prior_full)
        # GT in solid magenta — visually distinct from the colored pred gradient.
        rgb = _draw(rgb, gt_boxes[i], color=(255, 0, 255), thick=2)
        for d in dets_per[i]:
            x1, y1 = int(round(d.x1)), int(round(d.y1))
            x2, y2 = int(round(d.x2)), int(round(d.y2))
            _draw_pred(rgb, x1, y1, x2, y2, float(d.score))
        rendered.append(rgb.transpose(2, 0, 1))
    return np.stack(rendered, axis=0)
