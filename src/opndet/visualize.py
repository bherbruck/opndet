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


def _draw_prior_trails_from_trails(
    rgb: np.ndarray,
    trails: list,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw stamp dots + connecting polyline per object directly from synth-
    reported stamp positions. trails is list[list[(cx, cy)]] in input pixel
    coords — outer list is per-object, inner list is the sequence of stamps
    for that object's trail (ordered oldest→newest).

    No heatmap reconstruction, no nearest-neighbor heuristics — the synth
    knows exactly which stamps belong to which trail. One line per stamp
    pair within a trail (N stamps → N-1 line segments). Head dot (newest)
    larger than tail dots.
    """
    if not trails:
        return rgb
    out = rgb.copy()
    for trail in trails:
        if len(trail) == 0:
            continue
        pts = [(int(round(cx)), int(round(cy))) for (cx, cy) in trail]
        # Lines connecting sequential stamps within this trail
        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i + 1], color, thickness, lineType=cv2.LINE_AA)
        # Dots: small for older stamps, larger for the head (last in list = k=1 = newest)
        for pt in pts[:-1]:
            cv2.circle(out, pt, 1, color, -1)
        cv2.circle(out, pts[-1], 2, color, -1)
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


def save_prior_overlay_png(prior_full: np.ndarray, path: str) -> None:
    """Save the prior heatmap as a JET-colored PNG with per-pixel alpha
    proportional to the prior amplitude. Dashboard's CSS opacity slider then
    composites this on top of the RGB base — fully transparent where
    prior=0, full opacity where prior=1.
    """
    p = np.clip(prior_full.astype(np.float32), 0.0, 1.0)
    h = (p * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(h, cv2.COLORMAP_JET)  # BGR
    alpha = (p * 255).astype(np.uint8)
    bgra = np.dstack([color_bgr, alpha])
    cv2.imwrite(str(path), bgra)


@torch.no_grad()
def save_layered_vis(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    gt_boxes: list[np.ndarray],
    img_h: int,
    img_w: int,
    stride: int,
    out_sub: object,           # Path
    db: object,                # MetricsDB or None
    tag: str,
    ep: int,
    threshold: float = 0.05,
    device: torch.device | str = "cpu",
    trails_per: list | None = None,
) -> None:
    """Save per-sample LAYERED vis components for the dashboard:
      - sample_<i>_rgb.png        : clean denormalized RGB (no boxes/overlay)
      - sample_<i>_prior.png      : transparent JET prior overlay (4-ch only)
      - DB rows: image, overlay (prior_heat), gt boxes, pred boxes (with scores)

    Trails (synth-reported per-object stamp positions) are saved as JSON-meta
    rows under boxes(kind='trail') so the viewer can draw the polyline on top.
    """
    import json
    from opndet.decode import decode_batch
    out_sub.mkdir(parents=True, exist_ok=True)
    model.eval()
    out = model(imgs.to(device))
    out_t = out["output"] if isinstance(out, dict) else out
    out_np = out_t.detach().cpu().numpy()
    dets_per = decode_batch(out_np, img_h, img_w, stride, threshold=threshold)

    has_prior = imgs.shape[1] >= 4
    for i in range(imgs.shape[0]):
        rgb = _denorm(imgs[i])  # clean RGB, no annotations
        rgb_path = out_sub / f"sample_{i}_rgb.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        prior_path = None
        if has_prior:
            prior_full = imgs[i, 3].detach().cpu().numpy()
            prior_path = out_sub / f"sample_{i}_prior.png"
            save_prior_overlay_png(prior_full, str(prior_path))

        if db is None:
            continue

        try:
            db.add_image(ep, tag, i, rgb_path)
            if prior_path is not None:
                db.add_overlay(ep, tag, i, "prior_heat", prior_path)
            if i < len(gt_boxes) and gt_boxes[i].shape[0] > 0:
                db.add_boxes(ep, tag, i, "gt", gt_boxes[i])
            if dets_per[i]:
                pb = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets_per[i]], dtype=np.float32)
                ps = np.array([d.score for d in dets_per[i]], dtype=np.float32)
                db.add_boxes(ep, tag, i, "pred", pb, ps)
            if trails_per is not None and i < len(trails_per):
                # Encode the per-object trail polyline as a single "trail"
                # box row per object: x1,y1 = trail head, x2,y2 = trail tail,
                # meta.points = full stamp polyline.
                for trail in trails_per[i]:
                    if len(trail) < 1:
                        continue
                    head = trail[-1]
                    tail = trail[0] if len(trail) > 0 else head
                    db.add_boxes(
                        ep, tag, i, "trail",
                        np.array([[tail[0], tail[1], head[0], head[1]]], dtype=np.float32),
                        scores=np.array([float(len(trail))], dtype=np.float32),
                        meta={0: {"points": [[float(x), float(y)] for x, y in trail]}},
                    )
        except Exception:
            pass


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
    trails_per: list | None = None,
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
            if trails_per is not None and i < len(trails_per):
                rgb = _draw_prior_trails_from_trails(rgb, trails_per[i])
        # GT in solid magenta — visually distinct from the colored pred gradient.
        rgb = _draw(rgb, gt_boxes[i], color=(255, 0, 255), thick=2)
        for d in dets_per[i]:
            x1, y1 = int(round(d.x1)), int(round(d.y1))
            x2, y2 = int(round(d.x2)), int(round(d.y2))
            _draw_pred(rgb, x1, y1, x2, y2, float(d.score))
        rendered.append(rgb.transpose(2, 0, 1))
    return np.stack(rendered, axis=0)
