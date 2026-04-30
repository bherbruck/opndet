from __future__ import annotations

import cv2
import numpy as np
import torch

from opndet.decode import decode_batch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm(img_t: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized tensor -> [H,W,3] uint8 RGB."""
    arr = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _draw(img: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], thick: int = 1) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    return out


@torch.no_grad()
def render_predictions(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    gt_boxes: list[np.ndarray],
    img_h: int,
    img_w: int,
    stride: int,
    threshold: float = 0.3,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Run model on a batch of normalized images; return [N, 3, H, W] uint8 grid-ready array.

    Each image overlays: green = predicted boxes, red = GT boxes.
    """
    model.eval()
    out = model(imgs.to(device))
    out_t = out["output"] if isinstance(out, dict) else out
    out_np = out_t.cpu().numpy()
    dets_per = decode_batch(out_np, img_h, img_w, stride, threshold=threshold)

    rendered = []
    for i in range(imgs.shape[0]):
        rgb = _denorm(imgs[i])
        rgb = _draw(rgb, gt_boxes[i], color=(255, 0, 0), thick=2)
        pred_arr = np.array(
            [[d.x1, d.y1, d.x2, d.y2] for d in dets_per[i]],
            dtype=np.float32,
        ) if dets_per[i] else np.zeros((0, 4), dtype=np.float32)
        rgb = _draw(rgb, pred_arr, color=(0, 255, 0), thick=1)
        rendered.append(rgb.transpose(2, 0, 1))  # CHW for tensorboard
    return np.stack(rendered, axis=0)
