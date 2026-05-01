from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from opndet.dataset import Sample, letterbox
from opndet.decode import decode_batch
from opndet.metrics import hungarian_match


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    """[H,W,3] uint8 RGB -> normalized [1,3,H,W] tensor."""
    f = img_rgb_uint8.astype(np.float32) / 255.0
    f = (f - _MEAN) / _STD
    return torch.from_numpy(f.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def _perturb(
    img_rgb: np.ndarray, gt_boxes: np.ndarray, rng: np.random.Generator,
    max_translate_px: int = 2, brightness_range: float = 0.05, contrast_range: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Translate (integer pixels) + brightness + contrast jitter on a letterboxed image.
    Returns (perturbed_img, perturbed_gt, (tx, ty)).
    """
    H, W = img_rgb.shape[:2]
    tx = float(rng.integers(-max_translate_px, max_translate_px + 1))
    ty = float(rng.integers(-max_translate_px, max_translate_px + 1))
    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    img_t = cv2.warpAffine(img_rgb, M, (W, H), borderValue=(114, 114, 114))
    bright = 1.0 + float(rng.uniform(-brightness_range, brightness_range))
    contrast = 1.0 + float(rng.uniform(-contrast_range, contrast_range))
    img_f = (img_t.astype(np.float32) - 128.0) * contrast + 128.0
    img_f = img_f * bright
    img_t = np.clip(img_f, 0, 255).astype(np.uint8)
    gt_t = gt_boxes.copy()
    if gt_t.shape[0] > 0:
        gt_t[:, [0, 2]] += tx
        gt_t[:, [1, 3]] += ty
    return img_t, gt_t, (tx, ty)


@torch.no_grad()
def perturbation_stability(
    model: torch.nn.Module,
    samples: list[Sample],
    img_h: int, img_w: int, stride: int,
    n_perturbations: int = 8,
    decode_threshold: float = 0.05,
    iou_thresh: float = 0.5,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict:
    """Per-object jitter under tiny perturbations of the input image.

    For each test sample: run the model on the original + N small perturbations
    (±2px translate, ±5% brightness, ±5% contrast). Hungarian-match preds to GT
    per perturbed-frame, transform pred centers back to the original frame, then
    compute per-GT stddev of (score, cx, cy, w, h) across the K+1 versions.

    Higher = flappier under tiny input variation -> likely flappier frame-to-frame.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    score_stds: list[float] = []
    cx_stds: list[float] = []
    cy_stds: list[float] = []
    w_stds: list[float] = []
    h_stds: list[float] = []
    n_matched = 0
    n_gt_total = 0

    for s in tqdm(samples, desc="stability", leave=False):
        img_bgr = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb, gt_lb = letterbox(img_rgb, s.boxes.copy(), img_h, img_w)
        if gt_lb.shape[0] == 0:
            continue
        n_gt_total += gt_lb.shape[0]

        # Build batch: [original, perturbation_1, ..., perturbation_K]
        versions: list[tuple[np.ndarray, tuple[float, float]]] = [(gt_lb.copy(), (0.0, 0.0))]
        tensors = [_to_tensor(img_lb)]
        for _ in range(n_perturbations):
            img_p, gt_p, (tx, ty) = _perturb(img_lb, gt_lb, rng)
            versions.append((gt_p, (tx, ty)))
            tensors.append(_to_tensor(img_p))

        x = torch.cat(tensors, dim=0).to(device, non_blocking=True)
        out = model(x)
        out_t = out["output"] if isinstance(out, dict) else out
        out_np = out_t.detach().cpu().numpy()
        dets_per = decode_batch(out_np, img_h, img_w, stride, threshold=decode_threshold)

        # tracks[gt_idx] = list of (score, cx_orig, cy_orig, w, h) across versions
        tracks: list[list[tuple[float, float, float, float, float]]] = [[] for _ in range(gt_lb.shape[0])]
        for v_idx, dets in enumerate(dets_per):
            gt_v, (tx, ty) = versions[v_idx]
            if not dets:
                continue
            pb = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
            m = hungarian_match(pb, gt_v, iou_thresh=iou_thresh)
            for pi, gj in m.pairs:
                d = dets[int(pi)]
                cx = (d.x1 + d.x2) * 0.5 - tx       # back to original frame
                cy = (d.y1 + d.y2) * 0.5 - ty
                tracks[int(gj)].append((float(d.score), float(cx), float(cy),
                                        float(d.x2 - d.x1), float(d.y2 - d.y1)))

        for rec in tracks:
            if len(rec) >= 2:
                arr = np.array(rec, dtype=np.float64)
                score_stds.append(float(arr[:, 0].std()))
                cx_stds.append(float(arr[:, 1].std()))
                cy_stds.append(float(arr[:, 2].std()))
                w_stds.append(float(arr[:, 3].std()))
                h_stds.append(float(arr[:, 4].std()))
                n_matched += 1

    if not score_stds:
        return {
            "n_objects_tracked": 0, "n_gt_total": n_gt_total,
            "track_completion_rate": 0.0, "n_perturbations": n_perturbations,
        }

    def summarize(arr: list[float]) -> dict:
        a = np.array(arr)
        return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    return {
        "n_objects_tracked": n_matched,
        "n_gt_total": n_gt_total,
        "track_completion_rate": n_matched / max(1, n_gt_total),
        "n_perturbations": n_perturbations,
        "score": summarize(score_stds),
        "center_x_px": summarize(cx_stds),
        "center_y_px": summarize(cy_stds),
        "w_px": summarize(w_stds),
        "h_px": summarize(h_stds),
    }
