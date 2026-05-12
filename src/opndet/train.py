from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import copy

from opndet.assigner import build_assigner
from opndet.augment import AugConfig, make_augment
from opndet.dataset import OpndetDataset, collate, load_datasets, split_samples
from opndet.decode import decode_batch
from opndet.encode import encode_targets, encode_targets_ltrb, encode_targets_obb
from opndet.loss import OpndetBboxLoss
from opndet.presets import resolve as _resolve_preset
from opndet.visualize import render_predictions
from opndet.yaml_build import build_model_from_yaml


class EMA:
    """Exponential moving average with progressive decay (YOLOv5/8 style).

    Effective decay ramps from 0 to `decay` over `tau` steps:
        d(t) = decay * (1 - exp(-t / tau))
    This avoids the pathological early-training lag where shadow weights
    are still close to random init at high decay (e.g. 0.999).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, tau: int = 2000):
        self.decay = decay
        self.tau = tau
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = self.decay * (1.0 - math.exp(-self.step / self.tau))
        for p_s, p in zip(self.shadow.parameters(), model.parameters()):
            p_s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for b_s, b in zip(self.shadow.buffers(), model.buffers()):
            if b_s.dtype == b.dtype and b_s.shape == b.shape:
                b_s.copy_(b)


class _RepeatSampler:
    """Wraps a sampler to yield from it forever (YOLOv5 trick)."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """DataLoader that calls super().__iter__() exactly once and reuses it for the entire run.

    Workers spawn on first __init__ and never die; prefetch pipeline never resets at epoch
    boundaries. Drop-in replacement: callers still write `for batch in loader:` and get one
    epoch's worth of batches per call.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # batch_sampler is the BatchSampler PyTorch built from sampler+batch_size+drop_last.
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.batch_sampler.sampler)  # number of batches per epoch

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


def _detect_peak_op(model: torch.nn.Module) -> tuple[int | None, float | None]:
    """Walk the built model, find the (Sigmoid)PeakSuppress layer and return its (k, eps).
    Returns (None, None) if the model has no peak op (custom arch)."""
    for m in model.modules():
        if type(m).__name__ in ("PeakSuppress", "SigmoidPeakSuppress"):
            return int(m.k), float(m.eps)
    return None, None


class _CfgShim:
    """encode_targets needs an object with img_h/img_w/stride/out_h/out_w (and an
    optional hm_blob_frac for the object-shaped-heatmap encoding)."""

    def __init__(self, img_h: int, img_w: int, stride: int, hm_blob_frac: float = 0.0,
                 hm_target: str = "gaussian", hm_ellipse_edge_margin: float = 0.1):
        self.img_h = img_h
        self.img_w = img_w
        self.stride = stride
        self.out_h = img_h // stride
        self.out_w = img_w // stride
        self.hm_blob_frac = float(hm_blob_frac)
        self.hm_target = str(hm_target)
        self.hm_ellipse_edge_margin = float(hm_ellipse_edge_margin)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = a_area[:, None] + b_area[None, :] - inter + 1e-9
    return inter / union


def _iou_ceiling_per_gt(gt_boxes: np.ndarray, noise_px: float) -> np.ndarray:
    """Architectural IoU ceiling per GT: best achievable IoU when the model places
    centers/edges within `noise_px` of GT (worst case in both axes). Below this, no
    architecturally-fair model could score, so mAP thresholds above the ceiling are
    requirement-impossible at current stride/resolution.

    Approximation: GT box shrunk by noise_px on each side vs. expanded by noise_px,
    intersection-over-union of the two.
    """
    gw = np.maximum(1.0, gt_boxes[:, 2] - gt_boxes[:, 0])
    gh = np.maximum(1.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    inter = np.maximum(0.0, gw - noise_px) * np.maximum(0.0, gh - noise_px)
    union = 2.0 * gw * gh - inter
    return (inter / np.maximum(1e-9, union)).astype(np.float64)


def _center_aligned_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """IoU between pred and GT after translating pred so its center matches GT's.
    Measures bbox SHAPE quality only — decouples mAP from center precision. Right
    metric when center accuracy is bounded by stride (model can't physically beat
    sub-cell precision under sigmoid saturation), and what we care about is "if the
    cell is right, how good is the box shape?"

    Both rectangles share a center, so:
      inter = min(pw, gw) * min(ph, gh)
      union = pw*ph + gw*gh - inter
    """
    pw = np.maximum(0.0, pred_boxes[:, None, 2] - pred_boxes[:, None, 0])
    ph = np.maximum(0.0, pred_boxes[:, None, 3] - pred_boxes[:, None, 1])
    gw = np.maximum(1.0, gt_boxes[None, :, 2] - gt_boxes[None, :, 0])
    gh = np.maximum(1.0, gt_boxes[None, :, 3] - gt_boxes[None, :, 1])
    inter = np.minimum(pw, gw) * np.minimum(ph, gh)
    union = pw * ph + gw * gh - inter
    return (inter / np.maximum(1e-9, union)).astype(np.float64)


def _accumulate_correct(
    per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iouv: np.ndarray,
    mode: str = "standard",
) -> tuple[np.ndarray, np.ndarray, int]:
    """For each image: compute IoU once, greedy-match by score desc per IoU threshold (vectorized over thresholds).
    Concatenate. Returns (all_scores [N], all_correct [N, n_iouv], total_n_gt).

    mode = "standard": classic IoU (matches COCO mAP semantics).
    mode = "shape":    center-aligned IoU — pred translated to share GT's center
                       before IoU. Decouples mAP from sub-pixel center precision.
                       Honest "if the cell is right, how good is the box?" metric;
                       useful when stride limits center precision (small objects
                       at stride=4 can't physically clear mAP@.95 even with perfect
                       w/h). Ours, not standard. Always reported alongside.
    """
    n_t = iouv.shape[0]
    parts_scores: list[np.ndarray] = []
    parts_correct: list[np.ndarray] = []
    total_gt = 0
    iou_fn = _center_aligned_iou if mode == "shape" else _iou_xyxy
    for scores, boxes, gt_boxes in per_image:
        total_gt += int(gt_boxes.shape[0])
        if boxes.shape[0] == 0:
            continue
        order = np.argsort(-scores)
        s_sorted = scores[order]
        parts_scores.append(s_sorted.astype(np.float32))
        if gt_boxes.shape[0] == 0:
            parts_correct.append(np.zeros((boxes.shape[0], n_t), dtype=bool))
            continue
        iou = iou_fn(boxes[order], gt_boxes)  # [n_p, n_g]
        correct = np.zeros((boxes.shape[0], n_t), dtype=bool)
        avail = np.ones((n_t, gt_boxes.shape[0]), dtype=bool)
        for i in range(boxes.shape[0]):
            row = iou[i]                             # [n_g]
            masked = row[None, :] * avail            # [n_t, n_g]
            best = masked.max(axis=1)                # [n_t]
            am = masked.argmax(axis=1)               # [n_t]
            ok = best > iouv                         # strict, matches prior semantics
            if ok.any():
                tis = np.where(ok)[0]
                correct[i, tis] = True
                avail[tis, am[tis]] = False
        parts_correct.append(correct)
    if not parts_scores:
        return np.zeros(0, dtype=np.float32), np.zeros((0, n_t), dtype=bool), total_gt
    return np.concatenate(parts_scores), np.concatenate(parts_correct, axis=0), total_gt


def _ap_from_correct(all_scores: np.ndarray, all_correct: np.ndarray, total_gt: int) -> np.ndarray:
    """COCO 101-point AP per IoU threshold. Returns array of length all_correct.shape[1]."""
    n_t = all_correct.shape[1]
    if total_gt == 0 or all_scores.shape[0] == 0:
        return np.zeros(n_t, dtype=np.float64)
    order = np.argsort(-all_scores)
    c = all_correct[order].astype(np.float64)
    cum_tp = np.cumsum(c, axis=0)
    cum_fp = np.cumsum(1.0 - c, axis=0)
    p = cum_tp / np.maximum(1.0, cum_tp + cum_fp)
    r = cum_tp / max(1, total_gt)
    levels = np.linspace(0, 1, 101)
    ap = np.zeros(n_t)
    for ti in range(n_t):
        rt = r[:, ti]; pt = p[:, ti]
        # for each recall level, max precision at recall >= level
        for level in levels:
            mask = rt >= level
            ap[ti] += (pt[mask].max() if mask.any() else 0.0) / 101.0
    return ap


@torch.no_grad()
def evaluate(model, loader, cfg_shim: _CfgShim, device: torch.device,
             score_thresh: float = 0.3, iou_thresh: float = 0.5,
             decode_threshold: float = 0.05) -> dict[str, float]:
    """Hungarian-matched eval. Computes IoU once per image, AP across the IoU grid in a single pass.

    decode_threshold filters peaks before AP; 0.05 is plenty since AP is rank-driven and dense
    scenes can produce thousands of low-conf peaks that contribute essentially nothing to AP.
    """
    from opndet.metrics import hungarian_match, obb_summary
    from opndet.decode import decode_obb_batch, gt_obbs_from_targets
    model.eval()
    tp = fp = fn = 0
    n_pred = n_gt = 0
    per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    is_obb_model = False
    obb_match_total = 0
    obb_iou_sum = 0.0
    ang_err_sum = 0.0
    ang_err_le_10 = 0
    ang_err_le_30 = 0
    obb_ious_all: list[np.ndarray] = []
    ang_errs_all: list[np.ndarray] = []
    for imgs, boxes_list, targets in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        out_t = out["output"] if isinstance(out, dict) else out
        out_np = out_t.cpu().numpy()
        dets_per_full = decode_batch(out_np, cfg_shim.img_h, cfg_shim.img_w, cfg_shim.stride, threshold=decode_threshold)

        # OBB-mode side channel: when the head emits 7-ch, also run rotated-IoU
        # + angle-error metrics. GT OBBs come from the encoded targets dict
        # (post-letterbox, same coords as boxes_list), preds via decode_obb_batch.
        if out_np.shape[1] == 6 and targets is not None and "obb" in targets and "pos" in targets:
            is_obb_model = True
            pred_obbs_per = decode_obb_batch(out_np, cfg_shim.img_h, cfg_shim.img_w, cfg_shim.stride, threshold=score_thresh)
            pos_np = targets["pos"].cpu().numpy()
            obb_np = targets["obb"].cpu().numpy()
            gt_obbs_per = gt_obbs_from_targets(pos_np, obb_np, cfg_shim.img_h, cfg_shim.img_w, cfg_shim.stride)
            for pred_dets, gt_obbs_i in zip(pred_obbs_per, gt_obbs_per):
                if not pred_dets:
                    continue
                pred_arr = np.array([[d.cx, d.cy, d.w, d.h, d.theta] for d in pred_dets], dtype=np.float32)
                summ = obb_summary(pred_arr, gt_obbs_i, iou_thresh=0.3)
                obb_match_total += summ["n_match"]
                obb_iou_sum += summ["obb_iou_sum"]
                ang_err_sum += summ["ang_err_sum"]
                ang_err_le_10 += summ["ang_err_le_10_count"]
                ang_err_le_30 += summ["ang_err_le_30_count"]
                if summ["obb_ious"].size:
                    obb_ious_all.append(summ["obb_ious"])
                    ang_errs_all.append(summ["ang_errs_deg"])

        for dets_full, gt in zip(dets_per_full, boxes_list):
            scores_full = np.array([d.score for d in dets_full], dtype=np.float32) if dets_full else np.zeros(0, dtype=np.float32)
            boxes_full = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets_full], dtype=np.float32) if dets_full else np.zeros((0, 4), dtype=np.float32)
            per_image.append((scores_full, boxes_full, gt.astype(np.float32)))

            keep = scores_full >= score_thresh
            pb = boxes_full[keep]
            n_pred += int(pb.shape[0]); n_gt += int(gt.shape[0])
            m = hungarian_match(pb, gt.astype(np.float32), iou_thresh=iou_thresh)
            tp_i = m.pairs.shape[0]
            tp += tp_i
            fp += int(pb.shape[0]) - tp_i
            fn += int(gt.shape[0]) - tp_i

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # IoU-free "did we find it close to center?" pass. Uses the same
    # score-thresh-filtered preds as P/R so it reflects deployment detections.
    # Hungarian-matches by center distance with R = max(8 px, 0.5 × min(GT_w,
    # GT_h)) — generous on shape, strict on locality. Surfaces:
    #   center_recall, center_precision, center_f1
    #   center_dist_mean_px, center_dist_p50_px, center_dist_p95_px
    #   count_off_by_le1: fraction of images where |n_pred - n_gt| <= 1
    from opndet.metrics import center_match
    cm_match_total = cm_pred_total = cm_gt_total = 0
    cm_dup_total = cm_ghost_total = 0
    cm_dists_px: list[np.ndarray] = []
    cm_dists_frac: list[np.ndarray] = []
    cnt_off_le1 = 0
    cnt_total = len(per_image)
    for scores_full, boxes_full, gt in per_image:
        keep = scores_full >= score_thresh
        pb = boxes_full[keep]
        if abs(int(pb.shape[0]) - int(gt.shape[0])) <= 1:
            cnt_off_le1 += 1
        m = center_match(pb, gt)
        cm_match_total += m["n_match"]
        cm_pred_total += m["n_pred"]
        cm_gt_total   += m["n_gt"]
        cm_dup_total  += m.get("n_dup", 0)
        cm_ghost_total += m.get("n_ghost", 0)
        if m["distances_px"].size > 0:
            cm_dists_px.append(m["distances_px"])
            cm_dists_frac.append(m["distances_frac"])
    center_recall    = cm_match_total / max(1, cm_gt_total)
    center_precision = cm_match_total / max(1, cm_pred_total)
    center_f1 = 2 * center_recall * center_precision / max(1e-9, center_recall + center_precision)
    # Lenient flavor: duplicates (pred near or enclosing a real GT) are
    # treated as not-FPs. Only ghost detections (phantoms in empty frame
    # space) count against precision. This matches the deployment KPI:
    # "no false positives where there's nothing there" — duplicates are
    # ugly but not wrong.
    center_precision_lenient = (cm_match_total + cm_dup_total) / max(1, cm_pred_total)
    center_f1_lenient = 2 * center_recall * center_precision_lenient / max(1e-9, center_recall + center_precision_lenient)
    center_ghost_rate = cm_ghost_total / max(1, cm_pred_total)
    center_dup_rate   = cm_dup_total / max(1, cm_pred_total)
    # Noise-floor-aware tier rates. The model can only place a center at a
    # stride-cell, with sub-cell offset regression — so anything within
    # stride/2 px of GT center IS already "perfectly on cell" given the
    # model's output resolution. Reading 2.1 px at stride=4 as "off by 2 px"
    # is misleading; it's actually as tight as the architecture allows.
    stride = int(cfg_shim.stride)
    perfect_thresh_px = stride * 0.5     # half-cell — fundamental quantization floor
    within_1cell_px   = float(stride)    # 1 cell — practical localization noise
    within_2cell_px   = stride * 2.0     # 2 cells — "in the neighborhood"
    if cm_dists_px:
        d_px = np.concatenate(cm_dists_px)
        d_fr = np.concatenate(cm_dists_frac)
        n = len(d_px)
        center_dist_mean_px = float(d_px.mean())
        center_dist_p50_px  = float(np.percentile(d_px, 50))
        center_dist_p95_px  = float(np.percentile(d_px, 95))
        center_dist_mean    = float(d_fr.mean())
        center_dist_p50     = float(np.percentile(d_fr, 50))
        center_dist_p95     = float(np.percentile(d_fr, 95))
        center_perfect_rate    = float((d_px <= perfect_thresh_px).sum()) / n
        center_within_1cell    = float((d_px <= within_1cell_px).sum()) / n
        center_within_2cell    = float((d_px <= within_2cell_px).sum()) / n
    else:
        center_dist_mean_px = center_dist_p50_px = center_dist_p95_px = 0.0
        center_dist_mean = center_dist_p50 = center_dist_p95 = 0.0
        center_perfect_rate = center_within_1cell = center_within_2cell = 0.0
    count_off_le1_frac = cnt_off_le1 / max(1, cnt_total)

    iouv = np.arange(0.5, 1.0, 0.05, dtype=np.float64)
    all_scores, all_correct, total_gt = _accumulate_correct(per_image, iouv, mode="standard")
    aps = _ap_from_correct(all_scores, all_correct, total_gt)
    map50 = float(aps[0])
    map_50_95 = float(aps.mean())
    # Shape-mAP: center-aligned IoU. Decouples from sub-pixel center precision so
    # the metric is honest at small object sizes / large strides. Self-converges
    # with standard mAP as image resolution grows (objects bigger in px → ceiling
    # rises → no clipping vs standard). Documented as ours, not COCO-standard.
    all_scores_s, all_correct_s, _ = _accumulate_correct(per_image, iouv, mode="shape")
    aps_s = _ap_from_correct(all_scores_s, all_correct_s, total_gt)
    map50_shape = float(aps_s[0])
    map_50_95_shape = float(aps_s.mean())

    # F1 sweep across thresholds. Uses the iou=0.5 correctness column we already computed.
    # Picks best operating point automatically — robust to over-prediction at score_thresh=0.2.
    if all_scores.shape[0] > 0 and total_gt > 0:
        is_tp = all_correct[:, 0].astype(np.int64)
        thresholds = np.arange(0.10, 0.91, 0.05)
        best_f1 = 0.0
        best_t = float(score_thresh)
        for t in thresholds:
            keep = all_scores >= t
            tp_t = int(is_tp[keep].sum())
            fp_t = int(keep.sum()) - tp_t
            fn_t = total_gt - tp_t
            denom = 2 * tp_t + fp_t + fn_t
            f1_t = 2.0 * tp_t / denom if denom > 0 else 0.0
            if f1_t > best_f1:
                best_f1 = f1_t
                best_t = float(t)
        f1_opt = best_f1
        threshold_opt = best_t
    else:
        f1_opt = 0.0
        threshold_opt = float(score_thresh)

    if is_obb_model and obb_match_total > 0:
        obb_iou_mean = obb_iou_sum / max(1, obb_match_total)
        ang_err_deg_mean = ang_err_sum / max(1, obb_match_total)
        ang_err_le_10_frac = ang_err_le_10 / max(1, obb_match_total)
        ang_err_le_30_frac = ang_err_le_30 / max(1, obb_match_total)
        all_ious = np.concatenate(obb_ious_all)
        all_angs = np.concatenate(ang_errs_all)
        obb_iou_p50 = float(np.median(all_ious))
        ang_err_deg_median = float(np.median(all_angs))
    else:
        obb_iou_mean = obb_iou_p50 = 0.0
        ang_err_deg_mean = ang_err_deg_median = 0.0
        ang_err_le_10_frac = ang_err_le_30_frac = 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "map50": map50, "map_50_95": map_50_95,
            "map50_shape": map50_shape, "map_50_95_shape": map_50_95_shape,
            "f1_opt": f1_opt, "threshold_opt": threshold_opt,
            "n_pred": float(n_pred), "n_gt": float(n_gt),
            "center_recall": float(center_recall),
            "center_precision": float(center_precision),
            "center_f1": float(center_f1),
            "center_precision_lenient": float(center_precision_lenient),
            "center_f1_lenient": float(center_f1_lenient),
            "center_ghost_rate": float(center_ghost_rate),
            "center_dup_rate": float(center_dup_rate),
            # Bbox-relative distances (primary — survive image-size changes).
            # Units: fraction of min(gt_w, gt_h). 0.0 = on center, 0.5 = on
            # smaller-side edge, >1.0 = outside the bbox.
            "center_dist_mean": center_dist_mean,
            "center_dist_p50": center_dist_p50,
            "center_dist_p95": center_dist_p95,
            # Pixel-units (informational — only directly comparable across
            # runs at the same input resolution).
            "center_dist_mean_px": center_dist_mean_px,
            "center_dist_p50_px": center_dist_p50_px,
            "center_dist_p95_px": center_dist_p95_px,
            # Stride-quantization-aware rates. perfect = within stride/2 of
            # GT center (= as tight as the architecture can place a peak).
            # within_1cell = within one stride cell. within_2cell = neighbor.
            "center_perfect_rate": float(center_perfect_rate),
            "center_within_1cell": float(center_within_1cell),
            "center_within_2cell": float(center_within_2cell),
            "count_off_le1_frac": float(count_off_le1_frac),
            # OBB diagnostics (only meaningful when 7-ch head; 0 otherwise).
            # Hungarian-matched on 1-rotated_iou cost @ 0.3 threshold.
            "obb_iou_mean": float(obb_iou_mean),
            "obb_iou_p50": float(obb_iou_p50),
            "angle_err_deg_mean": float(ang_err_deg_mean),
            "angle_err_deg_median": float(ang_err_deg_median),
            "angle_err_le_10_frac": float(ang_err_le_10_frac),
            "angle_err_le_30_frac": float(ang_err_le_30_frac)}


def _bundle_run(out_dir: Path, include_tb: bool = False) -> Path | None:
    """Zip the run dir at end of training. Skips tfevents by default (huge). On Colab, also
    triggers a browser download. Returns the zip path."""
    import zipfile
    bundle = out_dir.parent / f"{out_dir.name}.zip"
    skipped_tb = 0
    n = 0
    try:
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            for f in out_dir.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(out_dir)
                if not include_tb and rel.parts and rel.parts[0] == "tb":
                    skipped_tb += 1
                    continue
                z.write(f, rel)
                n += 1
        sz_mb = bundle.stat().st_size / 1024 / 1024
        msg = f"bundled run -> {bundle} ({sz_mb:.1f} MB, {n} files"
        if not include_tb:
            msg += f", skipped {skipped_tb} tb events — pass bundle_include_tb: true to include"
        msg += ")"
        print(msg)
    except Exception as e:
        print(f"bundle failed: {e}")
        return None

    # Auto-download — works when training is invoked in-process from a notebook
    # cell (`from opndet.train import train; train('cfg.yaml')`) because the
    # IPython kernel context is available. Fails silently for subprocess
    # invocations (`!opndet train ...`) since `files.download` needs the kernel.
    _colab_download(bundle)
    return bundle


def _colab_download(path: Path) -> None:
    """Trigger a browser download of `path` if running in a Colab notebook kernel; no-op otherwise."""
    try:
        from google.colab import files  # type: ignore
        files.download(str(path))
    except ImportError:
        pass  # not on Colab
    except Exception:
        pass  # subprocess without a kernel; the file is still on disk


def _download_run(out_dir: Path, c: dict) -> None:
    """End-of-training Colab download. Controlled by `download:` in the config:
      "best"   (default) — just `<name>_best.pt` + `metrics.duckdb` (the things you actually
                 want off the box; the run dir's vis/ PNGs are dashboard artifacts served live,
                 not worth GB of zip).
      "bundle"          — zip the whole run dir (minus tb/) and download it (the old behaviour).
      "none"            — download nothing (the run stays on the box's disk).
    Back-compat: if `download` is absent but `auto_bundle` is set, true→"bundle", false→"none"."""
    if "download" in c:
        mode = str(c["download"]).lower()
    elif "auto_bundle" in c:
        mode = "bundle" if c["auto_bundle"] else "none"
    else:
        mode = "best"
    if mode == "none":
        return
    if mode == "bundle":
        _bundle_run(out_dir, include_tb=bool(c.get("bundle_include_tb", False)))
        return
    # mode == "best"
    targets: list[Path] = []
    best = next(iter(sorted(out_dir.glob("*_best.pt"))), None) or next(iter(sorted(out_dir.glob("*.pt"))), None)
    if best is not None:
        targets.append(best)
    db = out_dir / "metrics.duckdb"
    if db.exists():
        targets.append(db)
    if not targets:
        print(f"download: nothing to download in {out_dir}")
        return
    for p in targets:
        print(f"download: {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    for p in targets:
        _colab_download(p)


def _resolve_out_dir(base: Path, auto_increment: bool = True) -> Path:
    """If base doesn't exist or is empty, return base. Otherwise pick base_2, base_3, ..."""
    if not auto_increment:
        return base
    if not base.exists() or not any(base.iterdir()):
        return base
    parent, stem = base.parent, base.name
    n = 2
    while (parent / f"{stem}_{n}").exists():
        n += 1
    return parent / f"{stem}_{n}"


def _save_layered_vis_path(run_dir, tag: str, ep: int):
    from pathlib import Path as _P
    return _P(run_dir) / "vis" / tag.replace("/", "_") / f"ep_{ep:03d}"


def cosine_lr(step: int, total: int, base: float, warmup: int = 200, min_factor: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * p)))


def train(cfg_path: str, run_name: str | None = None, runs_dir: str | None = None,
          resume: str | None = None, teacher: str | None = None, self_distill: bool = False) -> None:
    with open(cfg_path) as f:
        user_cfg = yaml.safe_load(f) or {}

    if "model_config" not in user_cfg:
        raise ValueError("train.yaml must specify model_config (e.g. bbox-m, bbox-x)")

    # Layered defaults: per-preset training defaults + user yaml on top.
    # User can omit anything they don't want to override.
    from opndet.training_defaults import defaults_for, deep_merge
    c = deep_merge(defaults_for(user_cfg["model_config"]), user_cfg)

    # bbox-*-seg presets (dense full-res dome head) have a different metric space
    # (Dice / area / count, not detection mAP), no in-graph peak op to calibrate,
    # no curriculum/repulsion/eval-threshold. Delegate to the dedicated seg loop.
    try:
        _pp = yaml.safe_load(open(_resolve_preset(c["model_config"])))
        _is_seg = isinstance(_pp, dict) and (_pp.get("model", {}) or {}).get("head") == "seg"
    except Exception:
        _is_seg = False
    if _is_seg:
        if teacher is not None or self_distill:
            raise ValueError("distillation is not supported for bbox-*-seg presets")
        from opndet.train_seg import train_seg
        train_seg(cfg_path, run_name=run_name, runs_dir=runs_dir, resume=resume)
        return

    if runs_dir is not None:
        c["runs_dir"] = runs_dir
    if run_name is not None:
        c["name"] = run_name

    resume_state = None
    if resume:
        resume_path = Path(resume)
        if resume_path.is_dir():
            resume_path = resume_path / "last.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"resume ckpt not found: {resume_path}")
        print(f"resume: loading {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        out_dir = resume_path.parent
    else:
        if "runs_dir" in c or "name" in c:
            rd = Path(c.get("runs_dir", "runs"))
            nm = c.get("name", "exp1")
            base = rd / nm
        else:
            base = Path(c.get("out_dir", "runs/exp1"))
        out_dir = _resolve_out_dir(base, auto_increment=bool(c.get("auto_increment", True)))
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tb"
    print(f"out_dir: {out_dir}")
    seed = int(c.get("seed", 0))
    torch.manual_seed(seed); np.random.seed(seed)

    # Tensorboard is optional. Colab occasionally ships a numpy/tensorboard
    # version mismatch (tensorboard's compat layer references private numpy
    # symbols that newer numpy removes). Fall back to a no-op writer when
    # import fails — scalars still flow to DuckDB and the dashboard via the
    # add_scalar wrapper below; only the .tfevents files are skipped.
    class _NoOpWriter:
        def add_scalar(self, *a, **kw): pass
        def add_images(self, *a, **kw): pass
        def add_image(self, *a, **kw): pass
        def add_histogram(self, *a, **kw): pass
        def add_text(self, *a, **kw): pass
        def flush(self, *a, **kw): pass
        def close(self, *a, **kw): pass
    # Default OFF: the dashboard reads the DuckDB store, not tfevents, so a TB
    # log is dead weight (and `auto_bundle` already skips it). Set `tensorboard:
    # true` if you actually want a .tfevents log.
    if bool(c.get("tensorboard", False)):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"tensorboard: {tb_dir}")
        except Exception as e:
            writer = _NoOpWriter()
            print(f"tensorboard disabled ({type(e).__name__}: {e}); scalars still flow to DB / dashboard")
    else:
        writer = _NoOpWriter()
        print("tensorboard: off (default — scalars/images go to the DuckDB store / dashboard; set tensorboard: true for a .tfevents log)")

    # DuckDB metrics store (queryable companion to TB). Writes alongside TB —
    # both stay in the run dir. Default on; opt out with `metrics_db: false`.
    db = None
    if bool(c.get("metrics_db", True)):
        try:
            from opndet.metrics_db import MetricsDB
            db = MetricsDB(out_dir)
            db.set_config(c)
            print(f"metrics_db: {db.path}")
        except Exception as e:
            print(f"metrics_db disabled ({e})")
            db = None

    # Wrap writer.add_scalar so every TB scalar also lands in the DB.
    _orig_add_scalar = writer.add_scalar
    def _add_scalar(tag, value, step):
        _orig_add_scalar(tag, value, step)
        if db is not None:
            db.add_scalar(int(step), str(tag), float(value))
    writer.add_scalar = _add_scalar

    # Auto-spawn the dashboard subprocess when `dashboard: true` (or the env
    # var OPNDET_DASHBOARD=1). In Colab, the dashboard auto-embeds as an
    # iframe in the calling cell. Killed at training exit via atexit.
    dashboard_proc = None
    if bool(c.get("dashboard", False)) or os.environ.get("OPNDET_DASHBOARD") == "1":
        try:
            from opndet.dashboard import spawn_background
            port = int(c.get("dashboard_port", 5000))
            # Point at the RUNS PARENT so the dashboard auto-discovers every
            # run (this run + any siblings/historical) and can compare them
            # in multi-run charts. Following TB's --logdir behavior.
            dashboard_proc = spawn_background(out_dir.parent, port=port)
            import atexit
            atexit.register(lambda: dashboard_proc.terminate() if dashboard_proc and dashboard_proc.poll() is None else None)
        except Exception as e:
            print(f"dashboard auto-spawn failed: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() and c.get("device", "auto") != "cpu" else "cpu")
    print(f"device: {device}")

    print("loading data ...")
    samples = load_datasets(c["data"]["sources"], image_filter=c["data"].get("image_filter"))
    print(f"total samples: {len(samples)}")
    # GT box short-side sizes (px) — used to auto-size resolution-relative knobs
    # (convexity_radius, auto_mine.patch_size) when the yaml leaves them "auto".
    _box_min_sides = np.array(
        [min(float(b[2] - b[0]), float(b[3] - b[1]))
         for s in samples if getattr(s, "boxes", None) is not None and len(s.boxes)
         for b in s.boxes],
        dtype=np.float32,
    )
    ratios = tuple(c["data"].get("split_ratios", [0.8, 0.1, 0.1]))
    train_s, val_s, test_s = split_samples(samples, ratios=ratios, seed=seed)
    print(f"split: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    aug_dict = dict(c.get("augment") or {})
    tp_cfg = aug_dict.pop("temporal_prior", None)

    # Auto hard-negative mining (config block `auto_mine:`). When on, point the
    # paste side at a run-local pool dir (unless the user already configured
    # `augment.hard_negative_pool`) so the mined patches feed straight back in.
    auto_mine_cfg = c.get("auto_mine")
    if isinstance(auto_mine_cfg, dict) and auto_mine_cfg.get("enabled", True):
        auto_mine_cfg = dict(auto_mine_cfg)
        # patch_size auto: a typical object's pixel footprint (so the crop carries
        # the texture). Resolution-relative → derive it from the data, not a fixed 32.
        if auto_mine_cfg.get("patch_size") in (None, "auto"):
            ps = int(round(float(np.median(_box_min_sides)))) if _box_min_sides.size else 32
            auto_mine_cfg["patch_size"] = max(16, min(128, ps))
            print(f"auto-mine: patch_size auto → {auto_mine_cfg['patch_size']} px (median GT min-side)")
        _am_pool = str(out_dir / "hard_negatives")
        aug_dict.setdefault("hard_negative_pool", _am_pool)
        aug_dict.setdefault("hard_negative_prob", float(auto_mine_cfg.get("paste_prob", 0.2)))
        aug_dict.setdefault("hard_negative_count", int(auto_mine_cfg.get("paste_count", 1)))
        print(f"auto-mine: ON  start_epoch={auto_mine_cfg.get('start_epoch', 50)} "
              f"every={auto_mine_cfg.get('every', 10)} source={auto_mine_cfg.get('source', 'val')} "
              f"pool={aug_dict['hard_negative_pool']}  (paste p={aug_dict['hard_negative_prob']} n={aug_dict['hard_negative_count']})")
    else:
        auto_mine_cfg = None

    aug_cfg = AugConfig(**aug_dict)
    hn_pool = []
    if aug_cfg.hard_negative_pool and aug_cfg.hard_negative_prob > 0:
        from opndet.mine_negatives import load_pool
        hn_pool = load_pool(aug_cfg.hard_negative_pool)
        print(f"  hard-negative pool: {aug_cfg.hard_negative_pool}  loaded {len(hn_pool)} patches "
              f"(p={aug_cfg.hard_negative_prob}, count={aug_cfg.hard_negative_count})")
        if not hn_pool and not auto_mine_cfg:
            print(f"  WARN: hard_negative_pool {aug_cfg.hard_negative_pool} is empty or missing")
    aug_fn = make_augment(aug_cfg, hn_pool=hn_pool)

    model_path = _resolve_preset(c["model_config"])
    _mc = c.get("model", {}) or {}
    model = build_model_from_yaml(model_path, img_h=_mc.get("img_h"), img_w=_mc.get("img_w")).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
    in_ch, img_h, img_w = model.input_shape
    stride = int(_mc.get("stride", 4))
    cfg_shim = _CfgShim(img_h, img_w, stride=stride, hm_blob_frac=float(c.get("hm_blob_frac", 0.0)),
                        hm_target=str(c.get("hm_target", "gaussian")),
                        hm_ellipse_edge_margin=float(c.get("hm_ellipse_edge_margin", 0.1)))
    n_params = sum(p.numel() for p in model.parameters())
    has_dist = "dist" in getattr(model, "aliases", {})
    print(f"model: {c['model_config']}  params={n_params/1e6:.2f}M  input={in_ch}x{img_h}x{img_w}{'  (dist head)' if has_dist else ''}")

    prior_synth = None
    if in_ch == 4:
        from opndet.augment_temporal_prior import TemporalPriorSynth
        prior_synth = TemporalPriorSynth(tp_cfg or {})
        print(f"  temporal prior synth ON (n_max={prior_synth.cfg['n_max']}, "
              f"motion_speed={prior_synth.cfg['motion_speed_range']})")
    elif tp_cfg is not None:
        print(f"  warning: augment.temporal_prior set but model in_ch={in_ch}; ignoring")

    # Detect head variant via the YAML alias graph:
    #   `ang_r` present  → 7-ch OBB head      → encode_targets_obb
    #   `ltrb_r` present → 5-ch ltrb head     → encode_targets_ltrb
    #   else             → 5-ch (cxy, wh) head → encode_targets
    aliases = getattr(model, "aliases", {})
    has_obb = "ang_r" in aliases
    has_ltrb = "ltrb_r" in aliases and not has_obb

    if has_obb:
        # OBB head is COMPLETELY unaware of AABB-from-COCO. Two enforcements:
        #   1. Drop samples without OBB sidecars (no silent θ=0 fallback).
        #   2. REPLACE each sample's boxes (loaded from COCO) with the enclosing
        #      AABB derived from its OBBs. After this point, NOTHING in the
        #      pipeline reads AABB-from-COCO for OBB models — Sample.boxes is
        #      strictly OBB-derived. SAM2/SAM-OBB output is the only source of
        #      truth.
        from opndet.encode import obb_to_aabb
        def _has_obb_label(s):
            return getattr(s, "obbs", None) is not None
        def _replace_boxes_with_obb_enclosing(s):
            if s.obbs is None or s.obbs.shape[0] == 0:
                s.boxes = np.zeros((0, 4), dtype=np.float32)
                return s
            xyxy = np.array([obb_to_aabb(*o) for o in s.obbs], dtype=np.float32)
            s.boxes = xyxy
            return s
        n_train_pre, n_val_pre, n_test_pre = len(train_s), len(val_s), len(test_s)
        train_s = [_replace_boxes_with_obb_enclosing(s) for s in train_s if _has_obb_label(s)]
        val_s   = [_replace_boxes_with_obb_enclosing(s) for s in val_s   if _has_obb_label(s)]
        test_s  = [_replace_boxes_with_obb_enclosing(s) for s in test_s  if _has_obb_label(s)]
        d_tr, d_v, d_te = n_train_pre - len(train_s), n_val_pre - len(val_s), n_test_pre - len(test_s)
        if d_tr or d_v or d_te:
            print(f"  OBB head: dropped samples without sidecar OBB labels "
                  f"(train -{d_tr}, val -{d_v}, test -{d_te})")
        if not train_s:
            raise RuntimeError(
                "OBB head selected but no training samples have OBB sidecars. "
                "Run `opndet sam-obb --coco ... --images ... --out <obb_dir>` "
                "first and point `data.sources[*].obb_dir` at it."
            )

    def _obb_encode_fn(boxes_xyxy, obbs=None):
        # OBB GT path: REQUIRES real OBB GT. Samples without OBB sidecars are
        # filtered out at startup (see filter below). If we still get here
        # without OBBs and have boxes, that's a pipeline bug — fail loud rather
        # than silently train against θ=0 AABB-as-OBB (the original collapse).
        if obbs is not None and len(obbs) > 0:
            return encode_targets_obb(obbs, cfg_shim)
        if boxes_xyxy.shape[0] == 0:
            return encode_targets_obb(np.zeros((0, 5), dtype=np.float32), cfg_shim)
        raise RuntimeError(
            f"OBB head got {boxes_xyxy.shape[0]} boxes but no OBB GT. "
            "Sample-filter should have dropped this; check dataset.OpndetDataset "
            "wiring or rerun `opndet sam-obb` to generate sidecars."
        )
    _obb_encode_fn._takes_obbs = True  # type: ignore[attr-defined]

    if has_obb:
        encode_fn = _obb_encode_fn
        print("  head variant: OBB (6-channel direct cxywhθ + ProbIoU loss)")
    elif has_ltrb:
        encode_fn = partial(encode_targets_ltrb, cfg=cfg_shim)
        print("  head variant: ltrb (5-channel)")
    else:
        encode_fn = partial(encode_targets, cfg=cfg_shim, dist_head=has_dist)
    cache = bool(c.get("cache_images", False))
    mosaic_prob = float(aug_cfg.mosaic_prob if hasattr(aug_cfg, "mosaic_prob") else 0.0)
    min_vis = float(aug_cfg.min_visible_frac if hasattr(aug_cfg, "min_visible_frac") else 0.5)
    # 4-ch eval policy: val/test datasets carry the prior synth too — eval
    # F1/mAP then reflect deployed (warm-prior) conditions, not RGB-only
    # cold-start. Selection on f1_opt_cal then picks the checkpoint that's
    # best AT DEPLOYMENT, and we get a real signal that the model is using
    # the prior. Set eval_cold_start: true to ALSO run a zero-prior pass
    # logged as val_cold/* and test_cold/*.
    train_ds = OpndetDataset(train_s, img_h, img_w, augment_fn=aug_fn, encode_fn=encode_fn,
                             cache_images=cache, mosaic_prob=mosaic_prob, min_visible_frac=min_vis,
                             in_ch=in_ch, prior_synth=prior_synth, stride=stride)
    val_ds = OpndetDataset(val_s, img_h, img_w, augment_fn=None, encode_fn=encode_fn, cache_images=cache,
                           in_ch=in_ch, prior_synth=prior_synth, stride=stride)
    test_ds = OpndetDataset(test_s, img_h, img_w, augment_fn=None, encode_fn=encode_fn, cache_images=cache,
                            in_ch=in_ch, prior_synth=prior_synth, stride=stride)
    cold_val_ds = cold_test_ds = None
    if in_ch == 4 and bool(c.get("eval_cold_start", False)):
        cold_val_ds = OpndetDataset(val_s, img_h, img_w, augment_fn=None, encode_fn=encode_fn,
                                    cache_images=cache, in_ch=in_ch, prior_synth=None, stride=stride)
        cold_test_ds = OpndetDataset(test_s, img_h, img_w, augment_fn=None, encode_fn=encode_fn,
                                     cache_images=cache, in_ch=in_ch, prior_synth=None, stride=stride)
    nw = int(c.get("num_workers", 2))
    if cache:
        # Pre-decode images into the cache HERE (main process) before the DataLoaders fork their
        # workers → workers inherit ONE shared, fully-populated, never-growing copy (copy-on-write)
        # instead of each worker growing its own cache toward the whole dataset over epochs
        # (≈num_workers× the dataset in RAM, rising every epoch — the leak). Capped by cache_max_mb.
        rem = float(c.get("cache_max_mb", 32768)); n_cached = 0
        for d in (train_ds, val_ds, test_ds):
            rem -= d.warm_cache(max_mb=max(0.0, rem)); n_cached += len(d._cache)
        n_tot = len(train_s) + len(val_s) + len(test_s)
        used_gb = (float(c.get("cache_max_mb", 32768)) - max(0.0, rem)) / 1024
        print(f"  cache_images: pre-decoded {n_cached}/{n_tot} images (~{used_gb:.1f} GB, shared via "
              f"copy-on-write across {nw} workers)" + ("" if n_cached >= n_tot else
              f" — hit cache_max_mb={c.get('cache_max_mb', 32768)}; the rest decode on the fly"))
    pf = int(c.get("prefetch_factor", 4)) if nw > 0 else None
    # persistent_workers: keep worker procs alive between epochs (default true,
    # faster but cv2/numpy heaps in workers slowly accumulate over many epochs
    # on long runs — set persistent_workers: false in train.yaml to bounce
    # workers each iter and trade ~few seconds/epoch for guaranteed RAM release).
    persist = bool(c.get("persistent_workers", True)) and nw > 0
    train_kw = dict(num_workers=nw, collate_fn=collate, pin_memory=device.type == "cuda",
                    persistent_workers=persist, prefetch_factor=pf)
    eval_kw = {**train_kw, "pin_memory": False}  # val/test don't need pinned memory
    train_kw = {k: v for k, v in train_kw.items() if v is not None}
    eval_kw = {k: v for k, v in eval_kw.items() if v is not None}
    train_loader = InfiniteDataLoader(train_ds, batch_size=int(c["batch_size"]), shuffle=True, **train_kw)
    val_loader = InfiniteDataLoader(val_ds, batch_size=int(c["batch_size"]), shuffle=False, **eval_kw)
    # test_loader runs once at end of training; no benefit to keeping it alive.
    test_loader = DataLoader(test_ds, batch_size=int(c["batch_size"]), shuffle=False, **eval_kw)
    # Cold loaders: num_workers=0 (main-process data loading). They run once per
    # epoch on the small val/test split; spawning workers here adds fork+import
    # overhead and creates a multi-minute freeze on the first iteration when
    # paired with the existing 3 InfiniteDataLoaders that already hold workers
    # alive. Single-threaded numpy/cv2 in main is fast enough for a sequential
    # pass over 846 val samples (~10-15s).
    cold_kw = {"collate_fn": collate, "num_workers": 0, "pin_memory": False}
    cold_val_loader = (
        DataLoader(cold_val_ds, batch_size=int(c["batch_size"]), shuffle=False, **cold_kw)
        if cold_val_ds is not None else None
    )
    cold_test_loader = (
        DataLoader(cold_test_ds, batch_size=int(c["batch_size"]), shuffle=False, **cold_kw)
        if cold_test_ds is not None else None
    )

    loss_kw = c.get("loss") or {}
    loss_kw.setdefault("img_h", img_h)
    loss_kw.setdefault("img_w", img_w)
    loss_kw.setdefault("stride", cfg_shim.stride)
    # convexity_radius is the *cap* on the per-object convexity window (heatmap
    # cells). The window itself is already box-derived (σ ∝ box/stride) and
    # auto-scales; the cap just needs to be ≥ the biggest object's radius so it
    # never clamps. "auto" (or unset) → size it from the dataset's p99 GT box.
    if loss_kw.get("convexity_radius") in (None, "auto"):
        if _box_min_sides.size:
            cr = math.ceil(float(np.percentile(_box_min_sides, 99)) / max(1, cfg_shim.stride) / 2.0) + 1
        else:
            cr = 4
        loss_kw["convexity_radius"] = int(max(2, min(12, cr)))  # cap at 12: (2·12+1)² unfold ~3.9 GB @bs128
        if float(loss_kw.get("convexity_weight", 0.0)) > 0:
            print(f"convexity: radius auto → {loss_kw['convexity_radius']} cells (p99 GT min-side / 2 / stride, capped 12)")
    # hm_target: ellipse already encodes "be object-shaped and centered", so the
    # convexity regularizer is largely redundant on top of it — default it OFF
    # (user can still set loss.convexity_weight explicitly to add it back).
    if str(c.get("hm_target", "gaussian")) == "ellipse":
        if "convexity_weight" not in loss_kw:
            loss_kw["convexity_weight"] = 0.0
            print("convexity: weight defaulted to 0.0 — hm_target: ellipse already enforces "
                  "centered, object-shaped peaks (set loss.convexity_weight to add the regularizer back)")
        elif float(loss_kw.get("convexity_weight", 0.0)) > 0.0:
            print(f"note: loss.convexity_weight={loss_kw['convexity_weight']} with hm_target: ellipse is "
                  "largely redundant — the ellipse target already centers the heatmap; consider lowering it")
    # Auto-route wh_loss to match the head variant. User can override.
    if has_obb:
        loss_kw.setdefault("wh_loss", "obb")
    elif has_ltrb:
        loss_kw.setdefault("wh_loss", "ltrb")
    # repulsion (RepGT, a box-geometry loss) is only wired into the standard
    # (cx,cy,w,h) head — the obb/ltrb branches don't construct the axis-aligned
    # pred boxes it needs, so `repulsion_weight` is a silent no-op there. Warn.
    if float(loss_kw.get("repulsion_weight", 0.0)) > 0.0 and loss_kw.get("wh_loss") in ("obb", "ltrb"):
        print(f"  WARN: loss.repulsion_weight={loss_kw['repulsion_weight']} is IGNORED for the "
              f"{loss_kw['wh_loss']} head (repulsion only runs on the cxy/wh head). "
              f"Use loss.peak_sharpen_weight for the touching-peak problem instead.")
    # Auto-mirror the model's peak op so count-aware loss sees the same sparse map as inference.
    # User can still override by setting peak_kernel/peak_eps explicitly in the yaml's `loss:` block.
    peak_k, peak_eps = _detect_peak_op(model)
    if peak_k is not None:
        loss_kw.setdefault("peak_kernel", peak_k)
        loss_kw.setdefault("peak_eps", peak_eps)
        if loss_kw.get("count_weight", 0.0) > 0:
            print(f"count-aware loss: peak_kernel={peak_k}, peak_eps={peak_eps} (auto-detected from model)")
    loss_fn = OpndetBboxLoss(**loss_kw)

    # ROADMAP §1.8 Phase 3: TAL/STAL assigner for regression-side per-cell
    # positive selection. `assigner: peak` (or omitted) keeps the original
    # single-cell-per-GT path. `tal` / `stal` enable dense top-k assignment.
    # cls supervision stays on the Gaussian heatmap target either way (see
    # docs/engineering-decisions.md "Assigner choice").
    assigner_cfg = c.get("assigner")
    _assign_mode = "obb" if has_obb else ("ltrb" if has_ltrb else None)
    if assigner_cfg is not None and _assign_mode is None:
        print(f"  WARN: assigner: '{assigner_cfg}' set but head is not ltrb/obb; ignoring")
        assigner = None
    elif has_obb and assigner_cfg not in (None, "peak"):
        # Direct cxywhθ head (post ProbIoU refactor) is fundamentally
        # single-positive: cx_off ∈ [0, 1] is the cell-relative offset, only
        # one cell can naturally encode it. TAL/STAL (multi-positive) breaks
        # the encoding. Fall back to peak with a warning.
        print(f"  WARN: assigner: '{assigner_cfg}' is incompatible with the direct "
              f"cxywhθ OBB head — falling back to peak (single-positive). "
              f"TAL/STAL only works with ltrb regression.")
        assigner = None
    else:
        assigner = build_assigner(assigner_cfg, mode=_assign_mode or "ltrb",
                                  img_h=img_h, img_w=img_w, stride=cfg_shim.stride)
    if assigner is not None:
        kind = "stal" if assigner.stal else "tal"
        print(f"assigner: {kind} (topk={assigner.topk}, alpha={assigner.alpha}, "
              f"beta={assigner.beta}, mode={assigner.mode})")

    # Loss-weight curriculum. Two YAML forms:
    #   curriculum: warmup_wh           # shorthand: ramp w_wh from 0 → final
    #                                   # over the first 20% of total epochs
    #   curriculum:                     # custom per-weight schedule
    #     w_wh:  {start_epoch: 0,  end_epoch: 15, start_value: 0.0, end_value: 1.5}
    #     w_cxy: {start_epoch: 0,  end_epoch:  5, start_value: 0.0, end_value: 1.0}
    # Each entry is a linear ramp; before start_epoch = start_value, after
    # end_epoch = end_value.
    curriculum_cfg = c.get("curriculum")
    curriculum_schedule: dict[str, dict] = {}
    _curriculum_epochs = int(c.get("epochs", 100))   # `epochs` not yet bound here
    # ProgLoss: auto-balance loss-term weights so each contributes equally.
    # `curriculum: progloss` shorthand or {progloss: {ema_decay, smooth, components}}.
    progloss_enabled = False
    progloss_ema_decay = 0.9
    progloss_smooth = 0.2  # fraction toward target per epoch
    progloss_components = ["l_hm", "l_wh", "l_cxy"]
    progloss_attr_map = {"l_hm": "w_hm", "l_wh": "w_wh", "l_cxy": "w_cxy"}
    progloss_min_weight = 0.05
    progloss_max_weight = 50.0
    if curriculum_cfg == "progloss":
        progloss_enabled = True
    elif isinstance(curriculum_cfg, dict) and "progloss" in curriculum_cfg:
        progloss_enabled = True
        pcfg = curriculum_cfg.pop("progloss")
        if isinstance(pcfg, dict):
            progloss_ema_decay = float(pcfg.get("ema_decay", progloss_ema_decay))
            progloss_smooth = float(pcfg.get("smooth", progloss_smooth))
            comps = pcfg.get("components")
            if comps:
                progloss_components = [str(x) for x in comps]
            progloss_min_weight = float(pcfg.get("min_weight", progloss_min_weight))
            progloss_max_weight = float(pcfg.get("max_weight", progloss_max_weight))
    if curriculum_cfg == "warmup_wh":
        warmup_end = max(1, int(_curriculum_epochs * 0.20))
        curriculum_schedule = {
            "w_wh":  {"start_epoch": 0, "end_epoch": warmup_end,
                      "start_value": 0.0, "end_value": float(loss_fn.w_wh)},
        }
        print(f"curriculum: warmup_wh — w_wh ramps 0 → {loss_fn.w_wh:.2f} over epochs 1-{warmup_end}")
    elif isinstance(curriculum_cfg, dict):
        for k, spec in curriculum_cfg.items():
            curriculum_schedule[k] = dict(spec)
        if curriculum_schedule:
            print(f"curriculum: custom schedule — {sorted(curriculum_schedule.keys())}")

    # Curriculum keys -> loss_fn attribute names. Constructor uses the long
    # names but stores into short attrs (e.g. repulsion_weight -> self.rep_w),
    # so hasattr() on the long name is False and the curriculum was a silent
    # no-op for these. Map them.
    _curriculum_attr_aliases = {
        "repulsion_weight":  "rep_w",
        "count_weight":      "count_w",
        "convexity_weight":  "convex_w",
        "dist_weight":       "dist_w",
    }
    # ProgLoss state. EMAs of (loss_total, loss_term_k) updated each batch by
    # _progloss_observe; new weights computed at epoch end by _progloss_step
    # so each component contributes ~equally to total.
    progloss_state: dict[str, float] = {"_total": 0.0, **{c: 0.0 for c in progloss_components}}
    progloss_init = {"_total": False, **{c: False for c in progloss_components}}
    progloss_w0 = {progloss_attr_map[k]: float(getattr(loss_fn, progloss_attr_map[k]))
                   for k in progloss_components if k in progloss_attr_map}
    if progloss_enabled:
        print(f"curriculum: progloss — auto-balancing {progloss_components} "
              f"(ema={progloss_ema_decay}, smooth={progloss_smooth})")

    def _progloss_observe(losses_d: dict) -> None:
        if not progloss_enabled:
            return
        # losses_d['loss'] is the scalar total (a tensor); component values are
        # detached scalars. Use the detached components only.
        total_v = float(losses_d.get("loss", torch.zeros(())).detach()) if hasattr(losses_d.get("loss", 0.0), "detach") else float(losses_d.get("loss", 0.0))
        if not progloss_init["_total"]:
            progloss_state["_total"] = total_v
            progloss_init["_total"] = True
        else:
            progloss_state["_total"] = progloss_ema_decay * progloss_state["_total"] + (1.0 - progloss_ema_decay) * total_v
        for k in progloss_components:
            v = losses_d.get(k)
            if v is None:
                continue
            v = float(v.detach()) if hasattr(v, "detach") else float(v)
            if not progloss_init[k]:
                progloss_state[k] = v
                progloss_init[k] = True
            else:
                progloss_state[k] = progloss_ema_decay * progloss_state[k] + (1.0 - progloss_ema_decay) * v

    def _progloss_step() -> None:
        if not progloss_enabled:
            return
        active = [k for k in progloss_components if progloss_init[k] and progloss_state[k] > 1e-9
                  and progloss_attr_map.get(k) and hasattr(loss_fn, progloss_attr_map[k])]
        if len(active) < 2:
            return
        n = len(active)
        # Target weight: each weighted component contributes 1/n of an arbitrary
        # reference (we use the mean of un-weighted component magnitudes so the
        # total scalar is roughly preserved, not exploded).
        ref = sum(progloss_state[k] for k in active) / n
        for k in active:
            attr = progloss_attr_map[k]
            base = getattr(loss_fn, attr)
            target = ref / max(progloss_state[k], 1e-9)
            target = max(progloss_min_weight, min(progloss_max_weight, target))
            new = (1.0 - progloss_smooth) * float(base) + progloss_smooth * target
            setattr(loss_fn, attr, float(new))

    def _apply_curriculum(epoch_1based: int) -> None:
        for name, spec in curriculum_schedule.items():
            s_ep = float(spec.get("start_epoch", 0))
            e_ep = float(spec.get("end_epoch", s_ep + 1))
            s_v  = float(spec.get("start_value", 0.0))
            e_v  = float(spec.get("end_value", 1.0))
            t = (epoch_1based - 1 - s_ep) / max(1e-9, e_ep - s_ep)
            t = max(0.0, min(1.0, t))
            v = s_v + (e_v - s_v) * t
            attr = _curriculum_attr_aliases.get(name, name)
            if hasattr(loss_fn, attr):
                setattr(loss_fn, attr, v)
            else:
                # Fail loud — silent no-op was the bug for repulsion/count/convex.
                if epoch_1based == 1:
                    print(f"  WARN: curriculum key '{name}' has no loss_fn attribute "
                          f"(tried '{attr}'); skipping. Available: w_hm w_cxy w_wh "
                          f"rep_w count_w convex_w dist_w")
    opt_name = str(c.get("optimizer", "adamw")).lower()
    if opt_name == "musgd":
        from opndet.optim_muon import MuSGD
        # Edge-tier guard: server-tier `-pro` is the intended target. Warn (don't
        # block) for known edge-tier or non-pro presets so the user has cover to
        # experiment without surprise.
        _SERVER_PRO = {"bbox-m-pro", "bbox-l-pro", "bbox-x-pro"}
        preset = str(c.get("model_config", "")).strip()
        if preset and preset not in _SERVER_PRO:
            print(
                f"  WARN: optimizer=musgd is intended for server-tier `-pro` presets "
                f"({sorted(_SERVER_PRO)}); got '{preset}'. Continuing — empirical wins "
                f"for vision are unproven; validate before promoting."
            )
        opt = MuSGD(
            model,
            lr=float(c["lr"]),
            weight_decay=float(c.get("weight_decay", 1e-4)),
            muon_momentum=float(c.get("muon_momentum", 0.95)),
            muon_lr_scale=float(c.get("muon_lr_scale", 1.0)),
            ns_steps=int(c.get("muon_ns_steps", 5)),
        )
        ps = opt.partition_summary()
        print(
            f"optimizer: musgd (muon_tensors={ps['muon_tensors']}, "
            f"adamw_tensors={ps['adamw_tensors']}, muon_params={ps['muon_params']:,}, "
            f"adamw_params={ps['adamw_params']:,}, "
            f"muon_momentum={c.get('muon_momentum', 0.95)}, "
            f"muon_lr_scale={c.get('muon_lr_scale', 1.0)})"
        )
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(c["lr"]), weight_decay=float(c.get("weight_decay", 1e-4)))
    else:
        raise ValueError(f"unknown optimizer '{opt_name}'; expected 'adamw' or 'musgd'")
    if resume_state is not None and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
    amp_dtype_str = str(c.get("amp_dtype", "fp16")).lower()
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}.get(amp_dtype_str, torch.float16)
    use_amp = device.type == "cuda" and c.get("amp", True)
    needs_scaler = use_amp and amp_dtype == torch.float16  # bf16 doesn't need GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    epochs = int(c["epochs"])
    total_steps = epochs * max(1, len(train_loader))
    base_lr = float(c["lr"])
    warmup = int(c.get("warmup_steps", 200))

    ema = None
    ema_decay = float(c.get("ema_decay", 0.0))
    # auto-scale tau so EMA warms up over ~2 epochs by default — better fit for short
    # opndet runs than YOLOv8's tau=2000 which is tuned for COCO-length schedules.
    default_tau = max(500, len(train_loader) * 2)
    ema_tau = int(c.get("ema_tau", default_tau))
    if ema_decay > 0:
        ema = EMA(model, decay=ema_decay, tau=ema_tau)
        if resume_state is not None and "ema" in resume_state and resume_state["ema"] is not None:
            ema.shadow.load_state_dict(resume_state["ema"])
            ema.step = int(resume_state.get("ema_step", 0))
        print(f"EMA enabled (decay={ema_decay}, tau={ema_tau})")

    teacher_model = None
    if teacher and self_distill:
        raise ValueError("--teacher and --self-distill are mutually exclusive")
    if teacher:
        from opndet.distill import load_teacher
        teacher_model, teacher_preset = load_teacher(teacher, device)
        if teacher_model.input_shape != model.input_shape:
            raise ValueError(f"teacher input_shape {teacher_model.input_shape} != student {model.input_shape}")
        print(f"distillation: teacher={teacher_preset} ({teacher})")
    elif self_distill:
        if ema is None:
            raise ValueError("--self-distill requires ema_decay > 0 in the config")
        print("self-distillation: teacher = EMA shadow")
    distill_cfg = (c.get("distill") or {}) if (teacher or self_distill) else {}
    distill_kw = {
        "hm_weight": float(distill_cfg.get("hm_weight", 1.0)),
        "reg_weight": float(distill_cfg.get("reg_weight", 0.5)),
        "conf_gate": float(distill_cfg.get("conf_gate", 0.5)),
        "full_distill": bool(distill_cfg.get("full_distill", False)),
        "neg_gate": float(distill_cfg.get("neg_gate", 0.0)),
        "kd_temperature": float(distill_cfg.get("kd_temperature", 1.0)),
    }

    auto_calibrate = bool(c.get("auto_calibrate", True))
    calibrate_every = int(c.get("calibrate_every", 0))
    test_every = int(c.get("test_every", 0))

    patience_smart = bool(c.get("patience_smart", False))
    patience_include_test = bool(c.get("patience_include_test", False))
    patience_min_delta = float(c.get("patience_min_delta", 0.003))
    # When patience_smart is True, patience fires only if NO tracked metric has improved
    # by patience_min_delta in `patience` epochs. Tracks both raw and calibrated f1/map.
    # patience_include_test adds test metrics to the watch list — slightly leaky (test
    # influences WHEN to stop, never WHAT to save), useful when test/val have noticeable lag.
    best_per_metric: dict[str, tuple[float, int]] = {}

    # Trajectory-patience: stop only when ALL key metrics' slopes flatten over a window.
    # Beats best-not-improved when one metric saturates while another is still climbing.
    # Curriculum-aware patience floor: don't fire trajectory-patience until all
    # curriculum stages have finished + a full window has passed. Prevents stops
    # like "ep 55 — flat" when a curriculum loss is mid-ramp at ep 50-90 (the new
    # signal hasn't kicked in yet but the existing metrics have plateaued).
    _last_curriculum_ep = 0
    if isinstance(curriculum_cfg, dict) and curriculum_schedule:
        for spec in curriculum_schedule.values():
            _last_curriculum_ep = max(_last_curriculum_ep, int(spec.get("end_epoch", 0)))
    patience_trajectory = bool(c.get("patience_trajectory", False))
    patience_window = int(c.get("patience_window", 15))
    patience_min_slope = float(c.get("patience_min_slope", 0.001))  # relative slope/epoch
    patience_rule = str(c.get("patience_rule", "any_climbing"))  # or "weighted_sum"
    # dict: metric_name -> direction (+1 higher-better, -1 lower-better). Magnitude = weight.
    patience_metrics_cfg: dict[str, float] = dict(c.get("patience_metrics", {
        "center_f1_lenient": 1.0,
        "map_50_95": 1.0,
        "center_ghost_rate": -1.0,
        "count_off_le1_frac": 0.5,
    }))
    metric_history: list[dict[str, float]] = []  # one dict per evaluated epoch
    if patience_trajectory:
        _floor = max(int(c.get("patience", 0)), _last_curriculum_ep + patience_window)
        print(f"trajectory-patience: window={patience_window} rule={patience_rule} "
              f"min_slope={patience_min_slope}  metrics={patience_metrics_cfg}  "
              f"earliest-fire-epoch={_floor} (curriculum_end={_last_curriculum_ep}+window)")

    n_vis = int(c.get("vis_samples", 4))
    vis_every = int(c.get("vis_every", 5))
    vis_imgs = []
    vis_boxes = []
    vis_trails: list[list] = []
    vis_obbs: list[np.ndarray] = []
    from opndet.decode import gt_obbs_from_targets as _gt_obbs_from_targets
    if n_vis > 0 and vis_every > 0:
        val_ds._return_trails = (in_ch == 4)
        for i in range(min(n_vis, len(val_ds))):
            r = val_ds[i]
            img_t, boxes, targets = r[0], r[1], r[2]
            vis_imgs.append(img_t)
            vis_boxes.append(boxes)
            vis_trails.append(r[3] if len(r) == 4 else [])
            if targets is not None and "obb" in targets and "pos" in targets:
                pos_b = targets["pos"].unsqueeze(0).numpy()
                obb_b = targets["obb"].unsqueeze(0).numpy()
                vis_obbs.append(_gt_obbs_from_targets(pos_b, obb_b, img_h, img_w, cfg_shim.stride)[0])
            else:
                vis_obbs.append(np.zeros((0, 5), dtype=np.float32))
        val_ds._return_trails = False
    vis_batch = torch.stack(vis_imgs, dim=0) if vis_imgs else None

    n_test_vis = int(c.get("test_samples", n_vis))
    test_vis_imgs = []
    test_vis_boxes = []
    test_vis_trails: list[list] = []
    test_vis_obbs: list[np.ndarray] = []
    if n_test_vis > 0 and test_every > 0 and len(test_ds) > 0:
        test_ds._return_trails = (in_ch == 4)
        for i in range(min(n_test_vis, len(test_ds))):
            r = test_ds[i]
            img_t, boxes, targets = r[0], r[1], r[2]
            test_vis_imgs.append(img_t)
            test_vis_boxes.append(boxes)
            test_vis_trails.append(r[3] if len(r) == 4 else [])
            if targets is not None and "obb" in targets and "pos" in targets:
                pos_b = targets["pos"].unsqueeze(0).numpy()
                obb_b = targets["obb"].unsqueeze(0).numpy()
                test_vis_obbs.append(_gt_obbs_from_targets(pos_b, obb_b, img_h, img_w, cfg_shim.stride)[0])
            else:
                test_vis_obbs.append(np.zeros((0, 5), dtype=np.float32))
        test_ds._return_trails = False
    test_vis_batch = torch.stack(test_vis_imgs, dim=0) if test_vis_imgs else None

    metric_for_best = str(c.get("metric_for_best", "f1"))
    valid_metrics = ("f1", "map50", "map_50_95", "f1_opt",
                     "f1_cal", "map50_cal", "map_50_95_cal", "f1_opt_cal",
                     "center_f1", "center_f1_lenient", "center_recall",
                     "obb_iou_mean", "obb_iou_p50",
                     "angle_err_deg_mean", "angle_err_deg_median",
                     "angle_err_le_10_frac", "angle_err_le_30_frac")
    if metric_for_best not in valid_metrics:
        raise ValueError(f"metric_for_best must be one of {valid_metrics}, got {metric_for_best}")
    metric_is_cal = metric_for_best.endswith("_cal")
    metric_base = metric_for_best.removesuffix("_cal")
    # If selecting on calibrated metric, force per-epoch calibration regardless of calibrate_every.
    if metric_is_cal and calibrate_every == 0:
        calibrate_every = 1
        print(f"metric_for_best={metric_for_best}: forcing calibrate_every=1")
    best_metric = -1.0
    best_epoch = 0
    patience = int(c.get("patience", 0))   # 0 = disabled

    # eval_threshold: numeric (static) OR "auto" (track threshold_opt with EMA).
    # Auto: each epoch's score_thresh = EMA of past threshold_opt values, so the
    # operating point self-tunes to where F1 actually peaks given the model's
    # current calibration. Reduces guesswork; trades 1-epoch lag for it.
    _eval_threshold_cfg = c.get("eval_threshold", 0.3)
    eval_threshold_auto = (isinstance(_eval_threshold_cfg, str)
                           and _eval_threshold_cfg.lower() == "auto")
    eval_threshold_ema_alpha = float(c.get("eval_threshold_ema_alpha", 0.3))
    eval_threshold_min = float(c.get("eval_threshold_min", 0.10))  # bootstrap floor
    cur_eval_threshold = eval_threshold_min if eval_threshold_auto else float(_eval_threshold_cfg)
    if eval_threshold_auto:
        print(f"eval_threshold: auto (EMA α={eval_threshold_ema_alpha}, "
              f"bootstrap={eval_threshold_min})")

    step = 0
    start_epoch = 0
    if resume_state is not None:
        start_epoch = int(resume_state.get("epoch", 0))
        best_metric = float(resume_state.get("best_metric", resume_state.get("best_f1", -1.0)))
        best_epoch = int(resume_state.get("best_epoch", start_epoch))
        step = int(resume_state.get("step", start_epoch * max(1, len(train_loader))))
        if needs_scaler and "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])
        print(f"resuming from epoch {start_epoch + 1}, best_{metric_for_best}={best_metric:.3f} @ epoch {best_epoch}, step={step}")

    # Skip vis/test on plateaued epochs. Fires only when best.pt has moved
    # since the last fire (or this epoch is the new best, or it's a boundary
    # epoch ep==1 / ep==epochs). Saves test-eval cost on flat plateaus and
    # keeps TB image logs sparse-but-meaningful.
    viz_only_on_improvement = bool(c.get("viz_only_on_improvement", True))
    last_vis_epoch = 0
    last_test_epoch = 0

    for epoch in range(start_epoch, epochs):
        _apply_curriculum(epoch + 1)
        _progloss_step()
        model.train()
        t0 = time.time()
        running: dict[str, float] = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for imgs, boxes_list, tgt in pbar:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, base_lr, warmup=warmup)
            imgs = imgs.to(device, non_blocking=True)
            tgt = {k: v.to(device, non_blocking=True) for k, v in tgt.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                raw5 = model.forward_with_alias(imgs, "raw")
                if assigner is not None:
                    # Forward gives raw logits; sigmoid for the assigner's
                    # alignment metric. Box channels are first 4 of the reg
                    # output (sigmoid'd). Run under no_grad inside the autocast.
                    with torch.no_grad():
                        pred_obj_s = torch.sigmoid(raw5[:, 0:1].float())
                        pred_box_s = torch.sigmoid(raw5[:, 1:5].float())
                    a_out = assigner(pred_obj_s, pred_box_s, boxes_list)
                    new_pos = a_out["pos"].to(tgt["pos"].dtype)
                    if has_obb and "obb" in tgt:
                        # Replace ltrb edges; propagate angle channels from
                        # each GT's center cell to all assigned cells.
                        ltrb_new = a_out["ltrb"].to(tgt["obb"].dtype)
                        ang_new = tgt["obb"][:, 4:6].clone()
                        new_ang_mask = torch.zeros_like(tgt["angle_mask"]) if "angle_mask" in tgt else None
                        H_out = ang_new.shape[-2]; W_out = ang_new.shape[-1]
                        for b in range(ang_new.shape[0]):
                            assigned_b = a_out["assigned_gt"][b]
                            unique_gts = torch.unique(assigned_b[assigned_b >= 0])
                            bxs = boxes_list[b]
                            if not isinstance(bxs, torch.Tensor):
                                bxs = torch.as_tensor(bxs, dtype=torch.float32)
                            for n in unique_gts.tolist():
                                if bxs.ndim != 2 or bxs.shape[0] <= n:
                                    continue
                                gx = bxs[n]
                                cx_g = int(min(max(((float(gx[0]) + float(gx[2])) * 0.5 / cfg_shim.stride), 0.0), W_out - 1))
                                cy_g = int(min(max(((float(gx[1]) + float(gx[3])) * 0.5 / cfg_shim.stride), 0.0), H_out - 1))
                                ang_val = tgt["obb"][b, 4:6, cy_g, cx_g]
                                sel = (assigned_b == n)
                                if sel.any():
                                    ang_new[b, :, sel] = ang_val.view(2, 1)
                                    if new_ang_mask is not None:
                                        is_non_round = float(tgt["angle_mask"][b, 0, cy_g, cx_g]) > 0
                                        if is_non_round:
                                            new_ang_mask[b, 0][sel] = 1.0
                        tgt["obb"] = torch.cat([ltrb_new, ang_new], dim=1)
                        if new_ang_mask is not None:
                            tgt["angle_mask"] = new_ang_mask
                    elif has_ltrb and "ltrb" in tgt:
                        tgt["ltrb"] = a_out["ltrb"].to(tgt["ltrb"].dtype)
                    tgt["pos"] = new_pos
                losses = loss_fn(raw5, tgt)

                if teacher_model is not None or self_distill:
                    from opndet.distill import distillation_loss
                    active_teacher = teacher_model if teacher_model is not None else ema.shadow
                    with torch.no_grad():
                        t_out = active_teacher(imgs)
                        t_out = t_out["output"] if isinstance(t_out, dict) else t_out
                    kd = distillation_loss(raw5, t_out, **distill_kw)
                    losses["loss"] = losses["loss"] + kd["l_kd"]
                    losses["l_kd_hm"] = kd["l_kd_hm"].detach()
                    losses["l_kd_reg"] = kd["l_kd_reg"].detach()
            # GradScaler expects a true torch.optim.Optimizer. MuSGD wraps two
            # of them; dispatch to each leg so unscale_/step work correctly.
            opt_legs = []
            if hasattr(opt, "muon") and hasattr(opt, "adamw"):
                if opt.muon is not None:
                    opt_legs.append(opt.muon)
                if opt.adamw is not None:
                    opt_legs.append(opt.adamw)
            else:
                opt_legs = [opt]
            if needs_scaler:
                scaler.scale(losses["loss"]).backward()
                for _o in opt_legs:
                    scaler.unscale_(_o)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                for _o in opt_legs:
                    scaler.step(_o)
                scaler.update()
            else:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                opt.step()
            if ema is not None:
                ema.update(model)
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    running[k] = running.get(k, 0.0) + float(v.detach())
            _progloss_observe(losses)
            step += 1
            if step % 5 == 0:
                pbar.set_postfix(loss=f"{losses['loss'].item():.3f}", lr=f"{opt.param_groups[0]['lr']:.1e}")

        n_iter = max(1, len(train_loader))
        avg = {k: v / n_iter for k, v in running.items()}
        dt = time.time() - t0
        eval_model = ema.shadow if ema is not None else model
        _t_phase = time.time()
        m = evaluate(eval_model, val_loader, cfg_shim, device, score_thresh=cur_eval_threshold)
        _t_val = time.time() - _t_phase
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1:3d}/{epochs}  lr={cur_lr:.2e}  loss={avg['loss']:.4f}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  F1_opt={m['f1_opt']:.3f}@{m['threshold_opt']:.2f}  mAP@.5/.95={m['map50']:.3f}/{m['map_50_95']:.3f}  shape=.{int(m['map50_shape']*1000):03d}/.{int(m['map_50_95_shape']*1000):03d}  (train {dt:.1f}s val {_t_val:.1f}s)")
        print(f"          center: R={m['center_recall']:.3f} P={m['center_precision']:.3f} (lenient {m['center_precision_lenient']:.3f}) F1={m['center_f1']:.3f} (lenient {m['center_f1_lenient']:.3f})  ghost={m['center_ghost_rate']:.1%} dup={m['center_dup_rate']:.1%}  hit(perf/1c/2c)={m['center_perfect_rate']:.1%}/{m['center_within_1cell']:.1%}/{m['center_within_2cell']:.1%}  count±1={m['count_off_le1_frac']:.1%}")
        if m.get("obb_iou_mean", 0.0) > 0 or m.get("angle_err_deg_mean", 0.0) > 0:
            print(f"          obb: iou_mean={m['obb_iou_mean']:.3f} iou_p50={m['obb_iou_p50']:.3f}  ang_err_deg(mean/median)={m['angle_err_deg_mean']:.1f}/{m['angle_err_deg_median']:.1f}  ang≤10°={m['angle_err_le_10_frac']:.1%}  ang≤30°={m['angle_err_le_30_frac']:.1%}")

        ep = epoch + 1
        writer.add_scalar("lr", cur_lr, ep)
        for k, v in avg.items():
            writer.add_scalar(f"train/{k}", v, ep)

        # Auto eval_threshold: EMA-track threshold_opt for next epoch.
        if eval_threshold_auto:
            new_t = float(m.get("threshold_opt", cur_eval_threshold))
            cur_eval_threshold = max(eval_threshold_min,
                                     (1 - eval_threshold_ema_alpha) * cur_eval_threshold
                                     + eval_threshold_ema_alpha * new_t)
            writer.add_scalar("eval_threshold", cur_eval_threshold, ep)
            print(f"          eval_threshold: {cur_eval_threshold:.3f} "
                  f"(EMA, this-ep optimum={new_t:.2f})")

        # Cold-start diagnostic: same val with zero priors. Quantifies how
        # much the prior is helping. Selection metric still uses warm `m`.
        if cold_val_loader is not None:
            _t_phase = time.time()
            m_cold = evaluate(eval_model, cold_val_loader, cfg_shim, device,
                              score_thresh=cur_eval_threshold)
            _t_cold = time.time() - _t_phase
            print(f"  cold (zero-prior): F1={m_cold['f1']:.3f}  F1_opt={m_cold['f1_opt']:.3f}@{m_cold['threshold_opt']:.2f}  mAP@.5={m_cold['map50']:.3f}")
            for k, v in m_cold.items():
                writer.add_scalar(f"val_cold/{k}", v, ep)
            lift_keys = ("f1", "f1_opt", "precision", "recall", "map50", "map_50_95")
            lifts = {k: float(m[k]) - float(m_cold[k]) for k in lift_keys if k in m and k in m_cold}
            for k, v in lifts.items():
                writer.add_scalar(f"prior_lift/val/{k}", v, ep)
            print(f"  prior lift (val): F1{'+' if lifts.get('f1', 0) >= 0 else ''}{lifts.get('f1', 0):+.3f}  F1_opt{lifts.get('f1_opt', 0):+.3f}  mAP{lifts.get('map50', 0):+.3f}/{lifts.get('map_50_95', 0):+.3f}  (cold {_t_cold:.1f}s)")
        for k, v in m.items():
            writer.add_scalar(f"val/{k}", v, ep)
        writer.add_scalar("time/epoch_s", dt, ep)

        # Auto hard-negative mining: every N epochs (or when the ghost rate
        # crosses a gate) after a warmup, mine confident phantom centers from a
        # data slice into the paste pool, then rebuild the train loader so the
        # next epoch's workers pick up the new patches.
        if auto_mine_cfg is not None:
            _am_start = int(auto_mine_cfg.get("start_epoch", 50))
            _am_every = max(1, int(auto_mine_cfg.get("every", 10)))
            _am_gate = float(auto_mine_cfg.get("ghost_rate_gate", 2.0))  # >1 ⇒ gate disabled
            _due = ep >= _am_start and ((ep - _am_start) % _am_every == 0
                                        or float(m.get("center_ghost_rate", 0.0)) > _am_gate)
            if _due:
                from opndet.mine_negatives import mine_into_pool
                _src = str(auto_mine_cfg.get("source", "val")).lower()
                _src_samples = train_s if _src == "train" else val_s
                _thr_cfg = auto_mine_cfg.get("threshold")
                _mine_thr = float(_thr_cfg) if _thr_cfg is not None else float(cur_eval_threshold)
                _t_mine = time.time()
                n_new = mine_into_pool(
                    eval_model, _src_samples,
                    img_h=img_h, img_w=img_w, in_ch=in_ch, stride=stride, encode_fn=encode_fn,
                    score_thresh=_mine_thr, pool_dir=aug_cfg.hard_negative_pool, device=device,
                    patch_size=int(auto_mine_cfg.get("patch_size", 32)),
                    top_k_per_sample=int(auto_mine_cfg.get("top_k_per_sample", 4)),
                    max_total=int(auto_mine_cfg.get("max_pool", 500)),
                    max_scan=int(auto_mine_cfg.get("max_scan", 400)),
                    ghost_radius_frac=float(auto_mine_cfg.get("ghost_radius_frac", 0.5)),
                    epoch_tag=ep,
                )
                from opndet.mine_negatives import load_pool as _load_pool
                _pool_now = len(_load_pool(aug_cfg.hard_negative_pool))
                print(f"  auto-mine: +{n_new} hard-neg patches @thr={_mine_thr:.2f} from {_src} "
                      f"({len(_src_samples)} imgs); pool now {_pool_now}  ({time.time()-_t_mine:.1f}s)")
                writer.add_scalar("auto_mine/patches_added", n_new, ep)
                writer.add_scalar("auto_mine/pool_size", _pool_now, ep)
                if _pool_now > 0:
                    train_ds.aug = make_augment(aug_cfg, hn_pool=_load_pool(aug_cfg.hard_negative_pool))
                    # Tear the old InfiniteDataLoader's persistent workers down
                    # *now*, in the main process, before forking the new ones —
                    # otherwise the new workers inherit the still-alive old
                    # iterator and, on exit, spew "AssertionError: can only test
                    # a child process" from its __del__ (harmless but ugly; Colab
                    # / torch+py3.12). Explicit shutdown + gc makes it go away.
                    _old_it = getattr(train_loader, "iterator", None)
                    train_loader = None
                    if _old_it is not None and hasattr(_old_it, "_shutdown_workers"):
                        try:
                            _old_it._shutdown_workers()
                        except Exception:
                            pass
                    del _old_it
                    import gc as _gc
                    _gc.collect()
                    train_loader = InfiniteDataLoader(train_ds, batch_size=int(c["batch_size"]),
                                                      shuffle=True, **train_kw)

        m_cal = None
        cur_T = 1.0
        if calibrate_every > 0 and ep % calibrate_every == 0:
            from opndet.calibrate import (apply_temperature, collect_calibration_data,
                                            fit_temperature)
            from opndet.metrics import calibration_bins
            _t_phase = time.time()
            apply_temperature(eval_model, 1.0)
            _logits, _labels = collect_calibration_data(eval_model, val_loader, cfg_shim, device)
            if _logits.shape[0] > 0:
                cur_T = fit_temperature(_logits, _labels)
                _sig_raw = (1.0 / (1.0 + np.exp(-_logits))).astype(np.float32)
                _sig_cal = (1.0 / (1.0 + np.exp(-_logits / cur_T))).astype(np.float32)
                _ece_pre = calibration_bins(_sig_raw, _labels)["ece"]
                _ece_post = calibration_bins(_sig_cal, _labels)["ece"]
                writer.add_scalar("eval/T", cur_T, ep)
                writer.add_scalar("eval/ece_pre", _ece_pre, ep)
                writer.add_scalar("eval/ece_post", _ece_post, ep)
                # Re-eval with T applied to get true calibrated metrics; needed when selecting on _cal,
                # also useful as a TB readout when calibrate_every fires.
                apply_temperature(eval_model, cur_T)
                m_cal = evaluate(eval_model, val_loader, cfg_shim, device,
                                 score_thresh=cur_eval_threshold)
                for k, v in m_cal.items():
                    writer.add_scalar(f"val_cal/{k}", v, ep)
                _t_calib = time.time() - _t_phase
                print(f"  calib: T={cur_T:.3f}  ECE {_ece_pre:.3f} -> {_ece_post:.3f}  "
                      f"F1_cal={m_cal['f1']:.3f}  mAP_cal={m_cal['map50']:.3f}/{m_cal['map_50_95']:.3f}  ({_t_calib:.1f}s)")
                # Restore T=1.0 so subsequent epochs' raw eval starts clean.
                apply_temperature(eval_model, 1.0)

        # Resolve vis threshold: explicit `vis_threshold: <num>` always wins.
        # Otherwise: when selecting on f1_opt*, track the dynamic per-epoch optimum
        # (so the grid shows what the model would actually deploy at). Else eval_threshold.
        _vt = c.get("vis_threshold", "auto")
        if isinstance(_vt, (int, float)):
            vis_thresh_now = float(_vt)
        elif metric_for_best.startswith("f1_opt") and "threshold_opt" in m:
            vis_thresh_now = float(m["threshold_opt"])
        else:
            vis_thresh_now = cur_eval_threshold

        # Selection metric (computed up here so we can gate test/vis on
        # is_new_best): calibrated value if metric_for_best ends in _cal AND
        # we got m_cal this epoch. Falls back to raw otherwise.
        if metric_is_cal and m_cal is not None:
            cur = float(m_cal[metric_base])
        else:
            cur = float(m[metric_base if metric_is_cal else metric_for_best])
        is_new_best = cur > best_metric
        is_boundary = (ep == 1) or (ep == epochs)

        def _should_fire(last_fire_epoch: int) -> bool:
            if not viz_only_on_improvement:
                return True
            if is_boundary or is_new_best:
                return True
            # best.pt was updated at some point since our last fire
            return best_epoch > last_fire_epoch

        if test_every > 0 and ep % test_every == 0 and len(test_ds) > 0 and _should_fire(last_test_epoch):
            mt = evaluate(eval_model, test_loader, cfg_shim, device, score_thresh=cur_eval_threshold)
            print(f"  test: P={mt['precision']:.3f} R={mt['recall']:.3f} F1={mt['f1']:.3f}  mAP@.5={mt['map50']:.3f} mAP@.5:.95={mt['map_50_95']:.3f}")
            for k, v in mt.items():
                writer.add_scalar(f"test/{k}", v, ep)
            if cold_test_loader is not None:
                mt_cold = evaluate(eval_model, cold_test_loader, cfg_shim, device,
                                   score_thresh=cur_eval_threshold)
                print(f"  test cold (zero-prior): F1={mt_cold['f1']:.3f}  F1_opt={mt_cold['f1_opt']:.3f}@{mt_cold['threshold_opt']:.2f}  mAP@.5={mt_cold['map50']:.3f}")
                for k, v in mt_cold.items():
                    writer.add_scalar(f"test_cold/{k}", v, ep)
                lift_keys = ("f1", "f1_opt", "precision", "recall", "map50", "map_50_95")
                lifts = {k: float(mt[k]) - float(mt_cold[k]) for k in lift_keys if k in mt and k in mt_cold}
                for k, v in lifts.items():
                    writer.add_scalar(f"prior_lift/test/{k}", v, ep)
                print(f"  prior lift (test): F1{lifts.get('f1', 0):+.3f}  F1_opt{lifts.get('f1_opt', 0):+.3f}  mAP{lifts.get('map50', 0):+.3f}/{lifts.get('map_50_95', 0):+.3f}")
            # patience hook: when patience_smart and patience_include_test, count test improvements.
            if patience_smart and patience_include_test:
                for k in ("f1", "f1_opt", "map50", "map_50_95"):
                    if k in mt:
                        v = float(mt[k])
                        tk = f"test_{k}"
                        prev_v, _ = best_per_metric.get(tk, (-1e9, 0))
                        if v > prev_v + patience_min_delta:
                            best_per_metric[tk] = (v, ep)
            if test_vis_batch is not None:
                grid = render_predictions(
                    eval_model, test_vis_batch, test_vis_boxes, img_h, img_w, cfg_shim.stride,
                    threshold=vis_thresh_now, device=device, trails_per=test_vis_trails,
                )
                writer.add_images("test/preds", grid, ep, dataformats="NCHW")
                if db is not None:
                    from opndet.visualize import save_layered_vis
                    save_layered_vis(eval_model, test_vis_batch, test_vis_boxes,
                                     img_h, img_w, cfg_shim.stride,
                                     _save_layered_vis_path(out_dir, "test/preds", ep),
                                     db, "test/preds", ep,
                                     threshold=vis_thresh_now, device=device,
                                     trails_per=test_vis_trails,
                                     gt_obbs_per=test_vis_obbs)
            last_test_epoch = ep

        if vis_batch is not None and (ep == 1 or ep % vis_every == 0 or ep == epochs) and _should_fire(last_vis_epoch):
            _t_phase = time.time()
            # Vis through the EMA shadow + current T applied → what the
            # deployed (calibrated) model actually outputs. Without this,
            # peaks land at raw-sigmoid values (~0.5–0.85) and the obj
            # heatmap looks dim. With T applied (typically ~0.3–0.5 for
            # this dataset) peaks saturate near 1.0 like the bbox-x
            # teacher checkpoint shows.
            from opndet.calibrate import apply_temperature as _apply_T
            vis_T = float(cur_T) if cur_T and cur_T != 1.0 else 1.0
            if vis_T != 1.0:
                _apply_T(eval_model, vis_T)
            grid = render_predictions(
                eval_model, vis_batch, vis_boxes, img_h, img_w, cfg_shim.stride,
                threshold=vis_thresh_now, device=device, trails_per=vis_trails,
            )
            writer.add_images("val/preds", grid, ep, dataformats="NCHW")
            if db is not None:
                from opndet.visualize import save_layered_vis
                save_layered_vis(eval_model, vis_batch, vis_boxes,
                                 img_h, img_w, cfg_shim.stride,
                                 _save_layered_vis_path(out_dir, "val/preds", ep),
                                 db, "val/preds", ep,
                                 threshold=vis_thresh_now, device=device,
                                 trails_per=vis_trails,
                                 gt_obbs_per=vis_obbs)
            if vis_T != 1.0:
                _apply_T(eval_model, 1.0)
            last_vis_epoch = ep
            print(f"  vis: val/preds rendered (T={vis_T:.3f}) ({time.time() - _t_phase:.1f}s)")
        # If EMA is on, save EMA weights as the deployed model — they're the eval-quality ones.
        deployed_state = ema.shadow.state_dict() if ema is not None else model.state_dict()
        ckpt = {
            "epoch": ep,
            "model": deployed_state,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if needs_scaler else None,
            "ema": ema.shadow.state_dict() if ema is not None else None,
            "ema_step": ema.step if ema is not None else 0,
            "step": step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "metric_for_best": metric_for_best,
            "metrics": m,
            "metrics_cal": m_cal,
            "temperature": float(cur_T),
            "config": c,
        }
        _t_phase = time.time()
        torch.save(ckpt, out_dir / "last.pt")
        if cur > best_metric:
            best_metric = cur
            best_epoch = ep
            # Deployment-clean ckpt: model weights + temperature + config +
            # metadata. NO optimizer/scaler (~3-4× smaller, ready to upload
            # without exposing training state). Filename includes the run
            # name so multiple runs' bests can sit in one folder without
            # clobbering each other.
            slim = {
                "model": deployed_state,
                "ema": ema.shadow.state_dict() if ema is not None else None,
                "epoch": ep,
                "step": step,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "metric_for_best": metric_for_best,
                "metrics": m,
                "metrics_cal": m_cal,
                "temperature": float(cur_T),
                "config": c,
            }
            torch.save(slim, out_dir / "best.pt")
            torch.save(slim, out_dir / f"{out_dir.name}_best.pt")
            print(f"  -> saved best ({metric_for_best}={best_metric:.3f}, T={cur_T:.3f})  (save {time.time() - _t_phase:.1f}s)")

        # Snapshot metrics for trajectory analysis (always tracked; only consulted
        # if patience_trajectory is on).
        if patience_trajectory:
            snap = {k: float(m.get(k, 0.0)) for k in patience_metrics_cfg}
            metric_history.append(snap)

        if patience > 0 and patience_trajectory:
            # Floor: never stop before `patience` epochs, before window full, OR
            # while curriculum is still ramping in new losses (would stop on
            # already-trained metrics flattening before late-stage signals kick in).
            _curriculum_floor = _last_curriculum_ep + patience_window
            if (ep >= patience and len(metric_history) >= patience_window
                    and ep > _curriculum_floor):
                window = metric_history[-patience_window:]
                xs = np.arange(patience_window, dtype=np.float64)
                slopes_rel: dict[str, float] = {}
                for k, weight in patience_metrics_cfg.items():
                    ys = np.array([h[k] for h in window], dtype=np.float64)
                    mean_abs = max(abs(ys.mean()), 1e-6)
                    slope = float(np.polyfit(xs, ys, 1)[0]) / mean_abs  # rel slope/epoch
                    slopes_rel[k] = slope * weight  # sign-correct + weight
                if patience_rule == "weighted_sum":
                    score = sum(slopes_rel.values()) / max(1, len(slopes_rel))
                    moving = score > patience_min_slope
                    verdict = f"weighted_sum={score:+.4f}"
                else:
                    moving = any(s > patience_min_slope for s in slopes_rel.values())
                    verdict = "any_climbing"
                slope_str = " ".join(f"{k.split('_', 1)[-1]}={s:+.4f}" for k, s in slopes_rel.items())
                writer.add_scalar("trajectory/score",
                                  sum(slopes_rel.values()) / max(1, len(slopes_rel)), ep)
                for k, s in slopes_rel.items():
                    writer.add_scalar(f"trajectory/{k}", s, ep)
                print(f"          trajectory[{patience_window}ep, {verdict}]: {slope_str}  -> {'moving' if moving else 'FLAT'}")
                if not moving:
                    print(f"early stop: trajectory flat over {patience_window} epochs "
                          f"(rule={patience_rule}, min_slope={patience_min_slope})")
                    break
        elif patience > 0:
            if patience_smart:
                # Update best for each tracked metric (raw + calibrated where available).
                for k in ("f1", "map50", "map_50_95"):
                    v = float(m.get(k, 0.0))
                    prev_v, _ = best_per_metric.get(k, (-1e9, 0))
                    if v > prev_v + patience_min_delta:
                        best_per_metric[k] = (v, ep)
                    if m_cal is not None and k in m_cal:
                        ck = f"{k}_cal"
                        cv = float(m_cal[k])
                        cprev_v, _ = best_per_metric.get(ck, (-1e9, 0))
                        if cv > cprev_v + patience_min_delta:
                            best_per_metric[ck] = (cv, ep)
                last_improvement = max(
                    [ep_ for _, ep_ in best_per_metric.values()] + [best_epoch],
                    default=best_epoch,
                )
                if (ep - last_improvement) >= patience:
                    last_table = ", ".join(f"{k}={v:.3f}@{e}" for k, (v, e) in sorted(best_per_metric.items()))
                    print(f"early stop: no metric improved by >={patience_min_delta} in {patience} epochs.  bests: {last_table}")
                    break
            else:
                if (ep - best_epoch) >= patience:
                    print(f"early stop: no {metric_for_best} improvement for {patience} epochs (best={best_metric:.3f} @ epoch {best_epoch})")
                    break

    print("running final test eval ...")
    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if "temperature" in state and float(state["temperature"]) != 1.0:
        from opndet.calibrate import apply_temperature
        apply_temperature(model, float(state["temperature"]))
    m = evaluate(model, test_loader, cfg_shim, device, score_thresh=cur_eval_threshold)
    print(f"TEST: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  n_pred={m['n_pred']:.0f}/n_gt={m['n_gt']:.0f}")
    for k, v in m.items():
        writer.add_scalar(f"test/{k}", v, epochs)

    if db is not None:
        db.close()

    if auto_calibrate:
        print("auto-calibrating best.pt on val ...")
        try:
            from opndet.calibrate import calibrate_ckpt
            res = calibrate_ckpt(out_dir / "best.pt", config_path=None, split="val", save=True)
            print(f"  T={res['temperature']:.4f}  ECE {res['ece_before']:.4f} -> {res['ece_after']:.4f}")
            writer.add_scalar("test_cal/T", res["temperature"], epochs)
            writer.add_scalar("test_cal/ece", res["ece_after"], epochs)
            # re-run test eval with the calibrated weights
            state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            from opndet.calibrate import apply_temperature
            apply_temperature(model, float(state.get("temperature", 1.0)))
            m_cal = evaluate(model, test_loader, cfg_shim, device, score_thresh=cur_eval_threshold)
            print(f"TEST(cal): P={m_cal['precision']:.3f} R={m_cal['recall']:.3f} F1={m_cal['f1']:.3f}  n_pred={m_cal['n_pred']:.0f}/n_gt={m_cal['n_gt']:.0f}")
            for k, v in m_cal.items():
                writer.add_scalar(f"test_cal/{k}", v, epochs)
        except Exception as e:
            print(f"  calibration failed: {e}")
    writer.close()

    _download_run(out_dir, c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None, help="Override config 'name'")
    ap.add_argument("--runs-dir", default=None, help="Override config 'runs_dir'")
    ap.add_argument("--resume", default=None, help="Path to ckpt .pt OR run dir (uses last.pt). Continues training in same dir.")
    args = ap.parse_args()
    train(args.config, run_name=args.run_name, runs_dir=args.runs_dir, resume=args.resume)


if __name__ == "__main__":
    main()
