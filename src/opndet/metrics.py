from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two box arrays. a:[N,4] b:[M,4] -> [N,M]."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
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


@dataclass
class MatchResult:
    pairs: np.ndarray            # [K, 2] int (pred_idx, gt_idx) for matches with IoU >= thresh
    pair_ious: np.ndarray        # [K] float
    unmatched_pred: np.ndarray   # [Np-K] int
    unmatched_gt: np.ndarray     # [Ng-K] int


def hungarian_match(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float = 0.5) -> MatchResult:
    """Globally optimal pred->gt assignment, then drop pairs below iou_thresh."""
    n_p, n_g = pred_boxes.shape[0], gt_boxes.shape[0]
    if n_p == 0 or n_g == 0:
        return MatchResult(
            pairs=np.zeros((0, 2), dtype=np.int64),
            pair_ious=np.zeros(0, dtype=np.float32),
            unmatched_pred=np.arange(n_p),
            unmatched_gt=np.arange(n_g),
        )
    ious = iou_xyxy(pred_boxes, gt_boxes)
    cost = 1.0 - ious
    pi, gi = linear_sum_assignment(cost)
    keep = ious[pi, gi] >= iou_thresh
    pairs = np.stack([pi[keep], gi[keep]], axis=1).astype(np.int64)
    pair_ious = ious[pi[keep], gi[keep]].astype(np.float32)
    matched_p = set(pi[keep].tolist())
    matched_g = set(gi[keep].tolist())
    unmatched_p = np.array([i for i in range(n_p) if i not in matched_p], dtype=np.int64)
    unmatched_g = np.array([j for j in range(n_g) if j not in matched_g], dtype=np.int64)
    return MatchResult(pairs=pairs, pair_ious=pair_ious, unmatched_pred=unmatched_p, unmatched_gt=unmatched_g)


def count_stats(per_image: list[tuple[int, int]]) -> dict:
    """per_image: list of (n_pred, n_gt). Returns abs-error and signed-bias percentiles."""
    if not per_image:
        return {"n_images": 0}
    arr = np.array(per_image, dtype=np.int64)
    abs_err = np.abs(arr[:, 0] - arr[:, 1]).astype(np.float64)
    signed = (arr[:, 0] - arr[:, 1]).astype(np.float64)
    return {
        "n_images": int(len(arr)),
        "abs_err_mean": float(abs_err.mean()),
        "abs_err_median": float(np.median(abs_err)),
        "abs_err_p95": float(np.percentile(abs_err, 95)),
        "abs_err_p99": float(np.percentile(abs_err, 99)),
        "abs_err_max": float(abs_err.max()),
        "signed_bias_mean": float(signed.mean()),
        "exact_count_frac": float((abs_err == 0).mean()),
    }


def error_breakdown(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, scores: np.ndarray,
    iou_thresh: float = 0.5, loc_iou_floor: float = 0.1,
) -> dict:
    """Split predictions into TP / FP_localization / FP_duplicate / FP_background, and GTs into matched / missed.

    FP categories:
      - FP_localization: pred's best GT exists at IoU in [loc_iou_floor, iou_thresh) -> right place, bad fit.
      - FP_duplicate: pred's best GT is at IoU >= iou_thresh but already matched by a stronger pred.
      - FP_background: best IoU < loc_iou_floor -> no nearby object.
    """
    n_p, n_g = pred_boxes.shape[0], gt_boxes.shape[0]
    out = {"tp": 0, "fp_localization": 0, "fp_duplicate": 0, "fp_background": 0,
           "fn_missed": 0, "n_pred": int(n_p), "n_gt": int(n_g)}
    if n_g == 0:
        out["fp_background"] = int(n_p)
        return out
    if n_p == 0:
        out["fn_missed"] = int(n_g)
        return out

    ious = iou_xyxy(pred_boxes, gt_boxes)
    order = np.argsort(-scores)
    matched_g = np.zeros(n_g, dtype=bool)
    for pi in order:
        gj = int(np.argmax(ious[pi]))
        best = float(ious[pi, gj])
        if best >= iou_thresh:
            if not matched_g[gj]:
                matched_g[gj] = True
                out["tp"] += 1
            else:
                out["fp_duplicate"] += 1
        elif best >= loc_iou_floor:
            out["fp_localization"] += 1
        else:
            out["fp_background"] += 1
    out["fn_missed"] = int((~matched_g).sum())
    return out


COCO_SMALL = 32 * 32
COCO_MEDIUM = 96 * 96


def size_label(box: np.ndarray) -> str:
    area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    if area < COCO_SMALL:
        return "small"
    if area < COCO_MEDIUM:
        return "medium"
    return "large"


def size_mask(boxes: np.ndarray) -> dict[str, np.ndarray]:
    if boxes.shape[0] == 0:
        return {k: np.zeros(0, dtype=bool) for k in ("small", "medium", "large")}
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return {
        "small": areas < COCO_SMALL,
        "medium": (areas >= COCO_SMALL) & (areas < COCO_MEDIUM),
        "large": areas >= COCO_MEDIUM,
    }


def loc_bias(matched_pred: np.ndarray, matched_gt: np.ndarray) -> dict:
    """Per-matched-pair localization stats. Both inputs [K,4] xyxy aligned by index."""
    if matched_pred.shape[0] == 0:
        return {"n": 0}
    p_cx = (matched_pred[:, 0] + matched_pred[:, 2]) * 0.5
    p_cy = (matched_pred[:, 1] + matched_pred[:, 3]) * 0.5
    g_cx = (matched_gt[:, 0] + matched_gt[:, 2]) * 0.5
    g_cy = (matched_gt[:, 1] + matched_gt[:, 3]) * 0.5
    p_w = matched_pred[:, 2] - matched_pred[:, 0]
    p_h = matched_pred[:, 3] - matched_pred[:, 1]
    g_w = np.clip(matched_gt[:, 2] - matched_gt[:, 0], 1e-6, None)
    g_h = np.clip(matched_gt[:, 3] - matched_gt[:, 1], 1e-6, None)
    dx = p_cx - g_cx
    dy = p_cy - g_cy
    dw = (p_w - g_w) / g_w
    dh = (p_h - g_h) / g_h
    return {
        "n": int(matched_pred.shape[0]),
        "center_bias_x_px": float(dx.mean()),
        "center_bias_y_px": float(dy.mean()),
        "center_scatter_x_px": float(dx.std()),
        "center_scatter_y_px": float(dy.std()),
        "scale_bias_w": float(dw.mean()),
        "scale_bias_h": float(dh.mean()),
        "scale_scatter_w": float(dw.std()),
        "scale_scatter_h": float(dh.std()),
    }


def calibration_bins(scores: np.ndarray, is_tp: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability-diagram data. is_tp is 0/1 per detection (1 if matched at fixed iou_thresh)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(n_bins, dtype=np.int64)
    mean_score = np.zeros(n_bins, dtype=np.float64)
    empirical_p = np.zeros(n_bins, dtype=np.float64)
    if scores.shape[0] == 0:
        return {"edges": edges, "centers": centers, "counts": counts,
                "mean_score": mean_score, "empirical_precision": empirical_p, "ece": 0.0}
    idx = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        counts[b] = c
        if c > 0:
            mean_score[b] = float(scores[mask].mean())
            empirical_p[b] = float(is_tp[mask].mean())
    total = max(1, int(counts.sum()))
    ece = float(np.sum(counts / total * np.abs(mean_score - empirical_p)))
    return {"edges": edges, "centers": centers, "counts": counts,
            "mean_score": mean_score, "empirical_precision": empirical_p, "ece": ece}


def pr_curve(scores: np.ndarray, is_tp: np.ndarray, n_gt: int, thresholds: np.ndarray | None = None) -> dict:
    """Precision/Recall/F1 swept over confidence thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    P = np.zeros_like(thresholds)
    R = np.zeros_like(thresholds)
    F = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        keep = scores >= t
        tp = float(is_tp[keep].sum())
        fp = float(keep.sum() - tp)
        fn = max(0.0, n_gt - tp)
        P[i] = tp / max(1.0, tp + fp)
        R[i] = tp / max(1.0, tp + fn)
        F[i] = 2 * P[i] * R[i] / max(1e-9, P[i] + R[i])
    return {"thresholds": thresholds, "precision": P, "recall": R, "f1": F}


def conf_iou_hist(scores: np.ndarray, ious: np.ndarray, n_bins: int = 20) -> dict:
    """2D density of (confidence, IoU-with-best-GT) for all predictions. Off-diagonal density = miscalibration."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    if scores.shape[0] == 0:
        H = np.zeros((n_bins, n_bins), dtype=np.int64)
    else:
        H, _, _ = np.histogram2d(np.clip(scores, 0, 1), np.clip(ious, 0, 1), bins=[edges, edges])
        H = H.astype(np.int64)
    return {"edges": edges, "hist": H}


def aggregate_per_image_dets(
    images: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iou_thresh: float = 0.5,
) -> dict:
    """Run Hungarian matching per image and accumulate flat arrays.

    images: list of (scores[N], pred_boxes[N,4], gt_boxes[M,4]).
    Returns flat per-detection arrays + per-image counts ready to feed downstream metrics.
    """
    all_scores: list[float] = []
    all_is_tp: list[int] = []
    all_best_iou: list[float] = []
    all_pred_boxes: list[np.ndarray] = []
    all_pred_size: list[str] = []
    matched_pred: list[np.ndarray] = []
    matched_gt: list[np.ndarray] = []
    all_gt_boxes: list[np.ndarray] = []
    all_gt_matched: list[int] = []
    all_gt_size: list[str] = []
    counts: list[tuple[int, int]] = []

    for scores, pb, gt in images:
        counts.append((int(pb.shape[0]), int(gt.shape[0])))
        if pb.shape[0] > 0:
            ious_all = iou_xyxy(pb, gt) if gt.shape[0] > 0 else np.zeros((pb.shape[0], 0))
            best = ious_all.max(axis=1) if gt.shape[0] > 0 else np.zeros(pb.shape[0])
            for i in range(pb.shape[0]):
                all_scores.append(float(scores[i]))
                all_best_iou.append(float(best[i]))
                all_pred_boxes.append(pb[i])
                all_pred_size.append(size_label(pb[i]))
        m = hungarian_match(pb, gt, iou_thresh=iou_thresh)
        is_tp_arr = np.zeros(pb.shape[0], dtype=np.int64)
        if m.pairs.shape[0] > 0:
            is_tp_arr[m.pairs[:, 0]] = 1
            matched_pred.append(pb[m.pairs[:, 0]])
            matched_gt.append(gt[m.pairs[:, 1]])
        all_is_tp.extend(is_tp_arr.tolist())
        gt_matched = np.zeros(gt.shape[0], dtype=np.int64)
        if m.pairs.shape[0] > 0:
            gt_matched[m.pairs[:, 1]] = 1
        for j in range(gt.shape[0]):
            all_gt_boxes.append(gt[j])
            all_gt_matched.append(int(gt_matched[j]))
            all_gt_size.append(size_label(gt[j]))

    return {
        "scores": np.array(all_scores, dtype=np.float32),
        "is_tp": np.array(all_is_tp, dtype=np.int64),
        "best_iou": np.array(all_best_iou, dtype=np.float32),
        "pred_boxes": np.stack(all_pred_boxes) if all_pred_boxes else np.zeros((0, 4), dtype=np.float32),
        "pred_size": np.array(all_pred_size),
        "matched_pred": np.concatenate(matched_pred, axis=0) if matched_pred else np.zeros((0, 4), dtype=np.float32),
        "matched_gt": np.concatenate(matched_gt, axis=0) if matched_gt else np.zeros((0, 4), dtype=np.float32),
        "gt_boxes": np.stack(all_gt_boxes) if all_gt_boxes else np.zeros((0, 4), dtype=np.float32),
        "gt_matched": np.array(all_gt_matched, dtype=np.int64),
        "gt_size": np.array(all_gt_size),
        "counts": counts,
    }


def stratified_recall(gt_size: np.ndarray, gt_matched: np.ndarray) -> dict[str, dict]:
    out = {}
    for k in ("small", "medium", "large"):
        mask = gt_size == k
        n = int(mask.sum())
        r = float(gt_matched[mask].mean()) if n else 0.0
        out[k] = {"n_gt": n, "recall": r}
    return out


def stratified_precision(pred_size: np.ndarray, is_tp: np.ndarray) -> dict[str, dict]:
    out = {}
    for k in ("small", "medium", "large"):
        mask = pred_size == k
        n = int(mask.sum())
        p = float(is_tp[mask].mean()) if n else 0.0
        out[k] = {"n_pred": n, "precision": p}
    return out
