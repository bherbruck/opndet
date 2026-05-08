from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from opndet._optim import linear_sum_assignment


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


def center_match(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    dist_frac: float = 0.5,
    min_dist_px: float = 8.0,
    cell_window: int = 4,
    stride: int = 4,
) -> dict:
    """IoU-free matcher: pair preds to GTs by center distance only.

    A pred matches a GT if its center is within R pixels of GT center,
    where R = max(min_dist_px, dist_frac × min(gt_w, gt_h)). Hungarian
    assignment over the distance cost matrix; pairs above R are forbidden.

    Why this exists: mAP@.5:.95 punishes box-shape errors hard. A model
    placing the center perfectly but predicting wh ±20% off scores poorly
    on IoU-based F1 even though "did we find the object" is yes. This
    metric answers the deployment question: was the detection close to
    where the object actually is, regardless of box shape?

    Defaults: dist_frac=0.5 (half the smaller GT side) means a 40px object
    is matched if the predicted center is within 20px. min_dist_px=8
    floors that for tiny GTs so single-pixel offsets don't fail at
    stride=4 quantization.

    Returns:
        recall, precision, f1, n_match, distances_px (per matched pair),
        radii_px (the per-GT R used).
    """
    n_pred = int(pred_boxes.shape[0])
    n_gt = int(gt_boxes.shape[0])
    if n_pred == 0 or n_gt == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "n_match": 0,
                "n_pred": n_pred, "n_gt": n_gt,
                "distances_px": np.zeros(0, dtype=np.float32),
                "radii_px": np.zeros(0, dtype=np.float32)}
    pcx = (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5
    pcy = (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
    gcx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gcy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = np.clip(gt_boxes[:, 2] - gt_boxes[:, 0], 1.0, None)
    gh = np.clip(gt_boxes[:, 3] - gt_boxes[:, 1], 1.0, None)
    # Match radius. Pure distance-based; ignores pred bbox size on purpose
    # (huge random init bboxes used to false-match every GT via containment).
    # Three components:
    #   1. cell_window * stride / 2 — half of the user's "4x4 cell box". At
    #      stride=4, cell_window=4 -> ±8 px ≈ 2 cells from GT center.
    #   2. dist_frac * min(gw, gh) — GT-size-relative; lets big objects
    #      tolerate larger center error.
    #   3. min_dist_px — absolute floor for tiny objects.
    # Take the MAX so any of the three can rescue a pair from "too far."
    cell_radius_px = float(cell_window * stride * 0.5)
    radii = np.maximum.reduce([
        np.full(gw.shape, min_dist_px, dtype=np.float32),
        np.full(gw.shape, cell_radius_px, dtype=np.float32),
        (dist_frac * np.minimum(gw, gh)).astype(np.float32),
    ])
    dx = pcx[:, None] - gcx[None, :]
    dy = pcy[:, None] - gcy[None, :]
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    cost = dist.copy()
    forbidden = dist > radii[None, :]
    cost[forbidden] = 1e9
    from opndet._optim import linear_sum_assignment
    row, col = linear_sum_assignment(cost)
    valid = cost[row, col] < 1e8
    n_match = int(valid.sum())
    matched_d_px = dist[row[valid], col[valid]]
    matched_min_side = np.minimum(gw[col[valid]], gh[col[valid]])
    matched_d_frac = matched_d_px / np.maximum(matched_min_side, 1.0)

    # Ghost vs duplicate split for unmatched preds.
    #   Hungarian assigns each GT to at most one pred. The other preds
    #   landing on the SAME object's stride cluster get tagged as FPs in
    #   strict precision — but they're "duplicate detections of a real
    #   thing", not ghosts. Real ghosts are unmatched preds whose nearest
    #   GT center is FAR from any GT (outside its match radius).
    matched_pred = set(int(r) for r, v in zip(row, valid) if v)
    unmatched_pred_mask = np.ones(n_pred, dtype=bool)
    for i in matched_pred:
        unmatched_pred_mask[i] = False
    n_dup = n_ghost = 0
    if unmatched_pred_mask.any():
        u_dist = dist[unmatched_pred_mask]                       # (n_unmatched, n_gt)
        nearest_gt = np.argmin(u_dist, axis=1)
        nearest_d = u_dist[np.arange(u_dist.shape[0]), nearest_gt]
        nearest_r = radii[nearest_gt]
        # Duplicate: another pred landed inside SOME GT's match radius (same
        # object, picked up twice). Ghost: far from every GT — phantom in
        # empty frame space. Pure distance-based; no bbox-size leniency.
        is_dup = nearest_d <= nearest_r
        n_dup = int(is_dup.sum())
        n_ghost = int((~is_dup).sum())

    p_strict  = n_match / max(1, n_pred)
    p_lenient = (n_match + n_dup) / max(1, n_pred)        # treat dups as not-ghosts
    r_metric  = n_match / max(1, n_gt)
    f1_strict  = 2 * p_strict  * r_metric / max(1e-9, p_strict + r_metric)
    f1_lenient = 2 * p_lenient * r_metric / max(1e-9, p_lenient + r_metric)
    ghost_rate = n_ghost / max(1, n_pred)
    duplicate_rate = n_dup / max(1, n_pred)

    return {"recall": float(r_metric),
            "precision": float(p_strict),
            "f1": float(f1_strict),
            "precision_lenient": float(p_lenient),
            "f1_lenient": float(f1_lenient),
            "ghost_rate": float(ghost_rate),
            "duplicate_rate": float(duplicate_rate),
            "n_match": n_match, "n_dup": n_dup, "n_ghost": n_ghost,
            "n_pred": n_pred, "n_gt": n_gt,
            "distances_px": matched_d_px,
            "distances_frac": matched_d_frac.astype(np.float32),
            "radii_px": radii}


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


def rotated_iou(cx1: float, cy1: float, w1: float, h1: float, t1_rad: float,
                cx2: float, cy2: float, w2: float, h2: float, t2_rad: float) -> float:
    """IoU between two oriented rectangles via cv2 polygon intersection.
    theta in radians = rotation of major axis from +x. cv2 angles are in degrees.
    Robust to near-zero areas; returns 0 on degenerate input.
    """
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    r1 = ((float(cx1), float(cy1)), (float(w1), float(h1)), float(math.degrees(t1_rad)))
    r2 = ((float(cx2), float(cy2)), (float(w2), float(h2)), float(math.degrees(t2_rad)))
    ret, inter = cv2.rotatedRectangleIntersection(r1, r2)
    if ret == 0 or inter is None or len(inter) < 3:
        return 0.0
    inter_area = float(cv2.contourArea(inter))
    a1 = float(w1) * float(h1)
    a2 = float(w2) * float(h2)
    return inter_area / max(a1 + a2 - inter_area, 1e-9)


def angle_err_rad(t1: float, t2: float) -> float:
    """Angular error wrapped to [0, π/2]. OBB has π/2 symmetry — flipping the
    major axis by π yields the same rectangle, so errors >π/2 fold back."""
    d = abs(float(t1) - float(t2)) % math.pi
    return min(d, math.pi - d)


def obb_summary(pred_obbs: np.ndarray, gt_obbs: np.ndarray,
                iou_thresh: float = 0.3) -> dict:
    """Per-image OBB diagnostics. pred/gt: [N, 5] = (cx, cy, w, h, θ_rad)
    where w, h are the rotated rectangle's OWN dims (not enclosing AABB).

    Hungarian-matches preds→GTs minimizing 1 - rotated_iou; matches below
    iou_thresh count as unmatched. Returns rotated_iou + angle_err on matched
    pairs only — diagnostic, not a Precision/Recall replacement.
    """
    n_p = int(pred_obbs.shape[0])
    n_g = int(gt_obbs.shape[0])
    out = {"n_match": 0, "obb_iou_sum": 0.0, "ang_err_sum": 0.0,
           "ang_err_le_10_count": 0, "ang_err_le_30_count": 0,
           "obb_ious": np.zeros(0, dtype=np.float32),
           "ang_errs_deg": np.zeros(0, dtype=np.float32)}
    if n_p == 0 or n_g == 0:
        return out
    iou_mat = np.zeros((n_p, n_g), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_g):
            iou_mat[i, j] = rotated_iou(
                pred_obbs[i, 0], pred_obbs[i, 1], pred_obbs[i, 2], pred_obbs[i, 3], pred_obbs[i, 4],
                gt_obbs[j, 0], gt_obbs[j, 1], gt_obbs[j, 2], gt_obbs[j, 3], gt_obbs[j, 4],
            )
    cost = 1.0 - iou_mat
    rows, cols = linear_sum_assignment(cost)
    iou_keep = []
    ang_keep = []
    for r, c in zip(rows, cols):
        if iou_mat[r, c] < iou_thresh:
            continue
        iou_keep.append(float(iou_mat[r, c]))
        e_rad = angle_err_rad(pred_obbs[r, 4], gt_obbs[c, 4])
        ang_keep.append(math.degrees(e_rad))
    if not iou_keep:
        return out
    iou_arr = np.asarray(iou_keep, dtype=np.float32)
    ang_arr = np.asarray(ang_keep, dtype=np.float32)
    out["n_match"] = int(iou_arr.shape[0])
    out["obb_iou_sum"] = float(iou_arr.sum())
    out["ang_err_sum"] = float(ang_arr.sum())
    out["ang_err_le_10_count"] = int((ang_arr <= 10.0).sum())
    out["ang_err_le_30_count"] = int((ang_arr <= 30.0).sum())
    out["obb_ious"] = iou_arr
    out["ang_errs_deg"] = ang_arr
    return out


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
