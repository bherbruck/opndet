from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from opndet.dataset import OpndetDataset, collate, load_datasets, split_samples
from opndet.decode import decode_batch
from opndet.encode import encode_targets
from opndet.metrics import (aggregate_per_image_dets, calibration_bins,
                             conf_iou_hist, count_stats, error_breakdown,
                             iou_xyxy, loc_bias, pr_curve,
                             stratified_precision, stratified_recall)
from opndet.presets import resolve as _resolve_preset
from opndet.yaml_build import build_model_from_yaml


class _CfgShim:
    def __init__(self, img_h: int, img_w: int, stride: int):
        self.img_h = img_h
        self.img_w = img_w
        self.stride = stride
        self.out_h = img_h // stride
        self.out_w = img_w // stride


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg_shim: _CfgShim,
    device: torch.device,
    decode_threshold: float = 0.05,
    max_dets_per_image: int = 1000,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Forward pass over loader. Returns list of (scores, pred_boxes_xyxy, gt_boxes_xyxy) per image.

    max_dets_per_image: caps the per-image det pool BEFORE metric compute to avoid
    pathological cold-start explosions (untrained 4-ch models can produce ~1000+
    dets per image at the 0.05 threshold; mAP sweep × Hungarian on that pool
    becomes 100M+ ops). Top-K by score. No-op for trained models.
    """
    model.eval()
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for imgs, boxes_list, _ in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        pred = model(imgs)
        pred_t = pred["output"] if isinstance(pred, dict) else pred
        pred_np = pred_t.detach().cpu().numpy()
        dets_per = decode_batch(pred_np, cfg_shim.img_h, cfg_shim.img_w, cfg_shim.stride, threshold=decode_threshold)
        for dets, gt in zip(dets_per, boxes_list):
            if dets:
                if len(dets) > max_dets_per_image:
                    dets = sorted(dets, key=lambda d: -d.score)[:max_dets_per_image]
                scores = np.array([d.score for d in dets], dtype=np.float32)
                pb = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
            else:
                scores = np.zeros(0, dtype=np.float32)
                pb = np.zeros((0, 4), dtype=np.float32)
            out.append((scores, pb, gt.astype(np.float32)))
    return out


def _ap_from_match(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """COCO 101-point AP from per-detection (score, is_tp) flags."""
    if n_gt == 0 or scores.shape[0] == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = is_tp[order].astype(np.float64)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    p = cum_tp / np.maximum(1.0, cum_tp + cum_fp)
    r = cum_tp / max(1, n_gt)
    ap = 0.0
    for level in np.linspace(0, 1, 101):
        mask = r >= level
        ap += (p[mask].max() if mask.any() else 0.0) / 101.0
    return float(ap)


def compute_full_report(
    images: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iou_thresh: float = 0.5,
    score_thresh: float = 0.3,
) -> dict:
    """All Part 1.4 metrics. Filters predictions by score_thresh for fixed-threshold metrics
    (counting, error breakdown, calibration); uses full set for AP/PR-sweep curves."""
    # Apply fixed score threshold for "deployment-style" metrics.
    fixed = []
    for s, pb, gt in images:
        keep = s >= score_thresh
        fixed.append((s[keep], pb[keep], gt))

    full = aggregate_per_image_dets(images, iou_thresh=iou_thresh)
    fix = aggregate_per_image_dets(fixed, iou_thresh=iou_thresh)

    # Per-IoU-threshold AP from the FULL pool (sweeps confidence implicitly via order).
    map50 = _ap_from_match(full["scores"], full["is_tp"], int(full["gt_matched"].shape[0]))
    iou_grid = np.arange(0.5, 1.0, 0.05)
    ap_per_iou = []
    for t in iou_grid:
        agg_t = aggregate_per_image_dets(images, iou_thresh=float(t))
        ap_per_iou.append(_ap_from_match(agg_t["scores"], agg_t["is_tp"], int(agg_t["gt_matched"].shape[0])))
    map_50_95 = float(np.mean(ap_per_iou))

    # Fixed-threshold P/R/F1
    n_pred_fix = int(fix["scores"].shape[0])
    n_gt = int(fix["gt_matched"].shape[0])
    tp_fix = int(fix["is_tp"].sum())
    fp_fix = n_pred_fix - tp_fix
    fn_fix = n_gt - int(fix["gt_matched"].sum())
    P = tp_fix / max(1, tp_fix + fp_fix)
    R = tp_fix / max(1, tp_fix + fn_fix)
    F1 = 2 * P * R / max(1e-9, P + R)

    # Error breakdown (aggregated across images at fixed threshold).
    eb = {"tp": 0, "fp_localization": 0, "fp_duplicate": 0, "fp_background": 0, "fn_missed": 0}
    for s, pb, gt in fixed:
        b = error_breakdown(pb, gt, s, iou_thresh=iou_thresh)
        for k in eb:
            eb[k] += b[k]

    # Counting accuracy (fixed threshold).
    counts = count_stats(fix["counts"])

    # Localization bias (only on TP pairs, fixed threshold).
    lb = loc_bias(fix["matched_pred"], fix["matched_gt"])

    # Calibration (over all preds at full pool, since calibration is a property of the score itself).
    cal = calibration_bins(full["scores"], full["is_tp"], n_bins=10)

    # Confidence-IoU 2D histogram.
    cih = conf_iou_hist(full["scores"], full["best_iou"], n_bins=20)

    # P-R sweep.
    pr = pr_curve(full["scores"], full["is_tp"], n_gt=int(full["gt_matched"].shape[0]))

    # F1-optimal operating point. Detects "your threshold is off the knee" — common with
    # KD/calibrated models that produce a sharply bimodal score distribution.
    if pr["f1"].size > 0 and pr["f1"].max() > 0:
        idx = int(np.argmax(pr["f1"]))
        opt = {
            "threshold": float(pr["thresholds"][idx]),
            "f1": float(pr["f1"][idx]),
            "precision": float(pr["precision"][idx]),
            "recall": float(pr["recall"][idx]),
        }
    else:
        opt = {"threshold": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Size strata at fixed threshold.
    s_recall = stratified_recall(fix["gt_size"], fix["gt_matched"])
    s_prec = stratified_precision(fix["pred_size"], fix["is_tp"])

    return {
        "iou_thresh": iou_thresh,
        "score_thresh": score_thresh,
        "summary": {
            "precision": P, "recall": R, "f1": F1,
            "map50": map50, "map_50_95": map_50_95,
            "n_pred": n_pred_fix, "n_gt": n_gt,
            "n_images": int(len(images)),
        },
        "counts": counts,
        "error_breakdown": eb,
        "loc_bias": lb,
        "calibration": cal,
        "conf_iou_hist": cih,
        "pr_curve": pr,
        "optimal_threshold": opt,
        "size_strata_recall": s_recall,
        "size_strata_precision": s_prec,
    }


def _save_reliability_png(cal: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    mask = cal["counts"] > 0
    ax.plot(cal["mean_score"][mask], cal["empirical_precision"][mask], "o-", lw=2, label="model")
    ax.set_xlabel("predicted confidence (bin mean)")
    ax.set_ylabel("empirical precision")
    ax.set_title(f"reliability diagram  (ECE={cal['ece']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _save_pr_png(pr: dict, path: Path, opt: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pr["thresholds"], pr["precision"], "-", label="precision")
    ax.plot(pr["thresholds"], pr["recall"], "-", label="recall")
    ax.plot(pr["thresholds"], pr["f1"], "--", label="F1")
    if opt is not None and opt.get("f1", 0) > 0:
        ax.axvline(opt["threshold"], color="red", linestyle=":", lw=1, alpha=0.6)
        ax.scatter([opt["threshold"]], [opt["f1"]], color="red", s=40, zorder=5,
                   label=f"F1 knee @ {opt['threshold']:.2f}: {opt['f1']:.3f}")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("metric")
    ax.set_title("P / R / F1 vs threshold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _save_conf_iou_png(cih: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    H = cih["hist"].astype(np.float32)
    ax.imshow(np.log1p(H.T), origin="lower", extent=(0, 1, 0, 1), aspect="equal", cmap="viridis")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.7)
    ax.set_xlabel("confidence"); ax.set_ylabel("best IoU with GT")
    ax.set_title("conf vs IoU 2D histogram (log)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _save_count_png(counts: dict, abs_errs: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(int(abs_errs.max()) + 2) if abs_errs.size else np.arange(2)
    ax.hist(abs_errs, bins=bins, color="steelblue", edgecolor="black")
    for q, c in [("p95", "orange"), ("p99", "red"), ("max", "purple")]:
        v = counts[f"abs_err_{q}"]
        ax.axvline(v, color=c, linestyle="--", lw=1, label=f"{q}={v:.0f}")
    ax.set_xlabel("|n_pred - n_gt| per image"); ax.set_ylabel("images")
    ax.set_title(f"count error  (mean={counts['abs_err_mean']:.1f}, exact={counts['exact_count_frac']:.1%})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _abs_count_errors(images: list[tuple[np.ndarray, np.ndarray, np.ndarray]], score_thresh: float) -> np.ndarray:
    out = []
    for s, pb, gt in images:
        n_pred = int((s >= score_thresh).sum())
        out.append(abs(n_pred - int(gt.shape[0])))
    return np.array(out, dtype=np.int64)


def write_report(report: dict, out_dir: Path, abs_errs: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_reliability_png(report["calibration"], out_dir / "reliability.png")
    _save_pr_png(report["pr_curve"], out_dir / "pr_curve.png", opt=report.get("optimal_threshold"))
    _save_conf_iou_png(report["conf_iou_hist"], out_dir / "conf_iou_hist.png")
    _save_count_png(report["counts"], abs_errs, out_dir / "count_error.png")

    s = report["summary"]
    eb = report["error_breakdown"]
    cs = report["counts"]
    lb = report["loc_bias"]
    sr = report["size_strata_recall"]
    sp = report["size_strata_precision"]

    md = []
    md.append(f"# eval-report")
    md.append("")
    md.append(f"images: **{s['n_images']}**  •  iou_thresh: **{report['iou_thresh']}**  •  score_thresh: **{report['score_thresh']}**")
    md.append("")

    opt = report.get("optimal_threshold")
    if opt is not None and opt["f1"] > 0:
        gap = opt["threshold"] - report["score_thresh"]
        f1_gap = opt["f1"] - s["f1"]
        md.append("## optimal operating point (F1-max across PR sweep)")
        md.append("")
        md.append(f"- F1-optimal threshold: **{opt['threshold']:.3f}**  (you chose: {report['score_thresh']:.3f}, delta: {gap:+.3f})")
        md.append(f"- at optimum: P={opt['precision']:.3f}, R={opt['recall']:.3f}, **F1={opt['f1']:.3f}**  (gap vs your threshold: {f1_gap:+.3f})")
        if abs(gap) >= 0.1:
            direction = "higher" if gap > 0 else "lower"
            md.append(f"- ⚠ your chosen threshold is meaningfully off the F1 knee. **Try score_thresh={opt['threshold']:.2f}** for a {direction} threshold that better matches the model's confidence distribution.")
        md.append("")
    md.append("## summary")
    md.append("")
    md.append(f"- precision = **{s['precision']:.3f}**, recall = **{s['recall']:.3f}**, F1 = **{s['f1']:.3f}**")
    md.append(f"- mAP@.5 = **{s['map50']:.3f}**,  mAP@.5:.95 = **{s['map_50_95']:.3f}**")
    md.append(f"- n_pred = {s['n_pred']}, n_gt = {s['n_gt']}")
    md.append("")
    md.append("## counting accuracy (fixed threshold)")
    md.append("")
    md.append(f"- exact-count images: **{cs['exact_count_frac']:.1%}**")
    md.append(f"- abs error: mean={cs['abs_err_mean']:.2f}  median={cs['abs_err_median']:.0f}  p95={cs['abs_err_p95']:.0f}  p99={cs['abs_err_p99']:.0f}  max={cs['abs_err_max']:.0f}")
    md.append(f"- signed bias mean: {cs['signed_bias_mean']:+.2f}")
    md.append("")
    md.append("![count error](count_error.png)")
    md.append("")
    md.append("## error breakdown (fixed threshold)")
    md.append("")
    md.append(f"| TP | FP_loc | FP_dup | FP_bg | FN_missed |")
    md.append(f"|----|--------|--------|-------|-----------|")
    md.append(f"| {eb['tp']} | {eb['fp_localization']} | {eb['fp_duplicate']} | {eb['fp_background']} | {eb['fn_missed']} |")
    md.append("")
    md.append("- FP_loc: matched a GT but IoU below threshold (bad fit)")
    md.append("- FP_dup: matched a GT already taken by a stronger prediction (peak-suppression failure)")
    md.append("- FP_bg: no nearby GT (background false positive)")
    md.append("")
    md.append("## localization bias (TP pairs only)")
    md.append("")
    if lb["n"] > 0:
        md.append(f"- center bias (px): x={lb['center_bias_x_px']:+.2f}  y={lb['center_bias_y_px']:+.2f}")
        md.append(f"- center scatter (px stddev): x={lb['center_scatter_x_px']:.2f}  y={lb['center_scatter_y_px']:.2f}")
        md.append(f"- scale bias (relative): w={lb['scale_bias_w']:+.3f}  h={lb['scale_bias_h']:+.3f}")
        md.append(f"- scale scatter (relative stddev): w={lb['scale_scatter_w']:.3f}  h={lb['scale_scatter_h']:.3f}")
    else:
        md.append("- (no matched pairs)")
    md.append("")
    md.append("## size-stratified")
    md.append("")
    md.append(f"| size | n_gt | recall | n_pred | precision |")
    md.append(f"|------|------|--------|--------|-----------|")
    for k in ("small", "medium", "large"):
        md.append(f"| {k} | {sr[k]['n_gt']} | {sr[k]['recall']:.3f} | {sp[k]['n_pred']} | {sp[k]['precision']:.3f} |")
    md.append("")
    md.append("## confidence calibration")
    md.append("")
    md.append(f"- expected calibration error (ECE) = **{report['calibration']['ece']:.4f}**  (lower is better; 0 = perfectly calibrated)")
    md.append("")
    md.append("![reliability](reliability.png)")
    md.append("")
    md.append("## P / R / F1 vs threshold")
    md.append("")
    md.append("![pr curve](pr_curve.png)")
    md.append("")
    md.append("## confidence vs IoU 2D")
    md.append("")
    md.append("![conf-iou hist](conf_iou_hist.png)")
    md.append("")

    stab = report.get("stability")
    if stab is not None:
        md.append("## perturbation stability (proxy for frame-to-frame flapping)")
        md.append("")
        if stab["n_objects_tracked"] == 0:
            md.append("- no objects tracked across perturbations (model didn't match any GT)")
        else:
            md.append(f"- N perturbations / image: **{stab['n_perturbations']}**  (each: ±2px translate, ±5% brightness, ±5% contrast)")
            md.append(f"- objects tracked: **{stab['n_objects_tracked']} / {stab['n_gt_total']}**  ({stab['track_completion_rate']:.1%} GT matched in ≥2 versions)")
            md.append("")
            md.append("Per-object stddev of detection across perturbed versions (lower = more stable):")
            md.append("")
            md.append("| signal | mean | p95 | max |")
            md.append("|--------|------|-----|-----|")
            for k, label in [("score", "score"), ("center_x_px", "center x (px)"),
                             ("center_y_px", "center y (px)"), ("w_px", "width (px)"), ("h_px", "height (px)")]:
                v = stab[k]
                md.append(f"| {label} | {v['mean']:.4f} | {v['p95']:.4f} | {v['max']:.4f} |")
            md.append("")
            md.append("- **score stddev** is the flapping signal — high = confidence wobbles under tiny input changes.")
            md.append("- **center x/y stddev** is jitter in the predicted center after correcting for the perturbation translate.")
            md.append("- **width/height stddev** is bbox-dimension wobble.")
        md.append("")

    path = out_dir / "eval-report.md"
    path.write_text("\n".join(md))

    # Machine-readable scalars (drop ndarrays).
    scalars = {
        "summary": s,
        "counts": cs,
        "error_breakdown": eb,
        "loc_bias": lb,
        "size_strata_recall": sr,
        "size_strata_precision": sp,
        "ece": report["calibration"]["ece"],
        "iou_thresh": report["iou_thresh"],
        "score_thresh": report["score_thresh"],
    }
    if "stability" in report:
        scalars["stability"] = report["stability"]
    if "optimal_threshold" in report:
        scalars["optimal_threshold"] = report["optimal_threshold"]
    (out_dir / "eval-report.json").write_text(json.dumps(scalars, indent=2))
    return path


def write_tb_scalars(report: dict, writer, step: int = 0) -> None:
    """Optional: log all scalar metrics to TensorBoard at the given step."""
    s = report["summary"]
    cs = report["counts"]
    eb = report["error_breakdown"]
    lb = report["loc_bias"]
    sr = report["size_strata_recall"]
    sp = report["size_strata_precision"]
    for k, v in s.items():
        writer.add_scalar(f"eval/{k}", float(v), step)
    for k, v in cs.items():
        writer.add_scalar(f"eval/count/{k}", float(v), step)
    for k, v in eb.items():
        writer.add_scalar(f"eval/err/{k}", float(v), step)
    for k, v in lb.items():
        if k == "n":
            continue
        writer.add_scalar(f"eval/loc/{k}", float(v), step)
    for sz in ("small", "medium", "large"):
        writer.add_scalar(f"eval/recall_{sz}", sr[sz]["recall"], step)
        writer.add_scalar(f"eval/precision_{sz}", sp[sz]["precision"], step)
    writer.add_scalar("eval/ece", report["calibration"]["ece"], step)
    stab = report.get("stability")
    if stab is not None and stab.get("n_objects_tracked", 0) > 0:
        for sig in ("score", "center_x_px", "center_y_px", "w_px", "h_px"):
            writer.add_scalar(f"eval/stability/{sig}_mean", stab[sig]["mean"], step)
            writer.add_scalar(f"eval/stability/{sig}_p95", stab[sig]["p95"], step)
        writer.add_scalar("eval/stability/track_completion", stab["track_completion_rate"], step)


def run_eval(
    ckpt_path: str | Path,
    config_path: str | Path | None = None,
    split: str = "val",
    out_dir: str | Path | None = None,
    score_thresh: float | None = None,
    iou_thresh: float = 0.5,
    batch_size: int | None = None,
    stability: bool = False,
    n_perturbations: int = 8,
    auto_threshold: bool = False,
) -> dict:
    """Stand-alone eval entry point (used by the CLI). When config_path is None,
    falls back to the ckpt's saved config (works when running on the same machine
    that trained the model)."""
    if config_path is None:
        sd_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(sd_peek, dict) or "config" not in sd_peek:
            raise ValueError(f"--config required: ckpt {ckpt_path} has no saved config block")
        c = sd_peek["config"]
        print(f"using saved config from ckpt (model_config={c.get('model_config')})")
    else:
        with open(config_path) as f:
            c = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and c.get("device", "auto") != "cpu" else "cpu")
    print(f"device: {device}")

    print("loading data ...")
    samples = load_datasets(c["data"]["sources"])
    ratios = tuple(c["data"].get("split_ratios", [0.8, 0.1, 0.1]))
    seed = int(c.get("seed", 0))
    train_s, val_s, test_s = split_samples(samples, ratios=ratios, seed=seed)
    sel = {"train": train_s, "val": val_s, "test": test_s}[split]
    print(f"split={split}  n_samples={len(sel)}")

    model_path = _resolve_preset(c["model_config"])
    model = build_model_from_yaml(model_path).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    T = float(sd.get("temperature", 1.0)) if isinstance(sd, dict) else 1.0
    if T != 1.0:
        from opndet.calibrate import apply_temperature
        apply_temperature(model, T)
        print(f"applied calibration temperature T={T:.4f}")
    in_ch, img_h, img_w = model.input_shape
    cfg_shim = _CfgShim(img_h, img_w, stride=int(c["model"].get("stride", 4)))

    encode_fn = partial(encode_targets, cfg=cfg_shim)
    ds = OpndetDataset(sel, img_h, img_w, augment_fn=None, encode_fn=encode_fn, cache_images=False)
    bs = int(batch_size or c.get("batch_size", 8))
    nw = int(c.get("num_workers", 2))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=False)

    score_thresh = float(score_thresh if score_thresh is not None else c.get("eval_threshold", 0.3))

    images = collect_predictions(model, loader, cfg_shim, device, decode_threshold=0.05)
    report = compute_full_report(images, iou_thresh=iou_thresh, score_thresh=score_thresh)
    abs_errs = _abs_count_errors(images, score_thresh)

    if auto_threshold:
        opt = report.get("optimal_threshold", {})
        opt_thresh = float(opt.get("threshold", score_thresh))
        if abs(opt_thresh - score_thresh) >= 0.01:
            print(f"--auto-threshold: chosen={score_thresh:.3f}, F1-optimal={opt_thresh:.3f} -> recomputing fixed-threshold metrics at the optimum")
            score_thresh = opt_thresh
            report = compute_full_report(images, iou_thresh=iou_thresh, score_thresh=score_thresh)
            abs_errs = _abs_count_errors(images, score_thresh)

    if stability:
        from opndet.stability import perturbation_stability
        print(f"perturbation stability ({n_perturbations} perturbations × {len(sel)} samples) ...")
        stab = perturbation_stability(
            model, sel, img_h=img_h, img_w=img_w, stride=cfg_shim.stride,
            n_perturbations=n_perturbations, decode_threshold=0.05,
            iou_thresh=iou_thresh, device=device,
        )
        report["stability"] = stab

    if out_dir is None:
        ckpt_p = Path(ckpt_path)
        out_dir = ckpt_p.parent / f"eval_{split}"
    out_dir = Path(out_dir)
    md_path = write_report(report, out_dir, abs_errs)
    print(f"report: {md_path}")
    return {"report": report, "out_dir": str(out_dir)}
