"""Grad-CAM-driven hard-negative mining.

Given a trained ckpt + dataset, run inference, identify ghost (unmatched) preds,
trace each ghost back to the input region that drove the prediction via input-
gradient saliency, and crop a fixed-size patch centered on that region. Cluster
the patches and dump them to disk as a "hard-negative pool" the user can paste
back into training as labeled-bg.

ROADMAP §1.7 + §1.8 Phase 5.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from opndet.analyze import gradcam_input
from opndet.dataset import OpndetDataset, collate, load_datasets, split_samples
from opndet.decode import decode
from opndet.encode import encode_targets
from opndet.metrics import center_match
from opndet.presets import resolve as _resolve_preset
from opndet.yaml_build import build_model_from_yaml


@dataclass
class PatchEntry:
    image_path: str
    sample_idx: int
    ghost_idx: int
    ghost_score: float
    ghost_bbox: list[float]               # xyxy in letterboxed coords
    attribution_centroid: list[float]     # (y, x) in input-image pixels
    patch_path: str
    cluster: int = -1


def _denormalize(x_t: torch.Tensor) -> np.ndarray:
    """ImageNet-normalized (3, H, W) -> uint8 BGR (H, W, 3). RGB channels only."""
    x = x_t[:3].permute(1, 2, 0).cpu().numpy()
    x = x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _saliency_centroid(sal: np.ndarray, ghost_yx: tuple[int, int],
                       window: int = 64) -> tuple[int, int]:
    """Mass-weighted centroid of saliency in a window centered at the ghost cell.
    Restricting to a window keeps the centroid local — global argmax can wander
    to unrelated bright spots. window in image pixels.
    """
    H, W = sal.shape
    cy, cx = ghost_yx
    y0 = max(0, cy - window // 2)
    y1 = min(H, cy + window // 2)
    x0 = max(0, cx - window // 2)
    x1 = min(W, cx + window // 2)
    crop = sal[y0:y1, x0:x1]
    if crop.sum() <= 0:
        return cy, cx
    # mass-weighted centroid
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    w = crop.astype(np.float64)
    total = w.sum()
    cy_local = float((yy * w).sum() / total)
    cx_local = float((xx * w).sum() / total)
    return int(round(y0 + cy_local)), int(round(x0 + cx_local))


def _crop_patch(img_bgr: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    half = size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size
    # clamp window into image; pad with edge replication if out-of-bounds
    pad_t = max(0, -y0)
    pad_l = max(0, -x0)
    pad_b = max(0, y1 - H)
    pad_r = max(0, x1 - W)
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(H, y1), min(W, x1)
    crop = img_bgr[y0c:y1c, x0c:x1c]
    if pad_t or pad_l or pad_b or pad_r:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    if crop.shape[:2] != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def _load_config_and_data(ckpt_path: Path, config_path: Path | None, split: str):
    if config_path is None:
        sd_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(sd_peek, dict) or "config" not in sd_peek:
            raise ValueError(f"--config required: ckpt {ckpt_path} has no saved config")
        c = sd_peek["config"]
        print(f"using saved config from ckpt (model_config={c.get('model_config')})")
    else:
        with open(config_path) as f:
            c = yaml.safe_load(f)
    samples = load_datasets(c["data"]["sources"])
    ratios = tuple(c["data"].get("split_ratios", [0.8, 0.1, 0.1]))
    seed = int(c.get("seed", 0))
    train_s, val_s, test_s = split_samples(samples, ratios=ratios, seed=seed)
    sel = {"train": train_s, "val": val_s, "test": test_s}[split]
    return c, sel


def mine(
    ckpt: str | Path,
    config: str | Path | None,
    out_dir: str | Path,
    split: str = "val",
    max_samples: int = 500,
    top_k_per_sample: int = 4,
    patch_size: int = 32,
    score_thresh: float | None = None,
    clusters: int = 8,
    device: str | None = None,
) -> dict:
    """Mine hard negatives from a trained ckpt. Returns summary dict."""
    ckpt = Path(ckpt)
    out_dir = Path(out_dir)
    patches_dir = out_dir / "patches"
    clusters_dir = out_dir / "clusters"
    patches_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)

    c, sel = _load_config_and_data(ckpt, Path(config) if config else None, split)
    if max_samples and len(sel) > max_samples:
        sel = sel[:max_samples]
    print(f"split={split}  n_samples={len(sel)}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() and c.get("device", "auto") != "cpu" else "cpu"
    dev_t = torch.device(device)
    print(f"device: {dev_t}")

    model_path = _resolve_preset(c["model_config"])
    model = build_model_from_yaml(model_path).to(dev_t).eval()
    sd = torch.load(ckpt, map_location=dev_t, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    T = float(sd.get("temperature", 1.0)) if isinstance(sd, dict) else 1.0
    if T != 1.0:
        from opndet.calibrate import apply_temperature
        apply_temperature(model, T)
        print(f"applied calibration temperature T={T:.4f}")
    in_ch, img_h, img_w = model.input_shape
    stride = int(c.get("model", {}).get("stride", 4))

    if score_thresh is None:
        et = c.get("eval_threshold", 0.3)
        score_thresh = 0.3 if et == "auto" else float(et)
    score_thresh = float(score_thresh)
    print(f"score_thresh={score_thresh:.3f}  patch_size={patch_size}  top_k_per_sample={top_k_per_sample}")

    class _Shim:
        def __init__(self):
            self.img_h, self.img_w, self.stride = img_h, img_w, stride
            self.out_h, self.out_w = img_h // stride, img_w // stride
    encode_fn = partial(encode_targets, cfg=_Shim())
    ds = OpndetDataset(sel, img_h, img_w, augment_fn=None, encode_fn=encode_fn,
                       cache_images=False, in_ch=in_ch, stride=stride)

    entries: list[PatchEntry] = []
    raw_patches: list[np.ndarray] = []
    n_with_ghosts = 0
    n_total_ghosts = 0

    for s_idx in range(len(ds)):
        img_t, gt_boxes, _ = ds[s_idx]
        x = img_t.unsqueeze(0).to(dev_t)
        with torch.no_grad():
            out = model(x)
            obj_t = out["output"] if isinstance(out, dict) else out
        out_np = obj_t[0].detach().cpu().numpy()
        H_out, W_out = out_np.shape[1], out_np.shape[2]
        det_stride = img_h // H_out
        dets = decode(out_np, img_h, img_w, det_stride, threshold=score_thresh)
        if not dets:
            continue
        pred_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
        pred_scores = np.array([d.score for d in dets], dtype=np.float32)

        # center_match needs (n_pred, 4) gt and pred. Pass them.
        m = center_match(pred_boxes, gt_boxes.astype(np.float32) if gt_boxes is not None
                         else np.zeros((0, 4), dtype=np.float32),
                         stride=det_stride)
        if m["n_ghost"] == 0:
            continue
        # find ghost indices: replicate the matcher's ghost split here. center_match
        # doesn't return per-pred labels, so recompute: nearest-gt distance > radius.
        ghost_idx = _identify_ghosts(pred_boxes, gt_boxes, det_stride)
        if len(ghost_idx) == 0:
            continue
        # rank ghosts by score; keep top-k
        ranked = sorted(ghost_idx, key=lambda i: -pred_scores[i])[:top_k_per_sample]
        n_with_ghosts += 1
        n_total_ghosts += len(ranked)

        img_bgr = _denormalize(img_t)
        for k_idx, p_i in enumerate(ranked):
            d = dets[p_i]
            cx_px = (d.x1 + d.x2) * 0.5
            cy_px = (d.y1 + d.y2) * 0.5
            cell_x = max(0, min(W_out - 1, int(cx_px / det_stride)))
            cell_y = max(0, min(H_out - 1, int(cy_px / det_stride)))
            sal = gradcam_input(model, x, cell_y, cell_x)
            cy, cx = _saliency_centroid(sal, (int(cy_px), int(cx_px)),
                                        window=max(64, patch_size * 2))
            patch = _crop_patch(img_bgr, cy, cx, patch_size)
            patch_name = f"sample_{s_idx:05d}_ghost_{k_idx:02d}.png"
            patch_path = patches_dir / patch_name
            cv2.imwrite(str(patch_path), patch)
            raw_patches.append(patch)
            entries.append(PatchEntry(
                image_path=str(sel[s_idx].image_path),
                sample_idx=s_idx, ghost_idx=int(p_i),
                ghost_score=float(d.score),
                ghost_bbox=[float(d.x1), float(d.y1), float(d.x2), float(d.y2)],
                attribution_centroid=[float(cy), float(cx)],
                patch_path=str(patch_path.relative_to(out_dir)),
            ))
        if (s_idx + 1) % 50 == 0:
            print(f"  [{s_idx+1}/{len(ds)}] ghosts mined: {len(entries)}")

    print(f"\nmined {len(entries)} patches across {n_with_ghosts} samples "
          f"({n_total_ghosts} ghosts kept)")

    cluster_labels = _cluster_patches(raw_patches, n_clusters=clusters,
                                      out_dir=clusters_dir) if entries else []
    for e, lbl in zip(entries, cluster_labels):
        e.cluster = int(lbl)

    manifest = {
        "ckpt": str(ckpt),
        "split": split,
        "n_samples_scanned": len(ds),
        "n_samples_with_ghosts": n_with_ghosts,
        "n_patches": len(entries),
        "patch_size": patch_size,
        "score_thresh": score_thresh,
        "clusters": int(clusters) if entries else 0,
        "patches": [asdict(e) for e in entries],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {out_dir / 'manifest.json'}")
    print(f"patches:  {patches_dir}/  ({len(entries)} files)")
    if entries:
        print(f"clusters: {clusters_dir}/  (k={clusters})")
    return manifest


def _identify_ghosts(pred_boxes: np.ndarray, gt_boxes: np.ndarray,
                     stride: int, dist_frac: float = 0.5,
                     min_dist_px: float = 8.0, cell_window: int = 4) -> list[int]:
    """Return indices of preds that are ghosts (far from every GT). Mirrors
    center_match's ghost rule but exposes per-pred labels.
    """
    n_p = pred_boxes.shape[0]
    n_g = gt_boxes.shape[0] if gt_boxes is not None else 0
    if n_p == 0:
        return []
    if n_g == 0:
        return list(range(n_p))
    pcx = (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5
    pcy = (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
    gcx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gcy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = np.clip(gt_boxes[:, 2] - gt_boxes[:, 0], 1.0, None)
    gh = np.clip(gt_boxes[:, 3] - gt_boxes[:, 1], 1.0, None)
    cell_radius_px = float(cell_window * stride * 0.5)
    radii = np.maximum.reduce([
        np.full(gw.shape, min_dist_px, dtype=np.float32),
        np.full(gw.shape, cell_radius_px, dtype=np.float32),
        (dist_frac * np.minimum(gw, gh)).astype(np.float32),
    ])
    dx = pcx[:, None] - gcx[None, :]
    dy = pcy[:, None] - gcy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    nearest_g = np.argmin(dist, axis=1)
    nearest_d = dist[np.arange(n_p), nearest_g]
    nearest_r = radii[nearest_g]
    is_ghost = nearest_d > nearest_r
    return [int(i) for i in np.where(is_ghost)[0]]


def _cluster_patches(patches: list[np.ndarray], n_clusters: int,
                     out_dir: Path) -> list[int]:
    """K-means on raw RGB pixels. Lazy sklearn import. Returns per-patch label."""
    if not patches or n_clusters <= 1:
        return [0] * len(patches)
    n_clusters = min(n_clusters, len(patches))
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("  sklearn not installed; skipping clustering "
              "(install with `pip install scikit-learn` for cluster support)")
        return [0] * len(patches)

    X = np.stack([p.reshape(-1).astype(np.float32) / 255.0 for p in patches], axis=0)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=4).fit(X)
    labels = km.labels_.astype(int).tolist()

    # Save the patch closest to each cluster centroid
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in range(n_clusters):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) == 0:
            continue
        center = km.cluster_centers_[c]
        dists = np.linalg.norm(X[idx] - center[None, :], axis=1)
        repr_idx = int(idx[int(np.argmin(dists))])
        cv2.imwrite(str(out_dir / f"cluster_{c:02d}_n{len(idx):03d}.png"),
                    patches[repr_idx])
    return labels


def mine_into_pool(
    model: torch.nn.Module,
    samples: list,
    *,
    img_h: int,
    img_w: int,
    in_ch: int,
    stride: int,
    encode_fn,
    score_thresh: float,
    pool_dir: str | Path,
    device,
    patch_size: int = 32,
    top_k_per_sample: int = 4,
    max_total: int = 500,
    max_scan: int = 400,
    epoch_tag: int = 0,
) -> int:
    """Lightweight in-training hard-negative miner (no Grad-CAM).

    Runs `model` over a slice of `samples`, takes objectness peaks above
    `score_thresh` whose center isn't near any GT center (stride-aware radius,
    matching the metrics' quantization floor), crops a `patch_size` patch around
    each, and appends them to `<pool_dir>/patches/`. Then trims that dir to the
    `max_total` most-recent files (oldest deleted) so the pool reflects the
    current model's failures, not ancient ones. Returns the number written.

    Cheap enough to call every few epochs — one forward pass per scanned image,
    no backward. Restores the model's train/eval mode on exit.
    """
    patches_dir = Path(pool_dir) / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    ds = OpndetDataset(samples, img_h, img_w, augment_fn=None, encode_fn=encode_fn,
                       cache_images=False, in_ch=in_ch, stride=stride)
    n = min(len(ds), max_scan) if max_scan else len(ds)
    was_training = model.training
    model.eval()
    written = 0
    try:
        with torch.no_grad():
            for s_idx in range(n):
                img_t, gt_boxes, _ = ds[s_idx]
                out = model(img_t.unsqueeze(0).to(device))
                obj_t = out["output"] if isinstance(out, dict) else out
                obj = obj_t[0, 0].detach().float().cpu().numpy()       # peak-suppressed [0,1]
                Hc, Wc = obj.shape
                det_stride = max(1, img_h // Hc)
                ys, xs = np.where(obj >= score_thresh)
                if len(ys) == 0:
                    continue
                scores = obj[ys, xs]
                pcx = (xs.astype(np.float32) + 0.5) * det_stride
                pcy = (ys.astype(np.float32) + 0.5) * det_stride
                gb = gt_boxes if (gt_boxes is not None and len(gt_boxes)) else np.zeros((0, 4), np.float32)
                gb = np.asarray(gb, dtype=np.float32).reshape(-1, 4)
                if gb.shape[0] == 0:
                    ghost = np.ones(len(ys), dtype=bool)
                else:
                    gcx = (gb[:, 0] + gb[:, 2]) * 0.5
                    gcy = (gb[:, 1] + gb[:, 3]) * 0.5
                    gw = np.clip(gb[:, 2] - gb[:, 0], 1.0, None)
                    gh = np.clip(gb[:, 3] - gb[:, 1], 1.0, None)
                    radii = np.maximum(
                        np.full(gw.shape, max(8.0, 2.0 * det_stride), dtype=np.float32),
                        (0.5 * np.minimum(gw, gh)).astype(np.float32),
                    )
                    dd = np.sqrt((pcx[:, None] - gcx[None, :]) ** 2 + (pcy[:, None] - gcy[None, :]) ** 2)
                    nearest = dd.argmin(axis=1)
                    ghost = dd[np.arange(len(ys)), nearest] > radii[nearest]
                gi = np.where(ghost)[0]
                if len(gi) == 0:
                    continue
                gi = gi[np.argsort(-scores[gi])][:max(1, top_k_per_sample)]
                img_bgr = _denormalize(img_t)
                for j in gi:
                    patch = _crop_patch(img_bgr, int(round(pcy[j])), int(round(pcx[j])), patch_size)
                    name = f"ep{epoch_tag:03d}_s{s_idx:05d}_g{int(j):03d}_{written:04d}.png"
                    cv2.imwrite(str(patches_dir / name), patch)
                    written += 1
    finally:
        model.train(was_training)
    # Age out: keep the most-recent `max_total` patches.
    if max_total:
        files = sorted(patches_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        for old in files[:-max_total]:
            try:
                old.unlink()
            except OSError:
                pass
    return written


def load_pool(pool_dir: str | Path) -> list[np.ndarray]:
    """Load all patches from a hard-negative pool directory. Used by augment.py."""
    pool_dir = Path(pool_dir)
    if not pool_dir.exists():
        return []
    # accept either the pool root (which has patches/) or a flat dir of pngs
    patch_root = pool_dir / "patches" if (pool_dir / "patches").exists() else pool_dir
    paths = sorted(p for p in patch_root.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out
