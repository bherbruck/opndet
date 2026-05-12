"""Training loop for the dense-dome segmentation head (`bbox-*-seg`).

Deliberately separate from train.py: a seg model has fundamentally different
metrics (Dice / IoU / per-object area & count, not detection mAP) and no in-graph
peak op to calibrate, no eval-threshold EMA, no curriculum / repulsion / convexity,
no temporal-prior cold-start. It reuses the dataset, augmentation, EMA, cosine-LR,
auto-bundle and run-dir machinery from train.py; everything detection-specific is
dropped. `opndet train` (CLI) and `opndet.train.train()` auto-dispatch here when
the resolved preset is a seg head (`head: seg` / a `dome` alias).

v1 trains the elliptical dome rendered from `opndet sam-obb` OBB sidecars (the same
labels the -obb presets need). When an `opndet sam-seg` mask exporter lands,
encode_targets_seg already accepts `masks=` for the true distance-transform dome.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from opndet.augment import AugConfig, make_augment
from opndet.dataset import OpndetDataset, collate, load_datasets, split_samples
from opndet.decode import decode_seg
from opndet.encode import encode_targets_seg, obb_to_aabb
from opndet.loss import SegDomeLoss
from opndet.presets import resolve as _resolve_preset
from opndet.train import EMA, _bundle_run, _resolve_out_dir, cosine_lr
from opndet.training_defaults import deep_merge, defaults_for
from opndet.yaml_build import build_model_from_yaml


class _SegCfgShim:
    def __init__(self, img_h: int, img_w: int, seg_stride: int = 1):
        self.img_h, self.img_w, self.seg_stride = int(img_h), int(img_w), int(seg_stride)


def _touches_edge(blob, H: int, W: int, margin: float) -> bool:
    """True if the blob's AABB comes within `margin·dim` of any frame edge — i.e. it's
    a clipped/partial object. margin <= 0 → only literal-border touch (>=1px in)."""
    mx = max(1.0, margin * W) if margin > 0 else 1.0
    my = max(1.0, margin * H) if margin > 0 else 1.0
    return blob.x1 <= mx or blob.y1 <= my or blob.x2 >= (W - mx) or blob.y2 >= (H - my)


def _match_count_area(pred_blobs, gt_blobs, H: int, W: int, edge_margin: float = 0.0):
    """Greedy nearest-centroid match → (count_abs_err, list of |a_pred-a_gt|/a_gt).

    Edge leniency: a GT egg whose bbox touches within `edge_margin·dim` of the frame is
    a clipped partial — its visible area isn't a meaningful measurement — so it (and its
    matched pred) is skipped for the area-MAPE list. It still counts toward count_abs_err
    (you still want to know if the model missed/hallucinated an edge egg).
    """
    n_err = abs(len(pred_blobs) - len(gt_blobs))
    if not gt_blobs or not pred_blobs:
        return n_err, []
    used = set()
    apes = []
    for g in gt_blobs:
        best, bd = -1, 1e18
        for i, p in enumerate(pred_blobs):
            if i in used:
                continue
            d = (p.cx - g.cx) ** 2 + (p.cy - g.cy) ** 2
            if d < bd:
                bd, best = d, i
        if best >= 0 and bd <= (max(8.0, math.sqrt(max(g.area_px, 1.0)))) ** 2:
            used.add(best)
            if _touches_edge(g, H, W, edge_margin) or _touches_edge(pred_blobs[best], H, W, edge_margin):
                continue   # clipped object → area is unmeasurable; don't ding the model on it
            apes.append(abs(pred_blobs[best].area_px - g.area_px) / max(g.area_px, 1.0))
    return n_err, apes


@torch.no_grad()
def evaluate_seg(model, loader, device, fg_thresh: float = 0.5, edge_margin: float = 0.0) -> dict:
    model.eval()
    inter = denom = inter_i = union_i = 0.0
    n_imgs = 0
    count_errs: list[int] = []
    area_apes: list[float] = []
    for batch in loader:
        imgs, _boxes, targets = batch
        imgs = imgs.to(device)
        dome_t = targets["dome"].to(device)             # [B,1,H,W]
        logit = model.forward_with_alias(imgs, "raw")   # [B,1,H,W]
        p = torch.sigmoid(logit)
        pf = (p > fg_thresh).float()
        tf = (dome_t > 1e-3).float()
        inter += float((pf * tf).sum()); denom += float(pf.sum() + tf.sum())
        inter_i += float((pf * tf).sum()); union_i += float(((pf + tf) > 0).float().sum())
        pn = p.cpu().numpy(); tn = dome_t.cpu().numpy()
        Hd, Wd = pn.shape[2], pn.shape[3]
        for b in range(pn.shape[0]):
            n_imgs += 1
            pb = decode_seg(pn[b, 0], threshold=fg_thresh, min_area=4)
            gb = decode_seg(tn[b, 0], threshold=fg_thresh, min_area=4)
            ne, apes = _match_count_area(pb, gb, Hd, Wd, edge_margin)
            count_errs.append(ne); area_apes.extend(apes)
    dice = (2.0 * inter) / max(denom, 1e-9)
    iou = inter_i / max(union_i, 1e-9)
    return {
        "dice": dice,
        "fg_iou": iou,
        "count_mae": float(np.mean(count_errs)) if count_errs else 0.0,
        "area_mape": float(np.mean(area_apes)) if area_apes else 0.0,
        "n_val": n_imgs,
    }


# Distinct, colourblind-ish per-instance colours (RGB).
_SEG_PALETTE = [(86, 180, 233), (230, 159, 0), (0, 158, 115), (213, 94, 0), (204, 121, 167),
                (240, 228, 66), (0, 114, 178), (160, 200, 120), (200, 120, 180), (120, 160, 220),
                (233, 86, 120), (130, 200, 200), (200, 170, 90), (170, 120, 220), (90, 200, 130)]
_SEG_BODY_ALPHA = 130     # translucent fill — you see the object through it
_SEG_BORDER_ALPHA = 255   # crisp same-hue rim. Flip these two if you want a fainter border.


def _render_seg_decoded(dome: np.ndarray, thr: float = 0.5, min_area: int = 4,
                        edge_margin: float = 0.0) -> np.ndarray:
    """Decode a [H,W] dome → an RGBA (BGRA, for cv2.imwrite) overlay: each connected
    component filled with a distinct colour at `body_alpha`, its contour drawn in the
    SAME hue at `border_alpha`, a `<N>px` area label + centroid dot. Transparent
    elsewhere. The "instance segmentation" view (vs the raw `dome` heatmap); the
    dashboard's overlay-opacity slider scales the whole thing — same machinery as the
    other overlays. Blobs touching within `edge_margin·dim` of the frame (clipped
    partials) get a thin border + an `·E` tag (their area isn't a real measurement)."""
    import cv2
    H, W = dome.shape
    fg = (dome >= float(thr)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=4)
    canvas = np.zeros((H, W, 4), dtype=np.uint8)   # BGRA
    mx = max(1.0, edge_margin * W) if edge_margin > 0 else 1.0
    my = max(1.0, edge_margin * H) if edge_margin > 0 else 1.0
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP])
        w, h = int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT])
        edge = x <= mx or y <= my or (x + w) >= (W - mx) or (y + h) >= (H - my)
        r, g, b = _SEG_PALETTE[(lab - 1) % len(_SEG_PALETTE)]
        m = labels == lab
        canvas[m] = (b, g, r, _SEG_BODY_ALPHA)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rim = np.zeros((H, W, 4), dtype=np.uint8)
        cv2.drawContours(rim, cnts, -1, (b, g, r, _SEG_BORDER_ALPHA), 1 if edge else 2)
        rmask = rim[:, :, 3] > 0
        canvas[rmask] = rim[rmask]
        cx, cy = int(round(cents[lab, 0])), int(round(cents[lab, 1]))
        cv2.circle(canvas, (cx, cy), 2, (b, g, r, 255), -1)
        cv2.putText(canvas, f"{area}px{'·E' if edge else ''}", (cx + 4, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (b, g, r, 255), 1, cv2.LINE_AA)
    return canvas


def _seg_vis(model, ds, run_dir: Path, ep: int, n: int, device, tag: str = "val/seg",
             fg_thresh: float = 0.5, edge_margin: float = 0.0, db=None) -> None:
    """Vis for the seg head, on the samples of `ds`, registered under `tag` (e.g. `val/seg`,
    `test/seg`). Per sample:
      base    : sample_<i>_rgb.png         — the clean letterboxed RGB (written ONCE)
      overlay : dome_pred / dome_gt        — the dome as a TURBO heatmap (pred: this epoch; gt: once)
      overlay : seg_pred  / seg_gt         — the DECODED instance view: filled blobs + same-hue
                                             borders + per-blob `<N>px` area (pred: this epoch; gt: once)
      boxes   : kind pred / gt             — per-blob AABB + score=peak + meta {area_px, cx, cy}
    Files go under vis/<tag-with-_>/ ; the dashboard auto-discovers the overlay kinds and its
    overlay-opacity slider scales them (same as obj_heat / prior_heat). GT is deterministic
    (no aug on val/test) → written once to a stable path, re-referenced each call. Only the
    caller decides *when* to run this (the loop runs it on a new-best epoch — see train_seg)."""
    import cv2

    from opndet.decode import decode_seg
    from opndet.visualize import _denorm, save_heatmap_overlay_png
    shared = run_dir / "vis" / tag.replace("/", "_")
    epdir = shared / f"ep_{ep:03d}"
    epdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            img_t, _boxes, targets = ds[i]
            ih, iw = int(img_t.shape[-2]), int(img_t.shape[-1])
            rgb_path = shared / f"sample_{i}_rgb.png"
            if not rgb_path.exists():
                cv2.imwrite(str(rgb_path), cv2.cvtColor(_denorm(img_t), cv2.COLOR_RGB2BGR))
            # --- GT dome: deterministic → render heatmap + decoded ONCE (stable path) ---
            gt = targets["dome"][0].numpy()
            if gt.shape != (ih, iw):
                gt = cv2.resize(gt, (iw, ih), interpolation=cv2.INTER_LINEAR)
            gt_heat = shared / f"sample_{i}_gt_heat.png"
            gt_seg = shared / f"sample_{i}_gt_seg.png"
            if not gt_heat.exists():
                save_heatmap_overlay_png(gt, str(gt_heat), colormap=cv2.COLORMAP_TURBO, gamma=0.5)
            if not gt_seg.exists():
                cv2.imwrite(str(gt_seg), _render_seg_decoded(gt, fg_thresh, edge_margin=edge_margin))
            # --- predicted dome: this epoch ---
            pred = torch.sigmoid(model.forward_with_alias(img_t.unsqueeze(0).to(device), "raw"))[0, 0].cpu().numpy()
            if pred.shape != (ih, iw):
                pred = cv2.resize(pred, (iw, ih), interpolation=cv2.INTER_LINEAR)
            pred_heat = epdir / f"sample_{i}_pred_heat.png"
            pred_seg = epdir / f"sample_{i}_pred_seg.png"
            save_heatmap_overlay_png(pred, str(pred_heat), colormap=cv2.COLORMAP_TURBO, gamma=0.5)
            cv2.imwrite(str(pred_seg), _render_seg_decoded(pred, fg_thresh, edge_margin=edge_margin))
            if db is None:
                continue
            try:
                db.add_image(ep, tag, i, rgb_path)
                db.add_overlay(ep, tag, i, "dome_pred", pred_heat)
                db.add_overlay(ep, tag, i, "seg_pred", pred_seg)
                db.add_overlay(ep, tag, i, "dome_gt", gt_heat)
                db.add_overlay(ep, tag, i, "seg_gt", gt_seg)
                for kind, arr in (("pred", pred), ("gt", gt)):
                    blobs = decode_seg(arr, threshold=fg_thresh, min_area=4)
                    if blobs:
                        aabbs = np.array([[b.x1, b.y1, b.x2, b.y2] for b in blobs], np.float32)
                        peaks = np.array([b.peak for b in blobs], np.float32)
                        meta = {j: {"area_px": int(b.area_px), "cx": round(b.cx, 1), "cy": round(b.cy, 1)}
                                for j, b in enumerate(blobs)}
                        db.add_boxes(ep, tag, i, kind, aabbs, scores=peaks, meta=meta)
            except Exception:
                pass


def train_seg(cfg_path: str, run_name: str | None = None, runs_dir: str | None = None,
              resume: str | None = None) -> str:
    with open(cfg_path) as f:
        user_cfg = yaml.safe_load(f) or {}
    if "model_config" not in user_cfg:
        raise ValueError("train.yaml must specify model_config (a bbox-*-seg preset)")
    c = deep_merge(defaults_for(user_cfg["model_config"]), user_cfg)
    if runs_dir is not None:
        c["runs_dir"] = runs_dir
    if run_name is not None:
        c["name"] = run_name

    resume_state = None
    if resume is not None:
        resume_state = torch.load(resume, map_location="cpu", weights_only=False)
        out_dir = Path(resume).parent
    else:
        base = Path(c.get("runs_dir", "runs")) / str(c.get("name", c["model_config"]))
        out_dir = _resolve_out_dir(base, auto_increment=bool(c.get("auto_increment", True)))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")

    db = None
    if bool(c.get("metrics_db", True)):
        try:
            from opndet.metrics_db import MetricsDB
            db = MetricsDB(out_dir)
            db.set_config(c)
            print(f"metrics_db: {db.path}")
        except Exception as e:
            print(f"metrics_db disabled ({type(e).__name__}: {e})")
            db = None

    device = torch.device(c.get("device", "cuda") if torch.cuda.is_available()
                          or c.get("device") == "cpu" else "cpu")
    seed = int(c.get("seed", 0))
    torch.manual_seed(seed); np.random.seed(seed)

    # --- model ---
    model_path = _resolve_preset(c["model_config"])
    _mc = c.get("model", {}) or {}
    model = build_model_from_yaml(model_path, img_h=_mc.get("img_h"), img_w=_mc.get("img_w")).to(device)
    if "dome" not in getattr(model, "aliases", {}):
        raise ValueError(f"{c['model_config']} is not a seg head (no 'dome' alias) — use opndet.train.train()")
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
    in_ch, img_h, img_w = model.input_shape
    seg_stride = int(c.get("seg_stride", 1))
    cfg_shim = _SegCfgShim(img_h, img_w, seg_stride)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {c['model_config']}  params={n_params/1e6:.2f}M  input={in_ch}x{img_h}x{img_w}  "
          f"seg head: dense dome, output [1,1,{img_h//seg_stride},{img_w//seg_stride}]")

    # --- data (requires OBB sidecars; v1 renders the elliptical dome from them) ---
    print("loading data ...")
    data_cfg = c.get("data", {}) or {}
    sources = data_cfg["sources"]
    if not any((s.get("obb_dir") if isinstance(s, dict) else None) for s in sources):
        raise RuntimeError("bbox-*-seg needs OBB sidecars: run `opndet sam-obb` and set "
                           "data.sources[*].obb_dir (v1 derives the dome from the OBBs).")
    all_s = load_datasets(sources, image_filter=data_cfg.get("image_filter"))
    n_pre = len(all_s)
    out_kept = []
    for s in all_s:
        if getattr(s, "obbs", None) is None or s.obbs.shape[0] == 0:
            continue
        s.boxes = np.array([obb_to_aabb(*o) for o in s.obbs], dtype=np.float32)  # for aug clip/min-visible
        out_kept.append(s)
    all_s = out_kept
    if not all_s:
        raise RuntimeError("no samples have OBB sidecars — run `opndet sam-obb` first.")
    if n_pre != len(all_s):
        print(f"  dropped {n_pre - len(all_s)} samples without OBB sidecars")
    ratios = tuple(data_cfg.get("split_ratios", (0.8, 0.1, 0.1)))
    train_s, val_s, test_s = split_samples(all_s, ratios=ratios, seed=seed)
    print(f"total samples: {len(all_s)}   split: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    def _seg_encode(boxes_xyxy, obbs=None):
        return encode_targets_seg(cfg_shim, obbs=obbs if obbs is not None else np.zeros((0, 5), np.float32))
    _seg_encode._takes_obbs = True  # type: ignore[attr-defined]

    aug_dict = dict(c.get("augment", {}) or {})
    aug_dict.pop("temporal_prior", None)
    aug_dict.pop("hard_negative_pool", None); aug_dict.pop("hard_negative_prob", None); aug_dict.pop("hard_negative_count", None)
    aug_cfg = AugConfig(**aug_dict)
    mosaic_p = float(getattr(aug_cfg, "mosaic_prob", 0.0))
    min_vis = float(getattr(aug_cfg, "min_visible_frac", 0.5))
    nw = int(c.get("num_workers", 8))
    pf = int(c.get("prefetch_factor", 4))
    dl_kw = dict(num_workers=nw, collate_fn=collate, pin_memory=(device.type == "cuda"))
    if nw > 0:
        dl_kw["prefetch_factor"] = pf; dl_kw["persistent_workers"] = True
    bs = int(c["batch_size"])
    common = dict(img_h=img_h, img_w=img_w, encode_fn=_seg_encode, cache_images=bool(c.get("cache_images", False)),
                  in_ch=in_ch, stride=int(_mc.get("stride", 4)))
    train_ds = OpndetDataset(train_s, augment_fn=make_augment(aug_cfg), mosaic_prob=mosaic_p, min_visible_frac=min_vis, **common)
    val_ds   = OpndetDataset(val_s,   augment_fn=None, mosaic_prob=0.0, min_visible_frac=min_vis, **common)
    test_ds  = OpndetDataset(test_s,  augment_fn=None, mosaic_prob=0.0, min_visible_frac=min_vis, **common)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **dl_kw)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, **dl_kw)

    # --- loss / optim / ema ---
    lc = c.get("loss", {}) or {}
    loss_fn = SegDomeLoss(qfl_beta=float(lc.get("qfl_beta", 2.0)),
                          w_qfl=float(lc.get("seg_w_qfl", lc.get("w_hm", 1.0))),
                          w_dice=float(lc.get("seg_w_dice", 1.0)))
    base_lr = float(c["lr"]); wd = float(c.get("weight_decay", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    epochs = int(c["epochs"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup = int(c.get("warmup_steps", min(500, total_steps // 20)))
    log_every = max(1, steps_per_epoch // 8)   # ~8 train/loss points per epoch in the dashboard
    seg_fg = float(c.get("seg_fg_thresh", 0.5))            # dome foreground cut for decode/metrics/vis
    seg_edge_margin = float(c.get("seg_edge_margin", 0.0)) # >0 → clipped (frame-edge) blobs are lenient
    viz_on_best = bool(c.get("viz_only_on_improvement", True))
    ema_decay = float(c.get("ema_decay", 0.999))
    ema = EMA(model, decay=ema_decay, tau=int(c.get("ema_tau", 2000))) if ema_decay > 0 else None
    use_amp = bool(c.get("amp", True)) and device.type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(str(c.get("amp_dtype", "fp16")).lower(), torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    metric_for_best = str(c.get("metric_for_best", "dice"))
    lower_better = metric_for_best in ("count_mae", "area_mape")
    best_metric = float("inf") if lower_better else -float("inf")
    best_epoch = 0
    start_epoch, step = 0, 0
    if resume_state is not None:
        if ema is not None and resume_state.get("ema"):
            ema.shadow.load_state_dict(resume_state["ema"])
        start_epoch = int(resume_state.get("epoch", 0))
        step = int(resume_state.get("step", 0))
        best_metric = float(resume_state.get("best_metric", best_metric))
        best_epoch = int(resume_state.get("best_epoch", 0))
        print(f"resuming from epoch {start_epoch + 1}, best_{metric_for_best}={best_metric:.4f} @ ep {best_epoch}")

    vis_n = int(c.get("vis_samples", 16))   # # of val/test samples to vis on a new-best epoch
    patience = int(c.get("patience", 0))
    ckpt_path = out_dir / f"{c.get('name', c['model_config'])}_best.pt"

    def _save(path, ep, metrics):
        torch.save({"model": model.state_dict(),
                    "ema": (ema.shadow.state_dict() if ema is not None else None),
                    "epoch": ep, "step": step, "best_metric": best_metric, "best_epoch": best_epoch,
                    "metric_for_best": metric_for_best, "metrics": metrics, "temperature": 1.0,
                    "config": c}, path)

    for ep in range(start_epoch + 1, epochs + 1):
        model.train()
        t0 = time.time(); run_loss = 0.0; nb = 0; lr = base_lr
        for batch in train_loader:
            imgs, _boxes, targets = batch
            imgs = imgs.to(device, non_blocking=True)
            dome_t = targets["dome"].to(device, non_blocking=True)
            lr = cosine_lr(step, total_steps, base_lr, warmup=warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logit = model.forward_with_alias(imgs, "raw")
                out = loss_fn(logit, {"dome": dome_t})
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.update(model)
            run_loss += float(loss.detach()); nb += 1; step += 1
            if db is not None and step % log_every == 0:
                try:
                    db.add_scalar(step, "train/loss", float(loss.detach()))
                    db.add_scalar(step, "lr", lr)
                    db.flush_scalars()   # MetricsDB buffers scalars to 64 before writing+CHECKPOINTing;
                                         # the seg loop emits too few per step, so flush eagerly or the
                                         # dashboard's file-copy never sees them (this is *the* "no
                                         # scalars logged" bug — train.py just emits >64/step so it never
                                         # hit it).
                except Exception:
                    pass
        eval_model = ema.shadow if ema is not None else model
        m = evaluate_seg(eval_model, val_loader, device, fg_thresh=seg_fg, edge_margin=seg_edge_margin)
        dt = time.time() - t0
        print(f"epoch {ep:3d}/{epochs}  lr={lr:.2e}  loss={run_loss/max(nb,1):.4f}  "
              f"dice={m['dice']:.3f}  iou={m['fg_iou']:.3f}  count_mae={m['count_mae']:.2f}  "
              f"area_mape={m['area_mape']:.3f}  (n_val={m['n_val']}, {dt:.1f}s)")
        if db is not None:
            try:
                for k in ("dice", "fg_iou", "count_mae", "area_mape"):
                    db.add_scalar(ep, f"val/{k}", float(m[k]))
                db.flush_scalars()
            except Exception:
                pass

        cur = m[metric_for_best]
        is_best = (cur < best_metric) if lower_better else (cur > best_metric)
        if is_best:
            best_metric, best_epoch = cur, ep
            _save(ckpt_path, ep, m)
            print(f"  -> saved best ({metric_for_best}={best_metric:.4f})  {ckpt_path}")
        _save(out_dir / "last.pt", ep, m)

        # Vis (val + test) only when it's worth it: a new best, or the first/last epoch.
        # No point dumping heatmaps + decoded-instance PNGs for an epoch that didn't beat
        # what we already have (matches the detector's viz_only_on_improvement default).
        boundary = (ep == start_epoch + 1) or (ep == epochs)
        if vis_n > 0 and (not viz_on_best or is_best or boundary):
            try:
                _seg_vis(eval_model, val_ds, out_dir, ep, vis_n, device, tag="val/seg",
                         fg_thresh=seg_fg, edge_margin=seg_edge_margin, db=db)
            except Exception as e:
                print(f"  (val vis skipped: {type(e).__name__}: {e})")
            if len(test_s) > 0:
                try:
                    mt = evaluate_seg(eval_model, test_loader, device, fg_thresh=seg_fg, edge_margin=seg_edge_margin)
                    print(f"  test:  dice={mt['dice']:.3f}  iou={mt['fg_iou']:.3f}  count_mae={mt['count_mae']:.2f}  area_mape={mt['area_mape']:.3f}")
                    if db is not None:
                        for k in ("dice", "fg_iou", "count_mae", "area_mape"):
                            db.add_scalar(ep, f"test/{k}", float(mt[k]))
                        db.flush_scalars()
                    _seg_vis(eval_model, test_ds, out_dir, ep, vis_n, device, tag="test/seg",
                             fg_thresh=seg_fg, edge_margin=seg_edge_margin, db=db)
                except Exception as e:
                    print(f"  (test vis skipped: {type(e).__name__}: {e})")
            if db is not None:
                db.flush_scalars(); db.checkpoint()   # push the image/overlay INSERTs out of the WAL

        if patience > 0 and (ep - best_epoch) >= patience:
            print(f"early stop: no {metric_for_best} improvement for {patience} epochs (best={best_metric:.4f} @ ep {best_epoch})")
            break

    if db is not None:
        db.close()
    if bool(c.get("auto_bundle", True)):
        try:
            _bundle_run(out_dir, include_tb=False)
        except Exception as e:
            print(f"  (bundle skipped: {type(e).__name__}: {e})")
    return str(ckpt_path)
