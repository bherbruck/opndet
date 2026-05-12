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


def _match_count_area(pred_blobs, gt_blobs):
    """Greedy nearest-centroid match → (count_abs_err, list of |a_pred-a_gt|/a_gt)."""
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
            apes.append(abs(pred_blobs[best].area_px - g.area_px) / max(g.area_px, 1.0))
    return n_err, apes


@torch.no_grad()
def evaluate_seg(model, loader, device, fg_thresh: float = 0.5) -> dict:
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
        for b in range(pn.shape[0]):
            n_imgs += 1
            pb = decode_seg(pn[b, 0], threshold=fg_thresh, min_area=4)
            gb = decode_seg(tn[b, 0], threshold=fg_thresh, min_area=4)
            ne, apes = _match_count_area(pb, gb)
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


def _seg_vis(model, val_ds, run_dir: Path, ep: int, n: int, device, fg_thresh: float = 0.5, db=None) -> None:
    """Per-epoch vis: stable RGB once per sample + predicted & GT dome heatmaps per epoch.
    Also registers them in the DuckDB store (tag `val/seg`, overlay kinds `dome_pred`
    / `dome_gt`) so the `opndet dashboard` image view shows seg runs."""
    import cv2

    from opndet.visualize import _denorm, save_heatmap_overlay_png
    shared = run_dir / "vis" / "seg"
    epdir = shared / f"ep_{ep:03d}"
    epdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(min(n, len(val_ds))):
            img_t, _boxes, targets = val_ds[i]
            rgb_path = shared / f"sample_{i}_rgb.png"
            if not rgb_path.exists():
                rgb = _denorm(img_t)
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            logit = model.forward_with_alias(img_t.unsqueeze(0).to(device), "raw")
            pred = torch.sigmoid(logit)[0, 0].cpu().numpy()
            pred_path = epdir / f"sample_{i}_pred.png"
            gt_path = epdir / f"sample_{i}_gt.png"
            save_heatmap_overlay_png(pred, str(pred_path), colormap=cv2.COLORMAP_TURBO, gamma=0.5)
            save_heatmap_overlay_png(targets["dome"][0].numpy(), str(gt_path),
                                     colormap=cv2.COLORMAP_TURBO, gamma=0.5)
            if db is not None:
                try:
                    db.add_image(ep, "val/seg", i, rgb_path)
                    db.add_overlay(ep, "val/seg", i, "dome_pred", pred_path)
                    db.add_overlay(ep, "val/seg", i, "dome_gt", gt_path)
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
    sources = c["data"]["sources"]
    if not any((s.get("obb_dir") if isinstance(s, dict) else None) for s in sources):
        raise RuntimeError("bbox-*-seg needs OBB sidecars: run `opndet sam-obb` and set "
                           "data.sources[*].obb_dir (v1 derives the dome from the OBBs).")
    all_s = load_datasets(sources)
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
    ratios = tuple((c.get("data", {}) or {}).get("split_ratios", (0.8, 0.1, 0.1)))
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

    vis_every = int(c.get("vis_every", 1)); vis_n = int(c.get("vis_samples", 16))
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
        m = evaluate_seg(eval_model, val_loader, device)
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
        if vis_every > 0 and (ep % vis_every == 0):
            try:
                _seg_vis(eval_model, val_ds, out_dir, ep, vis_n, device, db=db)
                if db is not None:
                    db.flush_scalars(); db.checkpoint()   # push the image/overlay INSERTs out of the WAL
            except Exception as e:
                print(f"  (vis skipped: {type(e).__name__}: {e})")
        cur = m[metric_for_best]
        is_best = (cur < best_metric) if lower_better else (cur > best_metric)
        if is_best:
            best_metric, best_epoch = cur, ep
            _save(ckpt_path, ep, m)
            print(f"  -> saved best ({metric_for_best}={best_metric:.4f})  {ckpt_path}")
        _save(out_dir / "last.pt", ep, m)
        if patience > 0 and (ep - best_epoch) >= patience:
            print(f"early stop: no {metric_for_best} improvement for {patience} epochs (best={best_metric:.4f} @ ep {best_epoch})")
            break

    # final test
    eval_model = ema.shadow if ema is not None else model
    mt = evaluate_seg(eval_model, test_loader, device)
    print(f"test:  dice={mt['dice']:.3f}  iou={mt['fg_iou']:.3f}  count_mae={mt['count_mae']:.2f}  area_mape={mt['area_mape']:.3f}")
    if db is not None:
        try:
            for k in ("dice", "fg_iou", "count_mae", "area_mape"):
                db.add_scalar(epochs, f"test/{k}", float(mt[k]))
        finally:
            db.close()
    if bool(c.get("auto_bundle", True)):
        try:
            _bundle_run(out_dir, include_tb=False)
        except Exception as e:
            print(f"  (bundle skipped: {type(e).__name__}: {e})")
    return str(ckpt_path)
