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

from opndet.augment import AugConfig, make_augment
from opndet.dataset import OpndetDataset, collate, load_datasets, split_samples
from opndet.decode import decode_batch
from opndet.encode import encode_targets
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
    """encode_targets needs an object with img_h/img_w/stride/out_h/out_w."""

    def __init__(self, img_h: int, img_w: int, stride: int):
        self.img_h = img_h
        self.img_w = img_w
        self.stride = stride
        self.out_h = img_h // stride
        self.out_w = img_w // stride


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


def _accumulate_correct(
    per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iouv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """For each image: compute IoU once, greedy-match by score desc per IoU threshold (vectorized over thresholds).
    Concatenate. Returns (all_scores [N], all_correct [N, n_iouv], total_n_gt)."""
    n_t = iouv.shape[0]
    parts_scores: list[np.ndarray] = []
    parts_correct: list[np.ndarray] = []
    total_gt = 0
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
        iou = _iou_xyxy(boxes[order], gt_boxes)  # [n_p, n_g]
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
    from opndet.metrics import hungarian_match
    model.eval()
    tp = fp = fn = 0
    n_pred = n_gt = 0
    per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for imgs, boxes_list, _ in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        out_t = out["output"] if isinstance(out, dict) else out
        out_np = out_t.cpu().numpy()
        dets_per_full = decode_batch(out_np, cfg_shim.img_h, cfg_shim.img_w, cfg_shim.stride, threshold=decode_threshold)
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

    iouv = np.arange(0.5, 1.0, 0.05, dtype=np.float64)
    all_scores, all_correct, total_gt = _accumulate_correct(per_image, iouv)
    aps = _ap_from_correct(all_scores, all_correct, total_gt)
    map50 = float(aps[0])
    map_50_95 = float(aps.mean())

    return {"precision": precision, "recall": recall, "f1": f1, "map50": map50, "map_50_95": map_50_95, "n_pred": float(n_pred), "n_gt": float(n_gt)}


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


def cosine_lr(step: int, total: int, base: float, warmup: int = 200, min_factor: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * p)))


def train(cfg_path: str, run_name: str | None = None, runs_dir: str | None = None,
          resume: str | None = None, teacher: str | None = None, self_distill: bool = False) -> None:
    with open(cfg_path) as f:
        c = yaml.safe_load(f)

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

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"tensorboard: {tb_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and c.get("device", "auto") != "cpu" else "cpu")
    print(f"device: {device}")

    print("loading data ...")
    samples = load_datasets(c["data"]["sources"])
    print(f"total samples: {len(samples)}")
    ratios = tuple(c["data"].get("split_ratios", [0.8, 0.1, 0.1]))
    train_s, val_s, test_s = split_samples(samples, ratios=ratios, seed=seed)
    print(f"split: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    aug_cfg = AugConfig(**(c.get("augment") or {}))
    aug_fn = make_augment(aug_cfg)

    model_path = _resolve_preset(c["model_config"])
    model = build_model_from_yaml(model_path).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
    in_ch, img_h, img_w = model.input_shape
    cfg_shim = _CfgShim(img_h, img_w, stride=int(c["model"].get("stride", 4)))
    n_params = sum(p.numel() for p in model.parameters())
    has_dist = "dist" in getattr(model, "aliases", {})
    print(f"model: {c['model_config']}  params={n_params/1e6:.2f}M  input={in_ch}x{img_h}x{img_w}{'  (dist head)' if has_dist else ''}")

    encode_fn = partial(encode_targets, cfg=cfg_shim, dist_head=has_dist)
    cache = bool(c.get("cache_images", False))
    mosaic_prob = float(aug_cfg.mosaic_prob if hasattr(aug_cfg, "mosaic_prob") else 0.0)
    min_vis = float(aug_cfg.min_visible_frac if hasattr(aug_cfg, "min_visible_frac") else 0.5)
    train_ds = OpndetDataset(train_s, img_h, img_w, augment_fn=aug_fn, encode_fn=encode_fn,
                             cache_images=cache, mosaic_prob=mosaic_prob, min_visible_frac=min_vis)
    val_ds = OpndetDataset(val_s, img_h, img_w, augment_fn=None, encode_fn=encode_fn, cache_images=cache)
    test_ds = OpndetDataset(test_s, img_h, img_w, augment_fn=None, encode_fn=encode_fn, cache_images=cache)
    nw = int(c.get("num_workers", 2))
    pf = int(c.get("prefetch_factor", 4)) if nw > 0 else None
    train_kw = dict(num_workers=nw, collate_fn=collate, pin_memory=device.type == "cuda",
                    persistent_workers=nw > 0, prefetch_factor=pf)
    eval_kw = {**train_kw, "pin_memory": False}  # val/test don't need pinned memory
    train_kw = {k: v for k, v in train_kw.items() if v is not None}
    eval_kw = {k: v for k, v in eval_kw.items() if v is not None}
    train_loader = InfiniteDataLoader(train_ds, batch_size=int(c["batch_size"]), shuffle=True, **train_kw)
    val_loader = InfiniteDataLoader(val_ds, batch_size=int(c["batch_size"]), shuffle=False, **eval_kw)
    # test_loader runs once at end of training; no benefit to keeping it alive.
    test_loader = DataLoader(test_ds, batch_size=int(c["batch_size"]), shuffle=False, **eval_kw)

    loss_kw = c.get("loss") or {}
    loss_kw.setdefault("img_h", img_h)
    loss_kw.setdefault("img_w", img_w)
    loss_kw.setdefault("stride", cfg_shim.stride)
    # Auto-mirror the model's peak op so count-aware loss sees the same sparse map as inference.
    # User can still override by setting peak_kernel/peak_eps explicitly in the yaml's `loss:` block.
    peak_k, peak_eps = _detect_peak_op(model)
    if peak_k is not None:
        loss_kw.setdefault("peak_kernel", peak_k)
        loss_kw.setdefault("peak_eps", peak_eps)
        if loss_kw.get("count_weight", 0.0) > 0:
            print(f"count-aware loss: peak_kernel={peak_k}, peak_eps={peak_eps} (auto-detected from model)")
    loss_fn = OpndetBboxLoss(**loss_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["lr"]), weight_decay=float(c.get("weight_decay", 1e-4)))
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
    }

    auto_calibrate = bool(c.get("auto_calibrate", True))
    calibrate_every = int(c.get("calibrate_every", 0))
    test_every = int(c.get("test_every", 0))

    patience_smart = bool(c.get("patience_smart", False))
    patience_min_delta = float(c.get("patience_min_delta", 0.003))
    # When patience_smart is True, patience fires only if NO tracked metric has improved
    # by patience_min_delta in `patience` epochs. Tracks both raw and calibrated f1/map.
    best_per_metric: dict[str, tuple[float, int]] = {}

    n_vis = int(c.get("vis_samples", 4))
    vis_every = int(c.get("vis_every", 5))
    vis_imgs = []
    vis_boxes = []
    if n_vis > 0 and vis_every > 0:
        for i in range(min(n_vis, len(val_ds))):
            img_t, boxes, _ = val_ds[i]
            vis_imgs.append(img_t)
            vis_boxes.append(boxes)
    vis_batch = torch.stack(vis_imgs, dim=0) if vis_imgs else None

    n_test_vis = int(c.get("test_samples", n_vis))
    test_vis_imgs = []
    test_vis_boxes = []
    if n_test_vis > 0 and test_every > 0 and len(test_ds) > 0:
        for i in range(min(n_test_vis, len(test_ds))):
            img_t, boxes, _ = test_ds[i]
            test_vis_imgs.append(img_t)
            test_vis_boxes.append(boxes)
    test_vis_batch = torch.stack(test_vis_imgs, dim=0) if test_vis_imgs else None

    metric_for_best = str(c.get("metric_for_best", "f1"))
    valid_metrics = ("f1", "map50", "map_50_95", "f1_cal", "map50_cal", "map_50_95_cal")
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

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running: dict[str, float] = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for imgs, _, tgt in pbar:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, base_lr, warmup=warmup)
            imgs = imgs.to(device, non_blocking=True)
            tgt = {k: v.to(device, non_blocking=True) for k, v in tgt.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                raw5 = model.forward_with_alias(imgs, "raw")
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
            if needs_scaler:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(opt)
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
            step += 1
            if step % 5 == 0:
                pbar.set_postfix(loss=f"{losses['loss'].item():.3f}", lr=f"{opt.param_groups[0]['lr']:.1e}")

        n_iter = max(1, len(train_loader))
        avg = {k: v / n_iter for k, v in running.items()}
        dt = time.time() - t0
        eval_model = ema.shadow if ema is not None else model
        m = evaluate(eval_model, val_loader, cfg_shim, device, score_thresh=float(c.get("eval_threshold", 0.3)))
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1:3d}/{epochs}  lr={cur_lr:.2e}  loss={avg['loss']:.4f}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  mAP@.5={m['map50']:.3f} mAP@.5:.95={m['map_50_95']:.3f}  ({dt:.1f}s)")

        ep = epoch + 1
        writer.add_scalar("lr", cur_lr, ep)
        for k, v in avg.items():
            writer.add_scalar(f"train/{k}", v, ep)
        for k, v in m.items():
            writer.add_scalar(f"val/{k}", v, ep)
        writer.add_scalar("time/epoch_s", dt, ep)

        m_cal = None
        cur_T = 1.0
        if calibrate_every > 0 and ep % calibrate_every == 0:
            from opndet.calibrate import (apply_temperature, collect_calibration_data,
                                            fit_temperature)
            from opndet.metrics import calibration_bins
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
                                 score_thresh=float(c.get("eval_threshold", 0.3)))
                for k, v in m_cal.items():
                    writer.add_scalar(f"val_cal/{k}", v, ep)
                print(f"  calib: T={cur_T:.3f}  ECE {_ece_pre:.3f} -> {_ece_post:.3f}  "
                      f"F1_cal={m_cal['f1']:.3f}  mAP_cal={m_cal['map50']:.3f}/{m_cal['map_50_95']:.3f}")
                # Restore T=1.0 so subsequent epochs' raw eval starts clean.
                apply_temperature(eval_model, 1.0)

        if test_every > 0 and ep % test_every == 0 and len(test_ds) > 0:
            mt = evaluate(eval_model, test_loader, cfg_shim, device, score_thresh=float(c.get("eval_threshold", 0.3)))
            print(f"  test: P={mt['precision']:.3f} R={mt['recall']:.3f} F1={mt['f1']:.3f}  mAP@.5={mt['map50']:.3f} mAP@.5:.95={mt['map_50_95']:.3f}")
            for k, v in mt.items():
                writer.add_scalar(f"test/{k}", v, ep)
            if test_vis_batch is not None:
                grid = render_predictions(
                    eval_model, test_vis_batch, test_vis_boxes, img_h, img_w, cfg_shim.stride,
                    threshold=float(c.get("eval_threshold", 0.3)), device=device,
                )
                writer.add_images("test/preds", grid, ep, dataformats="NCHW")

        if vis_batch is not None and (ep == 1 or ep % vis_every == 0 or ep == epochs):
            grid = render_predictions(
                model, vis_batch, vis_boxes, img_h, img_w, cfg_shim.stride,
                threshold=float(c.get("eval_threshold", 0.3)), device=device,
            )
            writer.add_images("val/preds", grid, ep, dataformats="NCHW")

        # Selection metric: calibrated value if metric_for_best ends in _cal AND we got m_cal this epoch.
        # Falls back to raw if calibration didn't fire (e.g. calibrate_every step skipped).
        if metric_is_cal and m_cal is not None:
            cur = float(m_cal[metric_base])
        else:
            cur = float(m[metric_base if metric_is_cal else metric_for_best])
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
        torch.save(ckpt, out_dir / "last.pt")
        if cur > best_metric:
            best_metric = cur
            best_epoch = ep
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> saved best ({metric_for_best}={best_metric:.3f}, T={cur_T:.3f})")

        if patience > 0:
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
    m = evaluate(model, test_loader, cfg_shim, device, score_thresh=float(c.get("eval_threshold", 0.3)))
    print(f"TEST: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  n_pred={m['n_pred']:.0f}/n_gt={m['n_gt']:.0f}")
    for k, v in m.items():
        writer.add_scalar(f"test/{k}", v, epochs)

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
            m_cal = evaluate(model, test_loader, cfg_shim, device, score_thresh=float(c.get("eval_threshold", 0.3)))
            print(f"TEST(cal): P={m_cal['precision']:.3f} R={m_cal['recall']:.3f} F1={m_cal['f1']:.3f}  n_pred={m_cal['n_pred']:.0f}/n_gt={m_cal['n_gt']:.0f}")
            for k, v in m_cal.items():
                writer.add_scalar(f"test_cal/{k}", v, epochs)
        except Exception as e:
            print(f"  calibration failed: {e}")
    writer.close()


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
