from __future__ import annotations

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

from opndet.augment import AugConfig, make_augment
from opndet.dataset import OpndetDataset, collate, load_datasets, split_samples
from opndet.decode import decode_batch
from opndet.encode import encode_targets
from opndet.loss import OpndetBboxLoss
from opndet.presets import resolve as _resolve_preset
from opndet.visualize import render_predictions
from opndet.yaml_build import build_model_from_yaml


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


@torch.no_grad()
def evaluate(model, loader, cfg_shim: _CfgShim, device: torch.device, score_thresh: float = 0.3, iou_thresh: float = 0.5) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    n_pred = n_gt = 0
    for imgs, boxes_list, _ in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        out_t = out["output"] if isinstance(out, dict) else out
        out_np = out_t.cpu().numpy()
        dets_per = decode_batch(out_np, cfg_shim.img_h, cfg_shim.img_w, cfg_shim.stride, threshold=score_thresh)
        for dets, gt in zip(dets_per, boxes_list):
            n_pred += len(dets)
            n_gt += gt.shape[0]
            if not dets or gt.shape[0] == 0:
                fn += gt.shape[0]
                fp += len(dets)
                continue
            pred_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
            ious = _iou_xyxy(pred_boxes, gt)
            matched_gt = set()
            for pi in np.argsort([-d.score for d in dets]):
                gj = ious[pi].argmax()
                if ious[pi, gj] >= iou_thresh and gj not in matched_gt:
                    matched_gt.add(int(gj))
                    tp += 1
                else:
                    fp += 1
            fn += gt.shape[0] - len(matched_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "n_pred": float(n_pred), "n_gt": float(n_gt)}


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


def train(cfg_path: str, run_name: str | None = None, runs_dir: str | None = None) -> None:
    with open(cfg_path) as f:
        c = yaml.safe_load(f)

    if runs_dir is not None:
        c["runs_dir"] = runs_dir
    if run_name is not None:
        c["name"] = run_name

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
    in_ch, img_h, img_w = model.input_shape
    cfg_shim = _CfgShim(img_h, img_w, stride=int(c["model"].get("stride", 4)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {c['model_config']}  params={n_params/1e6:.2f}M  input={in_ch}x{img_h}x{img_w}")

    encode_fn = partial(encode_targets, cfg=cfg_shim)
    train_ds = OpndetDataset(train_s, img_h, img_w, augment_fn=aug_fn, encode_fn=encode_fn)
    val_ds = OpndetDataset(val_s, img_h, img_w, augment_fn=None, encode_fn=encode_fn)
    test_ds = OpndetDataset(test_s, img_h, img_w, augment_fn=None, encode_fn=encode_fn)
    nw = int(c.get("num_workers", 2))
    pf = int(c.get("prefetch_factor", 4)) if nw > 0 else None
    common = dict(num_workers=nw, collate_fn=collate, pin_memory=device.type == "cuda",
                  persistent_workers=nw > 0, prefetch_factor=pf)
    common = {k: v for k, v in common.items() if v is not None}
    train_loader = DataLoader(train_ds, batch_size=int(c["batch_size"]), shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=int(c["batch_size"]), shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=int(c["batch_size"]), shuffle=False, **common)

    loss_fn = OpndetBboxLoss(**(c.get("loss") or {}))
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["lr"]), weight_decay=float(c.get("weight_decay", 1e-4)))
    use_amp = device.type == "cuda" and c.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(c["epochs"])
    total_steps = epochs * max(1, len(train_loader))
    base_lr = float(c["lr"])
    warmup = int(c.get("warmup_steps", 200))

    n_vis = int(c.get("vis_samples", 8))
    vis_imgs = []
    vis_boxes = []
    for i in range(min(n_vis, len(val_ds))):
        img_t, boxes, _ = val_ds[i]
        vis_imgs.append(img_t)
        vis_boxes.append(boxes)
    vis_batch = torch.stack(vis_imgs, dim=0) if vis_imgs else None

    best_f1 = -1.0
    step = 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "l_hm": 0.0, "l_cxy": 0.0, "l_wh": 0.0}
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for imgs, _, tgt in pbar:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, base_lr, warmup=warmup)
            imgs = imgs.to(device, non_blocking=True)
            tgt = {k: v.to(device, non_blocking=True) for k, v in tgt.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                raw5 = model.forward_with_alias(imgs, "raw")
                losses = loss_fn(raw5, tgt)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(opt)
            scaler.update()
            for k in running:
                running[k] += float(losses.get(k, torch.tensor(0.0)).detach()) if k in losses else 0.0
            step += 1
            if step % 5 == 0:
                pbar.set_postfix(loss=f"{losses['loss'].item():.3f}", lr=f"{opt.param_groups[0]['lr']:.1e}")

        n_iter = max(1, len(train_loader))
        avg = {k: v / n_iter for k, v in running.items()}
        dt = time.time() - t0
        m = evaluate(model, val_loader, cfg_shim, device, score_thresh=float(c.get("eval_threshold", 0.3)))
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1:3d}/{epochs}  lr={cur_lr:.2e}  loss={avg['loss']:.4f}  hm={avg['l_hm']:.4f} cxy={avg['l_cxy']:.4f} wh={avg['l_wh']:.4f}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  n_pred={m['n_pred']:.0f}/n_gt={m['n_gt']:.0f}  ({dt:.1f}s)")

        ep = epoch + 1
        writer.add_scalar("lr", cur_lr, ep)
        for k, v in avg.items():
            writer.add_scalar(f"train/{k}", v, ep)
        for k, v in m.items():
            writer.add_scalar(f"val/{k}", v, ep)
        writer.add_scalar("time/epoch_s", dt, ep)

        if vis_batch is not None:
            grid = render_predictions(
                model, vis_batch, vis_boxes, img_h, img_w, cfg_shim.stride,
                threshold=float(c.get("eval_threshold", 0.3)), device=device,
            )
            writer.add_images("val/preds", grid, ep, dataformats="NCHW")

        ckpt = {"epoch": ep, "model": model.state_dict(), "metrics": m, "config": c}
        torch.save(ckpt, out_dir / "last.pt")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> saved best (F1={best_f1:.3f})")

    print("running final test eval ...")
    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    m = evaluate(model, test_loader, cfg_shim, device, score_thresh=float(c.get("eval_threshold", 0.3)))
    print(f"TEST: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  n_pred={m['n_pred']:.0f}/n_gt={m['n_gt']:.0f}")
    for k, v in m.items():
        writer.add_scalar(f"test/{k}", v, epochs)
    writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None, help="Override config 'name'")
    ap.add_argument("--runs-dir", default=None, help="Override config 'runs_dir'")
    args = ap.parse_args()
    train(args.config, run_name=args.run_name, runs_dir=args.runs_dir)


if __name__ == "__main__":
    main()
