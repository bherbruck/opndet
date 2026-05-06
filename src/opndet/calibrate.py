from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from opndet.dataset import OpndetDataset, collate, load_datasets, split_samples
from opndet.decode import decode_batch
from opndet.encode import encode_targets
from opndet.metrics import calibration_bins, hungarian_match
from opndet.presets import resolve as _resolve_preset
from opndet.yaml_build import build_model_from_yaml


class _CfgShim:
    def __init__(self, img_h: int, img_w: int, stride: int):
        self.img_h = img_h
        self.img_w = img_w
        self.stride = stride
        self.out_h = img_h // stride
        self.out_w = img_w // stride


def apply_temperature(model: torch.nn.Module, T: float) -> int:
    """Set temperature on all calibration-aware layers (SigmoidPeakSuppress, SigmoidT).
    Returns count of layers updated."""
    n = 0
    for m in model.modules():
        if type(m).__name__ in ("SigmoidPeakSuppress", "SigmoidT"):
            m.temperature = float(T)
            n += 1
    return n


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Platt-style scalar fit. Minimizes NLL of sigmoid(logit/T) against binary labels.
    Numerically stable via logaddexp.
    """
    if logits.shape[0] == 0:
        return 1.0
    labels = labels.astype(np.float64)
    logits = logits.astype(np.float64)

    def nll(t: float) -> float:
        z = logits / max(t, 1e-6)
        # log(sigmoid(z)) = -log(1 + exp(-z));  log(1 - sigmoid(z)) = -log(1 + exp(z))
        log_sig = -np.logaddexp(0.0, -z)
        log_1msig = -np.logaddexp(0.0, z)
        return float(-np.mean(labels * log_sig + (1.0 - labels) * log_1msig))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded", options={"xatol": 1e-4})
    return float(res.x)


def _logit_from_score(s: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = np.clip(s, eps, 1.0 - eps)
    return np.log(s / (1.0 - s))


@torch.no_grad()
def collect_calibration_data(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg_shim: _CfgShim,
    device: torch.device,
    iou_thresh: float = 0.5,
    decode_threshold: float = 0.05,
    max_dets_per_image: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over the loader, decode peaks, Hungarian-match to GT, return (logits, labels).

    Important: model.SigmoidPeakSuppress.temperature should be 1.0 during data collection so
    we recover raw-logit/score pairs. Caller is responsible for that.

    max_dets_per_image: top-K cap to avoid Hungarian-matching huge det pools
    on untrained models (warm 4-ch input + low threshold = thousands of dets).
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for imgs, boxes_list, _ in tqdm(loader, desc="calib", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        out_t = out["output"] if isinstance(out, dict) else out
        out_np = out_t.detach().cpu().numpy()
        dets_per = decode_batch(out_np, cfg_shim.img_h, cfg_shim.img_w, cfg_shim.stride, threshold=decode_threshold)
        for dets, gt in zip(dets_per, boxes_list):
            if not dets:
                continue
            if len(dets) > max_dets_per_image:
                dets = sorted(dets, key=lambda d: -d.score)[:max_dets_per_image]
            scores = np.array([d.score for d in dets], dtype=np.float32)
            pb = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
            m = hungarian_match(pb, gt.astype(np.float32), iou_thresh=iou_thresh)
            labels = np.zeros(pb.shape[0], dtype=np.int64)
            if m.pairs.shape[0] > 0:
                labels[m.pairs[:, 0]] = 1
            all_logits.append(_logit_from_score(scores))
            all_labels.append(labels)
    if not all_logits:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.concatenate(all_logits), np.concatenate(all_labels)


def calibrate_ckpt(
    ckpt_path: str | Path,
    config_path: str | Path | None = None,
    split: str = "val",
    save: bool = True,
) -> dict:
    """Fit a Platt-style temperature on `split`, write it back to the ckpt under key 'temperature'.
    Returns a dict with the fitted T plus before/after ECE. When config_path is None, uses the
    ckpt's saved config (works when running on the same machine that trained the model)."""
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

    samples = load_datasets(c["data"]["sources"])
    ratios = tuple(c["data"].get("split_ratios", [0.8, 0.1, 0.1]))
    seed = int(c.get("seed", 0))
    train_s, val_s, test_s = split_samples(samples, ratios=ratios, seed=seed)
    sel = {"train": train_s, "val": val_s, "test": test_s}[split]

    model_path = _resolve_preset(c["model_config"])
    model = build_model_from_yaml(model_path).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)

    # Force T=1.0 during data collection so we recover raw logits.
    apply_temperature(model, 1.0)

    in_ch, img_h, img_w = model.input_shape
    cfg_shim = _CfgShim(img_h, img_w, stride=int(c["model"].get("stride", 4)))
    encode_fn = partial(encode_targets, cfg=cfg_shim)
    ds = OpndetDataset(sel, img_h, img_w, augment_fn=None, encode_fn=encode_fn, cache_images=False)
    bs = int(c.get("batch_size", 8))
    nw = int(c.get("num_workers", 2))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=False)

    print(f"calibrating on split={split}  ({len(sel)} samples)")
    logits, labels = collect_calibration_data(model, loader, cfg_shim, device)
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    print(f"collected {labels.shape[0]} predictions: {n_pos} TP, {n_neg} FP")

    if labels.shape[0] == 0:
        print("no predictions decoded — model is probably untrained. T=1.0")
        return {"temperature": 1.0, "ece_before": 0.0, "ece_after": 0.0, "n_samples": 0}

    # ECE before calibration (on raw scores)
    scores_before = 1.0 / (1.0 + np.exp(-logits))
    cal_before = calibration_bins(scores_before.astype(np.float32), labels, n_bins=10)
    ece_before = float(cal_before["ece"])

    T = fit_temperature(logits, labels)

    scores_after = 1.0 / (1.0 + np.exp(-logits / T))
    cal_after = calibration_bins(scores_after.astype(np.float32), labels, n_bins=10)
    ece_after = float(cal_after["ece"])

    print(f"fitted T = {T:.4f}  (T<1 sharpens, T>1 softens)")
    print(f"ECE before: {ece_before:.4f}   after: {ece_after:.4f}   delta: {ece_after - ece_before:+.4f}")

    if save:
        sd["temperature"] = float(T)
        torch.save(sd, ckpt_path)
        print(f"wrote temperature into {ckpt_path}")

    return {"temperature": T, "ece_before": ece_before, "ece_after": ece_after, "n_samples": int(labels.shape[0])}
