"""Quick visual QA for TemporalPriorSynth output.

Renders n samples as side-by-side (RGB image, prior heatmap overlay) tiles.
Run after wiring synth into the dataset to eyeball-confirm priors look like
plausible past-frame trails (not e.g. stamps at current GT, or wrong-direction
trails, or all zeros).

Usage:
    python scripts/visualize_prior_synth.py \\
        --config train.yaml \\
        --n 16 \\
        --out prior_samples.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from opndet.augment import AugConfig, make_augment
from opndet.augment_temporal_prior import TemporalPriorSynth
from opndet.dataset import OpndetDataset, load_datasets, split_samples
from opndet.training_defaults import deep_merge, defaults_for


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm_rgb(t):
    arr = t[:3].numpy().transpose(1, 2, 0)
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _heatmap_overlay(rgb: np.ndarray, prior_full: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h = (np.clip(prior_full, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, color, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="train.yaml path")
    ap.add_argument("--n", type=int, default=16, help="number of samples")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default="prior_samples.png")
    args = ap.parse_args()

    with open(args.config) as f:
        user_cfg = yaml.safe_load(f)
    c = deep_merge(defaults_for(user_cfg["model_config"]), user_cfg)

    sources = c["data"]["sources"]
    samples = load_datasets(sources)
    train_s, _, _ = split_samples(samples, ratios=tuple(c["data"].get("split_ratios", [0.8, 0.1, 0.1])))

    aug_dict = dict(c.get("augment") or {})
    tp_cfg = aug_dict.pop("temporal_prior", None) or {}
    aug_cfg = AugConfig(**aug_dict)
    aug_fn = make_augment(aug_cfg)
    synth = TemporalPriorSynth(tp_cfg)

    img_h = c.get("model", {}).get("img_h", 384)
    img_w = c.get("model", {}).get("img_w", 512)
    stride = int(c.get("model", {}).get("stride", 4))

    ds = OpndetDataset(
        train_s, img_h, img_w,
        augment_fn=aug_fn,
        in_ch=4, prior_synth=synth, stride=stride,
        mosaic_prob=float(aug_cfg.mosaic_prob),
        min_visible_frac=float(aug_cfg.min_visible_frac),
    )

    tiles = []
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    for i in idxs:
        t, boxes, _ = ds[int(i)]
        rgb = _denorm_rgb(t)
        prior_full = t[3].numpy()
        overlay = _heatmap_overlay(rgb, prior_full)
        for b in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in b[:4])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tiles.append(np.concatenate([rgb, overlay], axis=1))

    cols = max(1, int(args.cols))
    while len(tiles) % cols != 0:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    grid = np.concatenate(rows, axis=0)

    out = Path(args.out)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote: {out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
