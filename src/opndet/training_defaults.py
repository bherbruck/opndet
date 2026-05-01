"""Per-preset training defaults. Merged under the user's train.yaml so the user only writes
what they want to change. Smart defaults scale with model size: bigger models get lower LR,
smaller batch, longer warmup, lighter aug for tiny / heavier aug for huge.

User yaml always wins. Anything you set there overrides these defaults.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


# ----------------------------------------------------------------------
# Common defaults: apply to every preset unless explicitly overridden.
# ----------------------------------------------------------------------
COMMON: dict[str, Any] = {
    "seed": 0,
    "device": "cuda",
    "amp": True,
    "amp_dtype": "fp16",

    "ema_decay": 0.999,

    # TB visualization defaults — small + infrequent. vis_threshold defaults to eval_threshold
    # (resolved at train-time, see train.py) so you see what passes deployment filter, with
    # color-coding making marginal preds (near the threshold) visually distinct from confident ones.
    "vis_samples": 4,
    "vis_every": 5,
    "test_samples": 4,
    "test_every": 0,
    # vis_threshold: <unset>  -> falls back to eval_threshold

    # Schedule defaults
    "epochs": 100,
    "patience": 30,
    "patience_smart": True,
    "patience_min_delta": 0.003,
    "metric_for_best": "f1_opt_cal",  # picks best.pt by calibrated F1 at the optimal threshold
    "calibrate_every": 1,             # forced anyway by _cal metric; explicit for clarity
    "auto_calibrate": True,
    "auto_bundle": True,              # zip the run dir at end; on Colab, also triggers download
    "bundle_include_tb": False,       # include tfevents in the bundle (big — opt in)

    "weight_decay": 1.0e-4,
    "eval_threshold": 0.2,
    "warmup_steps": 200,
    "num_workers": 8,
    "prefetch_factor": 4,
    "cache_images": False,

    "model": {"stride": 4},
    "data": {"split_ratios": [0.8, 0.1, 0.1]},

    "loss": {
        "w_hm": 1.0, "w_cxy": 1.0, "w_wh": 1.5,
        "focal_alpha": 2.0, "focal_beta": 4.0,
        "wh_loss": "ciou",
        "cls_loss": "vfl",
        "vfl_alpha": 0.75, "vfl_gamma": 2.0,
        "repulsion_weight": 0.0,
        "count_weight": 0.0,
        "convexity_weight": 0.0,
        "nwd_c": 12.8,
    },

    "distill": {
        "hm_weight": 1.0, "reg_weight": 0.5, "conf_gate": 0.5,
        "full_distill": False, "neg_gate": 0.0, "kd_temperature": 1.0,
    },
}


# ----------------------------------------------------------------------
# Per-tier augmentation profiles. Tiny models can't absorb mosaic/cutout.
# ----------------------------------------------------------------------
_AUG_BASE = {
    "enabled": True,
    "brightness": 0.4, "contrast": 0.4, "gamma": [0.6, 1.6],
    "hue": 20, "saturation": 0.5, "grayscale_prob": 0.2,
    "blur_prob": 0.1, "noise_sigma": 0.02,
    "hflip_prob": 0.5, "vflip_prob": 0.5, "rotate90_prob": 0.5,
    "scale_jitter": [0.7, 1.3], "translate_frac": 0.1,
    "cutout_count": 3, "cutout_size_frac": [0.05, 0.20],
    "min_visible_frac": 0.5,
}

_AUG_LIGHT = {**_AUG_BASE, "mosaic_prob": 0.0, "cutout_prob": 0.0}    # f, p
_AUG_MEDIUM = {**_AUG_BASE, "mosaic_prob": 0.05, "cutout_prob": 0.1}  # n
_AUG_FULL = {**_AUG_BASE, "mosaic_prob": 0.1, "cutout_prob": 0.3}     # s, m, l, x
_AUG_HEAVY = {**_AUG_BASE, "mosaic_prob": 0.3, "cutout_prob": 0.5}    # h, g, t

# Slightly stronger KD weights when student is much smaller than teacher.
_DISTILL_TIGHTER = {"hm_weight": 1.5, "reg_weight": 0.75, "conf_gate": 0.5}


# ----------------------------------------------------------------------
# Per-preset overrides. Scaling rule: roughly halve LR per ~4x param jump.
# ----------------------------------------------------------------------
PER_SIZE: dict[str, dict[str, Any]] = {
    # tiny tier — light aug, full LR, big batches (cheap compute), tight KD.
    "bbox-f": {"lr": 3.0e-3, "batch_size": 256, "augment": _AUG_LIGHT, "distill": _DISTILL_TIGHTER},
    "bbox-p": {"lr": 3.0e-3, "batch_size": 256, "augment": _AUG_LIGHT, "distill": _DISTILL_TIGHTER},
    "bbox-n": {"lr": 3.0e-3, "batch_size": 128, "augment": _AUG_MEDIUM, "distill": _DISTILL_TIGHTER},

    # mid tier — full aug, full LR, standard KD.
    "bbox-s":      {"lr": 3.0e-3, "batch_size": 128, "augment": _AUG_FULL},
    "bbox-m":      {"lr": 3.0e-3, "batch_size": 64,  "augment": _AUG_FULL},
    "bbox-m-dist": {"lr": 3.0e-3, "batch_size": 64,  "augment": _AUG_FULL},

    # large tier — lower LR, longer warmup. Watch fp16 NaN risk.
    "bbox-l":      {"lr": 2.0e-3, "batch_size": 64, "warmup_steps": 400, "augment": _AUG_FULL},
    "bbox-l-dist": {"lr": 2.0e-3, "batch_size": 64, "warmup_steps": 400, "augment": _AUG_FULL},
    "bbox-x":      {"lr": 1.5e-3, "batch_size": 32, "warmup_steps": 400, "augment": _AUG_FULL},
    "bbox-x-dist": {"lr": 1.5e-3, "batch_size": 32, "warmup_steps": 400, "augment": _AUG_FULL},

    # jumbo tier — very low LR, small batch, longer warmup, heavier aug.
    "bbox-h":      {"lr": 1.0e-3, "batch_size": 32, "warmup_steps": 600, "augment": _AUG_HEAVY},
    "bbox-h-dist": {"lr": 1.0e-3, "batch_size": 32, "warmup_steps": 600, "augment": _AUG_HEAVY},
    "bbox-g":      {"lr": 7.0e-4, "batch_size": 16, "warmup_steps": 600, "augment": _AUG_HEAVY},
    "bbox-g-dist": {"lr": 7.0e-4, "batch_size": 16, "warmup_steps": 600, "augment": _AUG_HEAVY},
    "bbox-t":      {"lr": 5.0e-4, "batch_size": 16, "warmup_steps": 800, "augment": _AUG_HEAVY},
    "bbox-t-dist": {"lr": 5.0e-4, "batch_size": 16, "warmup_steps": 800, "augment": _AUG_HEAVY},
}


def deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge b into a; b's values win. Returns a new dict (no mutation)."""
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def defaults_for(preset_name: str) -> dict:
    """Return merged training defaults (COMMON + per-preset overrides) for a model preset."""
    size = PER_SIZE.get(preset_name, {})
    return deep_merge(COMMON, size)
