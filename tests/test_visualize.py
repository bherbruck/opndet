"""save_layered_vis — the clean RGB is written once per (tag,sample), not per epoch."""
import numpy as np
import torch

from opndet.presets import resolve
from opndet.visualize import save_layered_vis
from opndet.yaml_build import build_model_from_yaml


def test_rgb_written_once_overlays_per_epoch(tmp_path):
    m = build_model_from_yaml(resolve("bbox-n-obb")).eval()
    ic, H, W = m.input_shape
    imgs = torch.zeros(2, ic, H, W)
    gt = [np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32)]
    tag_dir = tmp_path / "vis" / "val_preds"

    for ep in (1, 2, 3):
        save_layered_vis(m, imgs, gt, H, W, 4, tag_dir / f"ep_{ep:03d}", None, "val/preds", ep, device="cpu")

    # rgb lives at the stable per-tag path, written once — not under any ep_NNN dir
    for i in (0, 1):
        assert (tag_dir / f"sample_{i}_rgb.png").exists()
        for ep in (1, 2, 3):
            assert not (tag_dir / f"ep_{ep:03d}" / f"sample_{i}_rgb.png").exists()
    # exactly 2 rgb PNGs total (one per sample), regardless of epoch count
    assert len(list(tag_dir.rglob("*_rgb.png"))) == 2

    # the model-output heatmap overlay IS per-epoch
    for ep in (1, 2, 3):
        for i in (0, 1):
            assert (tag_dir / f"ep_{ep:03d}" / f"sample_{i}_obj_heat.png").exists()
    assert len(list(tag_dir.rglob("*_obj_heat.png"))) == 6  # 2 samples × 3 epochs
