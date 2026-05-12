"""Tests for hard-negative mining + paste augmentation. ROADMAP §1.7 / §1.8 Phase 5."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opndet.augment import AugConfig, _hard_negative_paste, make_augment
from opndet.mine_negatives import (_crop_patch, _identify_ghosts,
                                    _saliency_centroid, load_pool)


def test_identify_ghosts_no_gt_means_all_ghosts():
    pred = np.array([[10, 10, 30, 30], [100, 100, 120, 120]], dtype=np.float32)
    gt = np.zeros((0, 4), dtype=np.float32)
    assert _identify_ghosts(pred, gt, stride=4) == [0, 1]


def test_identify_ghosts_overlap_with_gt_not_ghost():
    pred = np.array([[100, 100, 140, 140]], dtype=np.float32)
    gt = np.array([[100, 100, 140, 140]], dtype=np.float32)  # exact match
    assert _identify_ghosts(pred, gt, stride=4) == []


def test_identify_ghosts_far_from_gt_is_ghost():
    pred = np.array([[400, 400, 432, 432]], dtype=np.float32)
    gt = np.array([[10, 10, 50, 50]], dtype=np.float32)
    assert _identify_ghosts(pred, gt, stride=4) == [0]


def test_saliency_centroid_finds_peak_region():
    sal = np.zeros((100, 100), dtype=np.float32)
    sal[60:70, 80:90] = 1.0  # bright square at (~65, 85)
    cy, cx = _saliency_centroid(sal, ghost_yx=(65, 85), window=64)
    # centroid should land near the bright region's center
    assert 60 <= cy <= 70
    assert 80 <= cx <= 90


def test_crop_patch_correct_size_center():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[95:105, 95:105] = 255  # bright pixel cluster at center
    patch = _crop_patch(img, cy=100, cx=100, size=32)
    assert patch.shape == (32, 32, 3)
    # the bright region should be visible in patch interior
    assert patch[12:20, 12:20].max() > 200


def test_crop_patch_handles_edge_with_pad():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = 50
    patch = _crop_patch(img, cy=2, cx=2, size=32)  # near top-left corner
    assert patch.shape == (32, 32, 3)


def test_hard_negative_paste_lands_at_known_spot_when_no_gt():
    rng = np.random.default_rng(0)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = np.zeros((0, 4), dtype=np.float32)
    # bright red BGR patch (cv2 imread returns BGR; aug converts internally)
    patch = np.full((16, 16, 3), 0, dtype=np.uint8)
    patch[..., 2] = 255  # BGR red -> after BGR2RGB conversion -> R=255, B=0
    cfg = AugConfig(hard_negative_prob=1.0, hard_negative_count=1, hard_negative_max_tries=32)
    out = _hard_negative_paste(img.copy(), boxes, [patch], cfg, rng)
    # somewhere in `out` there should be a 16x16 red region (R=255 in RGB)
    red_mask = (out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 0)
    assert red_mask.sum() == 16 * 16


def test_hard_negative_paste_does_not_overwrite_gt():
    rng = np.random.default_rng(0)
    H, W = 200, 200
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    # GT covers most of the image; only a corner is free
    boxes = np.array([[0, 0, 180, 180]], dtype=np.float32)
    patch_size = 16
    patch = np.full((patch_size, patch_size, 3), 0, dtype=np.uint8)
    patch[..., 2] = 255  # red in BGR
    cfg = AugConfig(hard_negative_prob=1.0, hard_negative_count=1, hard_negative_max_tries=64)
    out = _hard_negative_paste(img.copy(), boxes, [patch], cfg, rng)
    # The GT region (0-180, 0-180) must not contain the paste
    gt_view = out[:180, :180]
    red_in_gt = ((gt_view[..., 0] == 255) & (gt_view[..., 1] == 0) & (gt_view[..., 2] == 0)).sum()
    assert red_in_gt == 0
    # The paste did happen in the free corner
    red_total = ((out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 0)).sum()
    assert red_total == patch_size * patch_size


def test_hard_negative_paste_skipped_when_image_full():
    rng = np.random.default_rng(0)
    H, W = 50, 50
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    boxes = np.array([[0, 0, H - 1, W - 1]], dtype=np.float32)  # GT fills frame
    patch = np.full((16, 16, 3), 0, dtype=np.uint8)
    patch[..., 2] = 255
    cfg = AugConfig(hard_negative_prob=1.0, hard_negative_count=1, hard_negative_max_tries=32)
    out = _hard_negative_paste(img.copy(), boxes, [patch], cfg, rng)
    # No paste happened — image unchanged
    assert np.array_equal(out, img)


def test_make_augment_with_pool_invokes_paste(tmp_path):
    # Pool with one bright-red 16x16 patch
    p_dir = tmp_path / "patches"
    p_dir.mkdir()
    patch = np.full((16, 16, 3), 0, dtype=np.uint8)
    patch[..., 2] = 255  # BGR red on disk (cv2 saves BGR)
    cv2.imwrite(str(p_dir / "p0.png"), patch)
    pool = load_pool(p_dir)
    assert len(pool) == 1
    cfg = AugConfig(
        enabled=True, brightness=0.0, contrast=0.0, gamma=None,
        hue=0, saturation=0.0, grayscale_prob=0.0, blur_prob=0.0, noise_sigma=0.0,
        hflip_prob=0.0, vflip_prob=0.0, rotate90_prob=0.0,
        scale_jitter=(1.0, 1.0), translate_frac=0.0, mosaic_prob=0.0,
        cutout_prob=0.0, hard_negative_prob=1.0, hard_negative_count=1,
    )
    aug = make_augment(cfg, hn_pool=pool)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = np.zeros((0, 4), dtype=np.float32)
    out, _, _, _ = aug(img, boxes)
    red_mask = (out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 0)
    assert red_mask.sum() == 16 * 16


def test_load_pool_empty_dir(tmp_path):
    assert load_pool(tmp_path) == []
    assert load_pool(tmp_path / "does_not_exist") == []


def test_load_pool_with_subdir_named_patches(tmp_path):
    sub = tmp_path / "patches"
    sub.mkdir()
    img = np.full((16, 16, 3), 50, dtype=np.uint8)
    cv2.imwrite(str(sub / "x.png"), img)
    # passing the parent dir should still find the patches/ subdir
    out = load_pool(tmp_path)
    assert len(out) == 1


@pytest.mark.skipif(
    not Path("/home/bherbruck/github/opndet/scratch/bbox-x-kitchen_10_best.pt").exists()
    or not Path("/mnt/c/Users/bherbruck/Downloads/Egg Dataset OBB.coco (1)/train/_annotations.coco.json").exists(),
    reason="requires user's local checkpoint + dataset; only runs in dev env",
)
def test_mine_end_to_end(tmp_path):
    """End-to-end mining smoke test against the user's local kitchen-sink ckpt.
    Skipped in CI; runs locally when the prerequisite files exist.
    """
    from opndet.mine_negatives import mine
    ckpt = "/home/bherbruck/github/opndet/scratch/bbox-x-kitchen_10_best.pt"
    coco = "/mnt/c/Users/bherbruck/Downloads/Egg Dataset OBB.coco (1)/train/_annotations.coco.json"
    img_dir = "/mnt/c/Users/bherbruck/Downloads/Egg Dataset OBB.coco (1)/train"
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "model_config: bbox-x\n"
        "model: { stride: 4 }\n"
        "seed: 0\n"
        "device: cpu\n"
        "data:\n"
        f"  sources:\n    - coco: {coco}\n      images: {img_dir}\n"
        "  split_ratios: [0.8, 0.15, 0.05]\n"
        "eval_threshold: 0.3\n"
    )
    out_dir = tmp_path / "pool"
    res = mine(
        ckpt=ckpt, config=str(cfg_yaml), out_dir=str(out_dir),
        split="val", max_samples=8, top_k_per_sample=2, patch_size=32,
        clusters=2, device="cpu",
    )
    assert (out_dir / "manifest.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["n_samples_scanned"] > 0
    # if the model has any ghosts, patches dir is non-empty
    if manifest["n_patches"] > 0:
        files = list((out_dir / "patches").glob("*.png"))
        assert len(files) == manifest["n_patches"]
        assert all("patch_path" in p for p in manifest["patches"])
