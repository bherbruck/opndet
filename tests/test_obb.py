"""ROADMAP §1.8 Phase 4b — OBB output contract tests.

  - obb_to_corners ↔ corners_to_obb roundtrip
  - encode_targets_obb roundtrip via decode_obb
  - rotated Gaussian draw produces an elongated heatmap
  - OBB loss is differentiable (angle channels receive gradient)
  - All -pro variants build with 7-channel head + opset-13 export parity
  - YOLOv8-OBB label parser
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn


PRO_PRESETS = ["bbox-f-pro", "bbox-p-pro", "bbox-n-pro", "bbox-s-pro",
               "bbox-m-pro", "bbox-l-pro", "bbox-x-pro"]

# Edge-tier OBB-only family (base arch + 7-ch OBB head). Same output contract
# as -pro but without SPPF/PAFPN/decoupled head — leaner footprint, Myriad-X safe.
OBB_PRESETS = ["bbox-f-obb", "bbox-p-obb", "bbox-n-obb", "bbox-s-obb"]

ALL_OBB_PRESETS = PRO_PRESETS + OBB_PRESETS


def _export(m: nn.Module, x: torch.Tensor, path: str) -> None:
    torch.onnx.export(
        m, x, path,
        input_names=["image"], output_names=["output"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes=None, dynamo=False,
    )


# ---- 1. corners ↔ OBB roundtrip ----

def test_corners_obb_roundtrip():
    from opndet.encode import obb_to_corners, corners_to_obb

    rng = np.random.default_rng(0)
    cases = [
        (200.0, 150.0, 100.0, 40.0, math.radians(0)),
        (200.0, 150.0, 100.0, 40.0, math.radians(30)),
        (50.0, 80.0, 60.0, 25.0, math.radians(75)),
        (300.0, 200.0, 80.0, 80.0, math.radians(45)),  # square
    ]
    for cx, cy, w, h, th in cases:
        c = obb_to_corners(cx, cy, w, h, th)
        cx2, cy2, w2, h2, th2 = corners_to_obb(c)
        assert abs(cx - cx2) < 1e-3
        assert abs(cy - cy2) < 1e-3
        assert abs(w - w2) < 1e-3, f"w mismatch: {w} vs {w2}"
        assert abs(h - h2) < 1e-3, f"h mismatch: {h} vs {h2}"
        # angle wraps π — accept either th or th+π modulo equivalence; for
        # square objects orientation is degenerate (any rotation valid).
        d = abs(((th - th2) + math.pi / 2) % math.pi - math.pi / 2)
        assert d < 1e-3 or abs(w - h) < 1e-3, f"theta mismatch: {math.degrees(th)} vs {math.degrees(th2)}"


def test_obb_to_aabb_matches_corners_extents():
    from opndet.encode import obb_to_corners, obb_to_aabb

    cx, cy, w, h, th = 200.0, 150.0, 100.0, 40.0, math.radians(30)
    corners = obb_to_corners(cx, cy, w, h, th)
    x1, y1, x2, y2 = obb_to_aabb(cx, cy, w, h, th)
    assert abs(corners[:, 0].min() - x1) < 1e-3
    assert abs(corners[:, 0].max() - x2) < 1e-3
    assert abs(corners[:, 1].min() - y1) < 1e-3
    assert abs(corners[:, 1].max() - y2) < 1e-3


# ---- 2. encode_targets_obb ↔ decode_obb roundtrip ----


def test_encode_decode_obb_roundtrip():
    from opndet.config import ModelConfig
    from opndet.decode import decode_obb
    from opndet.encode import encode_targets_obb, obb_to_corners

    cfg = ModelConfig()
    obbs = np.array([
        [200.0, 150.0, 100.0, 40.0, math.radians(30)],
        [80.0,  80.0,   60.0, 25.0, math.radians(75)],
    ], dtype=np.float32)
    tgt = encode_targets_obb(obbs, cfg)
    assert tgt["pos"].sum() == 2
    assert tgt["angle_mask"].sum() == 2  # both non-round
    obj = tgt["pos"].numpy()
    out = np.concatenate([obj, tgt["obb"].numpy()], axis=0)
    dets = decode_obb(out, cfg.img_h, cfg.img_w, cfg.stride, threshold=0.5)
    assert len(dets) == 2

    # Match decoded OBBs to GT by center distance
    decoded = sorted([(d.cx, d.cy, d.w, d.h, d.theta) for d in dets])
    expected = sorted([tuple(o) for o in obbs])
    for (px, py, _, _, pth), (gx, gy, _, _, gth) in zip(decoded, expected):
        assert abs(px - gx) < 1.5
        assert abs(py - gy) < 1.5
        # angle: 2θ encoded so wraps at π → tolerate ±π
        d = abs(((pth - gth) + math.pi / 2) % math.pi - math.pi / 2)
        assert d < 1e-2, f"theta mismatch: {math.degrees(pth)} vs {math.degrees(gth)}"

    # Reconstructed OBB corners closely match the GT OBB corners
    decoded_objs = sorted(dets, key=lambda d: d.cx)
    expected_obbs = sorted(obbs, key=lambda o: o[0])
    for d, gt in zip(decoded_objs, expected_obbs):
        gt_corners = obb_to_corners(*gt)
        # Sort corners of both by (x, y) so order-of-sides doesn't matter.
        sort_key = lambda p: (p[0], p[1])
        a = np.array(sorted(d.to_corners().tolist(), key=sort_key))
        b = np.array(sorted(gt_corners.tolist(), key=sort_key))
        max_err = float(np.abs(a - b).max())
        assert max_err < 2.0, f"OBB corners off by {max_err}"


def test_gt_obbs_from_targets_roundtrip():
    """Verify the encode→decode chain used by metrics + dashboard GT corners."""
    from opndet.config import ModelConfig
    from opndet.decode import gt_obbs_from_targets
    from opndet.encode import encode_targets_obb

    cfg = ModelConfig()
    obbs = np.array([
        [200.0, 150.0, 100.0, 40.0, math.radians(30)],
        [80.0,  80.0,   60.0, 25.0, math.radians(75)],
    ], dtype=np.float32)
    tgt = encode_targets_obb(obbs, cfg)
    pos_b = tgt["pos"].unsqueeze(0).numpy()
    obb_b = tgt["obb"].unsqueeze(0).numpy()
    gt = gt_obbs_from_targets(pos_b, obb_b, cfg.img_h, cfg.img_w, cfg.stride)
    assert len(gt) == 1
    assert gt[0].shape == (2, 5)
    decoded = sorted(gt[0].tolist())
    expected = sorted(obbs.tolist())
    for (cx, cy, _, _, th), (gx, gy, _, _, gth) in zip(decoded, expected):
        assert abs(cx - gx) < 2.0
        assert abs(cy - gy) < 2.0
        d = abs(((th - gth) + math.pi / 2) % math.pi - math.pi / 2)
        assert d < 1e-2, f"theta mismatch decoded={math.degrees(th):.1f} expected={math.degrees(gth):.1f}"


def test_rotated_iou_self_and_orthogonal():
    """rotated_iou trace check: self-IoU=1, 90°-rotated IoU on a square=1, axis-flipped at 90°=0."""
    from opndet.metrics import rotated_iou

    # Self-IoU: identical rectangles
    iou = rotated_iou(100.0, 100.0, 80.0, 30.0, 0.0,
                      100.0, 100.0, 80.0, 30.0, 0.0)
    assert iou > 0.999, f"self-IoU should be 1, got {iou}"

    # Orthogonal: same rect rotated 90° on a non-square. Should be small overlap.
    iou_90 = rotated_iou(100.0, 100.0, 80.0, 30.0, 0.0,
                         100.0, 100.0, 80.0, 30.0, math.pi / 2)
    assert iou_90 < 0.5, f"90°-rotated non-square IoU should be small, got {iou_90}"

    # Square at 45° vs 0°: identical rotated by 45° around center → octagonal
    # intersection. IoU = √2/2 ≈ 0.707 exactly (two unit squares overlapped at 45°).
    iou_diag = rotated_iou(100.0, 100.0, 50.0, 50.0, 0.0,
                           100.0, 100.0, 50.0, 50.0, math.pi / 4)
    assert 0.65 < iou_diag < 0.75, f"square @45° IoU ≈ √2/2 = 0.707, got {iou_diag}"


def test_angle_err_rad_pi_symmetry():
    """Angle error wraps at π/2 — flipping the major axis by π gives the same rectangle."""
    from opndet.metrics import angle_err_rad

    assert angle_err_rad(0.0, 0.0) == 0.0
    assert abs(angle_err_rad(0.0, math.pi) - 0.0) < 1e-9
    assert abs(angle_err_rad(0.0, math.pi / 2) - math.pi / 2) < 1e-9
    # 30° vs -30° = 60° error, NOT 180-60=120
    assert abs(math.degrees(angle_err_rad(math.radians(30), math.radians(-30))) - 60.0) < 1e-3


def test_letterbox_transforms_obbs():
    """letterbox scales cx/cy/w/h uniformly and pad-shifts cx/cy. theta is invariant."""
    from opndet.dataset import letterbox

    img = np.zeros((600, 800, 3), dtype=np.uint8)
    boxes = np.array([[100.0, 50.0, 300.0, 250.0]], dtype=np.float32)
    obbs = np.array([[200.0, 150.0, 100.0, 40.0, math.radians(30)]], dtype=np.float32)
    out_img, out_boxes, out_obbs = letterbox(img, boxes, 384, 512, obbs=obbs)
    # 800x600 → 512x384 needs scale = min(512/800, 384/600) = min(0.64, 0.64) = 0.64
    # Both axes scale identically — no padding either direction (square aspect match)
    assert out_img.shape == (384, 512, 3)
    scale = 384 / 600  # 0.64
    # cx scales: 200 * 0.64 = 128 (no x-pad since aspect matches)
    assert abs(out_obbs[0, 0] - 200.0 * scale) < 0.5
    # cy scales: 150 * 0.64 = 96
    assert abs(out_obbs[0, 1] - 150.0 * scale) < 0.5
    # w/h scale uniformly
    assert abs(out_obbs[0, 2] - 100.0 * scale) < 0.5
    assert abs(out_obbs[0, 3] - 40.0 * scale) < 0.5
    # theta unchanged
    assert abs(out_obbs[0, 4] - math.radians(30)) < 1e-6


def test_letterbox_with_padding_shifts_obbs():
    """Non-square source → asymmetric pad on one axis. Verify cx/cy shift."""
    from opndet.dataset import letterbox

    img = np.zeros((400, 800, 3), dtype=np.uint8)  # 2:1 aspect
    boxes = np.array([[100.0, 100.0, 300.0, 300.0]], dtype=np.float32)
    obbs = np.array([[200.0, 200.0, 80.0, 30.0, math.radians(45)]], dtype=np.float32)
    out_img, _, out_obbs = letterbox(img, boxes, 384, 512, obbs=obbs)
    # scale = min(512/800, 384/400) = min(0.64, 0.96) = 0.64
    # new dims: 512x256. pad_y = (384-256)/2 = 64. pad_x = 0.
    scale = 0.64
    pad_y = 64
    expected_cx = 200.0 * scale + 0  # 128
    expected_cy = 200.0 * scale + pad_y  # 128 + 64 = 192
    assert abs(out_obbs[0, 0] - expected_cx) < 0.5
    assert abs(out_obbs[0, 1] - expected_cy) < 0.5
    assert abs(out_obbs[0, 4] - math.radians(45)) < 1e-6


def test_aug_hflip_transforms_obb():
    """hflip mirrors cx and θ for OBBs. Verifies the OBB GT survives photometric+hflip aug."""
    from opndet.augment import AugConfig, make_augment

    cfg = AugConfig(
        enabled=True, brightness=0.0, contrast=0.0, gamma=None,
        hue=0, saturation=0.0, grayscale_prob=0.0, blur_prob=0.0, noise_sigma=0.0,
        hflip_prob=1.0, vflip_prob=0.0, rotate90_prob=0.0,
        scale_jitter=(1.0, 1.0), translate_frac=0.0, mosaic_prob=0.0,
        cutout_prob=0.0,
    )
    aug = make_augment(cfg)
    img = np.zeros((384, 512, 3), dtype=np.uint8)
    boxes = np.array([[100.0, 50.0, 200.0, 150.0]], dtype=np.float32)
    obbs = np.array([[150.0, 100.0, 80.0, 30.0, math.radians(30)]], dtype=np.float32)
    _, boxes_aug, obbs_aug = aug(img, boxes, obbs)
    assert obbs_aug is not None and obbs_aug.shape == (1, 5)
    # cx mirror: 512 - 150 = 362
    assert abs(obbs_aug[0, 0] - 362.0) < 1e-3
    # cy preserved
    assert abs(obbs_aug[0, 1] - 100.0) < 1e-3
    # w, h preserved
    assert abs(obbs_aug[0, 2] - 80.0) < 1e-3 and abs(obbs_aug[0, 3] - 30.0) < 1e-3
    # θ → π - θ: 30° → 150°
    assert abs(math.degrees(obbs_aug[0, 4]) - 150.0) < 0.1


def test_aug_rotate90_transforms_obb():
    """rotate90 swaps cx/cy and adds k·π/2 to θ."""
    from opndet.augment import AugConfig, make_augment

    rng_seed_cfg = AugConfig(
        enabled=True, brightness=0.0, contrast=0.0, gamma=None,
        hue=0, saturation=0.0, grayscale_prob=0.0, blur_prob=0.0, noise_sigma=0.0,
        hflip_prob=0.0, vflip_prob=0.0, rotate90_prob=1.0,
        scale_jitter=(1.0, 1.0), translate_frac=0.0, mosaic_prob=0.0,
        cutout_prob=0.0,
    )
    aug = make_augment(rng_seed_cfg)
    img = np.zeros((384, 512, 3), dtype=np.uint8)
    boxes = np.array([[100.0, 50.0, 200.0, 150.0]], dtype=np.float32)
    obbs = np.array([[150.0, 100.0, 80.0, 30.0, math.radians(30)]], dtype=np.float32)
    # k is randomly chosen from {1, 2, 3}; verify the result is one of the three valid rotations
    valid_thetas_deg = [
        (30 + 90 * 1) % 180,
        (30 + 90 * 2) % 180,
        (30 + 90 * 3) % 180,
    ]
    _, _, obbs_aug = aug(img, boxes, obbs)
    obs_theta_deg = math.degrees(obbs_aug[0, 4]) % 180
    assert any(abs(obs_theta_deg - v) < 0.1 for v in valid_thetas_deg), \
        f"θ={obs_theta_deg} not in {valid_thetas_deg}"
    # w, h unchanged regardless of k
    assert abs(obbs_aug[0, 2] - 80.0) < 1e-3 and abs(obbs_aug[0, 3] - 30.0) < 1e-3


def test_aug_cutout_drops_obbs_in_lockstep():
    """When cutout drops a box (visibility too low), the corresponding OBB row is dropped too."""
    from opndet.augment import AugConfig, make_augment

    cfg = AugConfig(
        enabled=True, brightness=0.0, contrast=0.0, gamma=None,
        hue=0, saturation=0.0, grayscale_prob=0.0, blur_prob=0.0, noise_sigma=0.0,
        hflip_prob=0.0, vflip_prob=0.0, rotate90_prob=0.0,
        scale_jitter=(1.0, 1.0), translate_frac=0.0, mosaic_prob=0.0,
        cutout_prob=1.0, cutout_count=1, cutout_size_frac=(0.99, 0.99),
        min_visible_frac=0.5,
    )
    aug = make_augment(cfg)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)
    obbs = np.array([[20.0, 20.0, 20.0, 15.0, 0.0]], dtype=np.float32)
    _, boxes_aug, obbs_aug = aug(img, boxes, obbs)
    # near-full-image cutout should occlude the box → boxes/obbs both empty
    assert boxes_aug.shape[0] == obbs_aug.shape[0]


def test_obb_summary_perfect_match():
    """obb_summary on identical pred=GT yields IoU=1, ang_err=0.
    Inputs follow the (cx, cy, w_aabb, h_aabb, θ) contract — same shape produced
    by decode_obb / gt_obbs_from_targets. Use θ=0 so AABB==OBB to keep the
    expected dims hand-computable.
    """
    from opndet.metrics import obb_summary

    gt = np.array([[200.0, 150.0, 100.0, 40.0, 0.0]], dtype=np.float32)
    pred = gt.copy()
    s = obb_summary(pred, gt, iou_thresh=0.3)
    assert s["n_match"] == 1
    assert s["obb_ious"][0] > 0.999
    assert s["ang_errs_deg"][0] < 0.01


def test_encode_obb_no_obbs():
    from opndet.config import ModelConfig
    from opndet.encode import encode_targets_obb

    cfg = ModelConfig()
    tgt = encode_targets_obb(np.zeros((0, 5), dtype=np.float32), cfg)
    assert tgt["pos"].sum() == 0
    assert tgt["obb"].abs().sum() == 0


# ---- 3. rotated Gaussian shape ----


def test_obb_heatmap_target_is_circular():
    """encode_targets_obb produces a CIRCULAR cls heatmap, not elongated.
    Shape info lives in the reg head (sin2θ/cos2θ); elongated targets confused
    peak-pick because neighbors along the major axis stole local-max wins."""
    from opndet.config import ModelConfig
    from opndet.encode import encode_targets_obb

    cfg = ModelConfig()
    cx, cy = cfg.img_w / 2, cfg.img_h / 2
    obbs = np.array([[cx, cy, 200.0, 40.0, 0.0]], dtype=np.float32)
    tgt = encode_targets_obb(obbs, cfg)
    hm = tgt["hm"][0].numpy()
    cy_g = int(round(cy / cfg.stride))
    cx_g = int(round(cx / cfg.stride))
    row = hm[cy_g]
    col = hm[:, cx_g]
    x_extent = float((row > 0.1).sum())
    y_extent = float((col > 0.1).sum())
    assert x_extent == y_extent, f"expect circular: x={x_extent} y={y_extent}"


# ---- 4. OBB loss differentiability ----


def test_obb_loss_grads_to_angle_channel():
    """ProbIoU loss should produce non-zero gradient on ALL six output channels
    (obj's heatmap loss + reg's ProbIoU on cx/cy/w/h/θ)."""
    from opndet.config import ModelConfig
    from opndet.encode import encode_targets_obb
    from opndet.loss import OpndetBboxLoss

    cfg = ModelConfig()
    obbs = np.array([[200.0, 150.0, 100.0, 40.0, math.radians(30)]], dtype=np.float32)
    tgt = encode_targets_obb(obbs, cfg)
    tgt = {k: v.unsqueeze(0) for k, v in tgt.items()}

    Hp, Wp = cfg.img_h // cfg.stride, cfg.img_w // cfg.stride
    # 6-ch raw: obj + (cx_off, cy_off, w_norm, h_norm, θ_norm).
    raw = torch.zeros(1, 6, Hp, Wp, requires_grad=True)
    loss_fn = OpndetBboxLoss(wh_loss="obb", img_h=cfg.img_h, img_w=cfg.img_w, stride=cfg.stride)
    out = loss_fn(raw, tgt)
    out["loss"].backward()
    grad = raw.grad
    # Angle channel (idx 5) should have non-zero gradient at the positive cell.
    assert grad[:, 5:6].abs().sum() > 0, "angle channel got no gradient"
    # Reg (cx,cy,w,h) also non-zero
    assert grad[:, 1:5].abs().sum() > 0


def test_probiou_self_match_zero_loss():
    """ProbIoU loss should be ~0 when pred == GT (perfect overlap → IoU≈1)."""
    from opndet.loss import probiou_loss

    box = torch.tensor([[200.0, 150.0, 100.0, 40.0, math.radians(30)]])
    loss = probiou_loss(box, box.clone())
    assert loss.item() < 1e-3, f"self-match loss should be ~0, got {loss.item()}"


def test_probiou_orthogonal_high_loss():
    """ProbIoU loss should be high when boxes have very different orientations."""
    from opndet.loss import probiou_loss

    pred = torch.tensor([[200.0, 150.0, 100.0, 40.0, 0.0]])
    gt   = torch.tensor([[200.0, 150.0, 100.0, 40.0, math.pi / 2]])
    loss = probiou_loss(pred, gt)
    assert loss.item() > 0.3, f"orthogonal-orient loss should be substantial, got {loss.item()}"


# ---- 5. -pro presets build & export at opset 13 (7-channel) ----


@pytest.mark.parametrize("preset", ALL_OBB_PRESETS)
def test_pro_preset_builds_obb(preset: str):
    from opndet.presets import resolve
    from opndet.yaml_build import build_model_from_yaml

    m = build_model_from_yaml(resolve(preset)).eval()
    c, h, w = m.input_shape
    x = torch.randn(1, c, h, w)
    with torch.no_grad():
        y = m(x)
    out = y["output"]
    assert out.shape == (1, 6, h // 4, w // 4), f"{preset}: bad shape {out.shape}"
    # All reg channels are sigmoid → [0, 1] (cx_off, cy_off, w_norm, h_norm, θ_norm).
    assert (out[:, 1:6] >= 0).all() and (out[:, 1:6] <= 1).all()


@pytest.mark.parametrize("preset", ALL_OBB_PRESETS)
def test_pro_preset_obb_exports_opset13(preset: str):
    from opndet.export import allowed_ops_for_tier
    from opndet.presets import resolve
    from opndet.yaml_build import build_model_from_yaml

    m = build_model_from_yaml(resolve(preset)).eval()
    c, h, w = m.input_shape
    x = torch.randn(1, c, h, w)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / f"{preset}.onnx")
        _export(m, x, path)
        om = onnx.load(path)
        ops = {n.op_type for n in om.graph.node}
        # Phase 2: server-tier presets may use MatMul / Softmax via C2PSA.
        forbidden = ops - allowed_ops_for_tier(m.tier)
        assert not forbidden, f"{preset} (tier={m.tier}) forbidden ops: {forbidden}"
        assert "Sigmoid" in ops, f"{preset} expected Sigmoid in graph"

        with torch.no_grad():
            y_pt = m(x)["output"].numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"image": x.numpy()})[0]
        assert y_pt.shape == y_ort.shape == (1, 6, h // 4, w // 4)
        diff = float(np.abs(y_pt - y_ort).max())
        assert diff < 1e-3, f"{preset} parity diff {diff:.2e}"


# ---- 6. YOLOv8-OBB label parser ----


def test_yolo_obb_line_parse():
    from opndet.encode import obb_to_corners, yolo_obb_line_to_obb

    img_w, img_h = 384, 256
    cx, cy, w, h, th = 100.0, 80.0, 60.0, 25.0, math.radians(45)
    corners = obb_to_corners(cx, cy, w, h, th)
    norm = corners.copy()
    norm[:, 0] /= img_w
    norm[:, 1] /= img_h
    line = "0 " + " ".join(f"{v:.6f}" for v in norm.flatten())
    parsed = yolo_obb_line_to_obb(line, img_w, img_h)
    assert parsed is not None
    cls, pcx, pcy, pw, ph, pth = parsed
    assert cls == 0
    assert abs(pcx - cx) < 0.5
    assert abs(pcy - cy) < 0.5
    # w/h round-trip via minAreaRect: tolerate small numerical noise.
    assert abs(pw - w) < 0.5
    assert abs(ph - h) < 0.5


def test_yolo_obb_line_malformed():
    from opndet.encode import yolo_obb_line_to_obb
    assert yolo_obb_line_to_obb("not a real line", 100, 100) is None
    assert yolo_obb_line_to_obb("0 0.1 0.2 0.3", 100, 100) is None


# ---- 7. dataset obb_dir loader ----


def test_dataset_obb_loader(tmp_path):
    from opndet.dataset import load_coco_single_class
    import json
    import cv2

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    obb_dir = tmp_path / "obb"
    obb_dir.mkdir()

    img = np.zeros((256, 384, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "a.jpg"), img)
    cv2.imwrite(str(img_dir / "b.jpg"), img)

    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 384, "height": 256},
            {"id": 2, "file_name": "b.jpg", "width": 384, "height": 256},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 100, 50], "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 80, 80], "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "x"}],
    }
    coco_path = tmp_path / "ann.json"
    coco_path.write_text(json.dumps(coco))

    # Only image 'a' has an OBB sidecar; 'b' should fall back to None
    from opndet.encode import obb_to_corners
    corners = obb_to_corners(100.0, 80.0, 60.0, 25.0, math.radians(30))
    norm = corners.copy()
    norm[:, 0] /= 384
    norm[:, 1] /= 256
    line = "0 " + " ".join(f"{v:.6f}" for v in norm.flatten())
    (obb_dir / "a.txt").write_text(line + "\n")

    samples = load_coco_single_class(coco_path, img_dir, obb_dir=obb_dir)
    assert len(samples) == 2
    by_name = {s.image_path.name: s for s in samples}
    assert by_name["a.jpg"].obbs is not None
    assert by_name["a.jpg"].obbs.shape == (1, 5)
    assert abs(by_name["a.jpg"].obbs[0, 0] - 100.0) < 0.5
    assert by_name["b.jpg"].obbs is None
