"""hm_target: ellipse encoding + cls_loss: soft_hm (QFL) loss path."""
import numpy as np
import torch

from opndet.encode import encode_targets, encode_targets_obb
from opndet.loss import OpndetBboxLoss, quality_focal_loss


class _Shim:
    img_h = 128
    img_w = 256
    stride = 4
    out_h = 32
    out_w = 64
    hm_blob_frac = 0.0
    hm_target = "ellipse"
    hm_ellipse_edge_margin = 0.1


def _shim(**kw):
    s = _Shim()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_ellipse_target_interior_box():
    # 40w x 60h box centered well inside the frame.
    cfg = _shim()
    boxes = np.array([[108, 34, 148, 94]], np.float32)
    hm = encode_targets(boxes, cfg)["hm"][0].numpy()
    cy, cx = 64 // 4, 128 // 4  # 16, 32
    assert hm[cy, cx] > 0.99, "peak ~1.0 at the box center"
    # box semi-axes ≈ 5 cells (x) × 7.5 cells (y) → past those it's 0
    assert hm[cy, cx + 7] == 0.0, "past the x semi-axis → 0"
    assert hm[cy + 9, cx] == 0.0, "past the y semi-axis → 0"
    # box corner region is outside the inscribed ellipse → 0
    assert hm[cy + 5, cx + 5] == 0.0, "box corner is outside the inscribed ellipse → 0"
    # taller box → taller heatmap
    yspread = int((hm[:, cx] > 0).sum())
    xspread = int((hm[cy, :] > 0).sum())
    assert yspread > xspread


def test_ellipse_target_edge_box_falls_back_to_gaussian():
    cfg = _shim()
    # box touching the left edge (x1 = 0) → within margin → plain Gaussian
    edge = encode_targets(np.array([[0, 34, 40, 94]], np.float32), cfg)["hm"][0].numpy()
    interior = encode_targets(np.array([[108, 34, 148, 94]], np.float32), cfg)["hm"][0].numpy()
    assert (edge > 0).sum() < (interior > 0).sum(), "edge box → tight Gaussian, fewer lit cells than the full ellipse"


def test_ellipse_target_gaussian_mode_unchanged():
    # hm_target: gaussian (default) keeps the legacy tight bump.
    cfg = _shim(hm_target="gaussian")
    hm = encode_targets(np.array([[108, 34, 148, 94]], np.float32), cfg)["hm"][0].numpy()
    cy, cx = 16, 32
    assert hm[cy, cx] > 0.99
    # legacy CornerNet σ is small → much tighter than the box's ~5x7.5 ellipse
    assert (hm > 0).sum() < 60


def test_ellipse_target_obb_rotated():
    # a 60x20 box rotated 90° → the heatmap should be taller than wide.
    cfg = _shim()
    obbs = np.array([[128.0, 64.0, 60.0, 20.0, np.pi / 2]], np.float32)  # (cx,cy,w,h,θ)
    hm = encode_targets_obb(obbs, cfg)["hm"][0].numpy()
    cy, cx = 16, 32
    assert hm[cy, cx] > 0.99
    yspread = int((hm[:, cx] > 0).sum())
    xspread = int((hm[cy, :] > 0).sum())
    assert yspread > xspread, "w along θ=90° → the box's long side is vertical → taller heatmap"


def test_quality_focal_loss_basic():
    # perfect prediction → ~0 loss; wrong prediction → > 0; scale ~ matches focal (÷ n_pos).
    tgt = torch.zeros(2, 1, 8, 8)
    tgt[:, 0, 4, 4] = 1.0
    tgt[:, 0, 3, 4] = tgt[:, 0, 5, 4] = tgt[:, 0, 4, 3] = tgt[:, 0, 4, 5] = 0.6
    logit_good = torch.logit(tgt.clamp(1e-4, 1 - 1e-4))
    logit_bad = torch.zeros_like(tgt)  # σ = 0.5 everywhere
    l_good = quality_focal_loss(logit_good, tgt)
    l_bad = quality_focal_loss(logit_bad, tgt)
    assert float(l_good) < 1e-3
    assert float(l_bad) > float(l_good) + 1.0


def test_soft_hm_cls_loss_runs():
    cfg = _shim()
    tgt = encode_targets(np.array([[108, 34, 148, 94]], np.float32), cfg)
    tgt = {k: (v.unsqueeze(0).repeat(2, *([1] * v.dim())) if v.dim() == 3 else v) for k, v in tgt.items()}
    loss = OpndetBboxLoss(cls_loss="soft_hm", wh_loss="ciou", qfl_beta=2.0, img_h=128, img_w=256, stride=4)
    raw = torch.zeros(2, 5, 32, 64, requires_grad=True)
    out = loss(raw, tgt)
    assert float(out["l_hm"]) > 0
    out["loss"].backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_peak_sharpen_loss():
    from opndet.loss import peak_sharpen_loss
    B, H, W = 2, 16, 16
    pos = torch.zeros(B, 1, H, W)
    pos[:, 0, 8, 8] = 1.0
    # adjacent near-tie: 0.95 peak, 0.94 neighbor → penalized (0.94 > 0.95-0.15)
    hm = torch.full((B, 1, H, W), 0.05)
    hm[:, 0, 8, 8] = 0.95
    hm[:, 0, 8, 9] = 0.94
    assert float(peak_sharpen_loss(hm, pos, k=5, margin=0.15)) > 0.1
    # clean single peak: neighbors well below the margin band → ~0
    hm2 = torch.full((B, 1, H, W), 0.05)
    hm2[:, 0, 7:10, 7:10] = 0.6
    hm2[:, 0, 8, 8] = 0.95
    assert float(peak_sharpen_loss(hm2, pos, k=5, margin=0.15)) < 1e-4


def test_peak_sharpen_wired_into_loss():
    B, H, W = 1, 16, 16
    pos = torch.zeros(B, 1, H, W)
    pos[:, 0, 8, 8] = 1.0
    tgt = {"pos": pos, "hm": torch.zeros(B, 1, H, W),
           "cxy": torch.zeros(B, 2, H, W), "wh": torch.full((B, 2, H, W), 0.1)}
    tgt["hm"][:, 0, 8, 8] = 1.0
    loss = OpndetBboxLoss(cls_loss="focal", wh_loss="ciou", peak_sharpen_weight=0.5,
                          peak_sharpen_margin=0.15, img_h=64, img_w=64, stride=4)
    raw = torch.zeros(1, 5, 16, 16, requires_grad=True)
    out = loss(raw, tgt)
    assert "l_peaksharp" in out
    out["loss"].backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
