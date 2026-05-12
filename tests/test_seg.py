"""bbox-*-seg: full-res dense-dome segmentation head — build, export parity, encode, loss."""
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from opndet.presets import resolve
from opndet.yaml_build import build_model_from_yaml


def _export(m, x, path):
    torch.onnx.export(m, x, path, input_names=["input"], output_names=["output"],
                      opset_version=13, do_constant_folding=True, dynamo=False)


# ---- 1. preset builds, forwards [1,1,H,W], exports opset-13 edge-clean, ORT==PT ----
@pytest.mark.parametrize("preset", ["bbox-f-seg", "bbox-n-seg", "bbox-s-seg"])
def test_seg_preset_builds_and_exports_opset13(preset):
    m = build_model_from_yaml(resolve(preset)).eval()
    c, h, w = m.input_shape
    assert "dome" in m.aliases, "seg head must expose the 'dome' alias (used for head detection)"
    x = torch.randn(1, c, h, w)
    with torch.no_grad():
        out = m(x)
    ot = out["output"] if isinstance(out, dict) else out
    assert ot.shape == (1, 1, h, w), f"{preset}: expected full-res [1,1,{h},{w}], got {tuple(ot.shape)}"
    assert float(ot.min()) >= 0.0 and float(ot.max()) <= 1.0
    # raw (pre-sigmoid 1-ch logit) used by SegDomeLoss
    assert m.forward_with_alias(x, "raw").shape == (1, 1, h, w)
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / f"{preset}.onnx")
        _export(m, x, p)
        from opndet.export import allowed_ops_for_tier
        om = onnx.load(p)
        allowed = allowed_ops_for_tier("edge")
        bad = sorted({n.op_type for n in om.graph.node} - allowed)
        assert not bad, f"{preset} (edge tier) exported forbidden ops: {bad}"
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        o2 = sess.run(None, {"input": x.numpy()})[0]
        assert float(np.abs(o2 - ot.numpy()).max()) < 1e-4


def test_seg_runs_at_non_default_size():
    # fully conv + nearest-resize → any size divisible by 32 works
    m = build_model_from_yaml(resolve("bbox-n-seg"), img_h=256, img_w=384).eval()
    out = m(torch.randn(1, 3, 256, 384))
    ot = out["output"] if isinstance(out, dict) else out
    assert ot.shape == (1, 1, 256, 384)


# ---- 2. encode_targets_seg: dense dome from OBBs / masks ----
class _Shim:
    img_h = 128
    img_w = 192
    stride = 4          # ignored by seg encode (it renders at seg_stride)
    seg_stride = 1
    hm_ellipse_edge_margin = 0.0


def test_encode_seg_dome_from_obb():
    from opndet.encode import encode_targets_seg
    cfg = _Shim()
    # one 40x24 box at the center, axis-aligned
    obbs = np.array([[96.0, 64.0, 40.0, 24.0, 0.0]], np.float32)
    t = encode_targets_seg(cfg, obbs=obbs)
    dome = t["dome"][0].numpy()
    assert dome.shape == (128, 192)
    assert dome[64, 96] > 0.99, "1.0 at the box center"
    assert dome[64, 96 + 22] == 0.0, "past the x semi-axis (20px) → 0"
    assert dome[64, 96 + 10] > 0.0, "inside → positive"
    # taller dimension is shorter here → x-spread > y-spread
    assert int((dome[64, :] > 0).sum()) > int((dome[:, 96] > 0).sum())


def test_encode_seg_dome_from_obb_rotated():
    from opndet.encode import encode_targets_seg
    obbs = np.array([[96.0, 64.0, 40.0, 16.0, np.pi / 2]], np.float32)  # long side now vertical
    dome = encode_targets_seg(_Shim(), obbs=obbs)["dome"][0].numpy()
    assert dome[64, 96] > 0.99
    assert int((dome[:, 96] > 0).sum()) > int((dome[64, :] > 0).sum())


def test_encode_seg_dome_from_masks_distance_transform():
    import cv2
    from opndet.encode import encode_targets_seg
    cfg = _Shim()
    # a 30px-radius disk → distance-transform dome: 1 at the center, ~linear to 0 at the rim
    mask = np.zeros((128, 192), np.uint8)
    cv2.circle(mask, (96, 64), 30, 1, -1)
    dome = encode_targets_seg(cfg, masks=[mask])["dome"][0].numpy()
    assert dome[64, 96] > 0.99
    assert 0.0 < dome[64, 96 + 20] < 0.6
    assert dome[64, 96 + 31] == 0.0  # outside the disk
    # two disks with a real gap between them, passed as two instance masks →
    # max-aggregated, so the gap column stays 0 (no bleed across it). Disk A
    # reaches x=83, disk B reaches x=97; x=90 is in the gap.
    da = np.zeros((128, 192), np.uint8); cv2.circle(da, (65, 64), 18, 1, -1)
    db = np.zeros((128, 192), np.uint8); cv2.circle(db, (115, 64), 18, 1, -1)
    d2 = encode_targets_seg(cfg, masks=[da, db])["dome"][0].numpy()
    assert d2[64, 65] > 0.99 and d2[64, 115] > 0.99
    assert d2[64, 90] == 0.0, "the gap column between the two disks is 0"


def test_encode_seg_empty():
    from opndet.encode import encode_targets_seg
    t = encode_targets_seg(_Shim(), obbs=np.zeros((0, 5), np.float32))
    assert float(t["dome"].abs().sum()) == 0.0
    assert t["dome"].shape == (1, 128, 192)


def test_encode_seg_stride_2():
    from opndet.encode import encode_targets_seg
    class S(_Shim):
        seg_stride = 2
    obbs = np.array([[96.0, 64.0, 40.0, 24.0, 0.0]], np.float32)
    dome = encode_targets_seg(S(), obbs=obbs)["dome"][0].numpy()
    assert dome.shape == (64, 96)
    assert dome[32, 48] > 0.99


# ---- 3. SegDomeLoss: QFL toward the dome + soft Dice on foreground ----
def test_seg_dome_loss_basic():
    from opndet.loss import SegDomeLoss
    B, H, W = 2, 32, 48
    dome = torch.zeros(B, 1, H, W)
    dome[:, 0, 12:20, 20:32] = torch.linspace(0.2, 1.0, 12).unsqueeze(0).repeat(8, 1)
    loss = SegDomeLoss(qfl_beta=2.0)
    good = torch.logit(dome.clamp(1e-4, 1 - 1e-4))
    bad = torch.zeros_like(dome)
    lo = loss(good, {"dome": dome})
    hi = loss(bad, {"dome": dome})
    assert float(lo["loss"]) < float(hi["loss"])
    assert "l_qfl" in lo and "l_dice" in lo
    bad.requires_grad_(True)
    out = loss(bad, {"dome": dome})
    out["loss"].backward()
    assert bad.grad is not None and torch.isfinite(bad.grad).all()
