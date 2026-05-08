"""Phase 1 (ROADMAP §1.8) tests:
  - SPPF parity (PT vs ORT, opset 13)
  - ltrb encode/decode roundtrip
  - Each -pro preset builds + forward + exports cleanly
"""

from __future__ import annotations

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


def _export(m: nn.Module, x: torch.Tensor, path: str) -> None:
    torch.onnx.export(
        m, x, path,
        input_names=["image"], output_names=["output"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes=None, dynamo=False,
    )


# ---- 1. SPPF parity --------------------------------------------------------


def test_sppf_parity_opset13():
    """Build a tiny SPPF-only model, export to ONNX opset 13, verify ORT==PT."""
    from opndet.primitives import SPPF

    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Conv2d(3, 16, 3, 2, 1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU6(inplace=True),
        SPPF(in_ch=16, out_ch=16, k=5),
    ).eval()

    x = torch.randn(1, 3, 96, 128)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "sppf.onnx")
        _export(m, x, path)
        # opset-13 op-set check.
        from opndet.export import ALLOWED_OPS
        om = onnx.load(path)
        ops = {n.op_type for n in om.graph.node}
        forbidden = ops - ALLOWED_OPS
        assert not forbidden, f"forbidden ops in SPPF: {forbidden}"
        # Parity.
        with torch.no_grad():
            y_pt = m(x).numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"image": x.numpy()})[0]
        assert y_pt.shape == y_ort.shape
        diff = np.abs(y_pt - y_ort).max()
        assert diff < 1e-4, f"SPPF parity diff {diff:.2e}"


def test_sppf_register():
    """SPPF is registered in the YAML registry."""
    from opndet.registry import get
    cls = get("SPPF")
    m = cls(in_ch=8, out_ch=8, k=5).eval()
    y = m(torch.randn(1, 8, 16, 16))
    assert y.shape == (1, 8, 16, 16)


# ---- 2. ltrb encode/decode roundtrip ---------------------------------------


def test_ltrb_encode_decode_roundtrip():
    from opndet.config import ModelConfig
    from opndet.decode import decode_ltrb
    from opndet.encode import encode_targets_ltrb

    cfg = ModelConfig()
    # Two boxes with different sizes / positions; both centers fall inside the
    # image and map to distinct cells.
    boxes = np.array([
        [40.0, 60.0, 120.0, 180.0],
        [200.0, 100.0, 350.0, 280.0],
    ], dtype=np.float32)

    tgt = encode_targets_ltrb(boxes, cfg)
    # Simulate a perfect "model output": obj=1 at positive cells, ltrb = encoded.
    H, W = cfg.img_h // cfg.stride, cfg.img_w // cfg.stride
    obj = tgt["pos"].numpy()  # [1, H, W]
    ltrb = tgt["ltrb"].numpy()  # [4, H, W]
    out = np.concatenate([obj, ltrb], axis=0)  # [5, H, W]

    dets = decode_ltrb(out, cfg.img_h, cfg.img_w, cfg.stride, threshold=0.5)
    assert len(dets) == 2

    # Each decoded box should match a GT box. Match by center-distance, then
    # check edges within tolerance (cell-quantization at GT center cell can
    # shift the decoded center by up to ~stride pixels).
    decoded = sorted([(d.x1, d.y1, d.x2, d.y2) for d in dets])
    expected = sorted([tuple(b) for b in boxes])
    for (px1, py1, px2, py2), (gx1, gy1, gx2, gy2) in zip(decoded, expected):
        # ltrb encoding is exact at the cell center → encoded then decoded
        # produces the original box edges (subject to clip in [0,1] which
        # doesn't trigger here since boxes lie inside the image).
        assert abs(px1 - gx1) < 1.0, f"x1 {px1} vs {gx1}"
        assert abs(py1 - gy1) < 1.0, f"y1 {py1} vs {gy1}"
        assert abs(px2 - gx2) < 1.0, f"x2 {px2} vs {gx2}"
        assert abs(py2 - gy2) < 1.0, f"y2 {py2} vs {gy2}"


def test_ltrb_encode_no_boxes():
    from opndet.config import ModelConfig
    from opndet.encode import encode_targets_ltrb

    cfg = ModelConfig()
    tgt = encode_targets_ltrb(np.zeros((0, 4), dtype=np.float32), cfg)
    assert tgt["pos"].sum() == 0
    assert tgt["ltrb"].abs().sum() == 0


def test_ltrb_decode_no_peaks():
    from opndet.config import ModelConfig
    from opndet.decode import decode_ltrb

    cfg = ModelConfig()
    H, W = cfg.img_h // cfg.stride, cfg.img_w // cfg.stride
    out = np.zeros((5, H, W), dtype=np.float32)
    assert decode_ltrb(out, cfg.img_h, cfg.img_w, cfg.stride) == []


# ---- 3. -pro preset build + forward + export ------------------------------


@pytest.mark.parametrize("preset", PRO_PRESETS)
def test_pro_preset_builds_and_forwards(preset: str):
    """Phase 1 → 4b: -pro presets now ship 7-channel OBB output.
    See test_obb.py for OBB-specific shape/range assertions.
    """
    from opndet.presets import resolve
    from opndet.yaml_build import build_model_from_yaml

    m = build_model_from_yaml(resolve(preset)).eval()
    c, h, w = m.input_shape
    x = torch.randn(1, c, h, w)
    with torch.no_grad():
        y = m(x)
    assert "output" in y
    out = y["output"]
    # Phase 4b: OBB head ships 7 channels (obj, l, t, r, b, sin2θ, cos2θ).
    assert out.shape == (1, 7, h // 4, w // 4), f"{preset}: bad shape {out.shape}"
    # ltrb channels are sigmoid'd → [0, 1].
    assert (out[:, 1:5] >= 0).all() and (out[:, 1:5] <= 1).all()


@pytest.mark.parametrize("preset", PRO_PRESETS)
def test_pro_preset_exports_opset13_clean(preset: str):
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
        # Phase 2: server-tier presets may use MatMul / Softmax via C2PSA. Use
        # the tier-aware allowlist.
        forbidden = ops - allowed_ops_for_tier(m.tier)
        assert not forbidden, f"{preset} (tier={m.tier}) has forbidden ops: {forbidden}"

        # Parity check (PT vs ORT).
        with torch.no_grad():
            y_pt = m(x)["output"].numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"image": x.numpy()})[0]
        diff = np.abs(y_pt - y_ort).max()
        assert diff < 1e-3, f"{preset} parity diff {diff:.2e}"


# ---- 4. base presets unaffected (regression) -------------------------------


@pytest.mark.parametrize("preset", ["bbox-f", "bbox-p", "bbox-n", "bbox-s",
                                    "bbox-m", "bbox-l", "bbox-x"])
def test_base_presets_still_build(preset: str):
    """Phase 1 must not regress base presets — they keep (cx,cy,w,h) semantics."""
    from opndet.presets import resolve
    from opndet.yaml_build import build_model_from_yaml

    m = build_model_from_yaml(resolve(preset)).eval()
    c, h, w = m.input_shape
    x = torch.randn(1, c, h, w)
    with torch.no_grad():
        y = m(x)
    out = y["output"]
    assert out.shape == (1, 5, h // 4, w // 4)
