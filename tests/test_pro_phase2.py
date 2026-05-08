"""Phase 2 (ROADMAP §1.8) tests:
  - SiLU parity (PT vs ORT, opset 13; produces Mul+Sigmoid)
  - C2PSA parity (PT vs ORT, opset 13)
  - ResizeBilinear2xHalfPixel parity + tier check (only server-tier accepts it)
  - Server-tier -pro presets (m/l/x) build + forward + export with tier-allowlist
  - Edge-tier -pro presets (f/p/n/s) still build + forward + export clean (no regression)
  - Tier rejection: edge-tier preset with server-only ops injected fails verify
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


EDGE_PRESETS = ["bbox-f-pro", "bbox-p-pro", "bbox-n-pro", "bbox-s-pro"]
SERVER_PRESETS = ["bbox-m-pro", "bbox-l-pro", "bbox-x-pro"]


def _export(m: nn.Module, x: torch.Tensor, path: str) -> None:
    torch.onnx.export(
        m, x, path,
        input_names=["image"], output_names=["output"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes=None, dynamo=False,
    )


def _ops(path: str) -> set[str]:
    return {n.op_type for n in onnx.load(path).graph.node}


# ---- 1. SiLU parity --------------------------------------------------------


def test_silu_register_and_decomposes_to_mul_sigmoid():
    from opndet.registry import get
    cls = get("SiLU")
    m = cls().eval()

    x = torch.randn(1, 8, 16, 16)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "silu.onnx")
        _export(m, x, path)
        ops = _ops(path)
        # SiLU exports as Mul + Sigmoid (both opset-13 base ALLOWED_OPS).
        assert "Mul" in ops, f"expected Mul, got {ops}"
        assert "Sigmoid" in ops, f"expected Sigmoid, got {ops}"

        with torch.no_grad():
            y_pt = m(x).numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"image": x.numpy()})[0]
        diff = np.abs(y_pt - y_ort).max()
        assert diff < 1e-4, f"SiLU parity diff {diff:.2e}"


def test_convbnact_silu_arg():
    from opndet.yaml_build import ConvBnAct
    m = ConvBnAct(in_ch=4, out_ch=8, k=3, act="silu").eval()
    y = m(torch.randn(1, 4, 16, 16))
    assert y.shape == (1, 8, 16, 16)


def test_convbnact_default_relu6_unchanged():
    from opndet.yaml_build import ConvBnAct
    m = ConvBnAct(in_ch=4, out_ch=8, k=3).eval()
    # walk the inner block; last child should be ReLU6 (default).
    last = list(m.block.children())[-1]
    assert isinstance(last, nn.ReLU6), f"default activation regressed: {type(last)}"


# ---- 2. C2PSA parity -------------------------------------------------------


def test_c2psa_register_and_forward():
    from opndet.registry import get
    cls = get("C2PSA")
    m = cls(in_ch=16, num_heads=4).eval()
    x = torch.randn(1, 16, 24, 32)
    y = m(x)
    assert y.shape == x.shape


def test_c2psa_parity_opset13():
    from opndet.primitives import C2PSA

    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Conv2d(3, 16, 3, 2, 1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU6(inplace=True),
        C2PSA(in_ch=16, num_heads=4),
    ).eval()

    x = torch.randn(1, 3, 48, 64)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "c2psa.onnx")
        _export(m, x, path)
        ops = _ops(path)
        # C2PSA emits MatMul + Softmax + Reshape + Transpose + Conv + Add + Mul.
        # All server-tier-allowed; none in base edge ALLOWED_OPS for MatMul/Softmax.
        assert "MatMul" in ops, f"C2PSA missing MatMul: {ops}"
        assert "Softmax" in ops, f"C2PSA missing Softmax: {ops}"
        # Server allowlist accepts these.
        from opndet.export import allowed_ops_for_tier
        server_allowed = allowed_ops_for_tier("server")
        forbidden = ops - server_allowed
        assert not forbidden, f"C2PSA emits server-disallowed ops: {forbidden}"
        # Edge allowlist rejects them.
        edge_allowed = allowed_ops_for_tier("edge")
        edge_forbidden = ops - edge_allowed
        assert edge_forbidden, "edge tier must reject C2PSA ops"
        assert "MatMul" in edge_forbidden or "Softmax" in edge_forbidden

        with torch.no_grad():
            y_pt = m(x).numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"image": x.numpy()})[0]
        diff = np.abs(y_pt - y_ort).max()
        assert diff < 1e-4, f"C2PSA parity diff {diff:.2e}"


# ---- 3. ResizeBilinear2xHalfPixel parity + tier check -----------------------


def test_resize_halfpixel_parity_and_attr():
    from opndet.primitives import ResizeBilinear2xHalfPixel

    torch.manual_seed(0)
    m = ResizeBilinear2xHalfPixel().eval()
    x = torch.randn(1, 4, 12, 16)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "rhp.onnx")
        _export(m, x, path)
        om = onnx.load(path)
        # at least one Resize with coord_transform != asymmetric.
        modes = []
        for n in om.graph.node:
            if n.op_type == "Resize":
                for a in n.attribute:
                    if a.name == "coordinate_transformation_mode":
                        modes.append(a.s.decode("utf-8") if isinstance(a.s, bytes) else str(a.s))
        assert any(mm and mm != "asymmetric" for mm in modes), f"got modes={modes}; expected non-asymmetric"

        with torch.no_grad():
            y_pt = m(x).numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"image": x.numpy()})[0]
        diff = np.abs(y_pt - y_ort).max()
        assert diff < 1e-4, f"halfpixel parity diff {diff:.2e}"


def test_resize_asymmetric_passes_edge_tier():
    """Sanity: the existing asymmetric ResizeNearest2x must pass edge-tier check."""
    from opndet.export import check_resize_attrs
    from opndet.primitives import ResizeNearest2x

    m = ResizeNearest2x().eval()
    x = torch.randn(1, 4, 12, 16)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "rasym.onnx")
        _export(m, x, path)
        om = onnx.load(path)
        # must not raise
        check_resize_attrs(om, tier="edge")


def test_resize_halfpixel_rejected_on_edge_tier():
    """Tier check: half_pixel resize must be rejected when tier=edge."""
    from opndet.export import check_resize_attrs
    from opndet.primitives import ResizeBilinear2xHalfPixel

    m = ResizeBilinear2xHalfPixel().eval()
    x = torch.randn(1, 4, 12, 16)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "rhp.onnx")
        _export(m, x, path)
        om = onnx.load(path)
        with pytest.raises(RuntimeError, match=r"coord_transform"):
            check_resize_attrs(om, tier="edge")
        # server tier accepts.
        check_resize_attrs(om, tier="server")


# ---- 4. Server-tier -pro presets export with tier allowlist ----------------


@pytest.mark.parametrize("preset", SERVER_PRESETS)
def test_server_pro_preset_exports_with_attention(preset: str):
    from opndet.export import allowed_ops_for_tier, check_resize_attrs
    from opndet.presets import resolve
    from opndet.yaml_build import build_model_from_yaml

    m = build_model_from_yaml(resolve(preset)).eval()
    assert m.tier == "server", f"{preset} must declare tier: server"

    c, h, w = m.input_shape
    x = torch.randn(1, c, h, w)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / f"{preset}.onnx")
        _export(m, x, path)
        ops = _ops(path)
        # server-tier allowlist must cover them.
        allowed = allowed_ops_for_tier("server")
        forbidden = ops - allowed
        assert not forbidden, f"{preset} forbidden under server tier: {forbidden}"
        # And C2PSA must contribute MatMul + Softmax (proves attention is wired in).
        assert "MatMul" in ops and "Softmax" in ops, \
            f"{preset} missing attention ops: {ops}"
        # Resize half_pixel check passes at server tier.
        check_resize_attrs(onnx.load(path), tier="server")

        with torch.no_grad():
            y_pt = m(x)["output"].numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"image": x.numpy()})[0]
        diff = np.abs(y_pt - y_ort).max()
        assert diff < 1e-3, f"{preset} parity diff {diff:.2e}"


# ---- 5. Edge-tier -pro presets unaffected (no regression) ------------------


@pytest.mark.parametrize("preset", EDGE_PRESETS)
def test_edge_pro_preset_no_regression(preset: str):
    """Edge -pro presets must still build, forward, export under the strict
    edge ALLOWED_OPS (Phase 1 contract)."""
    from opndet.export import ALLOWED_OPS, allowed_ops_for_tier, check_resize_attrs
    from opndet.presets import resolve
    from opndet.yaml_build import build_model_from_yaml

    m = build_model_from_yaml(resolve(preset)).eval()
    assert m.tier == "edge", f"{preset} must declare tier: edge"

    c, h, w = m.input_shape
    x = torch.randn(1, c, h, w)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / f"{preset}.onnx")
        _export(m, x, path)
        ops = _ops(path)
        forbidden = ops - allowed_ops_for_tier("edge")
        assert not forbidden, f"{preset} edge regression: {forbidden}"
        # MatMul / Softmax / SiLU-fused must NOT appear on edge tier.
        assert "MatMul" not in ops, f"{preset} unexpectedly has MatMul"
        assert "Softmax" not in ops, f"{preset} unexpectedly has Softmax"
        # No half_pixel resize.
        check_resize_attrs(onnx.load(path), tier="edge")
        # ALLOWED_OPS (the base edge set) accepts everything.
        assert not (ops - ALLOWED_OPS), f"{preset} ops outside base ALLOWED_OPS: {ops - ALLOWED_OPS}"


# ---- 6. Tier rejection: edge model with server-only op fails verify --------


def test_edge_tier_rejects_injected_halfpixel():
    """Build a tiny edge-style model with a half_pixel Resize injected and
    verify the export-time tier check rejects it.
    """
    from opndet.export import check_resize_attrs
    from opndet.primitives import ResizeBilinear2xHalfPixel

    m = nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        ResizeBilinear2xHalfPixel(),
    ).eval()

    x = torch.randn(1, 3, 16, 16)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "edge_inject.onnx")
        _export(m, x, path)
        om = onnx.load(path)
        with pytest.raises(RuntimeError, match=r"edge-tier"):
            check_resize_attrs(om, tier="edge")


def test_edge_tier_rejects_injected_attention():
    """Build a tiny edge-style model with C2PSA injected; the edge allowlist
    must reject the resulting MatMul/Softmax ops.
    """
    from opndet.export import allowed_ops_for_tier
    from opndet.primitives import C2PSA

    m = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        C2PSA(in_ch=16, num_heads=4),
    ).eval()

    x = torch.randn(1, 3, 16, 16)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "edge_inject_attn.onnx")
        _export(m, x, path)
        ops = _ops(path)
        edge_allowed = allowed_ops_for_tier("edge")
        forbidden = ops - edge_allowed
        assert forbidden, "edge tier must flag attention ops as forbidden"
        # The classic offenders.
        assert ("MatMul" in forbidden) or ("Softmax" in forbidden), \
            f"expected MatMul/Softmax in forbidden set, got {forbidden}"


def test_default_tier_is_edge():
    """A YAML without a `tier:` field defaults to edge — preserves backwards
    compat for user-authored presets that predate Phase 2.
    """
    import textwrap
    from opndet.yaml_build import build_model_from_yaml

    yaml_str = textwrap.dedent("""
    model:
      in_ch: 3
      img_h: 32
      img_w: 32
      layers:
        - {name: stem, from: 0, module: ConvBnAct, args: {in_ch: 3, out_ch: 4, k: 3}}
      outputs:
        - {name: output, from: stem, activation: none}
    """)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "no_tier.yaml"
        path.write_text(yaml_str)
        m = build_model_from_yaml(path)
        assert m.tier == "edge", f"default tier regression: {m.tier}"
