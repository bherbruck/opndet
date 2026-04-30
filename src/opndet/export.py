from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from opndet.config import ModelConfig
from opndet.model import OpndetBbox

ALLOWED_OPS = {
    "Conv", "BatchNormalization", "Relu", "Clip", "Add", "Mul", "Sub", "Div",
    "Concat", "Split", "Slice", "Resize", "MaxPool", "Sigmoid", "Tanh",
    "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
    "Cast", "Constant", "Identity", "Reshape", "Transpose", "Shape", "Gather",
    "Unsqueeze", "Squeeze", "Where", "Pad",
}


def build_pt_model(ckpt_path: str | None, cfg: ModelConfig) -> OpndetBbox:
    m = OpndetBbox(cfg).eval()
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        m.load_state_dict(sd["model"] if "model" in sd else sd)
    return m


class _InputNormalizer(torch.nn.Module):
    """Wraps a model to accept raw uint8-range fp32 inputs [0, 255] and apply
    ImageNet (or arbitrary) mean/std normalization in-graph. Makes the exported
    ONNX consumable by embedded runtimes (depthai DetectionNetwork, OpenVINO,
    TFLite Micro) that pass raw camera frames without preprocessing.
    """

    def __init__(self, model: torch.nn.Module, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.model = model
        m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1) * 255.0
        s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1) * 255.0
        self.register_buffer("mean255", m)
        self.register_buffer("std255", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean255) / self.std255
        return self.model(x)


def export_onnx(
    pt_model: OpndetBbox,
    out_path: str,
    cfg: ModelConfig | None = None,
    opset: int = 13,
    bake_input_norm: bool = False,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> str:
    cfg = cfg or pt_model.cfg
    dummy = torch.randn(1, cfg.in_ch, cfg.img_h, cfg.img_w)
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    export_model = pt_model
    if bake_input_norm:
        export_model = _InputNormalizer(pt_model, mean=mean, std=std).eval()
        # dummy input is now in raw 0-255 range to exercise the new prefix
        dummy = torch.rand(1, cfg.in_ch, cfg.img_h, cfg.img_w) * 255.0

    torch.onnx.export(
        export_model,
        dummy,
        str(out_path_p),
        input_names=["image"],
        output_names=["det"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    return str(out_path_p)


def verify_onnx(model_path: str, pt_model: OpndetBbox, atol: float = 1e-4) -> dict:
    cfg = pt_model.cfg
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    used_ops = {n.op_type for n in onnx_model.graph.node}
    forbidden = used_ops - ALLOWED_OPS
    if forbidden:
        raise RuntimeError(f"forbidden ops in graph: {sorted(forbidden)}")

    x = torch.randn(1, cfg.in_ch, cfg.img_h, cfg.img_w)
    with torch.no_grad():
        y_pt = pt_model(x).numpy()

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {"image": x.numpy()})[0]

    diff = np.abs(y_pt - y_ort).max()
    cos = float(
        (y_pt.flatten() @ y_ort.flatten())
        / (np.linalg.norm(y_pt) * np.linalg.norm(y_ort) + 1e-12)
    )

    return {
        "ops": sorted(used_ops),
        "forbidden": sorted(forbidden),
        "max_abs_diff": float(diff),
        "cosine_sim": cos,
        "pt_shape": y_pt.shape,
        "ort_shape": y_ort.shape,
        "atol_pass": bool(diff < atol),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="opndet.onnx")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()
    cfg = ModelConfig()
    m = build_pt_model(args.ckpt, cfg)
    path = export_onnx(m, args.out, cfg=cfg, opset=args.opset)
    print(f"exported: {path}")
    info = verify_onnx(path, m)
    print(f"ops: {info['ops']}")
    print(f"max_abs_diff: {info['max_abs_diff']:.2e}  cosine_sim: {info['cosine_sim']:.6f}  parity: {info['atol_pass']}")


if __name__ == "__main__":
    main()
