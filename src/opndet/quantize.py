from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


def _list_images(image_dir: str | Path, n: int, seed: int = 0) -> list[Path]:
    image_dir = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"no images found under {image_dir}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n]


def _preprocess(img_path: Path, h: int, w: int) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    src_h, src_w = img.shape[:2]
    scale = min(w / src_w, h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    pad_x = (w - new_w) // 2
    pad_y = (h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    f = canvas.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    f = (f - mean) / std
    return f.transpose(2, 0, 1)[None, ...]   # [1,3,H,W]


class _ImageCalibReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], h: int, w: int, input_name: str):
        self.paths = image_paths
        self.h, self.w = h, w
        self.input_name = input_name
        self._iter = iter(self.paths)

    def get_next(self):
        try:
            p = next(self._iter)
        except StopIteration:
            return None
        return {self.input_name: _preprocess(p, self.h, self.w)}

    def rewind(self):
        self._iter = iter(self.paths)


def _input_shape_from_onnx(model_path: str) -> tuple[str, int, int, int, int]:
    m = onnx.load(model_path)
    inp = m.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4:
        raise ValueError(f"expected 4D NCHW input, got shape {dims}")
    return inp.name, dims[0], dims[1], dims[2], dims[3]


def quantize_onnx(
    onnx_path: str,
    out_path: str,
    calib_dir: str,
    n_calib: int = 100,
    quant_format: str = "qdq",
    seed: int = 0,
) -> dict:
    """Static int8 PTQ. quant_format: 'qdq' (OpenVINO + ORT compatible) or 'qoperator'."""
    name, _, _, h, w = _input_shape_from_onnx(onnx_path)
    calib_paths = _list_images(calib_dir, n_calib, seed=seed)
    reader = _ImageCalibReader(calib_paths, h, w, input_name=name)

    fmt = QuantFormat.QDQ if quant_format == "qdq" else QuantFormat.QOperator
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=onnx_path,
        model_output=str(out_path_p),
        calibration_data_reader=reader,
        quant_format=fmt,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=None,  # default — picks Conv, MatMul, Add, Mul, etc.
    )

    fp32_size = Path(onnx_path).stat().st_size
    int8_size = out_path_p.stat().st_size
    return {
        "fp32_path": onnx_path,
        "int8_path": str(out_path_p),
        "fp32_bytes": fp32_size,
        "int8_bytes": int8_size,
        "compression": fp32_size / max(1, int8_size),
        "n_calibration_images": len(calib_paths),
    }


def parity_check(
    fp32_path: str,
    int8_path: str,
    calib_dir: str,
    n_check: int = 16,
    seed: int = 1,
) -> dict:
    """Run both ONNX models on the same images, report mean abs diff on the obj channel
    (most sensitive to quantization due to peak suppression)."""
    name, _, _, h, w = _input_shape_from_onnx(fp32_path)
    paths = _list_images(calib_dir, n_check, seed=seed)
    sess32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    diffs_obj = []
    diffs_all = []
    for p in paths:
        x = _preprocess(p, h, w)
        y32 = sess32.run(None, {name: x})[0]
        y8 = sess8.run(None, {name: x})[0]
        diffs_obj.append(float(np.abs(y32[0, 0] - y8[0, 0]).mean()))
        diffs_all.append(float(np.abs(y32 - y8).mean()))
    return {
        "n": len(paths),
        "obj_mean_abs_diff": float(np.mean(diffs_obj)),
        "all_mean_abs_diff": float(np.mean(diffs_all)),
    }
