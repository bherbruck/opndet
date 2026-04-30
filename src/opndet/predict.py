from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from opndet.dataset import letterbox
from opndet.decode import decode
from opndet.yaml_build import build_model_from_yaml


def load_model(model_config: str, ckpt: str | None, device: str = "cpu") -> torch.nn.Module:
    m = build_model_from_yaml(model_config).to(device).eval()
    if ckpt:
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        m.load_state_dict(sd["model"] if "model" in sd else sd)
    return m


def preprocess(img_bgr: np.ndarray, h: int, w: int) -> tuple[torch.Tensor, dict]:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb, _ = letterbox(img, np.zeros((0, 4), dtype=np.float32), h, w)
    f = img_lb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    f = (f - mean) / std
    t = torch.from_numpy(f.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    orig_h, orig_w = img.shape[:2]
    scale = min(w / orig_w, h / orig_h)
    pad_x = (w - int(round(orig_w * scale))) // 2
    pad_y = (h - int(round(orig_h * scale))) // 2
    return t, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "orig_w": orig_w, "orig_h": orig_h}


def unletterbox_box(x1: float, y1: float, x2: float, y2: float, info: dict) -> tuple[float, float, float, float]:
    s, px, py = info["scale"], info["pad_x"], info["pad_y"]
    return ((x1 - px) / s, (y1 - py) / s, (x2 - px) / s, (y2 - py) / s)


def predict_image(
    image_path: str | Path,
    model_config: str,
    ckpt: str | None = None,
    threshold: float = 0.3,
    device: str = "cpu",
    save_path: str | Path | None = None,
    stride: int = 4,
) -> list[dict]:
    m = load_model(model_config, ckpt, device=device)
    _, h, w = m.input_shape
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    t, info = preprocess(img_bgr, h, w)
    with torch.no_grad():
        out = m(t.to(device))
    out_t = out["output"] if isinstance(out, dict) else out
    out_np = out_t.cpu().numpy()
    dets = decode(out_np[0], h, w, stride, threshold=threshold)

    results = []
    for d in dets:
        x1, y1, x2, y2 = unletterbox_box(d.x1, d.y1, d.x2, d.y2, info)
        x1 = max(0.0, min(info["orig_w"] - 1, x1))
        y1 = max(0.0, min(info["orig_h"] - 1, y1))
        x2 = max(0.0, min(info["orig_w"] - 1, x2))
        y2 = max(0.0, min(info["orig_h"] - 1, y2))
        results.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": d.score})

    if save_path is not None:
        vis = img_bgr.copy()
        for r in results:
            cv2.rectangle(vis, (int(r["x1"]), int(r["y1"])), (int(r["x2"]), int(r["y2"])), (0, 255, 0), 2)
            cv2.putText(vis, f"{r['score']:.2f}", (int(r["x1"]), int(r["y1"]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(save_path), vis)
    return results
