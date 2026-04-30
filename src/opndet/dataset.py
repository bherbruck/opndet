from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    image_path: Path
    boxes: np.ndarray  # [N, 4] xyxy in original pixel coords
    img_w: int
    img_h: int


def load_coco_single_class(coco_path: str | Path, image_root: str | Path) -> list[Sample]:
    """Load one COCO json + image dir; collapses all categories to a single class."""
    coco_path = Path(coco_path)
    image_root = Path(image_root)
    with open(coco_path) as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    boxes_by_image: dict[int, list[list[float]]] = {im_id: [] for im_id in images_by_id}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        boxes_by_image[ann["image_id"]].append([x, y, x + w, y + h])

    samples: list[Sample] = []
    for im_id, im in images_by_id.items():
        path = image_root / im["file_name"]
        if not path.exists():
            continue
        boxes = np.array(boxes_by_image[im_id], dtype=np.float32) if boxes_by_image[im_id] else np.zeros((0, 4), dtype=np.float32)
        samples.append(Sample(image_path=path, boxes=boxes, img_w=int(im["width"]), img_h=int(im["height"])))
    return samples


def load_datasets(sources: list[dict] | list[tuple[str, str]]) -> list[Sample]:
    """Load and concatenate multiple COCO sources.

    sources: list of either {"coco": path, "images": dir} dicts or (coco, dir) tuples.
    Returns a single merged Sample list. Single-class collapse is per-source then merged.
    """
    out: list[Sample] = []
    for src in sources:
        if isinstance(src, dict):
            coco, root = src["coco"], src["images"]
        else:
            coco, root = src
        before = len(out)
        out.extend(load_coco_single_class(coco, root))
        print(f"  loaded {len(out) - before} samples from {coco}")
    return out


def letterbox(img: np.ndarray, boxes: np.ndarray, target_h: int, target_w: int, pad_value: int = 114) -> tuple[np.ndarray, np.ndarray]:
    """Resize keeping aspect ratio, pad to target. Update boxes (xyxy)."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=img.dtype)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    if boxes.shape[0] > 0:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
    return canvas, boxes


class OpndetDataset(Dataset):
    """Returns (image_tensor [3,H,W], boxes [N,4] in resized pixel coords)."""

    def __init__(
        self,
        samples: Sequence[Sample],
        img_h: int,
        img_w: int,
        augment_fn=None,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.samples = list(samples)
        self.img_h = img_h
        self.img_w = img_w
        self.aug = augment_fn
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray]:
        s = self.samples[idx]
        img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {s.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = s.boxes.copy()

        if self.aug is not None:
            img, boxes = self.aug(img, boxes)

        img, boxes = letterbox(img, boxes, self.img_h, self.img_w)

        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.img_w - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.img_h - 1)
            keep = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
            boxes = boxes[keep]

        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - self.mean) / self.std
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).contiguous()
        return img_t, boxes


def collate(batch: list[tuple[torch.Tensor, np.ndarray]]) -> tuple[torch.Tensor, list[np.ndarray]]:
    imgs = torch.stack([b[0] for b in batch], dim=0)
    boxes = [b[1] for b in batch]
    return imgs, boxes


def split_samples(
    samples: Sequence[Sample],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, f"ratios must sum to 1, got {ratios}"
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = [samples[i] for i in idx[:n_train]]
    val = [samples[i] for i in idx[n_train:n_train + n_val]]
    test = [samples[i] for i in idx[n_train + n_val:]]
    return train, val, test
