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

# Single-threaded cv2 inside DataLoader workers. Without this, cv2's internal
# OpenMP/TBB pools fork-explode when num_workers is large, causing intermittent
# hangs between epochs (workers stuck inside cv2 internal mutexes).
cv2.setNumThreads(0)


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
        x, y, w, h = (float(v) for v in ann["bbox"])
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
    c = img.shape[2] if img.ndim == 3 else 1
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, c), pad_value, dtype=img.dtype) if img.ndim == 3 else np.full((target_h, target_w), pad_value, dtype=img.dtype)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    if boxes.shape[0] > 0:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
    return canvas, boxes


class OpndetDataset(Dataset):
    """Returns (image_tensor [C,H,W], boxes_xyxy, encoded_targets_dict).

    C=3 for snapshot models. C=4 when in_ch=4 (temporal-prior variants):
    channels 0:3 are normalized RGB and channel 3 is the prior heatmap
    upsampled from stride-4 to (H,W) via nearest. Prior is unnormalized,
    in [0,1]. If prior_synth is None and in_ch=4, prior channel is zeros
    (cold-start eval).

    GT encoding runs inside the worker so the main training loop just stacks
    pre-encoded tensors — no CPU work between batches blocks the GPU.

    cache_images=True keeps decoded uint8 RGB arrays in worker RAM after first
    read; subsequent epochs skip the JPEG decode entirely. Aug still runs each
    epoch on a fresh copy of the cached array.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        img_h: int,
        img_w: int,
        augment_fn=None,
        encode_fn=None,
        cache_images: bool = False,
        mosaic_prob: float = 0.0,
        min_visible_frac: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        in_ch: int = 3,
        prior_synth=None,
        stride: int = 4,
    ):
        self.samples = list(samples)
        self.img_h = img_h
        self.img_w = img_w
        self.aug = augment_fn
        self.encode = encode_fn
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.cache_images = cache_images
        self._cache: dict[int, np.ndarray] = {}
        self.mosaic_prob = mosaic_prob
        self.min_visible_frac = min_visible_frac
        self.in_ch = int(in_ch)
        self.prior_synth = prior_synth
        self.stride = int(stride)
        if self.in_ch not in (3, 4):
            raise ValueError(f"OpndetDataset supports in_ch in (3, 4); got {self.in_ch}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, idx: int, path) -> np.ndarray:
        if self.cache_images and idx in self._cache:
            return self._cache[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.cache_images:
            self._cache[idx] = img
        return img

    def _mosaic(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """4-image mosaic. Output dims = (img_h, img_w). Each quadrant filled by one
        sample, scaled to fit. Boxes transformed and filtered by min_visible_frac.
        """
        H, W = self.img_h, self.img_w
        rng = random.Random()
        idxs = [idx] + [rng.randrange(len(self.samples)) for _ in range(3)]
        cy = rng.randint(H // 4, 3 * H // 4)
        cx = rng.randint(W // 4, 3 * W // 4)
        # quadrant target rects: TL, TR, BL, BR
        rects = [(0, 0, cx, cy), (cx, 0, W, cy), (0, cy, cx, H), (cx, cy, W, H)]
        canvas = np.full((H, W, 3), 114, dtype=np.uint8)
        all_boxes = []
        for ix, (x1, y1, x2, y2) in zip(idxs, rects):
            qw, qh = x2 - x1, y2 - y1
            if qw <= 0 or qh <= 0:
                continue
            s = self.samples[ix]
            src = self._load_rgb(ix, s.image_path)
            sh, sw = src.shape[:2]
            # scale to fit quadrant preserving aspect
            scale = min(qw / sw, qh / sh)
            new_w, new_h = int(round(sw * scale)), int(round(sh * scale))
            if new_w <= 0 or new_h <= 0:
                continue
            resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            ox = x1 + (qw - new_w) // 2
            oy = y1 + (qh - new_h) // 2
            canvas[oy:oy + new_h, ox:ox + new_w] = resized
            if s.boxes.shape[0] > 0:
                bx = s.boxes.copy()
                orig_w = (bx[:, 2] - bx[:, 0]).clip(min=0)
                orig_h = (bx[:, 3] - bx[:, 1]).clip(min=0)
                orig_area = orig_w * orig_h
                bx[:, [0, 2]] = bx[:, [0, 2]] * scale + ox
                bx[:, [1, 3]] = bx[:, [1, 3]] * scale + oy
                # clip to quadrant (so boxes don't bleed across the cut center)
                bx[:, 0] = bx[:, 0].clip(x1, x2)
                bx[:, 1] = bx[:, 1].clip(y1, y2)
                bx[:, 2] = bx[:, 2].clip(x1, x2)
                bx[:, 3] = bx[:, 3].clip(y1, y2)
                new_w_px = (bx[:, 2] - bx[:, 0]).clip(min=0)
                new_h_px = (bx[:, 3] - bx[:, 1]).clip(min=0)
                new_area = new_w_px * new_h_px
                # min_visible_frac is in image-coord area, so compare against scaled-orig
                scaled_orig_area = orig_area * (scale ** 2)
                keep = (scaled_orig_area <= 0) | (new_area / np.maximum(scaled_orig_area, 1e-9) >= self.min_visible_frac)
                bx = bx[keep]
                if bx.shape[0]:
                    all_boxes.append(bx)
        out_boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        return canvas, out_boxes

    def _finish(self, img: np.ndarray, boxes: np.ndarray, do_letterbox: bool = True):
        if self.aug is not None:
            img, boxes = self.aug(img, boxes)

        # always letterbox — aug (rotate90 with k=1 or 3) can transpose dims, so the
        # final canvas size must be reasserted regardless of where img came from.
        if do_letterbox or img.shape[:2] != (self.img_h, self.img_w):
            img, boxes = letterbox(img, boxes, self.img_h, self.img_w)

        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.img_w - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.img_h - 1)
            keep = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
            boxes = boxes[keep]

        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - self.mean) / self.std
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).contiguous()

        if self.in_ch == 4:
            if self.prior_synth is not None:
                prior = self.prior_synth(boxes, self.img_h, self.img_w)
            else:
                prior = np.zeros((self.img_h // self.stride, self.img_w // self.stride), dtype=np.float32)
            prior_full = cv2.resize(prior, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
            prior_t = torch.from_numpy(prior_full).unsqueeze(0).contiguous()
            img_t = torch.cat([img_t, prior_t], dim=0)

        targets = self.encode(boxes) if self.encode is not None else None
        return img_t, boxes, targets

    def __getitem__(self, idx: int):
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            img, boxes = self._mosaic(idx)
            return self._finish(img, boxes, do_letterbox=False)
        s = self.samples[idx]
        img = self._load_rgb(idx, s.image_path)
        if self.cache_images:
            img = img.copy()
        boxes = s.boxes.copy()
        return self._finish(img, boxes, do_letterbox=True)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    boxes = [b[1] for b in batch]
    targets_list = [b[2] for b in batch]
    if targets_list[0] is None:
        return imgs, boxes, None
    targets = {k: torch.stack([t[k] for t in targets_list], dim=0) for k in targets_list[0]}
    return imgs, boxes, targets


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
